"""由契约脚本调用的课程回归检查，复用已启动的 SimulationApp。"""

from types import SimpleNamespace

import torch


def verify_odin_design(check):
    from rp_wd_rl.tasks.locomotion.velocity.mdp.observations import odin_depth_memory
    from rp_wd_rl.tasks.locomotion.velocity.mdp.odin_course import (
        OdinCourseVelocityCommand, odin_safe_terrain_levels, odin_joint_target_excess,
    )
    from isaaclab.envs.mdp.commands import UniformVelocityCommand
    from unittest.mock import patch
    from rp_wd_rl.tasks.locomotion.velocity.odin_terrains import (
        OdinCourseTerrainCfg, odin_course_terrain, ODIN_BASIC_TERRAINS_CFG,
    )
    from benchmark_metrics import benchmark_attempt_metrics
    from rp_wd_rl.tasks.locomotion.velocity.mdp.rewards import _course_scale
    scaling_env = SimpleNamespace(command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(course_mask=torch.tensor([False, True]))))
    check("平地滚动约束保持全额，专项允许抬轮", torch.equal(
        _course_scale(scaling_env, torch.ones(2), .25), torch.tensor([1., .25])))

    # 重现真实Manager顺序：command.reset执行时回合步数尚未清零。
    command = object.__new__(OdinCourseVelocityCommand)
    command._env = SimpleNamespace(num_envs=2, device="cpu",
        episode_length_buf=torch.tensor([700, 1100]),
        scene=SimpleNamespace(env_origins=torch.tensor([[12., 3., 0.], [6., 8., .2]])))
    command.robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[12., 3., .4], [6., 8., .6]])))
    command.cfg = SimpleNamespace(goal_distance=4.4)
    command.goal_w = torch.full((2, 3), -100.)
    command.previous_distance = torch.tensor([999., 777.])
    command.progress_speed = torch.tensor([8., 9.])
    command.max_nonfoot_force = torch.tensor([300., 400.])
    command._update_command = lambda: None
    with patch.object(UniformVelocityCommand, "reset", return_value={}):
        command.reset(torch.tensor([0]))
        check("非零回合步数时复位仍刷新终点、距离和撞击峰值",
              torch.allclose(command.goal_w[0], torch.tensor([16.4, 3., 0.]))
              and abs(command.previous_distance[0].item() - 4.4) < 1e-5
              and command.progress_speed[0] == 0 and command.max_nonfoot_force[0] == 0)
        check("单环境复位保留其余环境的赛道状态",
              command.previous_distance[1] == 777 and command.max_nonfoot_force[1] == 400)
        command.reset()
        check("全环境复位和部分复位采用相同初始化规则",
              torch.allclose(command.previous_distance, torch.full((2,), 4.4)))
    limits = torch.tensor([[[-1., 1.], [-2., 0.]]])
    target_env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
        joint_pos_target=torch.tensor([[2., -1.]]), joint_pos_limits=limits))})
    check("无效目标约束只统计越过物理关节范围的部分", torch.allclose(
        odin_joint_target_excess(target_env, SimpleNamespace(name="robot", joint_ids=[0, 1])), torch.tensor([.5])))

    camera = SimpleNamespace(frame=torch.zeros(2, dtype=torch.long), data=SimpleNamespace(
        output={"distance_to_image_plane": torch.full((2, 4, 6, 1), 0.3)}))
    env = SimpleNamespace(num_envs=2, device="cpu", common_step_counter=0,
                          episode_length_buf=torch.zeros(2, dtype=torch.long),
                          scene=SimpleNamespace(sensors={"odin_depth_camera": camera}))
    params = dict(grid_size=(2, 3), rotation_k=0, noise_std=0, dropout_prob=0,
                  random_crop=False, memory_frames=3, memory_stride=1,
                  augmentation_warmup_steps=0, augmentation_ramp_steps=1)
    first = odin_depth_memory(env, **params).clone()
    env.common_step_counter = 1
    env.episode_length_buf[:] = 1
    camera.frame += 1
    camera.data.output["distance_to_image_plane"][:] = 0.6
    second = odin_depth_memory(env, **params).clone()
    check("深度只在新相机帧到达时推进，顺序旧→新",
          torch.allclose(second[:, :-1], first[:, 1:]) and torch.allclose(second[:, -1], torch.full((2, 2, 3), .1)))
    camera.data.output["distance_to_image_plane"][:] = 1.2
    check("同一相机帧重复取观测不改变历史", torch.equal(odin_depth_memory(env, **params), second))
    env.common_step_counter = 2
    env.episode_length_buf[:] = 2
    camera.frame += 1
    check("模拟丢包保持完整历史而非复制旧帧推进", torch.equal(
        odin_depth_memory(env, **params, packet_loss_prob=1.0), second))
    env.common_step_counter = 3
    env.episode_length_buf[0] = 0
    camera.frame[0] += 1
    reset = odin_depth_memory(env, **params)
    check("单环境重置不污染其他环境历史",
          torch.allclose(reset[0], torch.full_like(reset[0], .2)) and torch.equal(reset[1], second[1]))

    # 8通道不增加ONNX维度，通过稀疏取帧扩大后腿地形记忆时域。
    params["memory_stride"] = 3
    env.episode_length_buf[:] = 0
    for index in range(7):
        env.common_step_counter = index + 10
        env.episode_length_buf[:] = index
        camera.frame[:] = index + 10
        camera.data.output["distance_to_image_plane"][:] = (index + 1) * 0.3
        sparse = odin_depth_memory(env, **params, memory_key="stride_probe")
    check("稀疏历史每3个相机帧取1张，最新帧仍即时更新",
          torch.allclose(sparse[0, :, 0, 0], torch.tensor([.05, .20, .35]), atol=1e-6))

    for kind in ("stairs_up", "stairs_down"):
        cfg = OdinCourseTerrainCfg(course_kind=kind, size=(6.0, 6.0))
        meshes, origin = odin_course_terrain(1.0, cfg)
        top = max(mesh.vertices[:, 2].max() for mesh in meshes)
        check(f"{kind}困难端为4级×12cm实体台阶", abs(top - .48) < 1e-6 and all(mesh.is_watertight for mesh in meshes))
        check(f"{kind}出生平台高度正确", abs(origin[2] - (.48 if kind == "stairs_down" else 0.0)) < 1e-6)
        easy, easy_origin = odin_course_terrain(0.0, cfg)
        check(f"{kind}简单端仅一级1cm", abs(max(mesh.vertices[:, 2].max() for mesh in easy) - .01) < 1e-6
              and abs(easy_origin[2] - (.01 if kind == "stairs_down" else 0.0)) < 1e-6)
        basic = ODIN_BASIC_TERRAINS_CFG.sub_terrains[kind]
        basic.size = (6., 6.)
        meshes, origin = odin_course_terrain(1.0, basic)
        check(f"{kind}基础专项上限为4级×8cm", abs(max(mesh.vertices[:, 2].max() for mesh in meshes) - .32) < 1e-6)

    levels = torch.tensor([0, 0, 1, 1])
    terrain = SimpleNamespace(terrain_levels=levels, terrain_types=torch.zeros(4, dtype=torch.long),
                              terrain_origins=torch.zeros(3, 1, 3), env_origins=torch.zeros(4, 3))
    def update(ids, up, down):
        terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + up.long() - down.long()).clamp_min(0)
    terrain.update_env_origins = update
    term = SimpleNamespace(refresh_progress=lambda: None, previous_distance=torch.tensor([.1, .1, 2., .1]),
                           cfg=SimpleNamespace(goal_tolerance=.35), max_nonfoot_force=torch.tensor([0., 0., 0., 150.]),
                           course_mask=torch.ones(4, dtype=torch.bool))
    manager = SimpleNamespace(
        terminated=torch.tensor([True, True, False, True]), active_terms=["fall", "goal_reached"],
        get_term=lambda name: (torch.tensor([False, True, False, False]) if name == "fall"
                               else torch.tensor([True, True, False, True])),
        get_term_cfg=lambda name: SimpleNamespace(time_out=False),
    )
    fake = SimpleNamespace(device="cpu", command_manager=SimpleNamespace(get_term=lambda name: term),
                           scene=SimpleNamespace(terrain=terrain), common_step_counter=24,
                           termination_manager=manager,
                           episode_length_buf=torch.full((4,), 100))
    odin_safe_terrain_levels(fake, torch.arange(4), ((0, 0), (24, 2)))
    check("到点安全升级、摔倒/失败降级、撞击到点不升级", levels.tolist() == [1, 0, 0, 1], str(levels.tolist()))
    levels[:] = 1
    term.max_nonfoot_force[:] = 0
    manager.get_term = lambda name: torch.zeros(4, dtype=torch.bool) if name == "fall" else torch.ones(4, dtype=torch.bool)
    odin_safe_terrain_levels(fake, torch.arange(4), ((0, 2),), rehearsal_stride=2, rehearsal_max_level=0)
    check("复习环境保持低难度，其他环境仍按安全成功升级", levels.tolist() == [0, 2, 0, 2], str(levels.tolist()))

    rows = []
    for episode, distance, terminated in ((0, 1.6, 0), (1, 3.4, 1), (2, 3.3, 0)):
        rows.append(dict(env_id=0, episode_id=episode, terrain_name="step_up_06cm",
                         forward_progress_m=distance, lateral_offset_m=0, terminated=terminated,
                         timeout=0, nonfoot_force_peak_n=0, wheel_force_peak_n=100,
                         root_clearance_m=.419, tilt_deg=0))
    result = benchmark_attempt_metrics(rows)
    check("通过率不把1.5m靠近或终止后的位移计为成功", result["completed_attempts"] == 1 and result["attempts"] == 3)
