"""由契约脚本调用的课程回归检查，复用已启动的 SimulationApp。"""

from types import SimpleNamespace

import torch


def verify_odin_design(check):
    from rp_wd_rl.tasks.locomotion.velocity.mdp.observations import odin_depth_memory
    from rp_wd_rl.tasks.locomotion.velocity.mdp.odin_course import odin_safe_terrain_levels
    from rp_wd_rl.tasks.locomotion.velocity.odin_terrains import OdinCourseTerrainCfg, odin_course_terrain
    from benchmark_metrics import benchmark_attempt_metrics
    from rp_wd_rl.tasks.locomotion.velocity.mdp.rewards import _course_scale
    scaling_env = SimpleNamespace(command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(course_mask=torch.tensor([False, True]))))
    check("平地滚动约束保持全额，专项允许抬轮", torch.equal(
        _course_scale(scaling_env, torch.ones(2), .25), torch.tensor([1., .25])))

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

    rows = []
    for episode, distance, terminated in ((0, 1.6, 0), (1, 3.4, 1), (2, 3.3, 0)):
        rows.append(dict(env_id=0, episode_id=episode, terrain_name="step_up_06cm",
                         forward_progress_m=distance, lateral_offset_m=0, terminated=terminated,
                         timeout=0, nonfoot_force_peak_n=0, wheel_force_peak_n=100,
                         root_clearance_m=.419, tilt_deg=0))
    result = benchmark_attempt_metrics(rows)
    check("通过率不把1.5m靠近或终止后的位移计为成功", result["completed_attempts"] == 1 and result["attempts"] == 3)
