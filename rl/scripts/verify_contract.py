"""sim2real 契约复核: 训练侧 (Isaac Lab) 与部署侧 (ROS2 推理节点) 的接口必须逐项一致.

跑法:
    /media/rmer/data/IsaacLab/isaaclab.sh -p rl/scripts/verify_contract.py

校验两类内容:
  A. 静态契约 (毫秒级, 读 YAML/URDF 对比训练侧常量)
     - 观测布局与维度 (53 × 帧栈 3 = 159)
     - 动作缩放 / 轮子索引 / USD↔URDF 关节顺序映射 / 默认姿态
     - 控制周期 dt × decimation
  B. 运行时语义 (需要启动仿真, 约 1 分钟)
     - articulation 关节顺序 == 部署端 usd2urdf 假设的顺序
     - 观测/动作维度
     - 动作映射: 给的 raw 动作 → 关节目标位置/速度 (含默认角度偏置与缩放)
     - 帧栈排布: term-major + 每个分量旧→新 (最容易写错、也最致命的一项)

任何一项 FAIL 都意味着上实机时策略输入/输出会对不上.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

# 输出重定向到文件时也实时刷新
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Verify RP_wd sim2real contract.")
parser.add_argument("--num_envs", type=int, default=2, help="校验用环境数 (小一点即可).")
parser.add_argument("--no-sim", action="store_true", default=False, help="只跑静态检查 (不启动仿真).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 校验脚本一律无头运行, 不需要图形窗口
args_cli.headless = True

if not args_cli.no_sim:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

import yaml  # noqa: E402
from sim_shutdown import shutdown  # noqa: E402

# 契约常量模块不依赖 isaaclab, 所以静态检查部分不需要启动仿真
from rp_wd_rl.assets.rp_wd_contract import (  # noqa: E402
    ACTION_CLIP,
    ACTION_SCALE,
    CMD_HEIGHT_RANGE,
    CMD_PITCH_RANGE,
    CMD_ROLL_RANGE,
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_WZ_RANGE,
    DAMPING,
    DECIMATION,
    DEFAULT_JOINT_POS,
    DEPLOY_CONFIG_PATH,
    DEPLOY_ROBOT_YAML_PATH,
    FRAME_STACK,
    JOINT_ORDER,
    JOINT_TYPES,
    NUM_JOINTS,
    OBS_LAYOUT,
    ONNX_INPUT_DIM,
    ONNX_OUTPUT_DIM,
    RP_WD_URDF_PATH,
    SIM_DT,
    STIFFNESS,
    URDF_JOINT_ORDER,
    USD2URDF,
    WHEEL_JOINT_INDICES,
    WHEEL_JOINT_NAMES,
)

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """记录一条检查结果."""
    print(f"  [{'OK' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def expand(value_by_type: dict[str, float], order: list[str]) -> list[float]:
    """按关节顺序把 {类型: 值} 展开成逐关节列表."""
    out = []
    for name in order:
        jtype = next(t for t in JOINT_TYPES if name.endswith(f"_{t}_JOINT"))
        out.append(value_by_type[jtype])
    return out


def close(a, b, tol: float = 1e-6) -> bool:
    """逐元素比较两个序列."""
    return len(a) == len(b) and all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# A. 静态契约
# ---------------------------------------------------------------------------
print("\n=== A. 静态契约 (YAML/URDF vs 训练侧常量) ===")

with open(DEPLOY_CONFIG_PATH, "r", encoding="utf-8") as f:
    deploy = yaml.safe_load(f)["inference_node"]["ros__parameters"]

# A1. 观测布局
layout = [part.strip() for part in deploy["obs_layouts"][0].split(",")]
layout_names = [part.split(":")[0] for part in layout]
layout_dims = [int(part.split(":")[1]) for part in layout]
obs_num = sum(layout_dims)
frame_stack = int(deploy["frame_stacks"][0])
check("观测分量顺序", layout_names == [name for name, _ in OBS_LAYOUT], f"{layout_names}")
check("观测分量维度", layout_dims == [dim for _, dim in OBS_LAYOUT], f"{layout_dims}")
check("帧栈长度", frame_stack == FRAME_STACK, f"{frame_stack} vs {FRAME_STACK}")
check("单帧观测 × 帧栈 = ONNX 输入", obs_num * frame_stack == ONNX_INPUT_DIM, f"{obs_num}×{frame_stack}={obs_num * frame_stack}")
check("输出维度 = 关节数", ONNX_OUTPUT_DIM == NUM_JOINTS == 16)
check("帧栈排布为 obs_major (term-major)", deploy["obs_stack_orders"][0] == "obs_major")
check("观测缩放全为 1.0 (无归一化)", all(
    float(deploy[key]) == 1.0
    for key in ("obs_scales_lin_vel", "obs_scales_ang_vel", "obs_scales_dof_pos", "obs_scales_dof_vel", "obs_scales_gravity_b")
))

# A2. 动作
expected_action_scales = expand(ACTION_SCALE, JOINT_ORDER)
check("动作缩放 (USD 顺序)", close(deploy["action_scales"], expected_action_scales), f"{deploy['action_scales']}")
check("轮关节 USD 索引", list(deploy["wheel_joint_indices"]) == WHEEL_JOINT_INDICES, f"{deploy['wheel_joint_indices']}")
check(
    "轮关节数量 = 16 - 12",
    len(WHEEL_JOINT_NAMES) == len(JOINT_ORDER) - 12,
    f"{WHEEL_JOINT_NAMES}",
)

# A3. USD ↔ URDF 关节顺序映射
check("usd2urdf 映射", list(deploy["usd2urdf"]) == USD2URDF, f"yaml={deploy['usd2urdf']} ours={USD2URDF}")

# A4. 关节默认姿态 (部署端是 URDF 顺序)
expected_defaults_urdf = expand(DEFAULT_JOINT_POS, URDF_JOINT_ORDER)
check(
    "关节默认角度 (URDF 顺序)",
    close(deploy["joint_default_angle"], expected_defaults_urdf, 1e-4),
    f"{deploy['joint_default_angle']}",
)

# A5. 时序 / 指令范围 / 裁剪
check("物理步长 dt", abs(float(deploy["dt"]) - SIM_DT) < 1e-9, f"{deploy['dt']} vs {SIM_DT}")
check("降采样 decimation", int(deploy["decimation"]) == DECIMATION, f"{deploy['decimation']} vs {DECIMATION}")
policy_dt = float(deploy["dt"]) * int(deploy["decimation"])
check("策略周期 = 20 ms (50 Hz)", abs(policy_dt - 0.02) < 1e-9, f"dt×decimation = {policy_dt}")
check("关节数", int(deploy["joint_num"]) == NUM_JOINTS, f"{deploy['joint_num']}")
check("动作裁剪 clip_actions", abs(float(deploy["clip_actions"]) - ACTION_CLIP) < 1e-9, f"{deploy['clip_actions']} vs {ACTION_CLIP}")
clip_cmd = [float(v) for v in deploy["clip_cmd"]]
check(
    "速度指令范围 (vx/vy/wz)",
    clip_cmd == [CMD_VX_RANGE[0], CMD_VX_RANGE[1], CMD_VY_RANGE[0], CMD_VY_RANGE[1], CMD_WZ_RANGE[0], CMD_WZ_RANGE[1]],
    f"{clip_cmd}",
)
clip_posture = [float(v) for v in deploy.get("clip_posture", [])]
check(
    "姿态/高度指令范围 (pitch/roll/h)",
    clip_posture
    == [
        CMD_PITCH_RANGE[0], CMD_PITCH_RANGE[1],
        CMD_ROLL_RANGE[0], CMD_ROLL_RANGE[1],
        CMD_HEIGHT_RANGE[0], CMD_HEIGHT_RANGE[1],
    ],
    f"{clip_posture}",
)
print(f"  [INFO] 部署端观测限幅 clip_observations = {deploy['clip_observations']} (训练侧不需要, 仅作记录)")

# A6. PD 增益 (部署端 robot.yaml 是 URDF/电机顺序)
with open(DEPLOY_ROBOT_YAML_PATH, "r", encoding="utf-8") as f:
    robot_yaml = yaml.safe_load(f)
expected_kp_urdf = expand(STIFFNESS, URDF_JOINT_ORDER)
expected_kd_urdf = expand(DAMPING, URDF_JOINT_ORDER)
check("PD 位置增益 kp (URDF 顺序)", close(robot_yaml["robot"]["kp"], expected_kp_urdf, 1e-6), f"{robot_yaml['robot']['kp']}")
check("PD 阻尼增益 kd (URDF 顺序)", close(robot_yaml["robot"]["kd"], expected_kd_urdf, 1e-6), f"{robot_yaml['robot']['kd']}")

# A7. URDF 关节顺序 (文档顺序) 自检
urdf_joints = [j.get("name") for j in ET.parse(RP_WD_URDF_PATH).getroot().findall("joint")]
check("URDF 关节顺序 (文档顺序)", sorted(urdf_joints) == sorted(URDF_JOINT_ORDER), f"{urdf_joints[:4]} ...")
check("URDF/训练侧关节集合一致", sorted(urdf_joints) == sorted(JOINT_ORDER))
print("  [INFO] 训练侧 (USD/articulation) 关节顺序:")
for i, name in enumerate(JOINT_ORDER):
    print(f"           {i:2d} {name:16s} -> URDF #{USD2URDF[i]:2d}")

# ---------------------------------------------------------------------------
# B. 运行时语义
# ---------------------------------------------------------------------------
if not args_cli.no_sim:
    print("\n=== B. 运行时语义 (启动仿真) ===")
    import gymnasium as gym  # noqa: E402
    import torch  # noqa: E402

    import rp_wd_rl.tasks  # noqa: E402, F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

    # 用 Play 变体: 域随机化全部关闭, 增益/默认姿态都是标称值, 便于逐项核对
    task = "Isaac-Velocity-Flat-RPWD-Play-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    robot = env.scene["robot"]

    # B1. 关节顺序
    check("articulation 关节顺序 == 部署端 USD 顺序", list(robot.joint_names) == JOINT_ORDER)
    if list(robot.joint_names) != JOINT_ORDER:
        print(f"        实际: {list(robot.joint_names)}")

    # B2. 维度
    check("观测向量维度 = 159", obs["policy"].shape[-1] == 159, f"{tuple(obs['policy'].shape)}")
    check("动作向量维度 = 16", env.action_manager.total_action_dim == 16, f"{env.action_manager.total_action_dim}")
    term_dim = env.observation_manager.group_obs_term_dim["policy"]
    blocks = [int(torch.tensor(d).prod()) for d in term_dim]  # 每项 flatten 后总长 (含帧栈)
    print(f"  [INFO] 观测项维度: {term_dim} -> 每项长度 {blocks}")
    expected_blocks = [dim * frame_stack for dim in layout_dims]
    check(
        "观测各项占用长度 = 部署端 layout × 帧栈",
        blocks == expected_blocks,
        f"实际 {blocks} / 期望 {expected_blocks}",
    )

    # B3. 动作映射 (raw → 关节目标, 含缩放与默认角度)
    action = torch.zeros(env.num_envs, 16, device=env.device)
    action[:, 0] = 1.0  # LB_ABAD: 位置控制
    action[:, 12] = 1.0  # LB_FOOT: 速度控制
    env.step(action)
    pos_target = robot.data.joint_pos_target[0].cpu()
    vel_target = robot.data.joint_vel_target[0].cpu()
    check(
        "腿位置目标 = raw×0.125 + 默认角度",
        abs(pos_target[0].item() - (DEFAULT_JOINT_POS["ABAD"] + 0.125)) < 1e-5,
        f"ABAD 目标 {pos_target[0].item():.5f} (期望 0.125)",
    )
    check(
        "腿位置目标未受轮公式影响 (HIP 保持默认)",
        abs(pos_target[4].item() - DEFAULT_JOINT_POS["HIP"]) < 1e-5,
        f"HIP 目标 {pos_target[4].item():.5f} (期望 0.72)",
    )
    check(
        "轮速度目标 = raw×5.0",
        abs(vel_target[12].item() - 5.0) < 1e-5,
        f"FOOT 速度目标 {vel_target[12].item():.5f} (期望 5.0)",
    )
    # 驱动器张量形状 (num_envs, num_joints_of_this_actuator)
    leg_kp = robot.actuators["legs"].stiffness[0]
    leg_kd = robot.actuators["legs"].damping[0]
    wheel_kp = robot.actuators["wheels"].stiffness[0]
    wheel_kd = robot.actuators["wheels"].damping[0]
    check(
        "腿部 PD: kp=100 / kd=3 (与部署端 robot.yaml 一致)",
        bool((leg_kp == 100.0).all()) and bool((leg_kd == 3.0).all()),
        f"kp={leg_kp.tolist()} kd={leg_kd.tolist()}",
    )
    check(
        "轮部 PD: kp=0 (纯阻尼) / kd=3",
        bool((wheel_kp == 0.0).all()) and bool((wheel_kd == 3.0).all()),
        f"kp={wheel_kp.tolist()} kd={wheel_kd.tolist()}",
    )

    # B4. 帧栈排布 (term-major, 每个分量旧→新, 最新在最后)
    #   reset 后 last_action 全零 -> step(ones) 后三帧 = [0, 0, 1]
    #                             -> step(zeros) 后三帧 = [0, 1, 0]
    env.reset()
    ones = torch.ones(env.num_envs, 16, device=env.device)
    env.step(ones)
    zeros = torch.zeros(env.num_envs, 16, device=env.device)
    obs2 = env.step(zeros)[0]  # ManagerBasedRLEnv 返回 5 元组 (obs, rew, terminated, truncated, info)
    # last_action 分量落在最后 frame_stack×16 = 48 维
    last_action_block = obs2["policy"][0, -16 * frame_stack :].reshape(frame_stack, 16)
    first_elems = [float(last_action_block[i, 0]) for i in range(frame_stack)]
    expected = [0.0, 1.0, 0.0]
    check(
        "帧栈顺序: 每个分量旧→新, 最新在最后 (部署端 update_stacked_obs 语义)",
        all(abs(a - b) < 1e-6 for a, b in zip(first_elems, expected)),
        f"三帧首元素 = {[round(v, 3) for v in first_elems]} (期望 {expected})",
    )
    check(
        "last_action 用的是原始动作 (未缩放)",
        bool(torch.allclose(last_action_block[1], ones[0])),
        f"第 2 帧取值 {last_action_block[1, :4].tolist()[:2]} (期望 1.0)",
    )
    # 分量之间不能串: cmd_vel 分量 (第 3 个 block, 3×3=9 维, 偏移 18..26)
    cmd_block = obs2["policy"][0, 18:27]
    check(
        "帧栈: 分量块之间无串扰 (cmd_vel 块三帧相同)",
        bool(torch.allclose(cmd_block, obs2["policy"][0, 18:21].repeat(frame_stack))),
        f"cmd 三帧首元素 = {cmd_block[::3].tolist()}",
    )
    del env
else:
    print("\n[INFO] 跳过运行时检查 (--no-sim)")

# ---------------------------------------------------------------------------
print("\n=== 结果 ===")
if _failures:
    print(f"FAIL: {len(_failures)} 项不通过:")
    for f in _failures:
        print(f"  - {f}")
else:
    print("PASS: 全部检查通过 —— 训练侧与部署侧接口一致.")

_exit_code = 1 if _failures else 0
if args_cli.no_sim:
    raise SystemExit(_exit_code)

# 关闭仿真并退出 (close() 偶发挂起, 见 sim_shutdown.py); 结果已打印, 直接退出
shutdown(simulation_app, exit_code=_exit_code)
