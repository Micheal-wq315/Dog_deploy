"""RP_wd 的 sim2real 契约常量 (纯 Python, **不导入 isaaclab**).

拆出来的原因: 这些常量要能在不启动 Isaac Sim 的情况下被校验脚本/工具引用
(isaaclab.sim 依赖 carb, 只能在 SimulationApp 启动后导入).

契约的"对方"是部署端:
  - src/inference/robots/wheel_quad/configs/default.yaml  (观测/动作/缩放/默认姿态)
  - src/inference/robots/wheel_quad/robot.yaml            (PD 增益/关节顺序)
  - src/inference/src/inference_node.cpp                  (观测拼接/动作后处理/CAN 顺序)

改任何一个常量, 都要同步改部署端, 并跑 rl/scripts/verify_contract.py 复核.
"""

from __future__ import annotations

import os

# 仓库根目录: rl/rp_wd_rl/assets/rp_wd_contract.py -> up 3
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RP_WD_USD_PATH = os.path.join(REPO_ROOT, "description", "RP_wd", "usd", "RP_wd.usd")
RP_WD_URDF_PATH = os.path.join(REPO_ROOT, "description", "RP_wd", "urdf", "RP_wd.urdf")
DEPLOY_ROBOT_DIR = os.path.join(REPO_ROOT, "src", "inference", "robots", "wheel_quad")
DEPLOY_CONFIG_PATH = os.path.join(DEPLOY_ROBOT_DIR, "configs", "default.yaml")
ODIN_DEPTH_DEPLOY_CONFIG_PATH = os.path.join(DEPLOY_ROBOT_DIR, "configs", "odin_depth.yaml")
DEPLOY_ROBOT_YAML_PATH = os.path.join(DEPLOY_ROBOT_DIR, "robot.yaml")
DEPLOY_MODEL_DIR = os.path.join(DEPLOY_ROBOT_DIR, "models")

##
# 关节顺序
##

LEG_NAMES = ("LB", "LF", "RB", "RF")  # 注意: 字母序, 不是 LF/RF/LB/RB
JOINT_TYPES = ("ABAD", "HIP", "KENN", "FOOT")

# 训练侧 (Isaac Lab articulation) 关节顺序 == 部署端 YAML 里的 "USD 顺序":
#   [ABAD×4, HIP×4, KENN×4, FOOT×4], 每组内 LB/LF/RB/RF
# 已用 isaacsim 实测确认 (2026-10): Articulation.joint_names 与该顺序一致.
JOINT_ORDER = [f"{leg}_{jt}_JOINT" for jt in JOINT_TYPES for leg in LEG_NAMES]
NUM_JOINTS = len(JOINT_ORDER)  # 16
LEG_JOINT_NAMES = [n for n in JOINT_ORDER if not n.endswith("_FOOT_JOINT")]  # 12
WHEEL_JOINT_NAMES = [n for n in JOINT_ORDER if n.endswith("_FOOT_JOINT")]  # 4
WHEEL_JOINT_INDICES = [JOINT_ORDER.index(n) for n in WHEEL_JOINT_NAMES]  # [12,13,14,15]

# 部署侧 (URDF/CAN) 关节顺序: 按腿分组 LF/RF/LB/RB, 每腿 [ABAD,HIP,KENN,FOOT]
# 对应 robot.yaml 的 can0/can1/can2/can3 顺序
URDF_JOINT_ORDER = [f"{leg}_{jt}_JOINT" for leg in ("LF", "RF", "LB", "RB") for jt in JOINT_TYPES]

# USD(训练) 索引 -> URDF(部署) 索引; 必须与 default.yaml 的 usd2urdf 完全一致
USD2URDF = [URDF_JOINT_ORDER.index(n) for n in JOINT_ORDER]

##
# 关节级契约
##

# 站立默认姿态 (rad) —— 部署端 joint_default_angle 的来源
DEFAULT_JOINT_POS = {"ABAD": 0.0, "HIP": 0.72, "KENN": -1.41, "FOOT": 0.0}

# 每类关节的动作缩放 —— 部署端 action_scales 的来源
#   ABAD/HIP/KENN: 位置控制 [rad/unit]; FOOT: 速度控制 [rad/s per unit] (kp=0 纯阻尼)
ACTION_SCALE = {"ABAD": 0.125, "HIP": 0.25, "KENN": 0.25, "FOOT": 5.0}

# PD 增益 —— 部署端 robot.yaml 的 kp/kd: 腿 kp=100 kd=3; 轮 kp=0 (纯阻尼) kd=3
STIFFNESS = {"ABAD": 100.0, "HIP": 100.0, "KENN": 100.0, "FOOT": 0.0}
DAMPING = {"ABAD": 3.0, "HIP": 3.0, "KENN": 3.0, "FOOT": 3.0}

# 力矩/速度限幅
#   腿部: URDF <limit effort="42" velocity="10.472">
#   轮子: RobStride RS02 准直驱模组, 峰值 17 N·m (src/motors/RS02使用说明书260428.pdf)
EFFORT_LIMIT_SIM = {"ABAD": 42.0, "HIP": 42.0, "KENN": 42.0, "FOOT": 17.0}
VELOCITY_LIMIT_SIM = {
    "ABAD": 10.472,
    "HIP": 10.472,
    "KENN": 10.472,
    # 部署端轮速命令限幅 = action(±10) × 5.0 = ±50 rad/s, 取同值
    "FOOT": 50.0,
}

# 站立高度 (base_link 原点离地) —— 默认姿态在平地实测 (2026-10), 用于 init_state 与高度奖励
STAND_HEIGHT = 0.419

# 轮子半径 [m] (URDF 里 FOOT 碰撞圆柱 radius=0.1025), 用于由轮速推算车速/评估打滑
WHEEL_RADIUS = 0.1025

##
# 时序 / 指令 / 裁剪
##

SIM_DT = 0.0025  # 物理步长 [s]
DECIMATION = 8  # 策略降采样 -> 1/(0.0025×8) = 50 Hz
POLICY_DT = SIM_DT * DECIMATION
FRAME_STACK = 3  # 观测帧栈长度 (部署端 frame_stacks)
POLICY_FREQ = 1.0 / POLICY_DT

# 速度指令范围 —— 部署端 clip_cmd: [vx_min,vx_max, vy_min,vy_max, wz_min,wz_max]
# 2026-10-20 训练课程重构: 速度降到 ±0.6 (强调控制精度而非极速), wz 保持 ±1.0.
# 注意: 这是**契约变更**, 必须同步改 src/inference/robots/wheel_quad/configs/default.yaml
# 的 clip_cmd (verify_contract.py 会逐项核对), 否则实机会按旧限幅截断指令.
CMD_VX_RANGE = (-0.6, 0.6)
CMD_VY_RANGE = (-0.6, 0.6)
CMD_WZ_RANGE = (-1.0, 1.0)

# 机身姿态指令 (rad): 俯仰 / 横滚。2026-10-16 新增 —— 摇杆2 纵向给 pitch, 拨轮给 roll。
#   符号约定 (由机体系重力反算): pitch = asin(g_x) 抬头为正; roll = atan2(g_y, -g_z) 右侧下沉为正
#   (静止时 gravity_b = (0,0,-1) -> pitch = roll = 0)
# 2026-10-20: 姿态指令收到 ±10° (运动时允许约 10° 倾斜, 但要求能主动控制这个倾角,
# 而不是被动地"被甩到多少算多少"); 另外增加**高度指令** (轮足靠腿部伸缩改变机身高度)。
CMD_PITCH_RANGE = (-0.1745, 0.1745)   # ±10°
CMD_ROLL_RANGE = (-0.1745, 0.1745)    # ±10°
CMD_HEIGHT_RANGE = (-0.07, 0.07)      # 相对站立高度 0.419 m 的偏移 [m]

# 部署端 clip_actions: 对模型**原始输出**裁剪 ±10 (在缩放与偏置之前)
ACTION_CLIP = 10.0

# 单帧观测各分量维度 (顺序 = 部署端 obs_layouts)
OBS_LAYOUT = (
    ("ang_vel", 3),
    ("gravity_b", 3),
    ("cmd_vel", 3),
    ("dof_pos", 12),
    ("dof_vel", 16),
    ("last_action", 16),
    # 机身姿态+高度指令 (pitch, roll, height_offset) —— 追加在末尾, 部署端的帧栈/拼接
    # 逻辑按分量列表通用处理, 所以老分量的偏移全不变。
    ("cmd_posture", 3),
)
OBS_NUM = sum(dim for _, dim in OBS_LAYOUT)  # 56 (includes cmd_posture:3)
ONNX_INPUT_DIM = OBS_NUM * FRAME_STACK  # 168, existing non-perception policy
ONNX_OUTPUT_DIM = NUM_JOINTS  # 16

# Odin1独立深度策略：逆时针90°校正、16×12网格、22帧缓冲间隔3帧取8张。
# 普通168维策略继续使用原配置；深度输出仍为独立1740维输入。
LEG_TORQUE_SCALE = 1.0 / 42.0
ODIN_DEPTH_IMAGE_SIZE = (160, 120)  # source image (width, height)
ODIN_DEPTH_ROTATION_DEG = 90        # CCW; normalize the side-mounted sensor image to portrait
ODIN_DEPTH_ROTATION_K = ODIN_DEPTH_ROTATION_DEG // 90
ODIN_DEPTH_GRID_SIZE = (12, 16)     # corrected output (width, height)
ODIN_DEPTH_MEMORY_FRAMES = 8        # 部署CNN的深度通道数
ODIN_DEPTH_MEMORY_STRIDE = 3        # 保留22帧原始历史，抽8张；覆盖1.4秒，最新图仍每帧更新
ODIN_DEPTH_FEATURE_DIM = ODIN_DEPTH_GRID_SIZE[0] * ODIN_DEPTH_GRID_SIZE[1]
ODIN_DEPTH_MEMORY_FEATURE_DIM = ODIN_DEPTH_FEATURE_DIM * ODIN_DEPTH_MEMORY_FRAMES
ODIN_DEPTH_OBS_LAYOUT = OBS_LAYOUT + (
    ("leg_torque", 12),
)
ODIN_DEPTH_OBS_NUM = sum(dim for _, dim in ODIN_DEPTH_OBS_LAYOUT)
ODIN_DEPTH_ONNX_INPUT_DIM = ODIN_DEPTH_OBS_NUM * FRAME_STACK + ODIN_DEPTH_MEMORY_FEATURE_DIM  # 1740
