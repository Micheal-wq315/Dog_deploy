"""RP_wd 平地速度跟踪任务配置 (manager-based).

设计原则: **一切与部署端 (src/inference/robots/wheel_quad) 的接口都必须是契约**,
契约表见 rl/README.md, 复核脚本 rl/scripts/verify_contract.py.

  sim:      dt=0.0025 s, decimation=8          -> 策略 50 Hz (部署端一致)
  obs:      ang_vel(3)+gravity(3)+cmd(3)+dof_pos(12)+dof_vel(16)+last_action(16)+cmd_posture(3) = 56
            帧栈 3 (term-major, 旧->新)         -> ONNX 输入 168
  action:   [腿 12 位置, 轮 4 速度] = USD 顺序, 缩放 [0.125/0.25/0.25/5.0], 原值裁剪 ±10
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from rp_wd_rl.assets.rp_wd import RP_WD_CFG
from rp_wd_rl.assets.rp_wd_contract import (
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_WZ_RANGE,
    DECIMATION,
    SIM_DT,
    STAND_HEIGHT,
    WHEEL_JOINT_NAMES,
)

from . import mdp

# 腿位置控制用的关节集合 (顺序即 USD/articulation 顺序: ABAD×4, HIP×4, KENN×4)
_LEG_JOINT_PATTERNS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]


##
# 场景
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """平地场景: 地面 + 机器人 + 接触力传感器 + 光照."""

    # 平地 (plane 类型不需要 terrain generator; env_spacing 由 InteractiveSceneCfg 提供)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = RP_WD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 接触力: 摔倒判定 + 轮子触地统计 (轮子是连续关节, 不需要 track_air_time)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False, debug_vis=False
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


##
# 指令
##


@configclass
class CommandsCfg:
    """速度指令 — 范围与部署端 clip_cmd 完全一致."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        # 10% 环境给零指令: 部署端松开摇杆时就是零指令, 必须学会原地站定
        rel_standing_envs=0.1,
        # 部署端直接给偏航角速度指令 (手柄/话题), 所以不用 heading 间接生成
        heading_command=False,
        rel_heading_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=CMD_VX_RANGE,
            lin_vel_y=CMD_VY_RANGE,
            ang_vel_z=CMD_WZ_RANGE,
            heading=(-math.pi, math.pi),
        ),
    )

    # 机身姿态+高度指令 (pitch/roll/height): 实机上由摇杆2 纵向、拨轮、按键给出。
    # 单独一项 (而不是塞进 base_velocity) -> 观测里是独立分量, 部署端只需加一项 layout。
    base_posture = mdp.UniformPostureCommandCfg()


##
# 动作
##


@configclass
class ActionsCfg:
    """动作空间: [12 个腿关节位置, 4 个轮子速度] —— 与部署端 action_scales 顺序一致.

    - `preserve_order` 保持默认 False: Isaac Lab 会按 articulation 的关节索引排序,
      而 articulation 顺序本身就是 [ABAD×4, HIP×4, KENN×4, FOOT×4] (已实测),
      与 scale 字典的键顺序一致 —— 两种排序结果相同.
    - 这里**不设** term 级 `clip`: Isaac Lab 的 term clip 作用在 scale+offset *之后*,
      而部署端是裁剪模型原始输出 (±10). 因此在 PPO 配置里用 `clip_actions=10.0`
      (RslRlVecEnvWrapper 对原始动作裁剪), 语义与部署端一致.
    """

    # 腿部 12 个关节: 位置控制, 目标 = raw × scale + 默认角度
    leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_LEG_JOINT_PATTERNS,
        scale={".*_ABAD_JOINT": 0.125, ".*_HIP_JOINT": 0.25, ".*_KENN_JOINT": 0.25},
        use_default_offset=True,  # offset 取 init_state 默认姿态 (0 / 0.72 / -1.41)
    )

    # 轮子 4 个关节: 速度控制 (纯阻尼 kp=0), 目标 = raw × 5.0
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_FOOT_JOINT"],
        scale={".*_FOOT_JOINT": 5.0},
        use_default_offset=False,
        offset=0.0,
    )


##
# 观测
##


@configclass
class ObservationsCfg:
    """普通策略单帧 56 维 (含 cmd_posture), 帧栈 3 -> 168 维."""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组 —— 顺序即部署端 obs_layout 顺序, 不可随意调整."""

        # 0) 机体角速度 (部署端: IMU 陀螺, 无零偏标定 -> 加常值零偏 DR)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_biased,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        # 1) 投影重力 (部署端: IMU 四元数把世界重力转到机体系, z 向上为 -1)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        # 2) 速度指令 [vx, vy, wz]
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # 3) 关节位置 12 维 (仅腿, 不含轮; 相对默认姿态; 叠加编码器零点误差 DR)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_biased,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINT_PATTERNS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 4) 关节速度 16 维 (含轮)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        # 5) 上一帧动作 16 维 (原始输出, 未缩放未偏置 —— 与部署端一致)
        actions = ObsTerm(func=mdp.last_action)
        # 6) 机身姿态+高度指令 (pitch, roll, height) 3 维 —— 必须放在最后, 与契约 OBS_LAYOUT 一致
        posture_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_posture"}
        )

        def __post_init__(self):
            # 训练时打开观测噪声 (play 时关闭); 噪声在 term 级配置
            self.enable_corruption = True
            # 帧栈 3: term-major, 每个分量自身 3 帧相邻 (旧->新) —— 即部署端 obs_major + frame_stack=3
            self.history_length = 3
            self.flatten_history_dim = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# 域随机化事件
##


@configclass
class EventCfg:
    """完整域随机化: 质量/质心/摩擦/电机增益/关节参数/传感器零偏/推力扰动."""

    # ---- 启动时 (整次训练固定) ----
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.5, 2.5),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        },
    )

    # 电机增益 (kp/kd) ±30%: 覆盖电池电压下降、温升导致的增益漂移
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )

    # 关节摩擦与电枢惯量: 减速箱阻力与转子惯量, 实机与理想仿真差别最大的两项
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.08),
            "armature_distribution_params": (0.0, 0.03),
            "operation": "abs",
        },
    )

    # ---- 每回合 reset ----
    sensor_bias = EventTerm(
        func=mdp.randomize_sensor_bias,
        mode="reset",
        params={
            "gyro_bias_range": (-0.05, 0.05),
            "joint_pos_offset_range": (-0.015, 0.015),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.08, 0.08),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # ---- 周期扰动 ----
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.4, 0.4), "z": (-0.2, 0.2)}},
    )


##
# 奖励
##


@configclass
class RewardsCfg:
    """速度跟踪为主, 姿态/平顺/轮子行为为辅."""

    # ---- 任务 ----
    # 权重给足: 跟踪必须是回报的主要来源, 否则"站着不动"或"早点摔倒"会成为局部最优.
    # (实测教训: 早期跟踪 0.5/回合 而惩罚合计 -5/回合, 策略 280 迭代时学会了摔倒)
    # σ 的取法: 配合**指令课程**用 0.5 —— 课程负责"从窄到宽"避免远处没梯度,
    # σ 负责"近处梯度够陡". (曾试过 σ=1.0 直接上 ±2.0: 奖励地形被抹平,
    # 策略卡在 error≈1.9 m/s 平了 400+ 迭代.)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # ---- 姿态与平顺 ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.03)
    # 姿态跟踪 (exp 核): 有姿态指令后 "保持水平" 不再是目标, 而是"跟踪指令的 pitch/roll".
    # 用 exp 核而不是 L2 —— 姿态误差只有 0.02~0.05 rad, L2 加权后量级远小于速度奖励,
    # 根本无法体现"重点突出控制能力".
    attitude_track_exp = RewTerm(
        func=mdp.attitude_track_exp,
        weight=0.6,
        params={"command_name": "base_posture", "std": 0.10, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 超过约 10° 的倾斜才罚: 允许运动时自然摆动, 但禁止大前倾/大侧倾
    attitude_limit_l2 = RewTerm(
        func=mdp.attitude_limit_l2,
        weight=-5.0,
        params={"limit": 0.1745, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 高度跟踪 (exp 核): 目标 = 站立高度 + 指令高度偏移
    height_track_exp = RewTerm(
        func=mdp.height_track_exp,
        weight=0.4,
        params={"command_name": "base_posture", "std": 0.04, "asset_cfg": SceneEntityCfg("robot")},
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-5.0e-8, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.004)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # 真实终止 (不含超时) 的固定惩罚: 堵死"主动摔倒以提前结束回合"的捷径
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # ---- 轮足特有 ----
    # 轮子空转 (打滑) 惩罚: 权重必须很小, 否则会压制正常行进.
    # 提速到 2 m/s 后轮速达 19.5 rad/s, Σω²≈1520 —— 权重从 -1e-3 降到 -1e-4,
    # 否则该项每回合要扣约 -30, 与"跑得快"直接对抗.
    wheel_vel_l2 = RewTerm(
        func=mdp.wheel_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # 零指令时轮子必须停住 (实机上表现为原地溜车); 该函数的返回值已截断, 见 mdp/rewards.py
    stand_still_wheel_vel = RewTerm(
        func=mdp.stand_still_wheel_vel,
        weight=-0.05,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # 轮电机力矩: RS02 额定 6 N·m, 长时间大扭矩会过热
    wheel_torque_l2 = RewTerm(
        func=mdp.wheel_torque_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # ★ 必须"真的滚起来": 惩罚 轮速×半径 与 轮心实际速度 的失配.
    # 没有这一项时, 策略会收敛到"用轮子当脚跑小步"(实测: 着地轮数 2.36/4, 轮速仅应有的 1/3),
    # 冲击大且浪费轮子. 权重按"小跑步态约 -30/回合、纯滚动约 0、乱动截断在 -50/回合"标定.
    wheel_rolling_slip = RewTerm(
        func=mdp.wheel_rolling_slip_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "wheel_joint_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "wheel_body_cfg": SceneEntityCfg("robot", body_names=[".*_FOOT_LINK"]),
        },
    )
    # 四个轮子同时着地 (滚动前进的必然结果): 满额 +10/回合
    # 注意用的是自定义的 wheels_in_contact_frac, 不是 mdp.desired_contacts (后者语义相反)
    all_wheels_contact = RewTerm(
        func=mdp.wheels_in_contact_frac,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT_LINK"]), "threshold": 1.0},
    )
    # 腿部关节角速度惩罚: 抑制"抬腿换位" —— 实测横移时策略会用对角腿交替抬起错开重心,
    # 仿真可行但 sim2real 很差 (冲击大、支撑面小、轮子侧向刮擦)
    leg_joint_vel_l2 = RewTerm(
        func=mdp.leg_joint_vel_l2,
        weight=-5.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
            )
        },
    )
    # 髋侧摆保持 0: 四条腿平行, 避免撇腿支撑 (权重加强, 配合上面的腿速惩罚)
    joint_deviation_abad = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ABAD_JOINT"])},
    )
    # 躯体/小腿触地 = 摔倒或姿态塌陷
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_KENN_LINK"]),
            "threshold": 1.0,
        },
    )


##
# 课程学习
##


@configclass
class CurriculumCfg:
    """指令范围课程: 直线速度从 ±0.7 起步, 随训练放大到契约满额 (±2.0 m/s)."""

    command_lin_vel = CurrTerm(
        func=mdp.command_lin_vel_curriculum,
        params={
            "command_name": "base_velocity",
            "start_scale": 0.35,      # ±0.7 m/s 起步, 与上一个稳定版本 (rolling_v4) 的能力衔接
            "end_scale": 1.0,         # 最终必须覆盖契约满额, 否则实机手柄一推就出分布
            "warmup_steps": 240,      # ≈10 迭代
            "ramp_steps": 24000,      # ≈1000 迭代 (1 迭代 = 24 策略步)
            "ramp_angular": False,    # 角速度一直保持满额 (实测偏航跟踪本就很好)
        },
    )

    # 姿态/高度指令幅度课程: 先学站住跑起来, 再逐步加姿态目标
    posture_command = CurrTerm(
        func=mdp.posture_command_curriculum,
        params={
            "command_name": "base_posture",
            "start_scale": 0.0,
            "end_scale": 1.0,
            "warmup_steps": 4800,   # ≈200 迭代
            "ramp_steps": 24000,    # ≈1000 迭代
        },
    )

    # 崎岖地形版才启用 (见 rp_wd_rough_env_cfg.py); 这里预声明字段以便子类赋值
    terrain_levels: CurrTerm | None = None
    # Odin1 感知课程版使用: 后期渐进开启腿部力矩惩罚
    leg_torque_penalty: CurrTerm | None = None


##
# 终止条件
##


@configclass
class TerminationsCfg:
    """终止条件 —— 与部署端安全逻辑对应."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 躯体触地
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    # 倾角 > 60°: 部署端 get_gravity_b_obs 里 gravity_z > -0.5 直接急停 (acos(0.5)=1.0472 rad)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0472})
    # 躯体高度过低 (站高 0.419 m, 掉到 0.18 m 以下即塌了)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.18})


##
# 环境
##


@configclass
class RPWDFlatEnvCfg(ManagerBasedRLEnvCfg):
    """RP_wd 平地速度跟踪 (训练)."""

    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 策略频率: 1 / (0.0025 × 8) = 50 Hz —— 必须与部署端 dt × decimation 一致
        self.decimation = DECIMATION
        self.sim.dt = SIM_DT
        self.episode_length_s = 20.0

        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # 大量刚体接触 (4 轮 + 地面) 时需要更大的 GPU patch 缓冲
        # 接触多时的 patch 容量 (崎岖地形实测: 该值 10->20 倍无收益, 瓶颈是三角网格窄相检测)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RPWDFlatEnvCfg_PLAY(RPWDFlatEnvCfg):
    """播放/评估: 关掉观测噪声与域随机化, 便于观察策略本身的表现."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # 关掉随机化, 只保留确定性环境
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_parameters = None
        self.events.sensor_bias = None
        self.events.push_robot = None
        # 评估时不强制站立指令混入
        self.commands.base_velocity.rel_standing_envs = 0.0
