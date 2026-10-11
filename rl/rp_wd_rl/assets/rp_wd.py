"""RP_wd 四轮足机器人资产配置 (Isaac Lab ArticulationCfg).

机器人本体: 16 关节 = 4 腿 × (ABAD 髋侧摆 / HIP 大腿 / KENN 小腿 / FOOT 轮)
资产来源: description/RP_wd/usd/RP_wd.usd (Isaac Lab UrdfConverter 从同名 URDF 转换, 含解析碰撞体).

契约常量 (关节顺序/缩放/增益/默认姿态/限幅) 全部在 rp_wd_contract.py 里, 本文件只负责
把它们组装成 Isaac Lab 的配置对象.

注意: 本模块 import isaaclab.sim -> 依赖 carb, **必须在 SimulationApp 启动后**才能导入.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .rp_wd_contract import (
    DAMPING,
    DEFAULT_JOINT_POS,
    EFFORT_LIMIT_SIM,
    RP_WD_USD_PATH,
    STAND_HEIGHT,
    STIFFNESS,
    VELOCITY_LIMIT_SIM,
)

# 重新导出契约常量, 方便 `from rp_wd_rl.assets.rp_wd import JOINT_ORDER` 这类用法
from .rp_wd_contract import *  # noqa: F401, F403

RP_WD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RP_WD_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 自碰撞保持关闭: 与 USD 转换时的 self_collision=false 一致, 且轮/小腿贴近时易抖动
            enabled_self_collisions=False,
            # 4 = Isaac Lab 默认值。曾用 8 (想让轮子连续接触更稳), 但在崎岖地形上
            # 接触数暴增时这一项要吃掉约 2x 的求解时间 (实测 7.4 -> 约 4 s/迭代)。
            solver_position_iteration_count=4,
            # 保留 1 次速度迭代: 轮子长时间贴地的滚动接触靠它稳住
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # z 取实测站立高度, 起手略微悬空由重力落定
        pos=(0.0, 0.0, STAND_HEIGHT + 0.02),
        joint_pos={
            ".*_ABAD_JOINT": DEFAULT_JOINT_POS["ABAD"],
            ".*_HIP_JOINT": DEFAULT_JOINT_POS["HIP"],
            ".*_KENN_JOINT": DEFAULT_JOINT_POS["KENN"],
            ".*_FOOT_JOINT": DEFAULT_JOINT_POS["FOOT"],
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 腿部: 位置伺服 (MIT 语义 τ = kp(p_des-p) + kd(0-v)), 对应部署端 kp=100/kd=3
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"],
            effort_limit_sim={
                ".*_ABAD_JOINT": EFFORT_LIMIT_SIM["ABAD"],
                ".*_HIP_JOINT": EFFORT_LIMIT_SIM["HIP"],
                ".*_KENN_JOINT": EFFORT_LIMIT_SIM["KENN"],
            },
            velocity_limit_sim={
                ".*_ABAD_JOINT": VELOCITY_LIMIT_SIM["ABAD"],
                ".*_HIP_JOINT": VELOCITY_LIMIT_SIM["HIP"],
                ".*_KENN_JOINT": VELOCITY_LIMIT_SIM["KENN"],
            },
            stiffness={
                ".*_ABAD_JOINT": STIFFNESS["ABAD"],
                ".*_HIP_JOINT": STIFFNESS["HIP"],
                ".*_KENN_JOINT": STIFFNESS["KENN"],
            },
            damping={
                ".*_ABAD_JOINT": DAMPING["ABAD"],
                ".*_HIP_JOINT": DAMPING["HIP"],
                ".*_KENN_JOINT": DAMPING["KENN"],
            },
            armature=0.01,
        ),
        # 轮子: 纯阻尼速度控制 (kp=0, kd=3); 仿真里就是 stiffness=0 的隐式驱动器
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT_JOINT"],
            effort_limit_sim={".*_FOOT_JOINT": EFFORT_LIMIT_SIM["FOOT"]},
            velocity_limit_sim={".*_FOOT_JOINT": VELOCITY_LIMIT_SIM["FOOT"]},
            stiffness={".*_FOOT_JOINT": STIFFNESS["FOOT"]},
            damping={".*_FOOT_JOINT": DAMPING["FOOT"]},
            armature=0.01,
        ),
    },
)
"""RP_wd 机器人配置 (prim_path 由环境配置替换为 {ENV_REGEX_NS}/Robot)."""
