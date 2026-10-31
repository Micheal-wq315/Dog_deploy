"""RP_wd 点足版资产：12 个腿部驱动关节 + 4 个无电机固定球脚。"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .rp_wd_contract import DAMPING, DEFAULT_JOINT_POS, EFFORT_LIMIT_SIM, STIFFNESS, VELOCITY_LIMIT_SIM

REPO_ROOT = Path(__file__).resolve().parents[3]
POINTFOOT_URDF_PATH = REPO_ROOT / "description" / "RP_wd_pointfoot" / "urdf" / "RP_wd_pointfoot.urdf"
POINTFOOT_STAND_HEIGHT = 0.32

# 通过 Isaac Lab 在仿真启动时从 URDF 转 USD，禁用 fixed-joint 合并，保留独立脚刚体以读取足端接触。
RP_WD_POINTFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(POINTFOOT_URDF_PATH),
        usd_dir="/tmp/isaaclab_rpwd_pointfoot",
        usd_file_name="RP_wd_pointfoot.usd",
        force_usd_conversion=False,
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=3.0,
            ),
        ),
    ),
    # Go2 几何按默认关节角估算约 0.32 m 站高；初始多留 2 cm 给重力 settle。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, POINTFOOT_STAND_HEIGHT + 0.02),
        joint_pos={
            ".*_ABAD_JOINT": DEFAULT_JOINT_POS["ABAD"],
            ".*_HIP_JOINT": DEFAULT_JOINT_POS["HIP"],
            ".*_KENN_JOINT": DEFAULT_JOINT_POS["KENN"],
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
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
    },
)
