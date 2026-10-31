"""RP_wd Go2 点足版：12 自由度运动控制与渐进式台阶/崎岖地形课程。"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp as locomotion_mdp

from rp_wd_rl.assets.rp_wd_contract import (
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_WZ_RANGE,
    DECIMATION,
    SIM_DT,
)
from rp_wd_rl.assets.rp_wd_pointfoot import RP_WD_POINTFOOT_CFG

from . import mdp

POINTFOOT_JOINTS = [".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
POINTFOOT_NAMES = ["LF", "RF", "LB", "RB"]
POINTFOOT_FEET = [f"{leg}_FOOT_LINK" for leg in POINTFOOT_NAMES]
POINTFOOT_NONFOOT_BODIES = [
    f"{leg}_{part}_LINK" for leg in POINTFOOT_NAMES for part in ("ABAD", "HIP", "KENN")
]


##
# Go2 点足课程：先低台阶与平缓起伏，难度行由地形表现课程逐步开放。
##

RPWD_POINTFOOT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    num_rows=6,
    num_cols=5,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.02, 0.16),
            step_width=0.34,
            platform_width=1.6,
            border_width=0.25,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.02, 0.14),
            step_width=0.34,
            platform_width=1.6,
            border_width=0.25,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.02, 0.12),
            platform_width=1.8,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            downsampled_scale=0.3,
            border_width=0.25,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.35),
            platform_width=1.8,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.075,
            slope_range=(0.0, 0.35),
            platform_width=1.8,
            border_width=0.25,
        ),
    },
)


@configclass
class PointfootSceneCfg(InteractiveSceneCfg):
    """点足地形场景，脚端接触独立可测。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RPWD_POINTFOOT_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
        ),
    )
    robot: ArticulationCfg = RP_WD_POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        update_period=0.02,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class PointfootCommandsCfg:
    """低速起步，最终覆盖本仓库已定义的平面速度指令范围。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        heading_command=False,
        rel_heading_envs=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=CMD_VX_RANGE,
            lin_vel_y=CMD_VY_RANGE,
            ang_vel_z=CMD_WZ_RANGE,
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class PointfootActionsCfg:
    """仅 12 个主动腿关节采用位置目标；四个球脚都是固定关节，不进入动作。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=POINTFOOT_JOINTS,
        scale={".*_ABAD_JOINT": 0.125, ".*_HIP_JOINT": 0.25, ".*_KENN_JOINT": 0.25},
        use_default_offset=True,
    )


@configclass
class PointfootObservationsCfg:
    """单帧 45 维本体观测，按分量堆叠 3 帧（策略输入 135 维）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_biased,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_biased,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=POINTFOOT_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=POINTFOOT_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.history_length = 3
            self.flatten_history_dim = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PointfootEventsCfg:
    """中等幅度域随机化；先覆盖摩擦、质量、执行器与传感器差异。"""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.65, 1.35),
            "dynamic_friction_range": (0.5, 1.2),
            "restitution_range": (0.0, 0.08),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 1.5),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.015, 0.015), "z": (-0.01, 0.01)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.06),
            "armature_distribution_params": (0.0, 0.02),
            "operation": "abs",
        },
    )
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
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.06, 0.06), "velocity_range": (-0.3, 0.3)},
    )


@configclass
class PointfootRewardsCfg:
    """以速度跟踪和步态接触为主；保留平顺/机身稳定项，不加入电机力矩惩罚。"""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POINTFOOT_JOINTS)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POINTFOOT_JOINTS)},
    )
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time,
        weight=0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=POINTFOOT_FEET),
            "command_name": "base_velocity",
            "threshold": 0.18,
        },
    )
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.12,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=POINTFOOT_FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=POINTFOOT_FEET),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=POINTFOOT_NONFOOT_BODIES),
            "threshold": 1.0,
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)


@configclass
class PointfootCurriculumCfg:
    command_lin_vel = CurrTerm(
        func=mdp.command_lin_vel_curriculum,
        params={
            "command_name": "base_velocity",
            "start_scale": 0.35,
            "end_scale": 1.0,
            "warmup_steps": 4_800,
            "ramp_steps": 24_000,
            "ramp_angular": False,
        },
    )
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={
            # 课程时钟只限制最高开放级别，能否升级仍由实际走过的距离决定。
            "max_level_schedule": (
                (0, 1),
                (24_000, 2),
                (72_000, 3),
                (144_000, 4),
                (240_000, 5),
            ),
        },
    )


@configclass
class PointfootTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(55.0)})
    root_height = DoneTerm(
        func=mdp.root_height_below_local_minimum,
        params={"minimum_height": 0.17},
    )


@configclass
class RPWDPointfootEnvCfg(ManagerBasedRLEnvCfg):
    """点足版崎岖地形训练环境。"""

    scene: PointfootSceneCfg = PointfootSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: PointfootObservationsCfg = PointfootObservationsCfg()
    actions: PointfootActionsCfg = PointfootActionsCfg()
    commands: PointfootCommandsCfg = PointfootCommandsCfg()
    rewards: PointfootRewardsCfg = PointfootRewardsCfg()
    terminations: PointfootTerminationsCfg = PointfootTerminationsCfg()
    events: PointfootEventsCfg = PointfootEventsCfg()
    curriculum: PointfootCurriculumCfg = PointfootCurriculumCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.sim.dt = SIM_DT
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
