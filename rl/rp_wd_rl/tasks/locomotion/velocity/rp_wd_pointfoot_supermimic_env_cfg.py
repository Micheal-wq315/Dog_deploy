"""点足版 SuperMimic：盲 Actor、特权高度图训练与定向上阶课程。"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewTerm,
    SceneEntityCfg,
)
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from rp_wd_rl.assets.rp_wd_contract import DECIMATION, SIM_DT
from rp_wd_rl.tasks.locomotion.velocity import mdp
from rp_wd_rl.tasks.locomotion.velocity.rp_wd_pointfoot_env_cfg import (
    PointfootActionsCfg,
    PointfootEventsCfg,
    PointfootObservationsCfg,
    PointfootRewardsCfg,
    PointfootSceneCfg,
    PointfootTerminationsCfg,
    RPWDPointfootEnvCfg,
)


SUPERMIMIC_SCAN_SIZE = (2.0, 1.2)  # 21 x 13 = 273 个前视高度点
SUPERMIMIC_SCAN_GRID = (21, 13)

POINTFOOT_SUPERMIMIC_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(6.0, 6.0),
    num_rows=6,
    num_cols=4,
    border_width=20.0,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "stairs_up": mdp.PointfootDirectedStairsCfg(proportion=0.50),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            downsampled_scale=0.3,
            border_width=0.25,
        ),
    },
)


@configclass
class PointfootSuperMimicSceneCfg(PointfootSceneCfg):
    """维持单射线净空测量，再加特权前视高度图供模型与 Critic 使用。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=POINTFOOT_SUPERMIMIC_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
        ),
    )
    supermimic_terrain_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.65, 0.0, 1.2)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=SUPERMIMIC_SCAN_SIZE),
        update_period=0.02,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class PointfootSuperMimicObservationsCfg(PointfootObservationsCfg):
    """Actor 仍只有135维本体历史；前视地形图单独暴露给训练循环。"""

    @configclass
    class PrivilegedTerrainCfg(ObsGroup):
        terrain_scan = ObsTerm(
            func=mdp.supermimic_terrain_scan,
            params={
                "sensor_cfg": SceneEntityCfg("supermimic_terrain_scanner"),
                "grid_size": SUPERMIMIC_SCAN_GRID,
                "height_clip": 0.5,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    supermimic_terrain: PrivilegedTerrainCfg = PrivilegedTerrainCfg()


@configclass
class PointfootSuperMimicCommandsCfg:
    """楼梯课程按 +X 定向前进；命令每回合固定，避免临近踏步时突然换向。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        heading_command=False,
        rel_heading_envs=0.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.20, 0.45),
            lin_vel_y=(-0.05, 0.05),
            ang_vel_z=(-0.08, 0.08),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class PointfootSuperMimicActionsCfg(PointfootActionsCfg):
    """采用 Go2W 上阶版本的更大髋/膝动作幅值，保持12维点足动作顺序。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"],
        scale={".*_ABAD_JOINT": 0.125, ".*_HIP_JOINT": 0.6, ".*_KENN_JOINT": 0.8},
        use_default_offset=True,
    )


@configclass
class PointfootSuperMimicEventsCfg(PointfootEventsCfg):
    """上阶赛道朝向固定，保留其它物理与传感器随机化。"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "yaw": (-0.08, 0.08)},
            "velocity_range": {
                "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.03, 0.03),
                "roll": (-0.03, 0.03), "pitch": (-0.03, 0.03), "yaw": (-0.05, 0.05),
            },
        },
    )
    reset_stair_events = EventTerm(
        func=mdp.reset_pointfoot_stair_state,
        mode="reset",
    )


@configclass
class PointfootSuperMimicRewardsCfg(PointfootRewardsCfg):
    """放松平地垂向约束，加入上阶抬脚、干净着陆和楼梯前进信号。"""

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stair_lift = RewTerm(func=mdp.pointfoot_stair_lift_reward, weight=6.0)
    stair_landing = RewTerm(func=mdp.pointfoot_stair_landing_reward, weight=10.0)
    stair_progress = RewTerm(func=mdp.pointfoot_stair_progress_reward, weight=0.5)


@configclass
class PointfootSuperMimicCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.pointfoot_terrain_curriculum)


@configclass
class RPWDPointfootSuperMimicEnvCfg(RPWDPointfootEnvCfg):
    """Go2W SuperMimic算法迁移到RP_wd固定点脚、12主动关节任务。"""

    scene: PointfootSuperMimicSceneCfg = PointfootSuperMimicSceneCfg(
        num_envs=64, env_spacing=2.5
    )
    observations: PointfootSuperMimicObservationsCfg = PointfootSuperMimicObservationsCfg()
    actions: PointfootSuperMimicActionsCfg = PointfootSuperMimicActionsCfg()
    commands: PointfootSuperMimicCommandsCfg = PointfootSuperMimicCommandsCfg()
    rewards: PointfootSuperMimicRewardsCfg = PointfootSuperMimicRewardsCfg()
    terminations: PointfootTerminationsCfg = PointfootTerminationsCfg()
    events: PointfootSuperMimicEventsCfg = PointfootSuperMimicEventsCfg()
    curriculum: PointfootSuperMimicCurriculumCfg = PointfootSuperMimicCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.render_interval = DECIMATION
        self.sim.dt = SIM_DT
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
