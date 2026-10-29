"""Odin1 深度感知轮足任务: 随机障碍/地形 + 深度增强 + 腿部力反馈."""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
import numpy as np
import trimesh
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from rp_wd_rl.assets.rp_wd_contract import (
    ODIN_DEPTH_GRID_SIZE,
    ODIN_DEPTH_IMAGE_SIZE,
    ODIN_DEPTH_MEMORY_FRAMES,
)

from . import mdp
from .odin_terrains import ODIN_COURSE_TERRAINS_CFG
from .rp_wd_env_cfg import ObservationsCfg
from .rp_wd_rough_env_cfg import RPWDRoughEnvCfg, RPWDRoughSceneCfg


ODIN_BENCHMARK_SCENARIOS = (
    "flat_reference",
    "step_up_06cm",
    "step_down_08cm",
    "low_box_12cm",
    "bypass_wall_45cm",
    "slalom_45cm",
)


def odin_fixed_benchmark_terrain(difficulty, cfg):
    """固定几何评测场景；出生点在 x=0.8m，前进方向为 +X。"""
    del difficulty
    size_x, size_y = cfg.size
    floor = trimesh.creation.box(
        extents=(size_x, size_y, 0.10),
        transform=trimesh.transformations.translation_matrix((size_x / 2, size_y / 2, -0.05)),
    )
    meshes = [floor]
    origin = np.array((0.8, size_y / 2, 0.0), dtype=np.float32)

    def add_box(x, y, z, dx, dy, dz):
        meshes.append(trimesh.creation.box(
            extents=(dx, dy, dz),
            transform=trimesh.transformations.translation_matrix((x, y, z)),
        ))

    scenario = cfg.scenario
    if scenario == "step_up_06cm":
        add_box(2.75, size_y / 2, 0.03, 1.5, size_y, 0.06)
    elif scenario == "step_down_08cm":
        add_box(1.1, size_y / 2, 0.04, 2.2, size_y, 0.08)
        origin[2] = 0.08
    elif scenario == "low_box_12cm":
        add_box(2.35, size_y / 2, 0.06, 0.40, 1.0, 0.12)
    elif scenario == "bypass_wall_45cm":
        # 留出两侧各约1m通道，要求策略绕过高于轮半径的障碍。
        add_box(2.65, size_y / 2, 0.225, 0.32, size_y - 2.0, 0.45)
    elif scenario == "slalom_45cm":
        # 前后交错摆放，至少需要一次明显的横向绕行。
        add_box(2.35, size_y * 0.66, 0.225, 0.42, 1.65, 0.45)
        add_box(3.95, size_y * 0.34, 0.225, 0.42, 1.65, 0.45)
    elif scenario != "flat_reference":
        raise ValueError(f"未知 Odin 固定评测场景: {scenario}")

    return meshes, origin


@configclass
class OdinBenchmarkTerrainCfg(SubTerrainBaseCfg):
    """固定障碍评测子地形配置。"""

    function = odin_fixed_benchmark_terrain
    scenario: str = "flat_reference"


RPWD_ODIN_DEPTH_TERRAINS_CFG = ODIN_COURSE_TERRAINS_CFG


@configclass
class RPWDOdinDepthSceneCfg(RPWDRoughSceneCfg):
    """rough 地形场景 + 前向 Odin1 仿真深度相机."""

    odin_depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/odin_depth_camera",
        offset=TiledCameraCfg.OffsetCfg(
            # 安装点先按机身前上方估计; 真机位置/俯角须按实际支架标定。
            pos=(0.18, 0.0, 0.10),
            # ROS 光轴 +Z 转到机体前方 +X, 并向下俯 10 度。
            rot=(0.6428, 0.0, 0.7660, 0.0),
            convention="ros",
        ),
        # ROS 深度图通常是沿光轴的Z深度；用 distance_to_image_plane 对齐其语义。
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=6.0,
            horizontal_aperture=20.8,
            vertical_aperture=12.0,
            clipping_range=(0.2, 6.0),
        ),
        width=ODIN_DEPTH_IMAGE_SIZE[0],
        height=ODIN_DEPTH_IMAGE_SIZE[1],
        update_period=1.0 / 15.0,
        depth_clipping_behavior="max",
    )


@configclass
class RPWDOdinDepthObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # USD 关节顺序中的 12 个腿关节力矩, 按 URDF effort limit 42 Nm 归一化。
        leg_torque = ObsTerm(
            func=mdp.leg_torque_normalized,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
                ),
                "effort_limit": 42.0,
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

    @configclass
    class DepthMemoryCfg(ObsGroup):
        """独立相机历史组，保持 [B, C=8, H=16, W=12] 供 CNN 编码。"""

        depth_memory = ObsTerm(
            func=mdp.odin_depth_memory,
            params={
                "sensor_cfg": SceneEntityCfg("odin_depth_camera"),
                "data_type": "distance_to_image_plane",
                "max_depth": 6.0,
                "grid_size": (ODIN_DEPTH_GRID_SIZE[1], ODIN_DEPTH_GRID_SIZE[0]),
                "noise_std": 0.03,
                "dropout_prob": 0.01,
                "crop_fraction_range": (0.95, 1.0),
                "random_crop": True,
                "augmentation_warmup_steps": 12_000,
                "augmentation_ramp_steps": 48_000,
                "memory_frames": ODIN_DEPTH_MEMORY_FRAMES,
                "episode_bias_m": 0.04,
                "scale_error": 0.02,
                "occlusion_prob": 0.30,
                "packet_loss_prob": 0.08,
                "max_latency_frames": 2,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 0
            self.flatten_history_dim = False
            self.concatenate_terms = True

    @configclass
    class TeacherDepthCfg(DepthMemoryCfg):
        """同视野的清晰深度标签；教师没有学生无法观测的背后高程信息。"""

        depth_memory = ObsTerm(
            func=mdp.odin_depth_memory,
            params={
                "sensor_cfg": SceneEntityCfg("odin_depth_camera"),
                "grid_size": (ODIN_DEPTH_GRID_SIZE[1], ODIN_DEPTH_GRID_SIZE[0]),
                "memory_key": "teacher", "noise_std": 0.0,
                "dropout_prob": 0.0, "random_crop": False,
                "episode_bias_m": 0.0, "scale_error": 0.0,
                "occlusion_prob": 0.0, "packet_loss_prob": 0.0,
                "max_latency_frames": 0,
            },
        )

    policy: ObsGroup = PolicyCfg()
    depth_memory: ObsGroup = DepthMemoryCfg()
    depth_teacher: ObsGroup = TeacherDepthCfg()


@configclass
class RPWDOdinDepthEnvCfg(RPWDRoughEnvCfg):
    """Odin1 深度策略: 256环境，先稳住运动再逐步增加地形与姿态难度。"""

    scene: RPWDOdinDepthSceneCfg = RPWDOdinDepthSceneCfg(num_envs=256, env_spacing=2.5)
    observations: RPWDOdinDepthObservationsCfg = RPWDOdinDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 深度任务不需要速度箭头 marker; 关闭后可避免依赖 Isaac UI Props USD。
        self.commands.base_velocity.debug_vis = False

        # 为本 profile 使用独立地形配置副本, 不改动原有 rough 任务。
        self.scene.terrain.terrain_generator = copy.deepcopy(RPWD_ODIN_DEPTH_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 0
        self.sim.physics_material = self.scene.terrain.physics_material

        self.episode_length_s = 30.0
        # 8个物理步覆盖完整20ms策略间隔，避免漏掉前几个子步的台阶冲击。
        self.scene.contact_forces.history_length = 8
        self.commands.base_velocity = mdp.OdinCourseVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.1, heading_command=False, debug_vis=False,
            ranges=copy.deepcopy(self.commands.base_velocity.ranges),
        )
        self.commands.base_posture.zero_on_course = True
        self.events.reset_base.func = mdp.reset_odin_course_root
        self.events.depth_memory_reset = EventTerm(func=mdp.reset_odin_depth_memory, mode="reset")
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.odin_safe_terrain_levels,
            params={"max_level_schedule": ((0, 0), (24_000, 2), (72_000, 5), (120_000, 8), (144_000, 11))},
        )
        self.curriculum.command_lin_vel.params.update(
            start_scale=0.4, end_scale=1.0, warmup_steps=12_000,
            ramp_steps=60_000, ramp_angular=True,
        )
        self.curriculum.posture_command.params.update(
            start_scale=0.0, end_scale=1.0,
            warmup_steps=72_000, ramp_steps=48_000,
        )
        self.events.push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
        }
        self.curriculum.push_curriculum = CurrTerm(
            func=mdp.push_velocity_curriculum,
            params={"start_step": 96_000, "ramp_steps": 36_000,
                    "max_push_x": 0.25, "max_push_y": 0.20, "max_push_z": 0.0},
        )
        self.rewards.track_lin_vel_xy_exp.func = mdp.odin_course_tracking
        self.rewards.track_lin_vel_xy_exp.params = {"std": 0.35}
        self.rewards.track_ang_vel_z_exp.func = mdp.odin_course_yaw_tracking
        self.rewards.track_ang_vel_z_exp.params = {"std": 0.5}
        self.rewards.goal_progress = RewTerm(func=mdp.odin_goal_progress, weight=2.0)
        self.rewards.goal_success = RewTerm(func=mdp.odin_goal_success, weight=250.0)
        self.rewards.termination_penalty.func = mdp.odin_failure
        self.rewards.height_track_exp.weight = 1.0
        self.rewards.height_track_exp.params["std"] = 0.08
        self.rewards.attitude_track_exp.weight = 0.3
        self.rewards.attitude_limit_l2.weight = -2.0
        self.rewards.all_wheels_contact.weight = 1.0
        self.rewards.all_wheels_contact.params["course_scale"] = 0.1
        self.rewards.leg_joint_vel_l2.weight = -5.0e-4
        self.rewards.leg_joint_vel_l2.params["course_scale"] = 0.4
        self.rewards.wheel_rolling_slip.weight = -10.0
        self.rewards.wheel_rolling_slip.params["course_scale"] = 0.25
        self.rewards.undesired_contacts.weight = -0.3
        self.rewards.undesired_contacts.params["threshold"] = 8.0
        self.rewards.wheel_impact = RewTerm(
            func=mdp.contact_load_excess, weight=-0.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_FOOT_LINK"]),
                    "threshold": 180.0, "scale": 180.0},
        )
        self.rewards.body_collision_load = RewTerm(
            func=mdp.contact_load_excess, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", ".*_ABAD_LINK", ".*_HIP_LINK", ".*_KENN_LINK"]),
                    "threshold": 20.0, "scale": 100.0},
        )
        # 到点是终态，不做超时价值bootstrap；失败惩罚显式排除成功项。
        self.terminations.goal_reached = DoneTerm(func=mdp.odin_goal_reached)
        self.terminations.course_bounds = DoneTerm(func=mdp.odin_course_out_of_bounds)
        self.terminations.root_height.params["minimum_height"] = 0.22


@configclass
class RPWDOdinDepthEnvCfg_TEACHER(RPWDOdinDepthEnvCfg):
    """8000轮清晰深度 PPO 教师，先学运动和赛道通过。"""

    def __post_init__(self):
        super().__post_init__()
        self.observations.depth_memory.depth_memory.params.update(
            noise_std=0.0, dropout_prob=0.0, random_crop=False,
            episode_bias_m=0.0, scale_error=0.0, occlusion_prob=0.0,
            packet_loss_prob=0.0, max_latency_frames=0,
        )
        self.curriculum.terrain_levels.params["max_level_schedule"] = (
            (0, 0), (24_000, 2), (60_000, 5), (96_000, 8), (108_000, 11)
        )


@configclass
class RPWDOdinDepthEnvCfg_STUDENT(RPWDOdinDepthEnvCfg):
    """2000轮在线蒸馏，教师使用同视野清晰历史，学生使用退化深度。"""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels.params["max_level_schedule"] = ((0, 11),)
        self.curriculum.command_lin_vel.params.update(start_scale=1.0, end_scale=1.0)
        self.curriculum.posture_command.params.update(start_scale=1.0, end_scale=1.0)
        self.curriculum.push_curriculum.params.update(start_step=12_000, ramp_steps=12_000)
        self.observations.depth_memory.depth_memory.params.update(
            augmentation_warmup_steps=0, augmentation_ramp_steps=24_000,
        )


@configclass
class RPWDOdinDepthEnvCfg_PLAY(RPWDOdinDepthEnvCfg):
    """视觉策略回放: 关闭训练增强/域随机化, 保留障碍地形和相机."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
        self.commands.base_velocity = UniformVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(8.0, 12.0),
            rel_standing_envs=0.0, heading_command=False, debug_vis=False,
            ranges=copy.deepcopy(self.commands.base_velocity.ranges),
        )
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params = {"command_name": "base_velocity", "std": 0.35}
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp
        self.rewards.track_ang_vel_z_exp.params = {"command_name": "base_velocity", "std": 0.5}
        self.rewards.goal_progress = self.rewards.goal_success = None
        self.terminations.goal_reached = self.terminations.course_bounds = None
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.depth_memory.depth_memory.params.update(
            {
                "noise_std": 0.0,
                "dropout_prob": 0.0,
                "random_crop": False,
                "episode_bias_m": 0.0, "scale_error": 0.0,
                "occlusion_prob": 0.0, "packet_loss_prob": 0.0,
                "max_latency_frames": 0,
            }
        )

        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_parameters = None
        self.events.sensor_bias = None
        self.events.push_robot = None

        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.curriculum = True
        self.curriculum.terrain_levels = None
        self.curriculum.command_lin_vel = None
        self.curriculum.posture_command = None
        self.curriculum.push_curriculum = None
        self.commands.base_velocity.rel_standing_envs = 0.0


@configclass
class RPWDOdinDepthEnvCfg_BENCHMARK(RPWDOdinDepthEnvCfg_PLAY):
    """固定障碍/台阶基准：每列一种场景，每场景默认2个并行环境。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(ODIN_BENCHMARK_SCENARIOS) * 2
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20261028,
            curriculum=True,
            size=(6.0, 6.0),
            num_rows=1,
            num_cols=len(ODIN_BENCHMARK_SCENARIOS),
            border_width=20.0,
            border_height=1.0,
            difficulty_range=(0.0, 0.0),
            use_cache=False,
            sub_terrains={
                name: OdinBenchmarkTerrainCfg(proportion=1.0, scenario=name)
                for name in ODIN_BENCHMARK_SCENARIOS
            },
        )
        self.scene.terrain.max_init_terrain_level = 0
        # 固定基准不根据行走距离升级/降级；每次复测都落在同一场景列。
        self.curriculum.terrain_levels = None
