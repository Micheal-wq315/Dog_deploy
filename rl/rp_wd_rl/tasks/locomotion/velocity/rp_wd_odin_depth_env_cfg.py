"""Odin1 深度感知轮足任务: 随机障碍/地形 + 深度增强 + 腿部力反馈."""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import numpy as np
import trimesh
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from .rp_wd_env_cfg import ObservationsCfg
from .rp_wd_rough_env_cfg import RPWD_ROUGH_TERRAINS_CFG, RPWDRoughEnvCfg, RPWDRoughSceneCfg


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


def curriculum_random_rough_terrain(difficulty, cfg):
    """官方 random_uniform 忽略 difficulty，这里让粗糙幅度随地形行逐级增加。"""
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain

    scaled_cfg = copy.deepcopy(cfg)
    amplitude = cfg.noise_range[0] + difficulty * (cfg.noise_range[1] - cfg.noise_range[0])
    scaled_cfg.noise_range = (-amplitude, amplitude)
    return random_uniform_terrain(difficulty, scaled_cfg)


# 在原轮足 rough 课程中加入独立随机箱体障碍，保留原地形配方并重新归一化占比。
RPWD_ODIN_DEPTH_TERRAINS_CFG = copy.deepcopy(RPWD_ROUGH_TERRAINS_CFG)
RPWD_ODIN_DEPTH_TERRAINS_CFG.num_rows = 12
RPWD_ODIN_DEPTH_TERRAINS_CFG.num_cols = 12
for _terrain_cfg in RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains.values():
    _terrain_cfg.proportion *= 0.82
for _name in ("pyramid_stairs", "pyramid_stairs_inv"):
    RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains[_name].step_height_range = (0.01, 0.12)
RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.01, 0.10)
for _name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
    RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains[_name].slope_range = (0.0, 0.40)
_rough_cfg = RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains["random_rough"]
_rough_cfg.function = curriculum_random_rough_terrain
_rough_cfg.noise_range = (0.005, 0.055)  # difficulty 对应的振幅上下限；wrapper 转成 ±振幅
_rough_cfg.noise_step = 0.005
RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains["obstacle_boxes"] = terrain_gen.MeshRepeatedBoxesTerrainCfg(
    proportion=0.20,
    object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=1,
        height=0.08,  # mesh box 以地面为中心，可见高度约 4 cm 起步
        size=(0.25, 0.25),
        max_yx_angle=0.0,
    ),
    object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=8,
        height=0.44,  # 可见约 22 cm，高箱体以绕行为主
        size=(0.65, 0.55),
        max_yx_angle=5.0,
    ),
    platform_width=1.6,
    # 给中心出生平台一个很小的非零厚度，避免生成退化的零厚度碰撞盒。
    platform_height=0.05,
)
# 重点训练上/下台阶，同时保留足量障碍、方格和坡地。
for _name, _proportion in {
    "pyramid_stairs": 0.23,
    "pyramid_stairs_inv": 0.23,
    "boxes": 0.12,
    "random_rough": 0.10,
    "hf_pyramid_slope": 0.06,
    "hf_pyramid_slope_inv": 0.06,
}.items():
    RPWD_ODIN_DEPTH_TERRAINS_CFG.sub_terrains[_name].proportion = _proportion


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
                "crop_fraction_range": (0.80, 1.0),
                "random_crop": True,
                "augmentation_warmup_steps": 12_000,
                "augmentation_ramp_steps": 48_000,
                "memory_frames": ODIN_DEPTH_MEMORY_FRAMES,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = 0
            self.flatten_history_dim = False
            self.concatenate_terms = True

    policy: ObsGroup = PolicyCfg()
    depth_memory: ObsGroup = DepthMemoryCfg()


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

        self.curriculum.terrain_levels.params["max_level_schedule"] = (
            (0, 1), (48_000, 2), (96_000, 4), (144_000, 6), (168_000, 8)
        )  # 0 / 2000 / 4000 / 6000 / 7000 iter；由行走能力决定是否升级
        self.curriculum.command_lin_vel.params.update(
            start_scale=0.2, end_scale=1.0, warmup_steps=4_800,
            ramp_steps=76_800, ramp_angular=True,
        )  # 200–3400 iter: 速度/转向从小到契约范围
        self.curriculum.posture_command.params.update(
            start_scale=0.0, end_scale=1.0,
            warmup_steps=72_000, ramp_steps=64_800,
        )  # 3000–5700 iter: 先学移动，再逐渐学习姿态与高度
        self.events.push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
        }
        self.curriculum.push_curriculum = CurrTerm(
            func=mdp.push_velocity_curriculum,
            params={"start_step": 96_000, "ramp_steps": 48_000},
        )
        self.rewards.height_track_exp.weight = 1.0


@configclass
class RPWDOdinDepthEnvCfg_PLAY(RPWDOdinDepthEnvCfg):
    """视觉策略回放: 关闭训练增强/域随机化, 保留障碍地形和相机."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.depth_memory.depth_memory.params.update(
            {
                "noise_std": 0.0,
                "dropout_prob": 0.0,
                "random_crop": False,
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
        self.scene.terrain.terrain_generator.curriculum = False
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
