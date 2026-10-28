"""Odin1 深度感知轮足任务: 随机障碍/地形 + 深度增强 + 腿部力反馈."""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from rp_wd_rl.assets.rp_wd_contract import ODIN_DEPTH_IMAGE_SIZE

from . import mdp
from .rp_wd_env_cfg import ObservationsCfg, SceneCfg
from .rp_wd_rough_env_cfg import RPWD_ROUGH_TERRAINS_CFG, RPWDRoughEnvCfg


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
    proportion=0.18,
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


@configclass
class RPWDOdinDepthSceneCfg(SceneCfg):
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
        data_types=["distance_to_camera"],
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
        # 每帧 12×16 = 192 个距离特征; 历史长度由父类设为 3 帧。
        perception = ObsTerm(
            func=mdp.odin_depth_grid,
            params={
                "sensor_cfg": SceneEntityCfg("odin_depth_camera"),
                "data_type": "distance_to_camera",
                "max_depth": 6.0,
                "grid_size": (12, 16),
                "noise_std": 0.03,
                "dropout_prob": 0.01,
                "crop_fraction_range": (0.80, 1.0),
                "random_crop": True,
                "augmentation_warmup_steps": 12_000,  # 500 iter 后渐入
                "augmentation_ramp_steps": 48_000,    # 2500 iter 达到完整增强
            },
        )

    policy: ObsGroup = PolicyCfg()


@configclass
class RPWDOdinDepthEnvCfg(RPWDRoughEnvCfg):
    """Odin1 深度策略: 256 env，8000 iter，地形与任务难度逐阶段解锁。"""

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
            (0, 1), (24_000, 3), (60_000, 5), (108_000, 8), (144_000, 11)
        )  # 0 / 1000 / 2500 / 4500 / 6000 iter；成功升级、失败降级
        self.curriculum.command_lin_vel.params.update(
            start_scale=0.2, end_scale=1.0, warmup_steps=2_400,
            ramp_steps=55_200, ramp_angular=True,
        )  # 100–2400 iter: ±0.12 → ±0.6 m/s，±0.2 → ±1 rad/s
        self.curriculum.posture_command.params.update(
            start_scale=0.0, end_scale=1.0,
            warmup_steps=36_000, ramp_steps=60_000,
        )  # 1500–4000 iter: 姿态与高度指令逐步达到完整范围
        self.events.push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
        }
        self.curriculum.push_curriculum = CurrTerm(
            func=mdp.push_velocity_curriculum,
            params={"start_step": 28_800, "ramp_steps": 91_200},
        )  # 1200–5000 iter: 周期推扰从零渐增

        # 后 30% 训练渐进加入节能约束，保留最后 1200 iter 稳定巩固。
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
        )
        self.curriculum.leg_torque_penalty = CurrTerm(
            func=mdp.late_reward_weight_ramp,
            params={
                "term_name": "dof_torques_l2",
                "start_step": 134_400,  # 5600 iter × 24 rollout steps
                "ramp_steps": 28_800,   # 1200 iter，6800 iter 达到目标
                "target_weight": -1.0e-5,
                "ramp_bins": 20,
            },
        )


@configclass
class RPWDOdinDepthEnvCfg_PLAY(RPWDOdinDepthEnvCfg):
    """视觉策略回放: 关闭训练增强/域随机化, 保留障碍地形和相机."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.policy.perception.params.update(
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
        self.curriculum.leg_torque_penalty = None
        self.commands.base_velocity.rel_standing_envs = 0.0
