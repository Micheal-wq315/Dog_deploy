"""RP_wd 崎岖地形速度跟踪任务 (manager-based).

**契约不变**: 观测仍是 53×3=159、动作仍是 16、仍是 50 Hz —— 崎岖地形靠的是
策略在训练中见过地形，不需要额外传感器，所以部署端零改动。

⚠️ 不要直接用 Isaac Lab 的 `ROUGH_TERRAINS_CFG`: 它是给腿足调的，台阶上限 0.23 m
≈ 2.25× 轮半径 (0.1025 m)，盲走的轮足根本不可能上去。下面是**轮足专用配方**：
台阶 0.02–0.08 m (0.2–0.8× 轮半径)、坡度 0–20°、随机粗糙 RMS 0.03–0.04 m。

数据依据 (盲走能力估计): 台阶 0.04–0.08 m 可靠 / 0.08–0.12 m 极限；
坡度 15–20° 可靠 / 25° 极限 —— 超过 0.12 m 或 25° 盲走不可行
(单轮抬升行程约 0.15 m + 三轮支撑稳定边界)。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .rp_wd_env_cfg import RPWDFlatEnvCfg

##
# 地形配方 (轮足专用)
##

RPWD_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    # 6×6 m 的地块, 6×6=36 块。升级阈值 = size/2 = 3 m, 而一回合能走 8–12 m, 够用;
    # 比官方默认的 8×8 m × 200 块省 10 倍三角面 (约 15–20 万 vs 200 万)。
    size=(6.0, 6.0),
    num_rows=6,
    num_cols=6,
    border_width=20.0,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,  # 改了配方就别吃旧缓存
    sub_terrains={
        # 上台阶: 0.02 起步让课程从平地缓升; 0.08 是首版上限 (纯滚动只能过 0.03–0.05,
        # 0.06+ 需要抬轮), 训练顺利后第二版可放宽到 0.10。阶宽 0.4 ≥ 2×轮直径(0.205)。
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.08),
            step_width=0.4,
            platform_width=2.0,
            border_width=0.25,
            holes=False,
        ),
        # 下台阶: 跌落冲击, 同样限 0.08
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.08),
            step_width=0.4,
            platform_width=2.0,
            border_width=0.25,
            holes=False,
        ),
        # 随机方格: 砖缝/井盖/路缘; 格宽 > 轮直径 0.205, 避免永远卡在缝里.
        # ⚠️ grid_width 必须**不能整除** size (0.4 整除 6.0 会让边框宽 = 0, Isaac Lab 直接
        #    RuntimeError); 0.45 → ⌊6/0.45⌋=13 块, 边框 0.15 m ✓
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.02, 0.07),
            platform_width=2.0,
        ),
        # 随机粗糙: 目标 RMS 0.03–0.04 m, 单点凸起 ≤ 0.05 m
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.04),
            noise_step=0.01,
            # downsampled_scale > horizontal_scale 会把噪声**平滑**到更大的尺度上。
            # 用默认值(0.1 m)时地面是 1 cm 级的碎尖刺, 轮子压上去会产生大量接触面
            # (PhysX 求解量暴增, 实测占地形训练耗时的大头), 对轮子也不真实 ——
            # 轮子要的是"起伏", 不是"尖刺"。
            downsampled_scale=0.3,
            border_width=0.25,
        ),
        # 上坡 / 下坡: 15–20° 盲走可靠, 22°+ 收益递减
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.35),  # 0–20°
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""轮足专用的地形生成配方 (台阶/方格/粗糙/坡度, 参数按轮半径 0.1025 m 标定)."""


##
# 环境
##


@configclass
class RPWDRoughEnvCfg(RPWDFlatEnvCfg):
    """崎岖地形 (训练): 相对平地版只改三处 —— 地形、地形课程、5 项奖励权重."""

    def __post_init__(self):
        super().__post_init__()

        # ---- 1) 换地形: 生成器 + 从低难度起步 (直接给高难度会一出生就摔) ----
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=RPWD_ROUGH_TERRAINS_CFG,
            max_init_terrain_level=2,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )
        # 地形换了, 仿真材质要跟着重设一次 (super() 里设的是平地那份)
        self.sim.physics_material = self.scene.terrain.physics_material

        # ---- 2) 地形课程 ----
        # 注意: 本配置直接继承 ManagerBasedRLEnvCfg, 没有 Isaac Lab velocity 任务里那段
        # "自动同步 terrain_generator.curriculum" 的逻辑 —— **必须手动打开**, 否则课程无效。
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        self.scene.terrain.terrain_generator.curriculum = True

        # ---- 3) 奖励调整 (这 5 项在平地版是对的, 在崎岖地形会把策略压成"不敢动") ----
        # 抬轮上台阶时, 轮子必然瞬时离地/失配, 权重 -10 会直接压制越障动作
        self.rewards.wheel_rolling_slip.weight = -2.5
        # 坡上姿态本身就难跟 (指令 pitch/roll 叠加坡面倾角), 跟踪权重略降
        self.rewards.attitude_track_exp.weight = 0.4
        # 坡上"绝对高度"跟踪意义弱 (地形起伏), 高度跟踪权重降到 0.2
        self.rewards.height_track_exp.weight = 0.2
        # 台阶上四轮同时着地比例天然下降, 别把爬台阶罚亏
        self.rewards.all_wheels_contact.weight = 0.3
        # 爬阶时小腿(KENN)蹭地是合法的, 阈值 1 N -> 5 N
        self.rewards.undesired_contacts.params["threshold"] = 5.0

        # ---- 4) 终止条件修正 (关键 bug) ----
        # 平地版用的是"世界绝对高度 < 0.18 m"; 生成地形的各地块平台高度不同
        # (实测 36 块从 -0.36 到 +0.40 m), 落在低处的环境会**一出生就被判死** ——
        # 实测 Play 里回合长度中位数 1 步、训练里 root_height 终止占 14-16%
        # (白白浪费约 1/3 的样本并扭曲地形课程)。改成相对本环境地形原点的高度。
        self.terminations.root_height = DoneTerm(
            func=mdp.root_height_below_local_minimum,
            params={"minimum_height": 0.18},
        )


@configclass
class RPWDRoughEnvCfg_PLAY(RPWDRoughEnvCfg):
    """崎岖地形 (测试/演示/遥控): 关随机化, 全难度地形都出现, 指令范围直接满额."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32  # 32 个环境撒到 36 块地形上, 基本各自一块, GUI 里看得清
        self.observations.policy.enable_corruption = False

        # 关域随机化
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.actuator_gains = None
        self.events.joint_parameters = None
        self.events.sensor_bias = None
        self.events.push_robot = None

        # 测试/演示: 关地形课程, 指令范围用契约满额.
        # 难度**默认钉在 2 级** (策略实际训练到的水平) —— 之前设 None(全难度随机, 含最难的 5 级)
        # 会让测试结果被"在硬地形上一出生就摔"的样本淹没. 想探边界就用
        # test_policy.sh --rough --terrain-level 5
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.curriculum.command_lin_vel = None
        self.commands.base_velocity.rel_standing_envs = 0.0
