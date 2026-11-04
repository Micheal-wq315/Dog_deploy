"""full-v3：六类双向地形、多方向命令与真实表现门控，保持225→12盲 Actor。"""

from __future__ import annotations

import math

from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from rp_wd_rl.supermimic.pointfoot_courses import COURSE_KINDS, COURSE_PROPORTIONS
from rp_wd_rl.supermimic.pointfoot_terrain import FULL_REWARD_WEIGHTS
from rp_wd_rl.tasks.locomotion.velocity import mdp
from rp_wd_rl.tasks.locomotion.velocity.mdp import pointfoot_full
from rp_wd_rl.tasks.locomotion.velocity.rp_wd_pointfoot_supermimic_env_cfg import (
    PointfootSuperMimicEventsCfg,
    PointfootSuperMimicSceneCfg,
    PointfootSuperMimicTerminationsCfg,
    RPWDPointfootSuperMimicEnvCfg,
)


POINTFOOT_FULL_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16., 16.),
    num_rows=6,
    num_cols=10,
    border_width=20.,
    curriculum=True,
    difficulty_range=(0., 1.),
    use_cache=False,
    sub_terrains={kind: pointfoot_full.PointfootFullCourseCfg(kind=kind, proportion=weight)
                  for kind, weight in zip(COURSE_KINDS, COURSE_PROPORTIONS)},
)


@configclass
class PointfootFullSceneCfg(PointfootSuperMimicSceneCfg):
    """继承原固定球脚资产、摩擦和两个特权测距器，仅替换地形生成器。"""

    def __post_init__(self):
        self.terrain = self.terrain.replace(
            terrain_generator=POINTFOOT_FULL_TERRAINS_CFG.copy(), max_init_terrain_level=0,
        )


@configclass
class PointfootFullCommandsCfg:
    base_velocity = pointfoot_full.PointfootFullVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3., 6.),
        rel_standing_envs=.10,
        heading_command=False,
        rel_heading_envs=0.,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-.25, .25), lin_vel_y=(-.15, .15), ang_vel_z=(-.4, .4),
        ),
    )


@configclass
class PointfootFullEventsCfg(PointfootSuperMimicEventsCfg):
    reset_base = EventTerm(func=pointfoot_full.reset_full_course_root, mode="reset")
    reset_stair_events = EventTerm(func=pointfoot_full.reset_full_course_state, mode="reset")


@configclass
class PointfootFullRewardsCfg:
    """权重和高度奖励由共享 PointfootTaskRewards(profile='full') 提供。"""

    def __post_init__(self):
        for name, weight in FULL_REWARD_WEIGHTS.items():
            setattr(self, name, RewTerm(func=mdp.pointfoot_model_reward, weight=weight, params={"term": name}))


@configclass
class PointfootFullCurriculumCfg:
    terrain_levels = CurrTerm(func=pointfoot_full.full_terrain_curriculum)


@configclass
class PointfootFullTerminationsCfg(PointfootSuperMimicTerminationsCfg):
    course_exit = DoneTerm(func=pointfoot_full.full_course_exit, time_out=True)
    course_stuck = DoneTerm(func=pointfoot_full.full_course_stuck, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.)})
    root_height = DoneTerm(func=pointfoot_full.full_low_support_height, params={"minimum_height": .12})


@configclass
class RPWDPointfootFullEnvCfg(RPWDPointfootSuperMimicEnvCfg):
    """保持原资产、物理 context、五帧本体观测、12关节动作和50Hz策略频率。"""

    supermimic_revision: int = 3
    scene: PointfootFullSceneCfg = PointfootFullSceneCfg(num_envs=64, env_spacing=2.5)
    commands: PointfootFullCommandsCfg = PointfootFullCommandsCfg()
    rewards: PointfootFullRewardsCfg = PointfootFullRewardsCfg()
    terminations: PointfootFullTerminationsCfg = PointfootFullTerminationsCfg()
    events: PointfootFullEventsCfg = PointfootFullEventsCfg()
    curriculum: PointfootFullCurriculumCfg = PointfootFullCurriculumCfg()
