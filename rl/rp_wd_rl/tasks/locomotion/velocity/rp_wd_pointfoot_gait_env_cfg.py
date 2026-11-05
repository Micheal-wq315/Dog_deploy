"""点足 v4：先学完整低频摆脚，再按真实通过质量升级地形。"""
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from rp_wd_rl.supermimic.pointfoot_gait import reset_gait_phase
from rp_wd_rl.supermimic.pointfoot_gait_rewards import GAIT_REWARD_WEIGHTS
from . import mdp
from .rp_wd_pointfoot_full_env_cfg import RPWDPointfootFullEnvCfg, PointfootFullEventsCfg


@configclass
class PointfootGaitEventsCfg(PointfootFullEventsCfg):
    gait_phase = EventTerm(func=reset_gait_phase, mode="reset")


@configclass
class PointfootGaitRewardsCfg:
    def __post_init__(self):
        for name, weight in GAIT_REWARD_WEIGHTS.items():
            setattr(self, name, RewTerm(func=mdp.pointfoot_model_reward,
                                       weight=weight, params={"term": name}))


@configclass
class RPWDPointfootGaitEnvCfg(RPWDPointfootFullEnvCfg):
    supermimic_revision: int = 4
    terrain_pass_rate: float = .80
    terrain_minimum_trials: int = 128
    pointfoot_step_heights: tuple = (.02, .04, .06, .08, .10, .12)
    rewards: PointfootGaitRewardsCfg = PointfootGaitRewardsCfg()
    events: PointfootGaitEventsCfg = PointfootGaitEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        # 保留六类地形，先保证足够平地样本学停止、全向和完整摆脚。
        for kind, proportion in zip(("flat", "up", "down", "platform", "slope", "rubble"),
                                    (.4, .2, .1, .1, .1, .1)):
            gen.sub_terrains[kind].proportion = proportion
            gen.sub_terrains[kind].step_heights = self.pointfoot_step_heights
