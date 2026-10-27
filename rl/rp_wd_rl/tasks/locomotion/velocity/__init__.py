"""注册 gym 环境 (Isaac Lab manager-based)."""

import gymnasium as gym

from . import agents

##
# 注册环境
##

gym.register(
    id="Isaac-Velocity-Flat-RPWD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_env_cfg:RPWDFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-RPWD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_env_cfg:RPWDFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDFlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Rough-RPWD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_rough_env_cfg:RPWDRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-RPWD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_rough_env_cfg:RPWDRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-RPWD-OdinDepth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_odin_depth_env_cfg:RPWDOdinDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDOdinDepthPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_odin_depth_env_cfg:RPWDOdinDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDOdinDepthPPORunnerCfg",
    },
)
