"""注册 gym 环境 (Isaac Lab manager-based)."""

import gymnasium as gym

from . import agents

for _stage, _env_class, _runner_class in (
    ("Teacher", "RPWDOdinDepthEnvCfg_TEACHER", "RPWDOdinTeacherRunnerCfg"),
    ("Student", "RPWDOdinDepthEnvCfg_STUDENT", "RPWDOdinStudentRunnerCfg"),
    (
        "SuperMimic-Teacher",
        "RPWDOdinSuperMimicTeacherEnvCfg",
        "RPWDOdinSuperMimicTeacherRunnerCfg",
    ),
    (
        "SuperMimic-Student",
        "RPWDOdinSuperMimicStudentEnvCfg",
        "RPWDOdinSuperMimicStudentRunnerCfg",
    ),
):
    gym.register(
        id=f"Isaac-Velocity-Rough-RPWD-OdinDepth-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rp_wd_odin_depth_env_cfg:{_env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_runner_class}",
        },
    )

# 世界模型阶段由 train_supermimic.py 自定义循环驱动，不走RSL-RL/PPO runner。
for _stage, _env_class in (
    ("Teacher", "RPWDOdinSuperMimicTeacherEnvCfg"),
    ("Student", "RPWDOdinSuperMimicStudentEnvCfg"),
):
    gym.register(
        id=f"Isaac-Velocity-Rough-RPWD-OdinDepth-SuperMimic-WM-{_stage}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rp_wd_odin_depth_env_cfg:{_env_class}",
        },
    )

gym.register(
    id="Isaac-Velocity-Rough-RPWD-OdinDepth-SuperMimic-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_odin_depth_env_cfg:RPWDOdinSuperMimicTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDOdinSuperMimicTeacherRunnerCfg",
    },
)

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
    id="Isaac-Velocity-Rough-RPWD-Pointfoot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_pointfoot_env_cfg:RPWDPointfootEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDPointfootPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-WM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.rp_wd_pointfoot_supermimic_env_cfg:RPWDPointfootSuperMimicEnvCfg"
        ),
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

gym.register(
    id="Isaac-Velocity-Rough-RPWD-OdinDepth-Benchmark-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rp_wd_odin_depth_env_cfg:RPWDOdinDepthEnvCfg_BENCHMARK",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RPWDOdinDepthPPORunnerCfg",
    },
)
