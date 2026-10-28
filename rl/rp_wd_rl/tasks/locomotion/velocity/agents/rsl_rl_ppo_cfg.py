"""RSL-RL PPO 超参 (基于 Isaac Lab 的 anymal_c 配置)."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RPWDFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """平地速度跟踪任务的 PPO 配置.

    说明:
      - `policy=` 是 Isaac Lab 2.3.2 自带任务统一使用的写法, 在 rsl-rl 5.x 下会被
        `handle_deprecated_rsl_rl_cfg` 自动迁移到新的 actor/critic 模型配置 (会有告警, 属正常).
      - `clip_actions=10.0` **不是**可有可无的: 部署端对模型原始输出裁剪 ±10 后再喂回
        `last_action` 观测, 这里必须一致 (见 default.yaml 的 clip_actions).
      - 关闭 obs normalization: 部署端推理不做归一化 (obs_scales 全为 1.0), 训练侧保持一致.
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "rp_wd_flat"
    clip_actions = 10.0

    policy = RslRlPpoActorCriticCfg(
        # 探索噪声 1.0 对轮足偏大 (raw 动作 ±10, 轮子 ×5 = ±50 rad/s):
        # 早期动作过猛会一路摔, 且动作率惩罚被噪声放大. 实测降到 0.7 更稳.
        init_noise_std=0.7,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RPWDRoughPPORunnerCfg(RPWDFlatPPORunnerCfg):
    """崎岖地形任务的 PPO 配置 (超参沿用平地版, 只换日志目录名)."""

    experiment_name = "rp_wd_rough"


@configclass
class RPWDOdinDepthPPORunnerCfg(RPWDRoughPPORunnerCfg):
    """Odin1 深度避障 PPO: CNN 编码8帧深度记忆，MLP融合本体观测."""

    max_iterations = 8000
    # 另开日志空间，避免误从旧780维 MLP checkpoint 续训。
    experiment_name = "rp_wd_odin_depth_memory"

    # policy 仍是 56维基础观测 + 12维力矩观测，每项按部署契约堆叠3帧。
    # CNN 对独立深度组 [B,8,16,12] 编码，空间维先保留到第二层再池化。
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.7),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[1, 1],
            padding="zeros",
            norm="none",
            activation="elu",
            max_pool=[False, True],
            global_pool="none",
            flatten=True,
        ),
    )
    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            stride=[1, 1],
            padding="zeros",
            norm="none",
            activation="elu",
            max_pool=[False, True],
            global_pool="none",
            flatten=True,
        ),
    )
    obs_groups = {
        "actor": ["policy", "depth_memory"],
        "critic": ["policy", "depth_memory"],
    }
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=False,
    )
