"""教师/学生 checkpoint 的训练加载与部署推理共用入口。"""

from __future__ import annotations

import torch


def initialize_student(runner, path):
    """同结构教师复制给学生；只给教师的 CNN 键换组名，其他权重严格对齐。"""
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise ValueError("--teacher_checkpoint 必须指向本版本 PPO 教师 checkpoint。")
    state = checkpoint["actor_state_dict"]
    teacher_state = {key.replace("cnns.depth_memory.", "cnns.depth_teacher."): value for key, value in state.items()}
    runner.alg._raw_teacher.load_state_dict(teacher_state, strict=True)
    runner.alg._raw_student.load_state_dict(state, strict=True)
    runner.alg.teacher_loaded = True


def initialize_supermimic_student(runner, path):
    """加载特权教师，并按共有观测/地形编码器迁移可兼容权重到深度学生。"""
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    if "actor_state_dict" not in checkpoint:
        raise ValueError("特权教师必须是本SuperMimic PPO实验的checkpoint。")

    source_state = checkpoint["actor_state_dict"]
    teacher = runner.alg._raw_teacher
    student = runner.alg._raw_student
    teacher.load_state_dict(source_state, strict=True)

    if len(teacher.obs_groups_2d) != 1 or len(student.obs_groups_2d) != 1:
        raise ValueError("特权教师与深度学生各自必须恰好有一个二维编码组。")
    teacher_grid, student_grid = teacher.obs_groups_2d[0], student.obs_groups_2d[0]
    target_state = student.state_dict()
    transferred = {}
    adapted_first_layer = False
    adapted_first_conv = False

    observations = runner.env.get_observations()
    teacher_input_dim = teacher.obs_dim + teacher.cnn_latent_dim
    student_input_dim = student.obs_dim + student.cnn_latent_dim
    teacher_first = next(
        (key for key, value in source_state.items()
         if key.startswith("mlp.") and value.ndim == 2 and value.shape[1] == teacher_input_dim),
        None,
    )
    student_first = next(
        (key for key, value in target_state.items()
         if key.startswith("mlp.") and value.ndim == 2 and value.shape[1] == student_input_dim),
        None,
    )

    for target_key, target_value in target_state.items():
        source_key = target_key.replace(f"cnns.{student_grid}.", f"cnns.{teacher_grid}.")
        source_value = source_state.get(source_key)
        if source_value is None:
            continue

        if target_key == student_first and source_key == teacher_first:
            if teacher.cnn_latent_dim != student.cnn_latent_dim:
                continue
            adapted = target_value.clone()
            source_offset = 0
            target_offset = 0
            for group in teacher.obs_groups:
                if group not in student.obs_groups:
                    source_offset += observations[group].shape[-1]
                    continue
                group_dim = observations[group].shape[-1]
                if observations[group].ndim != 2 or group not in teacher.obs_groups:
                    raise ValueError(f"无法对齐一维共享观测组: {group}")
                adapted[:, target_offset : target_offset + group_dim] = source_value[
                    :, source_offset : source_offset + group_dim
                ]
                source_offset += group_dim
                target_offset += group_dim
            adapted[:, student.obs_dim :] = source_value[:, teacher.obs_dim :]
            if source_value.shape[0] == target_value.shape[0]:
                transferred[target_key] = adapted
                adapted_first_layer = True
            continue

        if target_key.endswith(".0.weight") and target_key.startswith(f"cnns.{student_grid}."):
            if (source_value.ndim == 4 and target_value.ndim == 4
                    and source_value.shape[0] == target_value.shape[0]
                    and source_value.shape[2:] == target_value.shape[2:]
                    and source_value.shape[1] == 1):
                transferred[target_key] = source_value.expand_as(target_value) / target_value.shape[1]
                adapted_first_conv = True
                continue

        if source_value.shape == target_value.shape:
            transferred[target_key] = source_value

    if not adapted_first_layer or not adapted_first_conv:
        raise ValueError(
            "教师/学生网络结构与预期不符：无法适配首层MLP或将单通道地形编码器迁移到8帧深度编码器。"
        )
    student.load_state_dict(transferred, strict=False)
    runner.alg.teacher_loaded = True
    copied = sum(value.numel() for value in transferred.values())
    total = sum(value.numel() for value in target_state.values())
    print(
        f"[INFO] 特权教师已严格加载；学生迁移了{len(transferred)}个参数张量 "
        f"({copied / total:.1%}参数)，首层已按共有本体观测/地形特征对齐。"
    )


def warm_start_policy(runner, path, reset_std: float = 0.7):
    """只迁移教师网络权重；不加载旧优化器、轮数或课程状态。"""
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    required = ("actor_state_dict", "critic_state_dict")
    if any(key not in checkpoint for key in required):
        raise ValueError("--warm_start_checkpoint 必须是带 actor/critic 的 PPO 教师 checkpoint。")
    runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    runner.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    distribution = getattr(runner.alg.actor, "distribution", None)
    std_param = getattr(distribution, "std_param", None)
    if std_param is not None:
        with torch.no_grad():
            std_param.fill_(reset_std)
    print(f"[INFO] 已热启动策略/价值网络，优化器、探索噪声与课程时钟重新初始化: {path}")


def load_inference_checkpoint(runner, path):
    """评测的 PPO 容器可加载在线蒸馏学生，导出仍为1740→16。"""
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    if "student_state_dict" in checkpoint:
        runner.alg.get_policy().load_state_dict(checkpoint["student_state_dict"], strict=True)
        print(f"[INFO] 已加载深度学生策略: {path}")
    else:
        runner.load(path)
