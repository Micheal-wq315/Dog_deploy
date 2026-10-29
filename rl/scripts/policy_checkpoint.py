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


def load_inference_checkpoint(runner, path):
    """评测的 PPO 容器可加载在线蒸馏学生，导出仍为1740→16。"""
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    if "student_state_dict" in checkpoint:
        runner.alg.get_policy().load_state_dict(checkpoint["student_state_dict"], strict=True)
        print(f"[INFO] 已加载深度学生策略: {path}")
    else:
        runner.load(path)
