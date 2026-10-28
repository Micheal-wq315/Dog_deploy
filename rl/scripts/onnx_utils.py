"""ONNX 导出后处理与契约校验 (play.py / test_policy.py 共用).

部署端只拷贝一个 .onnx 文件, 所以导出必须满足:
  1. 权重**内嵌** (torch 2.11 的导出器默认把权重写成 <name>.onnx.data, 必须合并)
  2. 输入维度按当前策略观测推导 / 输出 16 维 (部署端会校验输入元素数)
"""

from __future__ import annotations

import os

import torch
from torch import nn

from rp_wd_rl.assets.rp_wd_contract import ONNX_INPUT_DIM

# 普通策略部署端契约: 输入 = 单帧观测 56 × 帧栈 3; 输出 = 16 个关节
EXPECTED_INPUT_DIM = ONNX_INPUT_DIM
EXPECTED_OUTPUT_DIM = 16


def consolidate_onnx(src: str, dst: str) -> None:
    """把外部权重合并进单文件."""
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    model = onnx.load(src)  # 默认会把外部权重读进内存
    convert_model_from_external_data(model)
    onnx.save_model(model, dst, save_as_external_data=False)


def verify_onnx(path: str, expected_input_dim: int = EXPECTED_INPUT_DIM) -> bool:
    """用 onnxruntime 校验导出的模型与部署端契约一致."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs, outputs = sess.get_inputs(), sess.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        print(f"[FAIL] 输入/输出张量数应为 1/1, 实际 {len(inputs)}/{len(outputs)}")
        return False

    inp, out = inputs[0], outputs[0]
    in_dim = int(np.prod([d for d in inp.shape if isinstance(d, int)]))
    out_dim = int(np.prod([d for d in out.shape if isinstance(d, int)]))
    print(f"[INFO] ONNX: 输入 '{inp.name}' {inp.shape} ({inp.type}) -> 输出 '{out.name}' {out.shape}")
    ok = True
    if in_dim != expected_input_dim:
        print(f"[FAIL] 输入元素数 {in_dim} != 期望 {expected_input_dim}")
        ok = False
    if out_dim != EXPECTED_OUTPUT_DIM:
        print(f"[FAIL] 输出元素数 {out_dim} != 期望 {EXPECTED_OUTPUT_DIM}")
        ok = False

    # 权重必须内嵌: 部署端只拷这一个文件, 依赖 <name>.data 会在实机上加载失败
    bare = onnx.load(path, load_external_data=False)
    if any(len(t.external_data) > 0 for t in bare.graph.initializer):
        print("[FAIL] 模型仍引用外部权重文件 (.data), 部署端只拷贝 .onnx 会加载失败")
        ok = False
    else:
        print("[INFO] 权重已内嵌 (单文件, 可直接拷到部署端)")

    # 跑一次全零输入, 确认能推理且无 NaN
    dummy = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)
    y = sess.run([out.name], {inp.name: dummy})[0]
    if not np.isfinite(y).all():
        print("[FAIL] 推理输出含 NaN/Inf")
        ok = False
    else:
        print(f"[INFO] 空输入推理输出: {np.array2string(y.reshape(-1)[:6], precision=4)} ...")
    print("[OK] ONNX 契约校验通过" if ok else "[FAIL] ONNX 契约校验失败")
    return ok


class _FlatCNNPolicy(nn.Module):
    """Wrap RSL-RL's split proprioception/image ONNX model as one flat input."""

    def __init__(self, model, main_dim: int, channels: int, height: int, width: int):
        super().__init__()
        self.model = model
        self.main_dim = main_dim
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprioception = obs[:, : self.main_dim]
        depth = obs[:, self.main_dim :].reshape(
            obs.shape[0], self.channels, self.height, self.width
        )
        return self.model(proprioception, depth)


def expected_runner_input_dim(runner) -> int:
    """Return flattened deployed input size for MLP or CNN RSL-RL policies."""
    actor = runner.alg.get_policy()
    if hasattr(actor, "obs_dims_2d"):
        if len(actor.obs_dims_2d) != 1:
            raise ValueError("部署端单深度输入只支持一个 CNN 观测组")
        height, width = actor.obs_dims_2d[0]
        return int(actor.obs_dim + actor.obs_channels_2d[0] * height * width)
    return int(actor.obs_dim)


def export_and_verify(
    runner,
    export_dir: str,
    expected_input_dim: int = EXPECTED_INPUT_DIM,
    output_filename: str = "policy_flat.onnx",
) -> str:
    """导出单文件 ONNX 并按当前策略的 I/O 契约校验; 返回文件路径."""
    os.makedirs(export_dir, exist_ok=True)
    actor = runner.alg.get_policy()
    src = os.path.join(export_dir, "policy_raw.onnx")
    if hasattr(actor, "obs_dims_2d"):
        if len(actor.obs_dims_2d) != 1:
            raise ValueError("部署端单输入 ONNX 只支持一个 CNN 深度观测组")
        onnx_model = actor.as_onnx(verbose=False).to("cpu").eval()
        height, width = onnx_model.obs_dims_2d[0]
        channels = onnx_model.obs_channels_2d[0]
        flat_model = _FlatCNNPolicy(
            onnx_model, onnx_model.obs_dim_1d, channels, height, width
        ).eval()
        flat_example = torch.zeros(1, expected_runner_input_dim(runner), dtype=torch.float32)
        torch.onnx.export(
            flat_model,
            flat_example,
            src,
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
        )
    else:
        runner.export_policy_to_onnx(path=export_dir, filename=os.path.basename(src))
    dst = os.path.join(export_dir, output_filename)
    consolidate_onnx(src, dst)
    # 清掉带外部权重的中间文件, 避免误拷
    for stale in (src, f"{src}.data"):
        if os.path.exists(stale):
            os.remove(stale)
    print(f"[INFO] 已导出(单文件): {dst}  ({os.path.getsize(dst) / 1024:.0f} KB)")
    expected = (
        expected_runner_input_dim(runner)
        if hasattr(actor, "obs_dims_2d")
        else expected_input_dim
    )
    if not verify_onnx(dst, expected_input_dim=expected):
        raise RuntimeError(f"ONNX 契约验证失败: {dst}")
    return dst
