"""ONNX 导出后处理与契约校验 (play.py / test_policy.py 共用).

部署端只拷贝一个 .onnx 文件, 所以导出必须满足:
  1. 权重**内嵌** (torch 2.11 的导出器默认把权重写成 <name>.onnx.data, 必须合并)
  2. 输入 159 维 / 输出 16 维 (部署端 inference_node.cpp 会校验输入元素数)
"""

from __future__ import annotations

import os

# 部署端契约: 输入 = 单帧观测 53 × 帧栈 3; 输出 = 16 个关节
EXPECTED_INPUT_DIM = 159
EXPECTED_OUTPUT_DIM = 16


def consolidate_onnx(src: str, dst: str) -> None:
    """把外部权重合并进单文件."""
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    model = onnx.load(src)  # 默认会把外部权重读进内存
    convert_model_from_external_data(model)
    onnx.save_model(model, dst, save_as_external_data=False)


def verify_onnx(path: str) -> bool:
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
    if in_dim != EXPECTED_INPUT_DIM:
        print(f"[FAIL] 输入元素数 {in_dim} != 期望 {EXPECTED_INPUT_DIM}")
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


def export_and_verify(runner, export_dir: str) -> str:
    """导出单文件 ONNX 到 export_dir/policy_flat.onnx 并校验; 返回文件路径."""
    os.makedirs(export_dir, exist_ok=True)
    runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    src = os.path.join(export_dir, "policy.onnx")
    dst = os.path.join(export_dir, "policy_flat.onnx")  # 与部署端 model_names 同名
    consolidate_onnx(src, dst)
    # 清掉带外部权重的中间文件, 避免误拷
    for stale in (src, f"{src}.data"):
        if os.path.exists(stale):
            os.remove(stale)
    print(f"[INFO] 已导出(单文件): {dst}  ({os.path.getsize(dst) / 1024:.0f} KB)")
    verify_onnx(dst)
    return dst
