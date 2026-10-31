"""在物理步之后、自动复位之前采样评测转移，保留终止帧。"""

from contextlib import contextmanager


@contextmanager
def capture_before_reset(manager, capture):
    """临时包装终止计算；即使抛异常也恢复原方法，不丢弃任何物理接触步。"""
    original = manager.compute
    snapshots = []

    def compute(*args, **kwargs):
        result = original(*args, **kwargs)
        snapshots.append(capture())
        return result

    manager.compute = compute
    try:
        yield snapshots
        if len(snapshots) != 1:
            raise RuntimeError(f"每个策略步必须采到一个复位前快照，实际{len(snapshots)}个。")
    finally:
        manager.compute = original
