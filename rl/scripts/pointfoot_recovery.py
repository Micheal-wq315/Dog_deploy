"""点足无效环境恢复；导入不依赖 Isaac Sim，可在 CPU 上验证。

依赖 Isaac Lab 2.3.2 的私有布局：
- ObservationManager._group_obs_term_history_buffer 为 group→term→CircularBuffer。
- CircularBuffer._buffer 为 [K,N,...]，_pointer 是整批共享的最新位置，
  _num_pushes 为 [N]。其 buffer getter 为
  roll(_buffer, K-_pointer-1, dim=0).transpose(0,1)。
- ManagerBasedEnv.reset(env_ids=...) 会对所有行 compute(update_history=True)，
  所以必须按 reset 后的新 pointer 恢复健康行，而不能直接还原旧物理槽位。

只修改无效行的回合失败标记和历史；不恢复无效物理状态，也不推进仿真。
"""

from __future__ import annotations

import torch


@torch.no_grad()
def recover_invalid_rows(env, bad: torch.Tensor):
    """将 bad:[N] bool 作为失败回合复位，返回已保护健康历史的观测 dict。

    调用方应在真实 step 后识别 bad，并继续将其从成功 archive 事件中排除。
    无 bad 行时仅重新读取观测，既不 reset，也不 append 历史。
    """
    env = getattr(env, "unwrapped", env)
    terminated = env.termination_manager.terminated
    if bad.dtype != torch.bool or bad.shape != (env.num_envs,):
        raise ValueError("bad 必须是 [num_envs] 的布尔张量")
    bad = bad.to(device=terminated.device)
    manager = env.observation_manager
    bad_ids = bad.nonzero(as_tuple=True)[0]
    if not bad_ids.numel():
        env.obs_buf = manager.compute(update_history=False)
        return env.obs_buf

    healthy_ids = (~bad).nonzero(as_tuple=True)[0]
    saved = []
    seen = set()
    if healthy_ids.numel():
        for terms in manager._group_obs_term_history_buffer.values():
            for buffer in terms.values():
                if id(buffer) in seen or buffer._buffer is None:
                    continue
                seen.add(id(buffer))
                ids = healthy_ids.to(device=buffer._buffer.device)
                # 高级索引产生独立副本；按时间排序保存，不能只复制旧原始槽位。
                saved.append((buffer, ids, buffer.buffer[ids], buffer._num_pushes[ids]))

    # _reset_idx 会先结算课程，必须在公共 reset 之前写入失败并清除成功锁存。
    terminated[bad] = True
    episode = getattr(env, "_pointfoot_full_episode", None)
    if episode is not None:
        episode["passed"][bad] = False
    success = getattr(env, "_pointfoot_success_event", None)
    if success is not None:
        success[bad] = False

    try:
        env.reset(env_ids=bad_ids)
    finally:
        for buffer, ids, ordered, pushes in saved:
            # getter 的逆变换；保留新 pointer，使 bad 行的真实复位帧保持正确。
            shift = buffer.max_length - buffer._pointer - 1
            buffer._buffer[:, ids] = torch.roll(ordered.transpose(0, 1), shifts=-shift, dims=0)
            buffer._num_pushes[ids] = pushes

    # 同时刷新 ManagerBasedEnv 和 ObservationManager 的缓存，不能再追加一帧。
    env.obs_buf = manager.compute(update_history=False)
    return env.obs_buf
