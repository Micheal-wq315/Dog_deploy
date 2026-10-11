"""Isaac Sim 进程退出的兜底逻辑.

`SimulationApp.close()` 在某些情况下会挂住不返回 —— 实测在**开过 GUI 窗口**后必现:
脚本结果都打印完了, 进程却不退出, 只能 Ctrl-C (而且 Ctrl-C 还会被 carb 吞掉).

这里用后台线程 + 超时兜底: 正常能关就正常关, 超过 timeout_s 还没关掉就强制退出.
校验脚本的结果、训练的 checkpoint 都已经落盘, 强制退出不会丢东西.
"""

from __future__ import annotations

import logging
import os
import sys
import threading


def shutdown(simulation_app, exit_code: int = 0, timeout_s: float = 5.0) -> None:
    """关闭 Isaac Sim 并结束进程 (本函数**不会返回**).

    Args:
        simulation_app: `AppLauncher(...).app` 返回的 SimulationApp 实例.
        exit_code: 进程退出码.
        timeout_s: 等待优雅关闭的最长时间 [s], 超时则 os._exit.
    """
    done = threading.Event()

    def _close() -> None:
        try:
            simulation_app.close()
        except Exception as exc:  # noqa: BLE001 — 关闭失败不影响结果, 记录即可
            print(f"[WARN] 关闭仿真时出错 (已忽略): {exc}")
        finally:
            done.set()

    threading.Thread(target=_close, daemon=True).start()
    if not done.wait(timeout_s):
        print(f"[WARN] Isaac Sim 关闭超时 ({timeout_s:.0f}s), 强制退出进程.")

    # os._exit 不刷 stdio 缓冲 —— 输出重定向到文件时 (>/log 2>&1) 会丢掉最后一段日志,
    # 这里必须先 flush.
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(exit_code)
