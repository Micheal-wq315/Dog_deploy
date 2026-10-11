"""任务定义 (gym.register 在这里的子包里执行).

导入方式 (必须在 SimulationApp 启动之后):
    import rp_wd_rl.tasks  # noqa: F401
"""

from isaaclab_tasks.utils import import_packages

# 递归导入子包 -> 触发各 config/__init__.py 里的 gym.register
_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
