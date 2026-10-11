"""机器人资产配置.

这里**刻意不导入** rp_wd / rp_wd_contract: rp_wd.py 依赖 isaaclab.sim (carb),
必须在 SimulationApp 启动后才能导入; 而 rp_wd_contract.py 是纯常量模块,
要能在没有仿真时被校验脚本引用. 请按需显式导入:

    from rp_wd_rl.assets.rp_wd_contract import JOINT_ORDER   # 无仿真依赖
    from rp_wd_rl.assets.rp_wd import RP_WD_CFG              # 需先启动仿真
"""
