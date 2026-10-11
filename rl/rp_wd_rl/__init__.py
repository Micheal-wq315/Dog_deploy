"""RP_wd 强化学习训练工程 (Isaac Lab 外部工程).

这个包只负责*训练侧*: 机器人资产配置 + MDP 任务定义 + PPO 超参.
训练产出的 ONNX 策略与 ../../src/inference/robots/wheel_quad 的推理节点对接.

注意: 本文件刻意不导入 tasks 子包 —— Isaac Lab 要求先启动 SimulationApp 再导入
任务定义 (见 rl/scripts/train.py 的导入顺序).
"""
