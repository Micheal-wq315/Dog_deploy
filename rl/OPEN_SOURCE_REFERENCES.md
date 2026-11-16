# 本仓库引用的开源项目

本文汇总本仓库代码、子模块、训练/部署文档及已克隆参考仓库中明确引用的开源项目。链接按用途分类，并标注它们在本仓库中的实际关系，避免把“算法参考”误认为运行时依赖。

## 算法与机器人训练

| 项目 | 开源链接 | 与本仓库的关系 |
|---|---|---|
| Go2W SuperMimic | [uwvwko-zzz/go2w-supermimic](https://github.com/uwvwko-zzz/go2w-supermimic) | 已克隆的原版 Go2W 轮足训练项目；本仓库用它进行独立对照训练。 |
| Go2 SuperMimic | [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic) | Go2W 项目声明的算法上游。 |
| UW-HIMLoco 轮足分支 | [uwvwko-zzz/uw-himloco/tree/wheel](https://github.com/uwvwko-zzz/uw-himloco/tree/wheel) | 已克隆的 HIM 轮足参考实现；另有 MuJoCo 跨仿真播放器。 |
| HIMLoco | [InternRobotics/HIMLoco](https://github.com/InternRobotics/HIMLoco) | HIM 估计器和训练架构的上游项目。 |
| Extreme Parkour | [chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) | 已克隆的视觉教师/学生与深度感知参考项目。 |
| Walk These Ways | [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) | 步态、支撑/摆动相位与奖励设计参考。 |
| legged_gym | [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | 足式机器人环境、接触与腾空奖励参考；也是 Extreme Parkour 的代码基础之一。 |
| Unitree RL mjlab | [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) | Go2W 项目声明的机器人资产来源。 |
| skrl | [Toni-SM/skrl](https://github.com/Toni-SM/skrl) | Extreme Parkour README 致谢的强化学习库；不是本仓库当前训练器依赖。 |
| Watch Your Step / Foothold Prediction | [leggedrobotics/foothold-prediction-model](https://github.com/leggedrobotics/foothold-prediction-model) | ETH Zurich 人形 Unitree G1 的落脚点预测（FPM）；论文 arXiv:2610.29612v1（2026-10）；中文导读 `output/pdf/watch_your_step_中文导读.pdf`，官方代码在主工作区根目录有本地克隆。研究参考，不是当前四足训练/部署依赖；人形→四足迁移需重做。 |
| go2_quad2hand | [Sang-SC/go2_quad2hand](https://github.com/Sang-SC/go2_quad2hand) | Unitree Go2 标准12关节四足+双足倒立；Isaac Gym + legged_gym + rsl_rl 1.0.2，PPO + 显式线速度估计器 + 非对称 Actor-Critic；MuJoCo/C++(SDK2,ONNX)部署；BSD-3；代码最后推送 2026-03-14。参考其估计器/倒立步态；平地训练、无台阶，不是当前依赖。 |

## 仿真与物理引擎

| 项目 | 开源链接 | 与本仓库的关系 |
|---|---|---|
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | RP_wd 训练环境使用的仿真与任务框架。 |
| mjlab | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | 原版 Go2W SuperMimic 使用的 MuJoCo 机器人学习框架。 |
| MuJoCo | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | HIMLoco 跨仿真验证及 Go2W 物理仿真的底层引擎。 |
| MuJoCo Warp | [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | Go2W 项目 GPU 并行物理仿真使用的组件。 |
| NVIDIA Warp | [NVIDIA/warp](https://github.com/NVIDIA/warp) | Go2W/mjlab GPU 计算依赖。 |

## 机器学习、数值计算与模型格式

| 项目 | 开源链接 | 与本仓库的关系 |
|---|---|---|
| RSL-RL | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | Isaac Lab PPO 训练器依赖。 |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | 训练、策略推理模型定义及导出所用框架。 |
| Gymnasium | [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | 强化学习环境接口依赖。 |
| ONNX | [onnx/onnx](https://github.com/onnx/onnx) | 训练策略的交换与部署模型格式。 |
| ONNX Runtime | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | ROS2 部署端加载并执行 ONNX 策略。 |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) | 训练、地形与数据处理依赖。 |
| SciPy | [scipy/scipy](https://github.com/scipy/scipy) | 数值和几何辅助计算依赖。 |
| trimesh | [mikedh/trimesh](https://github.com/mikedh/trimesh) | 网格地形生成及处理依赖。 |

## ROS2 部署、导航及 C++ 依赖

| 项目 | 开源链接 | 与本仓库的关系 |
|---|---|---|
| roboparty_inference | [wentywenty/roboparty_inference](https://github.com/wentywenty/roboparty_inference) | `src/inference/debian/copyright` 中记录的源码地址；该上游地址当前可访问性尚未确认。 |
| jie_3d_nav | [6-robot/jie_3d_nav](https://github.com/6-robot/jie_3d_nav) | Odin1 深度导航集成方案引用的 ROS2 导航项目。 |
| mujoco-gui | [uwvwko-zzz/mujoco-gui](https://github.com/uwvwko-zzz/mujoco-gui) | UW-HIMLoco 中包含的浏览器 MuJoCo GUI 项目。 |
| ArenaX | [Lain-Ego0/ArenaX](https://github.com/Lain-Ego0/ArenaX) | mujoco-gui 文档提及的早期设计参考。 |
| ROS 2 | [ros2/ros2](https://github.com/ros2/ros2) | 部署端使用的 ROS2 框架；本机部署目标为 Jazzy。 |
| cnpy | [rogersce/cnpy](https://github.com/rogersce/cnpy) | `src/inference/.gitmodules` 中登记的 Git 子模块。 |
| yaml-cpp | [jbeder/yaml-cpp](https://github.com/jbeder/yaml-cpp) | 部署端 CMake 获取的 YAML 解析库。 |
| Eigen | [libeigen/eigen](https://gitlab.com/libeigen/eigen) | 部署端 C++ 线性代数依赖。 |
| fmt | [fmtlib/fmt](https://github.com/fmtlib/fmt) | 部署端 C++ 格式化库依赖。 |
| spdlog | [gabime/spdlog](https://github.com/gabime/spdlog) | 部署端日志库依赖。 |
| Boost | [boostorg/boost](https://github.com/boostorg/boost) | 部署端 C++ 系统组件依赖。 |

## 主要出处

- 近期开源补充：Watch Your Step 中文导读 `output/pdf/watch_your_step_中文导读.pdf`；官方代码本地克隆 `foothold-prediction-model/`；Go2 四足/倒立参考 `Sang-SC/go2_quad2hand`。
- 本仓库训练与算法说明：`rl/GO2W_REFERENCE.md`、`rl/GO2W_SUPERMIMIC_ANALYSIS.md`、`rl/HIMLOCO_SIM2SIM.md`、`rl/SUPERMIMIC_POINTFOOT_GAIT.md`。
- 深度导航方案：`docs/odin1_nav_integration_plan.md`。
- 部署端依赖与源码地址：`src/inference/.gitmodules`、`src/inference/thirdparty/CMakeLists.txt`、`src/inference/debian/copyright` 及 `src/inference/CMakeLists.txt`。
- 本表记录的是仓库中引用到的开源项目入口；“参考”不表示已经移植或作为当前运行依赖。实际使用状态以各行说明及对应配置为准。
