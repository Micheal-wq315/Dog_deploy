# RP_wd Go2 点足试验模型

这是一份独立的点足 URDF 变体。它复用 RP_wd 机身和腿部网格，把原来的四个轮子替换为 Go2 点足几何；原模型 `description/RP_wd/urdf/RP_wd.urdf` 保持不变。

## 模型内容

- 保留四条腿的 ABAD、HIP、KENN 关节，共 **12 个驱动关节**。
- 每个脚端使用 Go2 URDF 的 `foot.dae` 外观，碰撞几何为半径 **2.2 cm** 的球体，质量 **0.04 kg**。
- 四个脚端用固定关节连接，不再包含轮子关节或轮速执行器。
- 脚端固定关节的 Z 偏移设为 `-0.322756 m`。按契约默认姿态（ABAD=0、HIP=0.72、KENN=-1.41 rad）计算，点脚球底与原轮足圆柱最低点均约在 base 高度 `0.424956 m` 时触地，几何高度差小于 0.001 mm。Isaac Lab 当前初始 base 高度为 `0.439 m`，因此启动时约有 14 mm 间隙，随后由重力落地。
- Go2 点足质量比原轮足 link 轻；四个脚端合计使 URDF 总质量减少约 **1.582 kg**。训练或动力学验证前，应确认这符合点足改型的实际质量分布。

Go2 脚网格取自本机 `genesis-world` 仓库随附的 Go2 URDF 资产，许可文件见 [`LICENSE_GO2_ASSET`](LICENSE_GO2_ASSET)。

## 静态检查

在包含 `check_urdf` 的 ROS 环境中运行：

```bash
check_urdf description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf
```

该模型只完成 URDF 几何改型和静态结构检查。它还没有接入 12 维点足训练配置，也没有在 Isaac Sim 中做动力学或步态验证。现有 16 维轮足策略、部署 YAML 与硬件执行器均不能直接用于此模型。
