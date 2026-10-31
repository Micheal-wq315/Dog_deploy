# RP_wd Go2 点足试验模型

这是一份独立的点足 URDF 变体，保留 RP_wd 机身和腿部 12 个驱动关节，并按 Go2 的小腿与脚端结构替换轮足末端。原轮足模型 `description/RP_wd/urdf/RP_wd.urdf` 保持不变。

## 点足结构

- 四个 `KENN_LINK` 使用 Go2 的 calf/calf_mirror 网格、质量、惯量和小腿碰撞几何，原 RP_wd 轮毂网格及其大圆柱碰撞体已移除。
- 脚端外观来自 Go2 `foot.dae`，碰撞为半径 **2.2 cm** 的球，质量 **0.04 kg**。
- 脚端固定在 calf 末端（`z=-0.213 m`），没有脚踝/轮子电机，也没有第四组脚端执行器；模型为 **12 个驱动关节 + 4 个固定脚端**。
- 按现有默认腿角（ABAD=0、HIP=0.72、KENN=-1.41 rad）计算，点脚球底触地时 base 高度约 **0.3403 m**。轮足版的 `STAND_HEIGHT=0.419 m` 不适用于此点足模型，后续需要独立设置点足站高和 12 维策略配置。
- 当前模型总质量约 **17.038 kg**，原轮足 URDF 约 **20.550 kg**，减少约 **3.512 kg**。训练前应按实际点足硬件重新确认质量与质心。

Go2 的 calf、calf_mirror 和 foot 网格取自本机 `genesis-world` 仓库随附的 Go2 URDF 资产，许可文件见 [`LICENSE_GO2_ASSET`](LICENSE_GO2_ASSET)。

## 检查

```bash
check_urdf description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf
```

现有 16 维轮足策略、部署 YAML 与硬件执行器不适用于本模型。点足版尚未接入训练配置；Isaac Sim 中的导入查看也不等于动力学或步态验收。
