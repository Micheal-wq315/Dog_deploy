# Go2W 原生 MuJoCo 键盘控制与三维测试场

2026-11-10。使用原版 Go2W SuperMimic 本机检查点，在官方 MuJoCo 原生查看器中运行 CPU 物理与 CPU Actor 推理。入口不创建 Warp GPU 仿真实例，当前60000轮GPU训练可以继续运行。

## 键盘启动

```bash
cd /home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm
# 自动选择本机正式运行中迭代数最高的已保存模型，默认8cm台阶。
./rl/scripts/play_go2w_mujoco.sh
# 指定模型、地图难度及速度。
./rl/scripts/play_go2w_mujoco.sh --checkpoint /绝对路径/model_23000.pt \
  --profile standard --lane stairs8 --speed 0.3 --lateral-speed 0.2 --turn-speed 0.5
```

点击MuJoCo窗口后，按住方向键控制前后/横移，Q/E控制正负角速度；松开按键或切出窗口立即归零。前后、横移与转向默认分别为0.3m/s、0.2m/s和0.5rad/s，斜向平移会归一化。相反方向同时按下相互抵消，空格优先停止。P键没有前进功能。

R复位当前赛道，K暂停/继续，Esc退出；数字1..8切换赛道并复位。程序读取X11当前按键状态以支持松键，需在本机X11桌面运行；依赖现有python-xlib。异常姿态或地面查询失败时暂停，按R复位。停止只将速度指令置零，不锁定机器人世界坐标。

## 地图与生成

| 数字 | 赛道 | standard规格 |
|---|---|---|
| 1 | flat | 平地速度、横移和转向基线 |
| 2 | stairs4 | 5级4cm上阶、1.2m平台及下阶，踏面30cm |
| 3 | stairs8 | 5级8cm上阶、平台及下阶 |
| 4 | stairs12 | 5级12cm上阶、平台及下阶 |
| 5 | ramp | 连续8°上坡、平台及下坡 |
| 6 | rubble | 按种子生成2–7cm实体石块 |
| 7 | mixed | 三级6cm上下阶后接2–6cm碎石 |
| 8 | slalom | 35cm高交错障碍，操作者遥控选路 |

赛道宽2.6m、中心间距4m。easy/standard/hard的高度比例为0.75/1/4⁄3，坡角按同一比例缩放；hard最高阶为16cm。地形具有真实碰撞，装饰标线没有碰撞。相同种子和难度可重复生成地图；模型与地图均可导出为scene.xml、assets目录及map.json，并保留上游机器人资产许可。

```bash
# 只生成可由官方MuJoCo查看器单独打开的场景。
./rl/scripts/play_go2w_mujoco.sh --mode generate --profile hard --seeds 7 \
  --output /media/rmer/系统/rpwd_training/go2w_supermimic/native_tests/my_hard_park
```

默认输出在`/media/rmer/系统/rpwd_training/go2w_supermimic/native_tests/`，每次创建独立目录；`scene/scene.xml`可独立加载，但单独打开XML不会自动执行神经网络策略，需要上述控制入口。

## 控制接口和验证

复用原Go2W EntityCfg与原生执行器构建机器人，再应用上游求解器设置：物理200Hz、策略50Hz，12个腿部位置动作和4个轮速动作。按关节名定位qpos、qvel和执行器；腿部动作缩放由v7配置读取，轮速缩放35rad/s。Actor仍为245维本体历史输入、16维规范顺序动作；last_action按checkpoint记录的原环境顺序重排，历史逐项按旧到新拼接，首帧复制填充。Actor没有地形或深度输入，绕障区需要人工选择方向。

验证记录：

- 原生适配14项检查通过：机器人质量/惯量、关节范围、armature/阻尼、执行器增益/偏置/限幅与上游一致，4/8/12cm台阶射线高度正确，根速度与原生IMU传感器一致，盲Actor不读取轮速；注入非轮接触可检出约533N碰撞。报告：`/media/rmer/系统/rpwd_training/go2w_supermimic/native_tests/native_contract_checks.json`。
- 第22500轮模型先通过平地和8cm上下阶基线，再进行standard地图、seed0/1/2、0.3m/s固定前进测试，共21试次全部到达终点，无10N以上非轮碰撞。测试不包含遥控绕障区。各赛道实际vx：平地约0.294m/s、4cm阶0.293–0.302、8cm阶0.305–0.312、12cm阶0.303–0.314、坡道约0.350、碎石0.283–0.314、组合0.284–0.296。坡道有速度超调，不能用到点成功替代速度跟踪质量。
- 报告及每50Hz逐帧CSV位于`/media/rmer/系统/rpwd_training/go2w_supermimic/native_tests/park_v1_model22500_3seeds/`；CSV含指令/实际速度、位置、倾角、机身净空、承载轮数及非轮碰撞力。
- 完成条件：到达赛道终点、横向保持在赛道范围、倾角小于25°、至少两轮承载、净空和垂向速度合理，并连续稳定0.2秒。翻倒、非轮碰撞、偏离赛道、8秒不前进或超时均记录具体失败原因。
- 键盘方向、Q/E符号、松键、反向对消、斜向归一化、空格优先、失焦和P无效共12项输入检查通过。官方查看器已加载第23000轮模型；键盘版未声称已完成该模型的21试次评测。

这些结果只覆盖低速、固定摩擦和有限种子的新地图；不等于实机表现。CPU原生MuJoCo与训练时MuJoCo Warp的数值结果也可能不同。

```bash
# CPU确定性评测；默认7类赛道、3个种子、每次最长45秒。
./rl/scripts/play_go2w_mujoco.sh --mode evaluate --checkpoint /绝对路径/model_23000.pt
# 测试更难的地图。
./rl/scripts/play_go2w_mujoco.sh --mode evaluate --profile hard \
  --lanes stairs8 stairs12 rubble mixed --seeds 0 1 2 --seconds 45
```
