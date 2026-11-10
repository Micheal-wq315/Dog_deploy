# 原版 Go2W SuperMimic 训练与效果对照

2026-11-09，用户要求先训练开源项目里的狗，观察原项目效果。

本次运行的是 `go2w-supermimic` 中的原版Go2W轮足机器人：12个腿部位置动作＋4个轮速动作，245维五帧盲Actor、350维特权地形世界模型、337维Critic。它不是RP_wd点足版，也没有使用本仓库v4步态参考、残差控制或奖励。

## 源码和环境

- [原仓库](https://github.com/uwvwko-zzz/go2w-supermimic)：`/home/rmer/Dog_deploy-main/Dog_deploy-main/go2w-supermimic`。
- 源码提交：`a299ed5f5571da4492b8a286d94acfe4a86128e1`。保留原始训练源码、机器人资产、奖励、随机化和v7课程，不修改参考仓库内用户PDF与翻译文件。
- 独立环境：`/media/rmer/data/rpwd_training/envs/go2w-supermimic/bin/python`。
- 使用 `venv --system-site-packages` 复用已可用的GPU PyTorch，新依赖只安装在数据盘venv中。复核原Isaac Lab环境仍为Warp1.13.0、trimesh4.5.1、rsl-rl-lib5.3.0。
- 新环境版本：mjlab1.6.0、MuJoCo3.11.0、MuJoCo Warp3.11.0、Warp1.19.0、PyTorch2.11.0+cu128。已完成导入与CUDA设备检查；GPU为RTX5070Ti、驱动580.159.03。
- 依赖约束在数据盘 `envs/go2w-constraints.txt`；下载包执行SHA256校验。安装和缓存全部使用单独目录，保留Isaac Lab原环境。

安装依据：[mjlab 1.6.0依赖定义](https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml)、[NVIDIA Warp安装要求](https://nvidia.github.io/warp/v1.19/user_guide/installation.html)。

## 训练配置与比较边界

沿用原仓库 `scripts/train_terrain.sh` 的v7盲轮足任务、seed0、512步回放、8步模型窗口、32步可微想象、模型每轮更新4次、Critic5轮/4小批次、优先回放与命令课程，以及Actor学习率调度。预热300步后开始更新Actor；`actor-hold-iterations=20000`是学习率平台期，不是冻结Actor20000轮。

先运行20000轮，和仓库提供的作者20000轮模型区分保存。本机最终使用4096环境、2048模型批量、4096想象批量：保留原真实采样环境数，仅把原8192想象批量减半。20000轮对应8192万真实转移，想象批量改变梯度采样量，不改变真实转移数。训练必须headless，使用防休眠和仿真进程守卫。

所有新日志、模型及编译缓存位于 `/media/rmer/data/rpwd_training/go2w_supermimic/`，避免系统盘剩余空间不足。参考项目自带的 `logs/go2w_terrain_v7_20000/model_20000.pt` 是作者随仓库提供的模型，不是本机刚刚训练出的结果。

## 使用命令

```bash
cd /home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm

# 原版Go2W从零训练，默认目标20000轮，参数可从命令末尾覆盖。
./rl/scripts/train_go2w_reference.sh
# 完整实验：训练完成后自动串行测试本机新模型，保存平地/4/8/12cm上下阶录像。
./rl/scripts/run_go2w_reference_experiment.sh

# 只在没有训练/其它仿真实例时打开浏览器控制。
# 默认使用作者随仓库提供的20000轮模型，CPU仿真，4cm上阶。
./rl/scripts/play_go2w_reference.sh
# 浏览器打开终端显示的http://127.0.0.1:8765/，可切换地形和难度。

# 测试本机新模型，显式指定checkpoint，避免和作者模型混淆。
./rl/scripts/play_go2w_reference.sh --checkpoint /绝对路径/model_20000.pt

# 原环境确定性测量并保存录像：默认作者模型，0.3m/s，无观测噪声。
./rl/scripts/evaluate_go2w_reference.sh --kinds up --levels 1 2 3 \
  --seconds 18 --video --output /media/rmer/data/rpwd_training/go2w_supermimic/evaluations/author_stairs
```

回放等级0..5分别为单级2/4/8/12/16/20cm。测试报告保存模型来源、源码提交、依赖版本、实际速度/误差、最大进度、翻倒/非轮碰撞、是否稳定到达末级，以及录像路径。`--max-steps`为工具冒烟，报告明确`smoke_only=true`，不能视作通过楼梯。

## 本次运行记录

- GPU依赖、原模型245→16输入输出以及全部权重有限检查通过。
- 原4096环境/8192想象批量预检在300步预热后出现Warp显存不足，峰值15723MiB；改为4096想象批量后完成305轮、5次Actor更新，跳过更新0、无效状态恢复0，峰值14094MiB，整个预检115.87秒。预检输出在`validation/`中，不作为正式新模型。
- 作者模型在单种子seed0、0.3m/s命令、18秒测量条件下通过4/8/12cm上阶，实际vx均值分别0.288/0.287/0.292m/s，无翻倒或非轮碰撞。这只是三个固定场景的基线，不是多种子鲁棒性结论。
- 作者录像及原始JSON位于`/media/rmer/data/rpwd_training/go2w_supermimic/evaluations/author_20k_stairs_20261109/`，对应`up_level1_seed0.mp4`、`up_level2_seed0.mp4`、`up_level3_seed0.mp4`。
- 正式实验已从零启动，未加载作者Actor或世界模型。服务：`go2w-supermimic-v7-20k-20261109.service`；日志：`/media/rmer/data/rpwd_training/go2w_supermimic/train_v7_20k_20261109.log`。
- 正式运行目录：`/media/rmer/data/rpwd_training/go2w_supermimic/2026-11-09_原版Go2W_v7_20k_seed0/`。每500轮保存模型；最终自动执行本机模型的平地/4/8/12cm上下阶测试并保存至该目录`evaluation/`。
- 启动检查至400轮：100次Actor更新，跳过更新0、无效状态恢复0；尚无本机新模型的最终效果结论。
- 首个正式`model_500.pt`已保存并CPU加载验证：500轮/200次Actor更新，实际4096环境、2048模型批量、4096想象批量、32步展开、v7/seed0，未使用resume或init_actor。`experiment_manifest.json`保存源码提交、服务、实际参数和作者基线位置。

```bash
# 只读查看训练，不新开仿真。
tail -f /media/rmer/data/rpwd_training/go2w_supermimic/train_v7_20k_20261109.log
systemctl --user status go2w-supermimic-v7-20k-20261109.service
# 需要人工GUI时先停止本次实验，使用最近500轮checkpoint继续训练。
systemctl --user stop go2w-supermimic-v7-20k-20261109.service
```

新模型效果以训练结束后的独立测试为准；作者预训练录像不会被标注成本机训练结果。

## 2026-11-10：从20000轮续训到40000轮

该40000轮目标已被用户随后明确的60000轮目标替代，以下保留为历史启动记录；当前服务与命令见下一节。

本机首轮20000轮已完成，训练耗时7203.6秒、19700次策略更新、跳过0次。自动评测在seed0、无观测噪声、vx=0.3m/s、18秒条件下完成平地及4/8/12cm上下阶七个场景，均无摔倒或非轮碰撞，六个楼梯场景到达末端。这是单种子低速结果，尚未验证多种子或16/20cm台阶。

本次使用原运行的`model_20000.pt`完整续训，总目标40000轮，即额外20000轮。恢复Actor、世界模型、Critic/目标Critic、优化器、地形等级和课程统计；保持seed0、4096环境、2048模型批量、4096想象批量、32步展开及原学习率调度。回放池不包含在checkpoint内，每次恢复先采集300轮，之后恢复策略更新。

重启后原`/media/rmer/data`因NTFS未正常关闭而只读挂载。用户尝试重新挂载后仍为只读，并授权继续训练，因此本次新模型、日志和缓存保存到另一块已通过写入检查的分区`/media/rmer/系统`（`/dev/sdb1`）；源码、Python环境和原检查点从原盘读取。该磁盘挂载状态只描述本次启动条件，后续启动应先检查。

- 服务：`go2w-supermimic-v7-resume40k-20261110.service`。
- 输出：`/media/rmer/系统/rpwd_training/go2w_supermimic/2026-11-10_原版Go2W_v7_续训40k_seed0/`。
- 日志：`/media/rmer/系统/rpwd_training/go2w_supermimic/train_v7_resume40k_20261110.log`。
- 缓存：`/media/rmer/系统/rpwd_training/go2w_supermimic/cache/`。
- 中途应用户要求停止过一次，停止前最新记录20250轮、尚未新增checkpoint；重新启动仍从原20000轮模型恢复。日志会保留两段启动记录，计算本次速度应使用后一次恢复后的指标。
- 训练每500轮保存，完成后自动串行执行平地及4/8/12cm上下阶录像评测，保存到新运行目录`evaluation/`。
- 启动验证至20350轮：完成300轮回放预热，策略更新从19700增至19750，跳过0次、无效环境恢复2次；显存13982MiB，后台服务持续运行。此记录只证明续训更新链路正常。

```bash
# 没有其他仿真进程时手动启动同一续训；后台服务运行期间不要重复执行。
cd /home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm
GO2W_CACHE=/media/rmer/系统/rpwd_training/go2w_supermimic/cache \
GO2W_RUN=/media/rmer/系统/rpwd_training/go2w_supermimic/2026-11-10_原版Go2W_v7_续训40k_seed0 \
./rl/scripts/run_go2w_reference_experiment.sh \
  --resume /media/rmer/data/rpwd_training/go2w_supermimic/2026-11-09_原版Go2W_v7_20k_seed0/model_20000.pt \
  --iterations 40000

# 查看当前后台服务和日志。
systemctl --user status go2w-supermimic-v7-resume40k-20261110.service
tail -f /media/rmer/系统/rpwd_training/go2w_supermimic/train_v7_resume40k_20261110.log
```

## 2026-11-10：当前总目标调整为60000轮

用户明确要求总计60000轮。已停止40000轮目标服务，从该运行最新保存的`model_20500.pt`完整恢复到新运行；停止前日志约20650轮，未保存的进度不纳入恢复。网络、优化器、课程及原有学习率调度继续沿用，只增加训练总目标。恢复后按原实现重新采集300轮回放，再恢复策略更新。

- 当前服务：`go2w-supermimic-v7-resume60k-20261110.service`。
- 当前输出：`/media/rmer/系统/rpwd_training/go2w_supermimic/2026-11-10_原版Go2W_v7_续训60k_seed0/`。
- 当前日志：`/media/rmer/系统/rpwd_training/go2w_supermimic/train_v7_resume60k_20261110.log`。
- 缓存仍使用`/media/rmer/系统/rpwd_training/go2w_supermimic/cache/`；训练结束后自动评测并保存录像。

```bash
# 仅在当前服务已停止且没有其他仿真进程时执行。
cd /home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm
GO2W_CACHE=/media/rmer/系统/rpwd_training/go2w_supermimic/cache \
GO2W_RUN=/media/rmer/系统/rpwd_training/go2w_supermimic/2026-11-10_原版Go2W_v7_续训60k_seed0 \
./rl/scripts/run_go2w_reference_experiment.sh \
  --resume /media/rmer/系统/rpwd_training/go2w_supermimic/2026-11-10_原版Go2W_v7_续训40k_seed0/model_20500.pt \
  --iterations 60000
```
