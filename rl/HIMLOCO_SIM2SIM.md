# UW-HIMLoco 的 MuJoCo sim2sim 环境

2026-11-09，用户要求下载并准备 MuJoCo，供后续 Isaac Lab 策略进行 sim2sim 测试。

## 当前状态与适用范围

- 当前后台 Go2W SuperMimic 实验使用 mjlab / MuJoCo Warp 训练，并非 Isaac Lab。
- UW-HIMLoco `wheel` 原版使用 Isaac Gym；迁移到 Isaac Lab 仍是待实现工作。
- 本次准备的是原 W1W 的 MuJoCo CPU 物理、CPU ONNX 推理及浏览器回放工具。
- 随仓库下载的 `policy_10000.onnx` 是作者模型，接口为348维输入、16维动作；它不兼容 RP_wd 点足的12动作模型。本次没有实现 RP_wd 的 MuJoCo 控制适配，也没有把作者模型当作本机训练成果。

## 安装位置

- 原始源码：`/media/rmer/data/rpwd_training/references/uw-himloco`，提交 `b723e87`，保持源码不变。
- 项目入口：`/home/rmer/Dog_deploy-main/Dog_deploy-main/uw-himloco`，软链接到上述目录。
- 独立 Python：`/media/rmer/data/rpwd_training/envs/uw-himloco-sim2sim/bin/python`。
- 安装记录、下载包和验证日志：`/media/rmer/data/rpwd_training/sim2sim/uw-himloco/`。

venv 通过 `--system-site-packages` 复用现有 NumPy、SciPy、CPU ONNX Runtime 等依赖，在数据盘 venv 中单独安装 MuJoCo；不替换 Isaac Lab 或 Go2W 训练环境的包。使用 MuJoCo 3.11.0、NumPy 1.26.4、SciPy 1.15.3、ONNX Runtime 1.26.0。

MuJoCo 官方 Python wheel 已包含物理引擎，不需要再安装 `mujoco-py` 或单独下载引擎压缩包。参见[官方安装说明](https://mujoco.readthedocs.io/en/stable/python.html#installation)。

## 启动命令

可在任意目录执行：

```bash
# 浏览器界面：原 W1W 作者模型，楼梯地图。
/home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm/rl/scripts/play_himloco_sim2sim.sh --map stairs

# 无窗口 CPU 检查：平地前进，模拟20秒。
/home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm/rl/scripts/play_himloco_sim2sim.sh \
  --headless --map flat --vx 0.5 --duration 20

# 指定后续导出的、接口兼容的 W1W 策略。
/home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm/rl/scripts/play_himloco_sim2sim.sh \
  --onnx /绝对路径/policy.onnx --map stairs
```

浏览器地址默认为 `http://127.0.0.1:8765/`；以播放器实际输出为准。启动器发现训练进程时会阻止图形回放；无窗口模式没有渲染器，物理和 ONNX 均运行在 CPU 上。

可用 `HIMLOCO_SOURCE`、`HIMLOCO_PYTHON`、`HIMLOCO_RUNTIME` 覆盖默认路径。临时文件放数据盘，禁止在参考仓库写入 Python 缓存。

## 兼容性与验证

按仓库说明先检查 MuJoCo 3.2.7，发现 `prepare_robot()` 的旧接口分支未得到有效的 base 刚体，2项模型测试失败；未修改参考源码，改用3.11.0验证。3.2.7 wheel 保留在下载目录供追溯，不作为当前运行版本。

验证包含官方最小模型的 CPU 物理步进、仓库自带 W1W_2 的5项接口/模型/控制/重置检查，以及作者 ONNX 策略的20秒无窗口平地回放，全部通过。0.5m/s指令下模拟20.02秒，最终位置为[9.89283413, 0.07975522, 0.45747891]m，状态有限。启动器Shell语法、帮助入口及训练期间拒绝图形回放的守卫也通过检查；参考源码工作树保持干净。具体输出见数据盘 `interface-tests-final.log`、`headless-flat20s.log`、`gui-guard.log`。无窗口启动检查不代表楼梯通过率；本次未启动浏览器渲染，也未中断后台训练。

后续 Isaac Lab → MuJoCo 验证还需逐项对齐机器人碰撞/惯量、关节顺序、动作方向与缩放、显式 PD、观测归一化与历史，以及50Hz策略/200Hz底层控制。只有完成对应机器人适配后，才可加载其模型进行跨仿真评测。
