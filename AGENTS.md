# AGENTS.md

## 语言与协作约定（用户 2026-10-28 明确要求）

- 后续回复、进度说明、开发记录和提交摘要/正文统一使用中文。
- 新增说明和代码注释优先使用中文；代码标识符、命令、路径、接口名称及规定的提交尾注保留原样。
- 开始工作时先查看 `git status`，按下方索引读取相关文档；保留用户已有改动，提交前核对实际差异。
- 完成可独立交付的改动后，按提交约定记录版本与验证结果；明确区分已实现功能、已验证结果和待实现方案。

## 提交约定（用户 2026-10-11 明确要求）

- **每次有"较成功的改动"或"更新量够一个版本"时就 commit 一次**
  （例：新步态收敛 / 契约变更 / 新工具链 / 地形训练接入 / 重要 bug 修复）
- 提交信息格式 `type: 中文摘要`（feat / fix / docs / refactor / train），正文用中文写清**动机**与**验证方式**
- 同时在 `rl/DEVELOPMENT.md` 的「版本记录」表格追加一行
- 结尾带 `Co-Authored-By: Codex <noreply@anthropic.com>`

## 项目速览

本仓库 = **部署端**（`src/inference`，ROS2 Jazzy + ONNX Runtime 推理节点）
＋ **训练端**（`rl/`，Isaac Lab 2.3.2 外部工程，产出与部署端逐项对齐的 ONNX）。

| 我要… | 看这里 |
|---|---|
| 常用命令（训练/测试/遥控/导出） | `rl/scripts/start.md` |
| sim2real 契约表、域随机化、调参提示 | `rl/README.md` |
| 开发记录、踩坑、训练历程、待办 | `rl/DEVELOPMENT.md` |
| 契约常量（唯一真相源） | `rl/rp_wd_rl/assets/rp_wd_contract.py` |
| 部署端构建与运行 | `README.md` |

## ★ 硬约束（都踩过坑）

1. **绝不同时开两个 Isaac Sim 实例** —— 共用一张卡会把训练进程死锁（2026-10-10 实测：
   训练空转 12.8 小时只跑到 101/3000）。GUI 测试走 `rl/scripts/gui_test.sh`（自带守卫），
   写脚本顺手跑仿真前先确认 `pgrep -f train.py` 为空。
2. **训练必须 headless**（`train.py` 内已强制）；训练期间用 `systemd-inhibit` 阻止休眠（`train.sh` 内已加）。
3. **改观测 / 动作 / 奖励 / 部署端 YAML 后必须跑** `./rl/scripts/verify_contract.sh`
   —— 含姿态指令的既有策略输入 168 维、Odin1 深度策略输入 780 维，动作均为 16 维；
   历史 159 维策略不含姿态指令。具体维度、关节顺序与指令范围以契约常量及对应部署配置为准，两端必须同步。
4. 杀 Isaac Sim 进程要 `pkill -9`（它吞 SIGTERM）；模式串写成 `[t]rain.py` 避免杀掉执行命令的 shell 自身。
