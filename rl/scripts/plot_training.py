"""从 TensorBoard 保存完整训练曲线、标量 CSV 和最近窗口统计，不启动仿真。"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from evaluation_recorder import setup_plotting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="包含 TensorBoard 事件文件的训练目录。")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    data = {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}
    with (out / "training_scalars.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "iteration", "value", "wall_time"])
        for tag, rows in data.items():
            writer.writerows((tag, r.step, r.value, r.wall_time) for r in rows)
    last = max(r.step for r in data["Train/mean_reward"])
    summary = {"run": str(run), "last_iteration": last,
               "generated_at": datetime.now().astimezone().isoformat(),
               "last100": {tag: float(np.mean([r.value for r in rows if r.step > last - 100])) for tag, rows in data.items() if any(r.step > last - 100 for r in rows)}}
    (out / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    plt = setup_plotting()
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), layout="constrained")

    def line(ax, tag, label, smooth=50):
        if tag not in data:
            return
        rows = data[tag]
        x = np.array([r.step for r in rows])
        y = np.array([r.value for r in rows])
        smooth = min(smooth, len(y))
        raw_line, = ax.plot(x, y, alpha=.12, linewidth=.6)
        ax.plot(x[smooth-1:], np.convolve(y, np.ones(smooth)/smooth, mode="valid"),
                color=raw_line.get_color(), label=label)

    line(axes[0, 0], "Train/mean_reward", "回报：50轮滑动均值")
    axes[0, 0].set_title("训练回报")
    line(axes[0, 1], "Train/mean_episode_length", "回合步数")
    axes[0, 1].axhline(1000, color="gray", ls="--")
    axes[0, 1].set_title("平均回合长度，1000步=20秒")
    line(axes[1, 0], "Curriculum/terrain_levels", "实际平均等级")
    axes[1, 0].set_title("实际地形等级（不是台阶通过率）")
    env_file, agent_file = run / "params/env.yaml", run / "params/agent.yaml"
    if env_file.exists() and agent_file.exists():
        # BaseLoader只读取字符串/容器，不执行YAML中的Python对象标签。
        env_cfg = yaml.load(env_file.read_text(), Loader=yaml.BaseLoader)
        agent_cfg = yaml.load(agent_file.read_text(), Loader=yaml.BaseLoader)
        terrain_cfg = env_cfg.get("curriculum", {}).get("terrain_levels") or {}
        schedule = terrain_cfg.get("params", {}).get("max_level_schedule", [])
        if schedule:
            rollout_steps = int(agent_cfg["num_steps_per_env"])
            points = [(int(step) / rollout_steps, int(level)) for step, level in schedule
                      if int(step) / rollout_steps <= last]
            if points:
                points.append((last, points[-1][1]))
                cap = axes[1, 0].twinx()
                cap.step(*zip(*points), where="post", ls="--", color="gray", label="保存配置中的允许上限")
                cap.set_ylabel("允许上限", color="gray")
                cap.legend(loc="upper right")
    for tag, label in [("command_lin_vel", "速度"), ("posture_command", "姿态/高度"), ("push_curriculum", "推扰"), ("leg_torque_penalty", "腿部力矩惩罚")]:
        line(axes[1, 1], "Curriculum/" + tag, label, 1)
    axes[1, 1].set_title("各项课程进度，相对最终幅度")
    for tag, label in [("track_lin_vel_xy_exp", "线速度"), ("track_ang_vel_z_exp", "偏航"), ("attitude_track_exp", "姿态"), ("height_track_exp", "高度")]:
        line(axes[2, 0], "Episode_Reward/" + tag, label)
    axes[2, 0].set_title("跟踪奖励（回合累计 / 固定20秒）")
    for tag, label in [("root_height", "高度条件"), ("bad_orientation", "倾角条件"), ("base_contact", "躯干接触")]:
        line(axes[2, 1], "Episode_Termination/" + tag, label)
    axes[2, 1].set_title("最近回合终止原因快照比例")
    line(axes[3, 0], "Loss/learning_rate", "学习率", 1)
    axes[3, 0].set_yscale("log")
    axes[3, 0].set_title("自适应学习率")
    line(axes[3, 1], "Policy/mean_std", "动作探索标准差")
    axes[3, 1].set_title("训练动作噪声（确定性测试不采样该噪声）")
    for ax in axes.flat:
        ax.set_xlabel("PPO迭代")
        ax.grid(alpha=.2)
        ax.legend(loc="best")
    fig.suptitle(f"Odin1 训练曲线 | {run.name} | 截至迭代 {last}\n原始标量保存在 training_scalars.csv", fontsize=15)
    fig.savefig(out / "training_curves.png", dpi=160)
    fig.savefig(out / "training_curves.svg")
    plt.close(fig)
    print(f"训练数据与曲线已保存: {out}")


if __name__ == "__main__":
    main()
