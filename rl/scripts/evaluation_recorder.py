"""保存策略测试的逐环境数据；绘图和统计均从同一份 CSV 生成。"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rp_wd_rl.assets.rp_wd_contract import (
    ODIN_DEPTH_GRID_SIZE,
    ODIN_DEPTH_ROTATION_K,
)
from rp_wd_rl.tasks.locomotion.velocity.mdp.terrain import terrain_height_under_root


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


class EvaluationRecorder:
    """采样转移前状态，终止标记对应接下来一步，避免读取自动重置后的假终止姿态。"""

    def __init__(self, directory, env, metadata, save_depth=False):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "samples.csv").exists():
            raise FileExistsError(f"已有测试数据，请换一个 --save-dir: {self.directory}")
        self.metadata = metadata
        self.save_depth = save_depth
        self.metadata.update(created_at=datetime.now().astimezone().isoformat(), step_dt=env.step_dt)
        self.metadata["notes"] = [
            "状态在物理转移前采样，终止原因来自随后的转移；可能多个原因同时触发。",
            "origin_height_m 是相对出生平台高度，不是机身到脚下地面的间隙。",
            "ImplicitActuator的applied_torque为裁剪后的PD估计力矩，不是PhysX直接测得的驱动力矩。",
            "测试为确定性策略、混合地形速度扫描，不代表指定高度台阶的通过率。",
            "零指令阶段的低速不计入低速比例；有指令时速度低于0.05m/s仅作为低速线索。",
        ]
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator
        self.names = {}
        if generator is not None:
            cfgs = list(generator.sub_terrains.items())
            cumulative = np.cumsum([cfg.proportion for _, cfg in cfgs])
            cumulative /= cumulative[-1]
            for col in range(generator.num_cols):
                index = int(np.searchsorted(cumulative, (col + .5) / generator.num_cols))
                self.names[col] = cfgs[index][0]
        self.metadata["terrain_columns"] = self.names
        if save_depth:
            (self.directory / "depth_frames").mkdir(exist_ok=True)
        self.term_names = list(env.termination_manager.active_terms)
        self.leg_ids = [i for i, name in enumerate(env.scene["robot"].joint_names) if "FOOT" not in name]
        self.wheel_ids = [i for i, name in enumerate(env.scene["robot"].joint_names) if "FOOT" in name]
        self.columns = ["cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "error_xy", "error_yaw",
                        "tilt_deg", "world_height_m", "ground_height_m", "root_clearance_m", "origin_height_m", "leg_torque_peak_nm",
                        "wheel_torque_peak_nm", "distance_origin_m", "speed_xy", "terrain_level", "terrain_column"]
        self.file = (self.directory / "samples.csv").open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=[
            "step", "time_s", "env_id", "case", "phase", "terrain_name", *self.columns,
            "terminated", "timeout", *["done_" + name for name in self.term_names],
        ])
        self.writer.writeheader()
        self.rows = []
        self.case = "continuous"
        self.phase = "measure"
        self.step = 0

    def capture(self, env):
        robot = env.scene["robot"].data
        cmd = env.command_manager.get_command("base_velocity")
        vel = robot.root_lin_vel_b
        omega = robot.root_ang_vel_b
        terrain = env.scene.terrain
        ground_height = terrain_height_under_root(env)
        zeros = torch.zeros(env.num_envs, device=env.device)
        values = torch.stack([
            cmd[:, 0], cmd[:, 1], cmd[:, 2], vel[:, 0], vel[:, 1], omega[:, 2],
            torch.linalg.vector_norm(cmd[:, :2] - vel[:, :2], dim=1), (cmd[:, 2] - omega[:, 2]).abs(),
            torch.rad2deg(torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1))),
            robot.root_pos_w[:, 2], ground_height, robot.root_pos_w[:, 2] - ground_height,
            robot.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
            robot.applied_torque[:, self.leg_ids].abs().amax(dim=1),
            robot.applied_torque[:, self.wheel_ids].abs().amax(dim=1),
            torch.linalg.vector_norm(robot.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1),
            torch.linalg.vector_norm(vel[:, :2], dim=1),
            getattr(terrain, "terrain_levels", zeros), getattr(terrain, "terrain_types", zeros),
        ], dim=1)
        return values.cpu().numpy()

    def record(self, env, values):
        dones = torch.stack([env.reset_terminated, env.reset_time_outs,
                             *[env.termination_manager.get_term(name) for name in self.term_names]], dim=1).cpu().numpy()
        for i, (row, flags) in enumerate(zip(values, dones)):
            entry = {"step": self.step, "time_s": self.step * env.step_dt,
                     "env_id": i, "case": self.case, "phase": self.phase,
                     "terrain_name": self.names.get(int(row[-1]), "plane")}
            entry.update(zip(self.columns, map(float, row)))
            entry.update(zip(["terminated", "timeout", *["done_" + name for name in self.term_names]], map(int, flags)))
            self.writer.writerow(entry)
            self.rows.append(entry)
        self.step += 1
        if self.step % 100 == 0:
            self.file.flush()
        if self.save_depth and self.step % 25 == 0:
            self._save_depth_preview(env)

    def _save_depth_preview(self, env):
        """保存原图、旋转校正图及策略使用的深度网格。"""
        camera = env.scene.sensors.get("odin_depth_camera")
        if camera is None:
            raise RuntimeError("请求保存深度数据，但场景中没有 odin_depth_camera 传感器。")
        depth = camera.data.output["distance_to_image_plane"]
        if depth.ndim == 4:
            depth = depth[..., 0]
        raw = depth[0].detach().float().cpu().numpy()
        valid = np.isfinite(raw) & (raw > 0.0)
        clipped = np.where(valid, np.clip(raw, 0.2, 6.0), 6.0).astype(np.float32)
        corrected = np.rot90(clipped, k=ODIN_DEPTH_ROTATION_K).copy()
        grid_h, grid_w = ODIN_DEPTH_GRID_SIZE[1], ODIN_DEPTH_GRID_SIZE[0]
        grid_m = -F.adaptive_max_pool2d(
            -torch.from_numpy(corrected)[None, None], (grid_h, grid_w)
        )[0, 0].numpy()
        grid_norm = np.clip(grid_m / 6.0, 0.0, 1.0)
        stem = f"depth_{self.step:06d}"
        np.save(self.directory / "depth_frames" / f"{stem}.npy", raw)
        np.save(self.directory / "depth_frames" / f"{stem}_rot90_ccw.npy", corrected)
        np.save(self.directory / "depth_frames" / f"{stem}_grid16x12_m.npy", grid_m)
        np.save(self.directory / "depth_frames" / f"{stem}_grid16x12_norm.npy", grid_norm)
        plt = setup_plotting()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, image, title in (
            (axes[0], clipped, "传感器原图（米）"),
            (axes[1], corrected, "旋转90°后的深度图（米）"),
            (axes[2], grid_m, "策略网格（16×12，米）"),
        ):
            plot = ax.imshow(image, cmap="turbo", vmin=0.2, vmax=6.0, interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, label="米")
        fig.suptitle(f"仿真步 {self.step} · 有效像素 {valid.mean():.1%}")
        fig.savefig(self.directory / "depth_frames" / f"{stem}.png", dpi=140)
        plt.close(fig)

    def finish(self, completed=False):
        self.file.flush()
        self.file.close()
        self.metadata["completed"] = completed
        (self.directory / "config.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2))
        measured = [r for r in self.rows if r["phase"] == "measure"]
        if not measured:
            return
        summaries = []
        for case in dict.fromkeys(r["case"] for r in measured):
            rows = [r for r in measured if r["case"] == case]
            summary = {"case": case, "samples": len(rows)}
            for name in self.columns[:-2]:
                summary[name] = float(np.mean([r[name] for r in rows]))
            summary["tilt_p95_deg"] = float(np.percentile([r["tilt_deg"] for r in rows], 95))
            summary["max_leg_torque_nm"] = max(r["leg_torque_peak_nm"] for r in rows)
            summary["leg_torque_ge_40_fraction"] = float(np.mean([r["leg_torque_peak_nm"] >= 40 for r in rows]))
            summary["low_speed_fraction"] = float(np.mean([
                np.hypot(r["cmd_vx"], r["cmd_vy"]) > .1 and r["speed_xy"] < .05 for r in rows]))
            for name in ["terminated", "timeout", *["done_" + n for n in self.term_names]]:
                summary[name] = sum(r[name] for r in rows)
            summaries.append(summary)
        with (self.directory / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        (self.directory / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
        # 以环境与时间为样本；终止次数是事件计数，不称作成功率。
        terrain_summary = {}
        for name in dict.fromkeys(r["terrain_name"] for r in measured):
            rows = [r for r in measured if r["terrain_name"] == name]
            terrain_summary[name] = {"samples": len(rows),
                "mean_error_xy": float(np.mean([r["error_xy"] for r in rows])),
                "mean_tilt_deg": float(np.mean([r["tilt_deg"] for r in rows])),
                "terminations": sum(r["terminated"] for r in rows)}
        (self.directory / "terrain_summary.json").write_text(json.dumps(terrain_summary, ensure_ascii=False, indent=2))
        self.plot(summaries)
        print(f"[INFO] 测试数据与图已保存: {self.directory}")

    def plot(self, summaries):
        plt = setup_plotting()
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained")
        groups = {}
        for row in self.rows:
            groups.setdefault(row["step"], []).append(row)
        times = [rows[0]["time_s"] for rows in groups.values()]
        for ax, command, actual, label in zip(axes.flat[:3], ["cmd_vx", "cmd_vy", "cmd_wz"], ["vx", "vy", "wz"], ["前进速度 m/s", "横向速度 m/s", "偏航角速度 rad/s"]):
            ax.plot(times, [np.mean([r[command] for r in rows]) for rows in groups.values()], label="指令", ls="--")
            ax.plot(times, [np.mean([r[actual] for r in rows]) for rows in groups.values()], label="实际均值")
            ax.set_title(label)
            ax.set_xlabel("仿真时间 s")
            ax.legend()
        axes[1, 1].plot(times, [np.mean([r["tilt_deg"] for r in rows]) for rows in groups.values()])
        axes[1, 1].axhline(10, color="gray", ls="--", label="10°参考")
        axes[1, 1].set_title("机身倾角均值 °")
        axes[1, 1].set_xlabel("仿真时间 s")
        axes[1, 1].legend()
        x = np.arange(len(summaries))
        axes[2, 0].bar(x, [s["error_xy"] for s in summaries], label="线速度误差 m/s")
        axes[2, 0].set_title("各指令测量阶段误差")
        axes[2, 0].legend()
        axes[2, 1].bar(x, [s["terminated"] for s in summaries], label="非超时终止")
        axes[2, 1].bar(x, [s["timeout"] for s in summaries], bottom=[s["terminated"] for s in summaries], label="超时")
        axes[2, 1].set_title("各指令终止事件数（允许自动重置，多次计数）")
        axes[2, 1].legend()
        for ax in axes[2]:
            ax.set_xticks(x, [s["case"].split("_")[0] for s in summaries])
            ax.set_xlabel("指令编号；对应 summary.csv")
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle(f"RP_wd 深度策略测试 | {Path(self.metadata['checkpoint']).name}\n种子 {self.metadata['seed']} · 地形等级 {self.metadata['terrain_level']} · {self.metadata['num_envs']} 环境", fontsize=15)
        fig.savefig(self.directory / "test_curves.png", dpi=160)
        fig.savefig(self.directory / "test_curves.svg")
        plt.close(fig)
