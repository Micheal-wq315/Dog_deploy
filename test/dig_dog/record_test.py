#!/usr/bin/env python3
"""DigDogv1 原生 MuJoCo 测试数据记录 v1。

逐 50Hz 控制步记录：机身位姿/速度/IMU、12 关节角/角速度/目标角/动作/扭矩、
四足位置/速度/接触力/撞击峰值/离地间隙、命令/相位、终止原因。
输出：test/dig_dog/runs/<时间>_<模型轮次>/<lane>_seed<k>_<direction>/
  trajectory.csv, observations.npy, summary.json
以及 run 级 meta.json / summary.json。
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import mujoco

WORKTREE = Path('/media/rmer/data/rpwd_training/worktrees/rpwd-digdogv1-mjlab')
RUN_DEFAULT = Path('/media/rmer/data/rpwd_training/digdog_mjlab/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2')
# 复用 worktree 中的原生适配器/地图
sys.path.insert(0, str(WORKTREE / 'rl'))
sys.path.insert(0, str(WORKTREE / 'rl' / 'scripts'))
import digdog_mujoco  # noqa: E402
from go2w_test_maps import LANES, terrain_height  # noqa: E402

LEGS = ('FL', 'FR', 'RL', 'RR')
SCRIPT_VERSION = 'dig_dog.record_test.v1'


class RecordingGo2W(digdog_mujoco.NativeGo2W):
    """在原原生适配器上补四足接触力与撞击峰值。"""

    def __init__(self, checkpoint, actor, seed=0, profile='standard'):
        super().__init__(checkpoint, actor, seed, profile)
        self.foot_bodies = [self.model.body(f'robot/{leg}_foot_link').id for leg in LEGS]
        self.foot_force = np.zeros((4, 3))
        self.foot_peak = np.zeros(4)
        self.foot_count = np.zeros(4, dtype=np.int32)

    def step(self, command):
        self.foot_force = np.zeros((4, 3))
        self.foot_peak = np.zeros(4)
        self.foot_count = np.zeros(4, dtype=np.int32)
        return super().step(command)

    def _contact_state(self):
        peak, wheels = super()._contact_state()
        force = np.zeros((4, 3))
        fpeak = np.zeros(4)
        fcount = np.zeros(4, dtype=np.int32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g0, g1 = int(c.geom[0]), int(c.geom[1])
            if g0 in self.wheel_geom and self.ground_geom[g1]:
                foot, sign = self.wheel_geom[g0], -1.
            elif g1 in self.wheel_geom and self.ground_geom[g0]:
                foot, sign = self.wheel_geom[g1], 1.
            else:
                continue
            cf = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, cf)
            world = sign * c.frame.reshape(3, 3).T @ cf[:3]
            force[foot] += world
            fpeak[foot] = max(fpeak[foot], float(world[2]))
            fcount[foot] += 1
        self.foot_force = force
        self.foot_peak = np.maximum(self.foot_peak, fpeak)
        self.foot_count = fcount
        return peak, wheels


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('robot_id') != 'rpwd_digdog':
        raise ValueError(f'不是 DigDogv1 检查点: {ckpt.get("robot_id")}')
    package = ckpt.get('backend_package') or 'rpwd_digdog_supermimic'
    policy = importlib.import_module(package + '.core.policy')
    return ckpt, policy, policy.load_actor(ckpt, 'cpu')


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float('nan')
    except Exception:
        return float('nan')


def build_row(sim, row, command, q_prev, tau_prev, action_prev, t0):
    d = sim.data
    state = sim.state()
    q = d.qpos[sim.qadr].copy()
    dq = d.qvel[sim.vadr].copy()
    qtar = d.ctrl[sim.actuators].copy()
    tau = d.actuator_force[sim.actuators].copy()
    action = sim.previous.copy()
    foot_pos = d.xpos[sim.foot_bodies].copy()
    foot_vel = d.cvel[sim.foot_bodies][:, 3:6].copy()
    ground = np.array([terrain_height(sim.model, sim.data, float(x), float(y))
                       for x, y in foot_pos[:, :2]])
    foot_clear = foot_pos[:, 2] - ground
    foot_slip = np.linalg.norm(foot_vel[:, :2], axis=1)
    phase = 2 * math.pi * sim.clock_frequency * float(d.time)
    base_quat = d.qpos[3:7].copy()
    out = {
        't': num(d.time), 't_since_start': num(d.time - t0),
        'x': num(d.qpos[0]), 'y': num(d.qpos[1]), 'z': num(d.qpos[2]),
        'qw': num(base_quat[0]), 'qx': num(base_quat[1]), 'qy': num(base_quat[2]), 'qz': num(base_quat[3]),
        'vx_world': num(d.qvel[0]), 'vy_world': num(d.qvel[1]), 'vz_world': num(d.qvel[2]),
        'wx_body': num(state[7]), 'wy_body': num(state[8]), 'wz_body': num(state[9]),
        'vx_body': num(state[4]), 'vy_body': num(state[5]), 'vz_body': num(state[6]),
        'gx': num(state[1]), 'gy': num(state[2]), 'gz': num(state[3]),
        'tilt_deg': num(row['tilt_degrees']), 'clearance': num(row['clearance']),
        'cmd_vx': num(command[0]), 'cmd_vy': num(command[1]), 'cmd_wz': num(command[2]),
        'phase': num(phase), 'phase_sin': num(math.sin(phase)), 'phase_cos': num(math.cos(phase)),
        'wheel_contacts': int(row['wheel_contacts']), 'ncon': int(d.ncon),
        'nonwheel_force': num(sim.peak_nonwheel), 'finite': int(bool(row['finite'])),
        'q_prev': q_prev, 'tau_prev': tau_prev, 'action_prev': action_prev,
        'q': q, 'dq': dq, 'q_target': qtar, 'tau': tau, 'action': action,
        'foot_pos': foot_pos, 'foot_vel': foot_vel, 'foot_ground': ground,
        'foot_clearance': foot_clear, 'foot_slip': foot_slip,
        'foot_force': sim.foot_force.copy(), 'foot_peak': sim.foot_peak.copy(),
        'foot_count': sim.foot_count.copy(), 'foot_contact': sim.contacts.copy(),
    }
    return out


def flatten(row, joint_names, q_prev, tau_prev, action_prev):
    out = {}
    for k, v in row.items():
        if k in ('q', 'dq', 'q_target', 'tau', 'action', 'q_prev', 'tau_prev', 'action_prev'):
            continue
        if k == 'foot_pos':
            for i, leg in enumerate(LEGS):
                for j, a in enumerate(('x', 'y', 'z')):
                    out[f'foot_{leg}_pos_{a}'] = num(v[i, j])
        elif k == 'foot_vel':
            for i, leg in enumerate(LEGS):
                for j, a in enumerate(('vx', 'vy', 'vz')):
                    out[f'foot_{leg}_vel_{a}'] = num(v[i, j])
        elif k == 'foot_ground':
            for i, leg in enumerate(LEGS):
                out[f'foot_{leg}_ground'] = num(v[i])
        elif k == 'foot_clearance':
            for i, leg in enumerate(LEGS):
                out[f'foot_{leg}_clearance'] = num(v[i])
        elif k == 'foot_slip':
            for i, leg in enumerate(LEGS):
                out[f'foot_{leg}_slip'] = num(v[i])
        elif k == 'foot_force':
            for i, leg in enumerate(LEGS):
                for j, a in enumerate(('x', 'y', 'z')):
                    out[f'foot_{leg}_force_{a}'] = num(v[i, j])
        elif k == 'foot_peak':
            for i, leg in enumerate(LEGS):
                out[f'foot_{leg}_peak'] = num(v[i])
        elif k == 'foot_count':
            for i, leg in enumerate(LEGS):
                out[f'foot_{leg}_count'] = int(v[i])
        elif k == 'foot_contact':
            for i, leg in enumerate(LEGS):
                out[f'foot_{leg}_contact'] = int(bool(v[i]))
        else:
            out[k] = v
    for i, name in enumerate(joint_names):
        key = name.replace('_joint', '')
        out[f'q_{key}'] = num(row['q'][i])
        out[f'dq_{key}'] = num(row['dq'][i])
        out[f'q_target_{key}'] = num(row['q_target'][i])
        out[f'tau_{key}'] = num(row['tau'][i])
        out[f'action_{key}'] = num(row['action'][i])
        out[f'dq_delta_{key}'] = num(row['q'][i] - q_prev[i])
        out[f'tau_delta_{key}'] = num(row['tau'][i] - tau_prev[i])
        out[f'action_delta_{key}'] = num(row['action'][i] - action_prev[i])
    return out


def run_one(sim, policy, ckpt, lane, seed, direction, seconds, speed, out_dir, write_obs=True):
    sim.reset(lane, seed)
    if direction == 'backward':
        sim.data.qpos[3:7] = np.array([0., 0., 0., 1.])  # yaw=pi，背对+x台阶，负vx向+x走
        mujoco.mj_forward(sim.model, sim.data)
    command = (speed, 0., 0.) if direction == 'forward' else (-speed, 0., 0.)
    q_prev = sim.data.qpos[sim.qadr].copy()
    tau_prev = sim.data.actuator_force[sim.actuators].copy()
    action_prev = np.zeros(12, dtype=np.float32)
    rows, obs_rows, scan_rows, records = [], [], [], []
    stable, reason, success, t0 = 0., 'timeout', False, None
    active_seconds = 0.
    n_steps = round((seconds + 2.) / sim.dt)
    for i in range(n_steps):
        active = i * sim.dt >= 2.
        step_cmd = command if active else (0., 0., 0.)
        row = sim.step(step_cmd)
        if t0 is None and active:
            t0 = float(sim.data.time)
        if active:
            rec = build_row(sim, row, step_cmd, q_prev.copy(), tau_prev.copy(), action_prev.copy(), t0)
            records.append(rec)
            rows.append(row)
            active_seconds += sim.dt
            if write_obs:
                obs_rows.append(sim.history.detach().cpu().numpy().reshape(-1).copy())
            if getattr(sim, 'scan_dim', 0) and sim.scan_cache is not None:
                scan_rows.append(sim.scan_cache.detach().cpu().numpy().reshape(-1).copy())
        q_prev = sim.data.qpos[sim.qadr].copy()
        tau_prev = sim.data.actuator_force[sim.actuators].copy()
        action_prev = sim.previous.copy()
        if not row['finite'] or row['tilt_degrees'] > 60:
            reason = 'fallen_or_nonfinite'; break
        if row['nonwheel_force'] > 15.:
            reason = 'nonwheel_collision'; break
        if abs(row['y'] - sim.lane['center_y']) > sim.lane['half_width']:
            reason = 'left_lane'; break
        reached = (row['x'] >= sim.lane['goal_x'] and row['tilt_degrees'] < 25
                   and row['wheel_contacts'] >= 2 and .15 < row['clearance'] < .70
                   and abs(sim.state()[6]) < .35)
        stable = stable + sim.dt if reached else 0.
        if stable >= .20 - 1e-9:
            success, reason = True, 'completed'; break
        if len(rows) > round(8. / sim.dt) and row['x'] - rows[-round(8. / sim.dt)]['x'] < .12:
            reason = 'stuck_8s'; break
    # 写 CSV/obs
    run_dir = out_dir / f'{lane}_seed{seed}_{direction}'
    run_dir.mkdir(parents=True, exist_ok=True)
    if records:
        flat = [flatten(r, sim.names[:12], r['q_prev'], r['tau_prev'], r['action_prev']) for r in records]
        with (run_dir / 'trajectory.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader(); writer.writerows(flat)
        if write_obs:
            np.save(run_dir / 'observations.npy', np.stack(obs_rows))
        if scan_rows:
            np.save(run_dir / 'height_scan.npy', np.stack(scan_rows))
    v = np.array([[r['vx'], r['vy'], r['wz']] for r in rows] or [[0., 0., 0.]])
    foot_peak = np.array([r['foot_peak'].max() for r in records] or [0.])
    tau_abs = np.array([np.abs(r['tau']).max() for r in records] or [0.])
    summary = dict(
        lane=lane, seed=seed, direction=direction, success=success, reason=reason,
        measured_seconds=len(rows) * sim.dt, max_x=max((r['x'] for r in rows), default=float(sim.data.qpos[0])),
        max_tilt=max((r['tilt_degrees'] for r in rows), default=0.),
        peak_nonwheel_force=max((r['nonwheel_force'] for r in rows), default=0.),
        actual_mean=v.mean(0).tolist(), velocity_mae=np.abs(v - np.array(command)).mean(0).tolist(),
        rows=len(rows), foot_peak_max=float(foot_peak.max()), tau_abs_max=float(tau_abs.max()),
        csv=str(run_dir / 'trajectory.csv'), observations=str(run_dir / 'observations.npy') if write_obs else None,
        height_scan=str(run_dir / 'height_scan.npy') if scan_rows else None,
        scan_mode=getattr(sim, 'scan_mode', 'none'), scan_dim=int(getattr(sim, 'scan_dim', 0)),
        specification=sim.lane,
    )
    (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=None,
                   help='检查点；默认取 RUN_DEFAULT 下最新 model_*.pt')
    p.add_argument('--lanes', nargs='+', choices=LANES, default=['stairs10', 'stairs12'])
    p.add_argument('--directions', nargs='+', choices=('forward', 'backward'), default=['forward', 'backward'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--seconds', type=float, default=50.)
    p.add_argument('--speed', type=float, default=.2)
    p.add_argument('--profile', choices=('easy', 'standard', 'hard'), default='standard')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--no-obs', action='store_true', help='不保存235维观测历史')
    args = p.parse_args()
    ckpt_path = args.checkpoint or max(RUN_DEFAULT.glob('model_*.pt'),
                                       key=lambda x: (int(x.stem.split('_')[-1]), x.stat().st_mtime_ns))
    ckpt, policy, actor = load_checkpoint(ckpt_path)
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.output or (Path(__file__).resolve().parent / 'runs' / f'{stamp}_model{ckpt["step"]}')
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        script=SCRIPT_VERSION, checkpoint=str(ckpt_path.resolve()), step=ckpt.get('step'),
        format=ckpt.get('format'), robot_id=ckpt.get('robot_id'), robot_signature=ckpt.get('robot_signature'),
        profile=args.profile, lanes=args.lanes, directions=args.directions, seeds=args.seeds,
        seconds=args.seconds, speed=args.speed, worktree=str(WORKTREE),
        git_commit=subprocess.run(['git', '-C', str(WORKTREE), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True).stdout.strip(),
        host=socket.gethostname(), platform=platform.platform(),
        python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
        mujoco=mujoco.__version__, cwd=os.getcwd(),
    )
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(f'[模型] {ckpt_path} (step {ckpt.get("step")})')
    print(f'[输出] {out_dir}')
    summaries = []
    for seed in args.seeds:
        sim = RecordingGo2W(ckpt, actor, seed, args.profile)
        for lane in args.lanes:
            for direction in args.directions:
                s = run_one(sim, policy, ckpt, lane, seed, direction, args.seconds, args.speed,
                            out_dir, write_obs=not args.no_obs)
                summaries.append(s)
                print(f"[结果] {lane} seed{seed} {direction}: success={s['success']} "
                      f"reason={s['reason']} max_x={s['max_x']:.2f} tilt={s['max_tilt']:.1f} "
                      f"force={s['peak_nonwheel_force']:.1f} foot_peak={s['foot_peak_max']:.1f} "
                      f"tau_max={s['tau_abs_max']:.2f}")
    (out_dir / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
    print(f'[完成] {out_dir}')


if __name__ == '__main__':
    main()
