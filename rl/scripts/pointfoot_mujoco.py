"""本机 Go2W checkpoint 的原生 MuJoCo 查看器、三维地图导出及 CPU 能力评测。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch

from go2w_test_maps import LANES, PROFILES, add_test_park, terrain_height

BASE = Path("/media/rmer/系统/rpwd_training/pointfoot_mjlab")
# 数据盘“系统”偶发未挂载或只读（NTFS 脏位），检查点可读但测试产物写不进去；
# 写盘失败时回退到同内容的可写数据盘，避免键盘测试刚加载完模型就 mkdir 崩溃。
OUTPUT_BASE = BASE if os.access(BASE, os.W_OK) else Path("/media/rmer/data/rpwd_training/pointfoot_mjlab")


def latest_checkpoint():
    candidates = []
    for base in (BASE, Path("/media/rmer/data/rpwd_training/go2w_supermimic")):
        for run in base.glob("*原版Go2W*"):
            candidates.extend(run.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError("没有找到本机训练的 Go2W 检查点，请指定 --checkpoint")
    return max(candidates, key=lambda p: (int(p.stem.split("_")[-1]), p.stat().st_mtime_ns))


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class HeldKeyboard:
    """查询当前窗口的实际按键状态，松键或失焦立即返回零速度。"""
    def __init__(self, speed, lateral, turn):
        from Xlib import display, XK
        self.display = display.Display()
        self.root = self.display.screen().root
        self.active_atom = self.display.intern_atom("_NET_ACTIVE_WINDOW")
        self.pid_atom = self.display.intern_atom("_NET_WM_PID")
        self.codes = {n: self.display.keysym_to_keycode(XK.string_to_keysym(n))
                      for n in ("Up", "Down", "Left", "Right", "q", "e", "space")}
        self.speed, self.lateral, self.turn = speed, lateral, turn

    def command(self):
        from Xlib import X, error
        try:
            active = self.root.get_full_property(self.active_atom, X.AnyPropertyType)
            if active is None or not len(active.value) or not active.value[0]:
                return np.zeros(3)
            window = self.display.create_resource_object("window", int(active.value[0]))
            pid = window.get_full_property(self.pid_atom, X.AnyPropertyType)
            if pid is None or int(pid.value[0]) != os.getpid():
                return np.zeros(3)
            bits = self.display.query_keymap()
        except (error.XError, OSError):
            return np.zeros(3)
        held = {n for n, c in self.codes.items() if bits[c // 8] & (1 << (c % 8))}
        if "space" in held:
            return np.zeros(3)
        x = int("Up" in held) - int("Down" in held)
        y = int("Left" in held) - int("Right" in held)
        divisor = max(1., math.hypot(x, y))
        return np.array([self.speed * x / divisor, self.lateral * y / divisor,
                         self.turn * (int("q" in held) - int("e" in held))])

    def close(self):
        self.display.close()


class NativeGo2W:
    """复用上游实体/执行器配置，直接使用原生 mj_step，无 Warp GPU 仿真实例。"""
    def __init__(self, checkpoint, actor, seed=0, profile="standard"):
        from mjlab.scene import Scene
        self.package = checkpoint.get('backend_package') or ('rpwd_pointfoot_supermimic' if checkpoint['format'].startswith('rpwd-pointfoot-') else 'go2w_supermimic')
        self.policy = importlib.import_module(self.package + '.core.policy')
        robot_config = importlib.import_module(self.package + '.robot.config')
        evaluation_env_cfg, checkpoint_wheel_vel = self.policy.evaluation_env_cfg, self.policy.checkpoint_wheel_vel
        LEG_JOINT_NAMES, WHEEL_JOINT_NAMES = robot_config.LEG_JOINT_NAMES, robot_config.WHEEL_JOINT_NAMES
        self.robot_id = checkpoint.get('robot_id', 'go2w')
        if checkpoint_wheel_vel(checkpoint) or checkpoint["config"].get("terrain_revision") != 7:
            raise ValueError("当前原生适配器只接受新点足235/508维Actor")
        self.checkpoint, self.actor, self.seed = checkpoint, actor, seed
        self.cfg = evaluation_env_cfg(checkpoint, 0, "flat")
        self.cfg.scene.num_envs = 1
        self.cfg.scene.terrain, self.cfg.scene.sensors = None, ()
        self.cfg.scene.spec_fn = None
        self.scene = Scene(self.cfg.scene, device="cpu")
        # 与训练场和 PDF 保持同一尺寸：不再把桩/胎环放大 1.4 倍，
        # 否则键盘测试的方块 280mm、缝隙 7cm，和训练用的 200mm、5cm 对不上。
        self.map = add_test_park(self.scene.spec, seed, profile, robot_scale=1.0)
        self.model = self.scene.compile()
        self.cfg.sim.mujoco.apply(self.model)
        # 把实际求解器参数同步到可导出的 MJCF，而非只修改内存中的模型。
        for field in fields(self.cfg.sim.mujoco):
            value = getattr(self.model.opt, field.name)
            if isinstance(value, np.ndarray):
                getattr(self.scene.spec.option, field.name)[:] = value
            else:
                setattr(self.scene.spec.option, field.name, value)
        self.data = mujoco.MjData(self.model)
        self.names = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
        self.joints = [self.model.joint("robot/" + name).id for name in self.names]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.actuators = []
        for joint in self.joints:
            found = np.where(self.model.actuator_trnid[:, 0] == joint)[0]
            if len(found) != 1:
                raise ValueError("每个点足关节必须对应唯一原生执行器")
            self.actuators.append(int(found[0]))
        self.base = self.model.body("robot/base_link").id
        self.home = self.model.key("init_state").qpos.copy()
        self.default = self.home[self.qadr].copy()
        self.order = checkpoint["env_action_order"]
        if sorted(self.order) != list(range(12)) or self.model.nu != 12:
            raise ValueError("动作映射不是12维排列")
        self.dt = self.model.opt.timestep * self.cfg.decimation
        if abs(self.dt - checkpoint["dt"]) > 1e-9:
            raise ValueError("原生模型与策略控制周期不一致")
        scale = self.cfg.actions["joint_pos"].scale
        import re
        self.scales = np.array([next(v for pattern, v in scale.items() if re.fullmatch(pattern, n))
                                for n in self.names[:12]])
        self.wheel_scale = 0.
        self.robot_geom = np.array([self.model.geom(i).name.startswith("robot/")
                                   for i in range(self.model.ngeom)])
        self.ground_geom = np.array([self.model.geom(i).name.startswith("test_")
                                    for i in range(self.model.ngeom)])
        self.wheel_geom = {self.model.geom(f"robot/{leg}_foot_collision").id: i
                           for i, leg in enumerate(("FL", "FR", "RL", "RR"))}
        # 273维真值高度图：与训练一致，机体 yaw 坐标系 21×13（x±1m, y±0.6m）。
        gx, gy = np.meshgrid(np.linspace(-1., 1., 21), np.linspace(-.6, .6, 13), indexing="ij")
        self.scan_offsets = np.stack((gx.flatten(), gy.flatten()), -1)
        self.scan_shape = (21, 13)
        self.scan_dim = int(getattr(actor, "scan_dim", 0) or 0)
        self.last_scan = None
        self.scan_world = None
        self.scan_show = True
        # 启动时把测试场静态几何栅格化成 heightfield；每步只做查表，
        # 不再逐点调用 mj_ray（那是键盘测试卡顿的主要来源）。
        # 盲模型也要建：训练 TerrainBridge 的 state z 是相对脚下地面的高度。
        TerrainField = importlib.import_module(self.package + '.core.terrain').TerrainField
        self.field = TerrainField.from_model(
            self.model, set(np.where(self.robot_geom)[0].tolist()),
            bounds=((-18., -44.), (18., 44.)))
        self.reset("stairs8")

    def reset(self, lane, seed=0):
        self.lane = next(x for x in self.map["lanes"] if x["name"] == lane)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.home
        self.data.qpos[:3] = self.lane["spawn"]
        if seed:
            rng = np.random.default_rng(seed)
            self.data.qpos[:2] += rng.uniform(-.03, .03, 2)
            yaw = rng.uniform(-.025, .025)
            self.data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.previous = np.zeros(12, dtype=np.float32)
        self.history = None
        self.last_scan = None
        self.scan_world = None
        self.data.ctrl[self.actuators[:12]] = self.default[:12]
        self.data.ctrl[self.actuators[12:]] = 0.
        self.peak_nonwheel, self.contacts = 0., np.zeros(4, dtype=bool)
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        r = self.data.xmat[self.base].reshape(3, 3)
        # 与训练 TerrainBridge 一致：state z 相对脚下地面，不是世界高度。
        ground = float(self.field.sample(
            torch.from_numpy(self.data.qpos[:2].astype(np.float32)[None])).item())
        # 自由关节前三维速度在世界系，角速度在根刚体系。
        return np.concatenate(([self.data.qpos[2] - ground], r.T @ np.array([0., 0., -1.]),
                               r.T @ self.data.qvel[:3], self.data.qvel[3:6],
                               self.data.qpos[self.qadr[:12]] - self.default[:12],
                               self.data.qvel[self.vadr[:12]], np.array([math.sin(2*math.pi*1.5*self.data.time),math.cos(2*math.pi*1.5*self.data.time)]))).astype(np.float32)

    def scan(self):
        """按训练契约计算273维局部高度图；旧235维Actor返回None。"""
        if not self.scan_dim:
            return None
        r = self.data.xmat[self.base].reshape(3, 3)
        yaw = math.atan2(r[1, 0], r[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = self.scan_offsets[:, 0], self.scan_offsets[:, 1]
        x, y = self.data.qpos[:2]
        wx, wy = x + c * ox - s * oy, y + s * ox + c * oy
        xy = torch.from_numpy(np.stack((wx, wy), -1).astype(np.float32))
        h = self.field.sample(xy).numpy()
        scan = np.clip(float(self.data.qpos[2]) - h, -1.5, 1.5).astype(np.float32)
        self.last_scan, self.scan_world = scan, np.stack((wx, wy, h), -1)
        return torch.from_numpy(scan[None])

    def observation(self, command):
        frame, push_history, TERMS = self.policy.frame, self.policy.push_history, self.policy.TERMS
        s = torch.from_numpy(self.state()[None])
        # 上游Actor输出用规范关节顺序；last_action观测使用原环境动作顺序。
        latest = frame(s, torch.from_numpy(self.previous[None]),
                       torch.as_tensor(np.asarray(command, dtype=np.float32)[None]), torch.zeros(1, 12), False)
        if self.history is None:
            self.history = torch.cat([v.repeat(1, self.checkpoint["history"])
                                      for v in latest.split(TERMS, dim=-1)], dim=-1)
        else:
            self.history = push_history(self.history, latest, self.checkpoint["history"])
        if self.history.shape != (1, 47 * self.checkpoint["history"]) or not torch.isfinite(self.history).all():
            raise RuntimeError("原生策略观测无效")
        return self.history

    def _contact_state(self):
        force_by_body = {}
        wheels = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g0, g1 = c.geom
            if self.robot_geom[g0] and self.ground_geom[g1]:
                robot, sign = g0, -1
            elif self.robot_geom[g1] and self.ground_geom[g0]:
                robot, sign = g1, 1
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if robot in self.wheel_geom:
                world = sign * c.frame.reshape(3, 3).T @ force[:3]
                wheels[self.wheel_geom[robot]] |= world[2] > 8.
            else:
                body = self.model.geom_bodyid[robot]
                world = sign * c.frame.reshape(3, 3).T @ force[:3]
                force_by_body[body] = force_by_body.get(body, np.zeros(3)) + world
        peak = max((np.linalg.norm(v) for v in force_by_body.values()), default=0.)
        return peak, wheels

    @torch.inference_mode()
    def step(self, command):
        observation = self.observation(command)
        scan = self.scan()
        action = (self.actor(observation, scan=scan) if scan is not None else self.actor(observation))[0].numpy()
        if not np.isfinite(action).all():
            raise RuntimeError("策略动作出现非有限值")
        self.data.ctrl[self.actuators[:12]] = self.default[:12] + action[:12] * self.scales
        self.data.ctrl[self.actuators[12:]] = np.clip(action[12:] * self.wheel_scale, -35., 35.)
        self.peak_nonwheel = 0.
        for _ in range(self.cfg.decimation):
            mujoco.mj_step(self.model, self.data)
            peak, self.contacts = self._contact_state()
            self.peak_nonwheel = max(self.peak_nonwheel, peak)
        mujoco.mj_forward(self.model, self.data)
        self.previous = action.copy()
        return self.measure(command)

    def measure(self, command=(0., 0., 0.)):
        s = self.state()
        x, y, z = self.data.qpos[:3]
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        clearance = float("nan")
        if finite:
            try:
                if self.field is not None:
                    h = float(self.field.sample(torch.tensor([[x, y]], dtype=torch.float32)).item())
                else:
                    h = terrain_height(self.model, self.data, x, y)
                clearance = float(z - h)
            except RuntimeError:
                finite = False
        return dict(t=float(self.data.time), x=float(x), y=float(y), z=float(z),
                    vx=float(s[4]), vy=float(s[5]), wz=float(s[9]),
                    tilt_degrees=math.degrees(math.acos(float(np.clip(-s[3], -1., 1.)))),
                    clearance=clearance,
                    wheel_contacts=int(self.contacts.sum()), nonwheel_force=float(self.peak_nonwheel),
                    finite=finite, cmd_vx=command[0], cmd_vy=command[1], cmd_wz=command[2])

    def export(self, output):
        config = importlib.import_module(self.package + '.robot.config')
        source_xml = config.RPWD_XML if self.robot_id == 'rpwd' else config.GO2W_XML
        source_assets = source_xml.parent / ('meshes' if self.robot_id == 'rpwd' else 'assets')
        self.scene.spec.key("init_state").qpos = self.data.qpos.copy()
        self.scene.write(output)
        # 上游从磁盘加载的网格不一定进入 MjSpec.assets；补齐导出XML引用的原始资产。
        for mesh in ET.parse(output / "scene.xml").getroot().iter("mesh"):
            if mesh.get("file"):
                target = output / "assets" / mesh.get("file")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    candidate=source_assets / Path(mesh.get('file')).name
                    if not candidate.exists(): candidate=config.REPO_ROOT/'description/RP_wd/urdf/meshes'/Path(mesh.get('file')).name
                    shutil.copyfile(candidate,target)
        license_file = source_xml.parent.parent / 'LICENSE_menagerie_go2'
        if license_file.exists():
            shutil.copyfile(license_file, output / 'LICENSE_menagerie_go2')
        dump_json(output / "map.json", self.map)
        # 确认导出资产完整，独立加载时也能获得相同机器人与物理步长。
        check = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
        if check.nu != 12 or abs(check.opt.timestep - self.model.opt.timestep) > 1e-12:
            raise RuntimeError("导出MJCF与运行模型不一致")


def evaluate(args, checkpoint, actor, report):
    for seed in args.seeds:
        sim = NativeGo2W(checkpoint, actor, seed, args.profile)
        if seed == args.seeds[0]:
            sim.export(args.output / "scene")
        for lane in args.lanes:
            sim.reset(lane, seed)
            rows, stable, reason, success = [], 0., "timeout", False
            command = (args.speed, 0., 0.)
            for i in range(round((args.seconds + 2.) / sim.dt)):
                active = i * sim.dt >= 2.
                row = sim.step(command if active else (0., 0., 0.))
                if active:
                    rows.append(row)
                if not row["finite"] or row["tilt_degrees"] > 60:
                    reason = "fallen_or_nonfinite"
                    break
                if row["nonwheel_force"] > 15.:
                    reason = "nonwheel_collision"
                    break
                if abs(row["y"] - sim.lane["center_y"]) > sim.lane["half_width"]:
                    reason = "left_lane"
                    break
                reached = (row["x"] >= sim.lane["goal_x"] and row["tilt_degrees"] < 25
                           and row["wheel_contacts"] >= 2 and .15 < row["clearance"] < .70
                           and abs(sim.state()[6]) < .35)
                stable = stable + sim.dt if reached else 0.
                if stable >= .20 - 1e-9:
                    success, reason = True, "completed"
                    break
                if len(rows) > round(8. / sim.dt) and row["x"] - rows[-round(8. / sim.dt)]["x"] < .12:
                    reason = "stuck_8s"
                    break
            path = args.output / f"{lane}_seed{seed}.csv"
            if rows:
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            v = np.array([[r["vx"], r["vy"], r["wz"]] for r in rows] or [[0., 0., 0.]])
            result = dict(lane=lane, seed=seed, success=success, reason=reason,
                          manual_navigation=sim.lane["manual_navigation"],
                          specification=sim.lane, measured_seconds=len(rows) * sim.dt,
                          actual_mean=v.mean(0).tolist(), velocity_mae=np.abs(v - command).mean(0).tolist(),
                          max_x=max((r["x"] for r in rows), default=float(sim.data.qpos[0])),
                          max_tilt=max((r["tilt_degrees"] for r in rows), default=0.),
                          peak_nonwheel_force=max((r["nonwheel_force"] for r in rows), default=0.), csv=str(path))
            report["trials"].append(result)
            dump_json(args.output / "summary.json", report)
            print("[能力测试] " + json.dumps(result, ensure_ascii=False), flush=True)


def _draw_scan(viewer, sim):
    """在查看器里用彩色小方块叠加273个scan点；红色=高地面，蓝色=低地面。"""
    scn = viewer.user_scn
    scn.ngeom = 0
    if not sim.scan_show or sim.scan_world is None:
        return
    for x, y, h in sim.scan_world[::2]:
        t = float(np.clip(h / .55, 0., 1.))
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array((.012, .012, .004)),
                            np.array((x, y, h + .006)),
                            np.eye(3).reshape(-1),
                            np.array((t, .15, 1. - t, .75), dtype=np.float32))
        scn.ngeom += 1


def _show_heatmap(sim, enabled, state):
    """可选2D高程热力图；方向为机体系 x向右、y向上。

    这个环境的 OpenCV 是 headless 构建（cv2.imshow 不可用），因此用
    matplotlib TkAgg；只在第一次调用时创建窗口，之后只更新图像。
    """
    if not enabled or not sim.scan_show or sim.last_scan is None:
        return
    h = (float(sim.data.qpos[2]) - sim.last_scan).reshape(sim.scan_shape).T
    if state["plt"] is None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        state["plt"] = plt
        state["fig"], state["ax"] = plt.subplots(figsize=(6.4, 4.2))
        state["im"] = state["ax"].imshow(h, vmin=0., vmax=.55, cmap="turbo", origin="lower")
        state["ax"].set_xlabel("x [m] (body right)")
        state["ax"].set_ylabel("y [m] (body up)")
        state["fig"].colorbar(state["im"], ax=state["ax"], label="ground height [m]")
        state["fig"].canvas.manager.set_window_title("height scan (body frame)")
        plt.show(block=False)
    else:
        state["im"].set_data(h)
    state["ax"].set_title(f"ground h  min {h.min():.2f}  max {h.max():.2f}  mean {h.mean():.2f}")
    state["fig"].canvas.draw_idle()
    state["fig"].canvas.flush_events()


def play(sim, args):
    import mujoco.viewer
    commands = np.zeros(3)
    events = queue.SimpleQueue()
    keyboard = HeldKeyboard(args.speed, args.lateral_speed, args.turn_speed)
    sim.reset(args.lane)
    heat = {"plt": None, "fig": None, "ax": None, "im": None}
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=events.put) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.base
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -25
        print(f"[就绪] 官方MuJoCo查看器，模型第{sim.checkpoint['step']}轮，场景{args.lane}", flush=True)
        print("按住方向键前后/横移，Q/E转向，松开归零；Space停止；R复位；1..8切赛道；K暂停；H高程显示；Esc退出", flush=True)
        paused = False
        frame = 0
        while viewer.is_running():
            began = time.monotonic()
            while not events.empty():
                key = events.get()
                if key in (256,):
                    return
                if key == 32:
                    commands[:] = 0.
                elif key == ord("R"):
                    commands[:] = 0.
                    sim.reset(sim.lane["name"])
                    paused = False
                elif ord("1") <= key <= ord("8"):
                    commands[:] = 0.
                    sim.reset(LANES[key - ord("1")])
                    paused = False
                    # 数字键也是原生查看器的组可见性快捷键，切赛道后保留地形和机器人可见。
                    viewer.opt.geomgroup[0] = viewer.opt.geomgroup[2] = viewer.opt.geomgroup[4] = True
                    viewer.opt.geomgroup[3] = False
                elif key in (ord("9"), ord("0")):
                    commands[:] = 0.
                    sim.reset("stones" if key == ord("9") else "tires")
                    paused = False
                    viewer.opt.geomgroup[0] = viewer.opt.geomgroup[2] = viewer.opt.geomgroup[4] = True
                    viewer.opt.geomgroup[3] = False
                elif key == ord("K"):
                    paused = not paused
                elif key == ord("H"):
                    sim.scan_show = not sim.scan_show
            commands[:] = keyboard.command() if not paused else 0.
            row = sim.measure(commands) if paused else sim.step(commands)
            if not row["finite"]:
                commands[:] = 0.
                paused = True
            scan_text = ""
            if sim.last_scan is not None:
                h = float(sim.data.qpos[2]) - sim.last_scan
                scan_text = (f"scan h {h.min():.2f}/{h.max():.2f}/{h.mean():.2f}m | "
                             f"H {'on' if sim.scan_show else 'off'}\n")
            # 控制器/物理仍50Hz；3D渲染25Hz、热力图10Hz，避免软件GL下过度占用。
            if frame % 2 == 0:
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 f"{sim.robot_id.upper()} {sim.checkpoint['step']} | {sim.lane['name']} | CPU native\n"
                                 "Hold Arrows: vx/vy | Q/E: yaw | Release: stop\nSpace stop  R reset\n"
                                 "1 Flat  2 Stairs4  3 Stairs6  4 Stairs8\n"
                                 "5 Stairs10  6 Stairs12  7 Stairs16  8 Ramp\n"
                                 "9 Stones  0 Tires\nK pause  H scan  Esc exit\n"
                                 f"cmd {commands.round(2)} | vx {row['vx']:.2f} | tilt {row['tilt_degrees']:.1f}\n"
                                 f"support {row['wheel_contacts']} | nonwheel {row['nonwheel_force']:.1f}N\n"
                                 + scan_text
                                 + ("PAUSED" if paused else ""), ""))
                _draw_scan(viewer, sim)
                viewer.sync()
            _show_heatmap(sim, args.scan_heatmap and frame % 5 == 0, heat)
            frame += 1
            time.sleep(max(0., sim.dt - (time.monotonic() - began)))
    keyboard.close()
    if heat["plt"] is not None:
        heat["plt"].close("all")


def main(checkpoint_finder=None, output_base=None, expected_robot=None, policy_backend=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("play", "evaluate", "generate"), default="play")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--profile", choices=PROFILES, default="standard")
    p.add_argument("--lane", choices=LANES, default="stairs8")
    p.add_argument("--lanes", nargs="+", choices=LANES, default=list(LANES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--seconds", type=float, default=45.)
    p.add_argument("--speed", type=float, default=.3)
    p.add_argument("--lateral-speed", type=float, default=.2)
    p.add_argument("--turn-speed", type=float, default=.5)
    p.add_argument("--scan-heatmap", action="store_true",
                   help="键盘测试时额外打开21×13高程热力图窗口")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not 0 < args.speed <= .6:
        p.error("测试时长必须为正数，前进速度必须在(0,0.6]内")
    if not 0 < args.lateral_speed <= .4 or not 0 < args.turn_speed <= 1.:
        p.error("横移速度须在(0,0.4]，角速度须在(0,1]内")
    torch.set_num_threads(1)
    args.checkpoint = (args.checkpoint or (checkpoint_finder or latest_checkpoint)()).resolve()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if expected_robot is not None and ckpt.get('robot_id', 'go2w') != expected_robot:
        raise ValueError('检查点机器人与请求不匹配')
    package = policy_backend or ('rpwd_pointfoot_supermimic' if ckpt['format'].startswith('rpwd-pointfoot-') else 'go2w_supermimic')
    actor = importlib.import_module(package + '.core.policy').load_actor(ckpt, 'cpu')
    metadata = {k: ckpt[k] for k in ("format", "step", "config", "history", "dt", "env_action_order")}
    metadata["actor_wheel_vel"] = False
    metadata["actor_scan_dim"] = int(ckpt.get("actor_scan_dim") or 0)
    metadata["actor_scan_hidden"] = ckpt.get("actor_scan_hidden")
    metadata["backend_package"] = package
    for name in ('robot_id', 'robot_signature'):
        if name in ckpt: metadata[name] = ckpt[name]
    del ckpt
    if args.output is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        args.output = (output_base or OUTPUT_BASE) / "native_tests" / f"{stamp}_{args.mode}_{metadata['step']}"
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  model_step=metadata["step"], engine="MuJoCo native CPU", mujoco_version=mujoco.__version__,
                  actor_observation=47 * metadata["history"] + metadata["actor_scan_dim"],
                  actor_action=12, policy_dt=metadata["dt"], observation_noise=False,
                  profile=args.profile, source=os.environ.get("GO2W_SOURCE", package), robot_id=metadata.get("robot_id", "go2w"),
                  speed=args.speed, seconds=args.seconds, seeds=args.seeds, trials=[])
    dump_json(args.output / "run.json", report)
    print(f"[模型] {args.checkpoint}\n[输出] {args.output}", flush=True)
    if args.mode == "evaluate":
        evaluate(args, metadata, actor, report)
    else:
        sim = NativeGo2W(metadata, actor, args.seeds[0], args.profile)
        sim.reset(args.lane)
        sim.export(args.output / "scene")
        if args.mode == "play":
            play(sim, args)
    print(f"[完成] {args.output}", flush=True)


if __name__ == "__main__":
    main()
