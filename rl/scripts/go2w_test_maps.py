"""原生 MuJoCo 三维能力测试场；实体尺寸以米计，随机地图可按种子复现。"""
from __future__ import annotations

import math
import mujoco
import numpy as np

LANES = ("flat", "stairs4", "stairs6", "stairs8", "stairs10", "stairs12", "stairs16", "ramp", "rubble", "mixed", "slalom", "stones", "tires")
PROFILES = {"easy": .75, "standard": 1., "hard": 4 / 3}


def add_test_park(spec, seed=0, profile="standard", robot_scale=1.0):
    rng = np.random.default_rng(seed)
    scale = PROFILES[profile]
    manifest = {"generator": "go2w-native-park-v1", "seed": seed, "profile": profile,
                "height_scale": scale, "robot_scale": robot_scale, "lanes": [], "geometry": []}
    colors = ((.33, .65, .72), (.35, .67, .43), (.72, .55, .35), (.83, .64, .26),
              (.45, .62, .68), (.78, .37, .31), (.58, .48, .72), (.40, .55, .79),
              (.62, .52, .42), (.62, .42, .73), (.68, .58, .33), (.55, .70, .48),
              (.30, .32, .35))
    world = spec.worldbody
    # 按比赛 PDF，梅花桩是 100mm 高的方块/平台，地面仍是平地；不再挖 1m 深坑。
    world.add_geom(name="test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=(18., 30., .1), rgba=(.24, .28, .32, 1), group=0,
                   contype=1, conaffinity=1, condim=3, friction=(.8, .02, .001))
    world.add_light(name="test_sun", pos=(0, 0, 10), dir=(-.3, -.2, -1),
                    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    castshadow=False, diffuse=(.8, .8, .8))
    spec.visual.quality.shadowsize = 1024
    spec.stat.extent = 30

    def box(name, x, y, height, length, width, color, yaw=0., base=0.):
        if height <= 0:
            return
        world.add_geom(name="test_" + name, type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=(x, y, base + height / 2), size=(length / 2, width / 2, height / 2),
                       quat=(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)),
                       rgba=(*color, 1), group=0, contype=1, conaffinity=1,
                       condim=3, friction=(.8, .02, .001))
        manifest["geometry"].append(dict(name=name, pos=[x, y, base + height / 2],
                                         size=[length, width, height], yaw=yaw))

    def stairs(name, y, h, color, n=5, start=2.2):
        tread, landing, width = .30, 1.20, 2.6
        for k in range(n):
            box(f"{name}_up{k}", start + (k + .5) * tread, y, (k + 1) * h,
                tread, width, color)
        top = start + n * tread
        box(name + "_landing", top + landing / 2, y, n * h, landing, width, color)
        for k in range(n - 1):
            box(f"{name}_down{k}", top + landing + (k + .5) * tread, y,
                (n - k - 1) * h, tread, width, color)
        return top + landing + (n - 1) * tread

    def rubble(name, y, color, start=2.2, end=6.6, max_h=.07):
        # 每格一块随机石块，避免叠加后高度超出测试规格。
        for ix, x in enumerate(np.arange(start, end, .48)):
            for iy, yy in enumerate((-.78, -.26, .26, .78)):
                a, b = rng.uniform(.32, .48, 2)
                box(f"{name}_{ix}_{iy}", x + rng.uniform(-.05, .05),
                    y + yy + rng.uniform(-.035, .035), rng.uniform(.02, max_h),
                    a, b, color, yaw=rng.uniform(-.30, .30))

    for i, (name, color) in enumerate(zip(LANES, colors)):
        y, goal = (i - (len(LANES) - 1) / 2) * 4., 7.5
        lane = dict(name=name, number=i + 1, center_y=y, spawn=[.5, y, .4],
                    goal_x=goal, half_width=1.3, manual_navigation=name == "slalom")
        # 标线只有视觉效果，不制造毫米级假台阶。
        for yy in (y - 1.5, y + 1.5):
            world.add_geom(name=f"mark_{name}_{yy}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=(5.5, yy, .002), size=(6., .025, .001),
                           rgba=(*color, 1), group=4, contype=0, conaffinity=0)
        if name.startswith("stairs"):
            h = int(name[6:]) / 100 * scale
            stairs(name, y, h, color)
            lane.update(step_height=h, risers=5, tread=.30,
                        description=f"{h * 100:.1f}cm上阶、平台及下阶")
        elif name == "ramp":
            angle = math.radians(8 * scale)
            height = 3 * math.tan(angle)
            for tag, x0, x1, z0, z1 in (("up", 2.2, 5.2, 0., height),
                                       ("down", 6.4, 9.4, height, 0.)):
                pitch = math.atan2(z1 - z0, x1 - x0)
                thick = .16
                world.add_geom(name=f"test_ramp_{tag}", type=mujoco.mjtGeom.mjGEOM_BOX,
                               pos=((x0 + x1) / 2 + math.sin(pitch) * thick / 2, y,
                                    (z0 + z1) / 2 - math.cos(pitch) * thick / 2),
                               size=(math.hypot(x1 - x0, z1 - z0) / 2, 1.3, thick / 2),
                               quat=(math.cos(pitch / 2), 0, -math.sin(pitch / 2), 0),
                               rgba=(*color, 1), group=0, contype=1, conaffinity=1,
                               condim=3, friction=(.8, .02, .001))
                manifest["geometry"].append(dict(name=f"ramp_{tag}", endpoints=[[x0, z0], [x1, z1]]))
            box("ramp_top", 5.8, y, height, 1.2, 2.6, color)
            lane.update(goal_x=10.2, slope_degrees=8 * scale, description="连续上坡、平台及下坡")
        elif name == "rubble":
            rubble(name, y, color, max_h=.07 * scale)
            lane.update(max_obstacle_height=.07 * scale, description="随机实体碎石")
        elif name == "mixed":
            stairs(name, y, .06 * scale, color, n=3)
            rubble(name, y, color, start=5.5, end=7.8, max_h=.06 * scale)
            lane.update(goal_x=8.8, step_height=.06 * scale,
                        max_obstacle_height=.06 * scale, description="上下台阶后接随机碎石")
        elif name == "slalom":
            for k in range(3):
                box(f"slalom_{k}", 2.5 + 1.6 * k, y + (.65 if k % 2 == 0 else -.65),
                    .35, .50, 1.05, color)
            lane.update(description="遥控绕障；需要操作者选择路线，盲策略没有自主避障输入")
        elif name == "stones":
            # 按用户要求：桩柱/平台恒定 1m，顶面与平台齐平；地面 0，桩间空隙 1m 深。
            # 平面尺寸仍按 PDF：200×200 方块 + 800×800 平台，间隙 100/200/400mm。
            top, side, plat = 1.0, .20 * robot_scale, .80 * robot_scale
            x0 = 2.3
            box("stones_start", x0 + plat / 2, y, top, plat, plat, color)
            box("stones_end", x0 + 3.5 - plat / 2, y, top, plat, plat, color)
            for yy, xs in ((y + .2 * robot_scale, (x0 + .9, x0 + 1.5)),
                           (y - .2 * robot_scale, (x0 + 1.2, x0 + 1.8))):
                for k, x in enumerate(xs):
                    box(f"stones_{k}_{yy:.2f}", x + side / 2, yy, top, side, side, color)
            lane.update(spawn=[x0 + plat / 2, y, top + .40],
                        goal_x=x0 + 3.5, max_obstacle_height=top,
                        description="按 PDF：200×200×100 方块 + 800×800×100 平台，顶面同平面 100mm，间隙 100/200/400mm")
        elif name == "tires":
            # 轮胎：平躺圆环胎（外径 D、胎孔 d、胎宽 w），多个紧挨成簇；踩胎面，胎孔可踩空。
            D = .40 * robot_scale
            hole = .20 * robot_scale
            w = .08 * robot_scale  # 先低胎面，训练课程从2cm起逐步到16cm
            rmid = (D + hole) / 4
            seg = (D - hole) / 2
            tang = 2 * math.pi * rmid / 12
            for ix, x in enumerate(np.arange(2.3, 6.8, D * .95)):
                for iy, yy in enumerate(np.array((-.9, -.45, 0., .45, .9)) * robot_scale):
                    if (ix + iy) % 2:
                        continue
                    cx, cy = x + rng.uniform(-.02, .02), y + yy + rng.uniform(-.02, .02)
                    for k in range(12):
                        ang = 2 * math.pi * k / 12
                        box(f"tires_{ix}_{iy}_{k}", cx + rmid * math.cos(ang),
                            cy + rmid * math.sin(ang), w, seg, tang, color,
                            yaw=ang)
            lane.update(max_obstacle_height=w,
                        description="平躺轮胎：圆环胎紧密排列，胎面孔隙需落脚")
        else:
            lane.update(description="平地速度、横移与转向基线")
        manifest["lanes"].append(lane)
    return manifest


def terrain_height(model, data, x, y):
    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([x, y, 5.]), np.array([0., 0., -1.]),
                             np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), True, -1, geom_id)
    if distance < 0:
        raise RuntimeError("测试场地面射线没有命中")
    return 5. - distance
