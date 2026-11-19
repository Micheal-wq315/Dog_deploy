"""原生 MuJoCo 三维能力测试场；实体尺寸以米计，随机地图可按种子复现。"""
from __future__ import annotations

import math
import mujoco
import numpy as np

LANES = ("flat", "stairs4", "stairs6", "stairs8", "stairs10", "stairs12", "stairs16", "ramp",
         "rubble", "mixed", "slalom", "lowframe", "wall", "trench", "brickpit", "bridge",
         "stones", "tires")
PROFILES = {"easy": .75, "standard": 1., "hard": 4 / 3}


def add_test_park(spec, seed=0, profile="standard", robot_scale=1.0):
    rng = np.random.default_rng(seed)
    scale = PROFILES[profile]
    manifest = {"generator": "go2w-native-park-v1", "seed": seed, "profile": profile,
                "height_scale": scale, "robot_scale": robot_scale, "lanes": [], "geometry": []}
    colors = ((.33, .65, .72), (.35, .67, .43), (.72, .55, .35), (.83, .64, .26),
              (.45, .62, .68), (.78, .37, .31), (.58, .48, .72), (.40, .55, .79),
              (.62, .52, .42), (.62, .42, .73), (.68, .58, .33), (.55, .70, .48),
              (.30, .32, .35), (.38, .62, .40), (.66, .44, .30), (.52, .60, .36),
              (.58, .40, .62), (.42, .58, .60))
    world = spec.worldbody
    # 按比赛 PDF，梅花桩是 100mm 高的方块/平台，地面仍是平地；不再挖 1m 深坑。
    world.add_geom(name="test_ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=(18., 44., .1), rgba=(.24, .28, .32, 1), group=0,
                   contype=1, conaffinity=1, condim=3, friction=(.8, .02, .001))
    world.add_light(name="test_sun", pos=(0, 0, 10), dir=(-.3, -.2, -1),
                    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    castshadow=False, diffuse=(.8, .8, .8))
    spec.visual.quality.shadowsize = 1024
    spec.stat.extent = 44

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

    def cylinder(name, x, y, radius, height, color, base=0.):
        world.add_geom(name="test_" + name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=(x, y, base + height / 2), size=(radius, height / 2, 0.),
                       rgba=(*color, 1), group=0, contype=1, conaffinity=1,
                       condim=3, friction=(.8, .02, .001))
        manifest["geometry"].append(dict(name=name, pos=[x, y, base + height / 2],
                                         size=[radius, height], kind="cylinder"))

    def marker(name, x, y, radius, color):
        # 比赛标线：无碰撞、不可见以外的视觉圆环（PDF 红色虚线必达区）。
        world.add_geom(name="test_" + name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=(x, y, .002), size=(radius, .001, 0.),
                       rgba=(*color, 1), group=4, contype=0, conaffinity=0)

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
            # PDF 绕桩：竖杆 Ø100mm、高 100mm、中心距 900mm，L 形布置；
            # 两端 Ø300mm 必达区用红色圆盘标记（仅视觉，无碰撞）。
            r, pole_h, pitch = .05 * robot_scale, .10 * robot_scale, .90 * robot_scale
            x0 = 2.6
            legs = [(x0 + pitch * k, y - pitch) for k in range(4)]
            legs += [(x0 + 3 * pitch, y - pitch + pitch * k) for k in range(1, 4)]
            for k, (px, py) in enumerate(legs):
                cylinder(f"slalom_{k}", px, py, r, pole_h, color)
            marker("slalom_start", x0 - .4, y - pitch, .15 * robot_scale, (.8, .1, .1))
            marker("slalom_end", x0 + 3 * pitch, y + 2 * pitch + .4,
                   .15 * robot_scale, (.8, .1, .1))
            lane.update(half_width=2.2 * robot_scale, manual_navigation=True,
                        description="PDF 绕桩：Ø100×100 竖杆、中心距 900mm、L 形，两端 Ø300 必达区")
        elif name == "lowframe":
            # PDF 限高框架：内高 290mm、内宽/长 800mm；立柱与顶框 60mm。
            inner, t, post_h, cx = .80 * robot_scale, .06 * robot_scale, .35 * robot_scale, 3.6
            for sx in (-1, 1):
                for sy in (-1, 1):
                    box(f"lowframe_post_{sx}_{sy}", cx + sx * (inner / 2 + t / 2),
                        y + sy * (inner / 2 + t / 2), post_h, t, t, color)
            for sy in (-1, 1):
                box(f"lowframe_topx_{sy}", cx, y + sy * (inner / 2 + t / 2),
                    t, inner + 2 * t, t, color, base=post_h - t)
            for sx in (-1, 1):
                box(f"lowframe_topy_{sx}", cx + sx * (inner / 2 + t / 2), y,
                    t, t, inner + 2 * t, color, base=post_h - t)
            lane.update(goal_x=5.2, max_obstacle_height=post_h,
                        description="PDF 限高框架：内高 290mm、内宽/长 800mm，需低姿态通过")
        elif name == "wall":
            # PDF 高墙：300mm 高、800mm 宽、80mm 厚，顶部 20mm 包钢。
            cx = 3.6
            box("wall", cx, y, .28, .08 * robot_scale, .80 * robot_scale, color)
            box("wall_steel", cx, y, .02, .08 * robot_scale, .80 * robot_scale,
                (.55, .58, .60), base=.28)
            lane.update(goal_x=5.2, max_obstacle_height=.30,
                        description="PDF 高墙：300×800×80mm，顶部包钢")
        elif name == "trench":
            # PDF 沟壑组合：3 级台阶(120mm 高/320mm 深) → 平台 A → 360mm 沟 → 平台 B → 20° 斜坡。
            step_h, tread, top = .12 * robot_scale, .32 * robot_scale, .36 * robot_scale
            width, gap = .80 * robot_scale, .36 * robot_scale
            x = 2.0
            for k in range(3):
                box(f"trench_step{k}", x + tread / 2, y, (k + 1) * step_h, tread, width, color)
                x += tread
            box("trench_platA", x + width / 2, y, top, width, width, color)
            x += width + gap
            box("trench_platB", x + width / 2, y, top, width, width, color)
            x += width
            angle = math.radians(20)
            run, thick, pitch = top / math.tan(angle), .08, -math.radians(20)
            world.add_geom(name="test_trench_ramp", type=mujoco.mjtGeom.mjGEOM_BOX,
                           pos=(x + run / 2 + math.sin(pitch) * thick / 2, y,
                                top / 2 - math.cos(pitch) * thick / 2),
                           size=(math.hypot(run, top) / 2, width / 2, thick / 2),
                           quat=(math.cos(pitch / 2), 0, -math.sin(pitch / 2), 0),
                           rgba=(*color, 1), group=0, contype=1, conaffinity=1,
                           condim=3, friction=(.8, .02, .001))
            manifest["geometry"].append(dict(name="trench_ramp",
                                             endpoints=[[x, top], [x + run, 0.]]))
            lane.update(goal_x=x + run, max_obstacle_height=top,
                        description="PDF 沟壑组合：台阶 120/320、平台 360、沟宽 360mm、20° 斜坡")
        elif name == "brickpit":
            # PDF 砖木坑：L 形 2000×2000（缺角 1000×1000），围栏高 150mm，
            # 坑内碎砖/松树皮按 10-20mm 高度随机铺。
            cx, half, rail_h, rail_t = 3.8, 1.0, .15, .04
            x0, x1, y0, y1 = cx - half, cx + half, y - half, y + half
            box("brickpit_rail_s", cx, y0, rail_h, 2 * half, rail_t, color)
            box("brickpit_rail_w", x0, y, rail_h, rail_t, 2 * half, color)
            box("brickpit_rail_n", cx - half / 2, y1, rail_h, half, rail_t, color)
            box("brickpit_rail_inner_h", cx + half / 2, y, rail_h, half, rail_t, color)
            box("brickpit_rail_inner_v", cx, y + half / 2, rail_h, rail_t, half, color)
            box("brickpit_rail_e", x1, y - half / 2, rail_h, rail_t, half, color)
            brick, bark = (.58, .30, .22), (.45, .33, .20)
            k = 0
            for gx in np.arange(x0 + .08, x1 - .05, .11):
                for gy in np.arange(y0 + .08, y1 - .05, .11):
                    if gx > cx + .04 and gy > y + .04:
                        continue
                    height = rng.uniform(.01, .02)
                    a, b = rng.uniform(.05, .09, 2)
                    box(f"brickpit_{k}", gx + rng.uniform(-.02, .02),
                        gy + rng.uniform(-.02, .02), height, a, b,
                        brick if rng.random() < .6 else bark,
                        yaw=rng.uniform(-.5, .5))
                    k += 1
            lane.update(max_obstacle_height=rail_h,
                        description="PDF 砖木坑：L 形 2×2m、围栏 150mm、坑内 10-20mm 碎砖/松树皮")
        elif name == "bridge":
            # PDF 单木桥：3000×400×100mm。
            box("bridge", 3.9, y, .10, 3.0 * robot_scale, .40 * robot_scale, color)
            lane.update(goal_x=6.8, max_obstacle_height=.10,
                        description="PDF 单木桥：3000×400×100mm")
        elif name == "stones":
            # 按用户要求：桩柱/平台恒定 1m，顶面与平台齐平；地面 0，桩间空隙 1m 深。
            # 平面尺寸按 PDF（200×200 方块 + 800×800 平台），间隙取训练课程最简单一档：
            # 横向 5cm、纵向 3cm，随难度逐步加到 PDF 的 200/400mm。
            top, side, plat = 1.0, .20 * robot_scale, .80 * robot_scale
            gap_x, gap_y = .05 * robot_scale, .03 * robot_scale
            x0 = 2.3
            box("stones_start", x0 + plat / 2, y, top, plat, plat, color)
            pitch_x, pitch_y = side + gap_x, side + gap_y
            columns = np.arange(x0 + plat + pitch_x / 2, x0 + plat + 4 * pitch_x, pitch_x)
            rows = np.arange(y - .46, y + .47, pitch_y)
            for ix, x in enumerate(columns):
                for iy, yy in enumerate(rows):
                    if (ix + iy) % 2:
                        continue
                    box(f"stones_{ix}_{iy}", x, yy, top, side, side, color)
            end_x = x0 + plat + 4 * pitch_x + gap_x
            box("stones_end", end_x + plat / 2, y, top, plat, plat, color)
            lane.update(spawn=[x0 + plat / 2, y, top + .40],
                        goal_x=end_x + plat, max_obstacle_height=top,
                        description="1m 齐平梅花桩：200×200 方块 + 800×800 平台，横隙 5cm、纵隙 3cm 起步（训练最简单档）")
        elif name == "tires":
            # PDF 165/70R13 常见轮胎：外径约 550mm、内径约 330mm、胎面宽 165mm；
            # 8 个轮胎平躺组成 2×4 矩阵（短边 1100mm、长边 2200mm）。
            D, hole, w = .55 * robot_scale, .33 * robot_scale, .165 * robot_scale
            rmid = (D + hole) / 4
            seg = (D - hole) / 2
            tang = 2 * math.pi * rmid / 12
            x0, y0 = 2.2, y - D  # 两排中心在 y±D/2，整体以赛道中心对称
            for ix in range(4):
                for iy in range(2):
                    cx, cy = x0 + D / 2 + ix * D, y0 + D / 2 + iy * D
                    for k in range(12):
                        ang = 2 * math.pi * k / 12
                        box(f"tires_{ix}_{iy}_{k}", cx + rmid * math.cos(ang),
                            cy + rmid * math.sin(ang), w, seg, tang,
                            (.12, .12, .13), yaw=ang)
            lane.update(goal_x=x0 + 4 * D, max_obstacle_height=w,
                        description="PDF 轮胎阵：165/70R13，Ø550/Ø330、胎宽 165mm，2×4 矩阵 2200×1100mm")
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
