"""垂直碰撞立面的直线专项赛道，尺寸以米为单位。"""

from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def odin_course_terrain(difficulty, cfg):
    size_x, size_y = cfg.size
    meshes = []

    def box(x0, x1, y0, y1, height):
        if height > 0:
            meshes.append(trimesh.creation.box(
                extents=(x1 - x0, y1 - y0, height),
                transform=trimesh.transformations.translation_matrix(
                    ((x0 + x1) / 2, (y0 + y1) / 2, height / 2)
                ),
            ))

    meshes.append(trimesh.creation.box(
        extents=(size_x, size_y, 0.10),
        transform=trimesh.transformations.translation_matrix((size_x / 2, size_y / 2, -0.05)),
    ))
    origin = np.array((0.8, size_y / 2, 0.0))
    h = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    width = 0.65 - 0.20 * difficulty
    if cfg.course_kind in ("stairs_up", "stairs_down"):
        # 先单级，再2/3/4级，避免最低难度也必须连续跨越4次。
        count = 1 + int(round(3 * difficulty))
        start = 1.8
        if cfg.course_kind == "stairs_up":
            for i in range(count):
                box(start + i * width, size_x, 0.0, size_y, h)
                # 嵌套实体箱的顶面是逐级累计高度。
                meshes[-1].apply_translation((0.0, 0.0, i * h))
        else:
            origin[2] = count * h
            box(0.0, start, 0.0, size_y, count * h)
            for i in range(count - 1):
                box(start + i * width, start + (i + 1) * width, 0.0, size_y, (count - 1 - i) * h)
    elif cfg.course_kind == "low_obstacles":
        # 横跨赛道，必须越过；高度上限12cm，不与45cm绕行墙混为一种任务。
        for x in (2.0, 3.8):
            box(x, x + 0.30 + 0.20 * difficulty, 0.0, size_y, h)
    elif cfg.course_kind == "bypass":
        # 明确不可跨越的45cm墙；困难端通道仍留有足够机身宽度。
        span = 0.7 + 2.3 * difficulty
        center = size_y / 2 + np.random.uniform(-0.25, 0.25)
        box(2.45, 2.80, center - span / 2, center + span / 2, 0.45)
    elif cfg.course_kind == "rough":
        for x in np.arange(1.8, 4.8, 0.45):
            box(x, x + 0.45, 0.0, size_y, np.random.uniform(0.0, h))
    elif cfg.course_kind == "slope":
        rise = 2.0 * (0.02 + 0.23 * difficulty)
        ramp = trimesh.Trimesh(
            vertices=[(1.8, 0, 0), (3.8, 0, 0), (3.8, 0, rise),
                      (1.8, size_y, 0), (3.8, size_y, 0), (3.8, size_y, rise)],
            faces=[(0, 2, 1), (3, 4, 5), (0, 1, 4), (0, 4, 3),
                   (1, 2, 5), (1, 5, 4), (0, 3, 5), (0, 5, 2)],
        )
        ramp.fix_normals()
        meshes.append(ramp)
        box(3.8, size_x, 0.0, size_y, rise)
    elif cfg.course_kind != "manual":
        raise ValueError(f"未知赛道类型: {cfg.course_kind}")
    if cfg.course_kind == "manual":
        origin[:2] = (size_x / 2, size_y / 2)
    return meshes, origin


@configclass
class OdinCourseTerrainCfg(SubTerrainBaseCfg):
    function = odin_course_terrain
    course_kind: str = "manual"
    step_height_range: tuple[float, float] = (0.01, 0.12)


ODIN_COURSE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0), num_rows=12, num_cols=13,
    border_width=10.0, border_height=1.0, curriculum=True,
    difficulty_range=(0.0, 1.0), use_cache=False,
    sub_terrains={
        # 13列按中心点分配：平地3、上阶3、下阶3、矮障碍1、绕行1、起伏1、坡地1。
        "flat_control": OdinCourseTerrainCfg(proportion=3.0, course_kind="manual"),
        "stairs_up": OdinCourseTerrainCfg(proportion=3.0, course_kind="stairs_up"),
        "stairs_down": OdinCourseTerrainCfg(proportion=3.0, course_kind="stairs_down"),
        "low_obstacles": OdinCourseTerrainCfg(proportion=1.0, course_kind="low_obstacles"),
        "bypass": OdinCourseTerrainCfg(proportion=1.0, course_kind="bypass"),
        "rough": OdinCourseTerrainCfg(proportion=1.0, course_kind="rough", step_height_range=(0.005, 0.07)),
        "slope": OdinCourseTerrainCfg(proportion=1.0, course_kind="slope"),
    },
)
