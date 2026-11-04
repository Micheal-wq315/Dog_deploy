"""full-v3 点足地形、命令采样及课程账本；只依赖 Python/numpy/torch/trimesh。

几何坐标遵循 Isaac Lab：网格在 [0, size] 内，返回未经中心化的出生原点。
训练元数据不进入盲 Actor。此模块可在 CPU 上独立导入，无需 SimulationApp。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
import trimesh


COURSE_KINDS = ("flat", "up", "down", "platform", "slope", "rubble")
COURSE_PROPORTIONS = (.1, .3, .2, .1, .1, .2)
STEP_HEIGHTS = (.02, .04, .08, .12, .16, .20)
MIN_ITERATIONS = (0, 1500, 4000, 8000, 12000, 16000)
COMMAND_LIMITS = ((.25, .15, .4), (.32, .20, .5), (.40, .25, .65),
                  (.48, .30, .8), (.55, .35, .9), (.60, .40, 1.0))
STAT_FIELDS = ("trials", "passed", "failed", "falls", "stuck", "neutral", "episodes")
HISTORY_WINDOW = 64
STAIR_START = 1.0
STAIR_TREAD = .30
NUM_STEPS = 5
STAIR_PASS_DISTANCE = 2.8
COURSE_PASS_DISTANCE = 5.4
EXIT_DISTANCE = 6.5
FOOT_RADIUS = .022


def difficulty_level(difficulty: float) -> int:
    """匹配 generator 的行内随机 difficulty，不使用 round 混合相邻等级。"""
    return min(len(STEP_HEIGHTS) - 1, max(0, math.floor(float(difficulty) * len(STEP_HEIGHTS))))


def column_kind_indices(sub_terrains: Mapping, num_cols: int) -> tuple[int, ...]:
    """逐字对应 TerrainGenerator 的 index/num_cols + .001 < cumsum 配方。

    必须读取配置插入顺序和归一化 proportion，不能把列编号当成地形种类。
    """
    cfgs = list(sub_terrains.values())
    weights = np.asarray([cfg.proportion for cfg in cfgs], dtype=np.float64)
    if num_cols <= 0 or not len(cfgs) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("地形列数和比例必须有效")
    if weights.sum() <= 0:
        raise ValueError("地形比例总和必须为正")
    cumulative = np.cumsum(weights / weights.sum())
    result = []
    for index in range(num_cols):
        selected = np.flatnonzero(index / num_cols + .001 < cumulative)
        if not len(selected):
            raise ValueError("地形列无法映射到子地形")
        kind = cfgs[int(selected[0])].kind
        if kind not in COURSE_KINDS:
            raise ValueError(f"未知 full-v3 地形 {kind!r}")
        result.append(COURSE_KINDS.index(kind))
    return tuple(result)


def pointfoot_full_course(difficulty: float, cfg):
    """生成 16×16 镜像实体赛道：中心出生，±X 上均有相同障碍。

    台阶使用不相交实体块；斜坡使用封闭楔体，起点连续接地。碎石静态固定，
    包含旋转底块和堆叠顶块，所有顶点严格限制在当前等级的高度以内。
    """
    sx, sy = cfg.size
    if sx < 15.0 or sy < 15.0:
        raise ValueError("full-v3 赛道需要至少 15×15 米")
    kind = cfg.kind
    if kind not in COURSE_KINDS:
        raise ValueError(kind)
    level = difficulty_level(difficulty)
    height = STEP_HEIGHTS[level]
    cx, cy = sx / 2, sy / 2
    bottom = -.20
    meshes = []

    def block(x0, x1, top, y0=0., y1=None, base=bottom):
        y1 = sy if y1 is None else y1
        mesh = trimesh.creation.box(extents=(x1 - x0, y1 - y0, top - base))
        mesh.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (top + base) / 2))
        meshes.append(mesh)

    def mirrored_interval(near, far, top):
        block(cx + near, cx + far, top)
        block(cx - far, cx - near, top)

    spawn_height = NUM_STEPS * height if kind == "down" else 0.
    if kind in ("up", "down"):
        tread = getattr(cfg, "stair_tread", STAIR_TREAD)
        block(cx - STAIR_START, cx + STAIR_START, spawn_height)
        for k in range(NUM_STEPS):
            near = STAIR_START + k * tread
            far = STAIR_START + (k + 1) * tread if k < NUM_STEPS - 1 else cx
            top = (k + 1) * height if kind == "up" else (NUM_STEPS - k - 1) * height
            mirrored_interval(near, far, top)
    elif kind == "platform":
        # 平台之间保留落地平面，最后一个平台后有足够长的稳定着陆区。
        edges = ((0., 1.15, 0.), (1.15, 2.05, height), (2.05, 3.15, 0.),
                 (3.15, 4.05, height), (4.05, cx, 0.))
        for near, far, top in edges:
            mirrored_interval(near, far, top)
    elif kind == "slope":
        length = 3.
        top = length * math.tan(math.radians(3 + 12 * level / 5))
        block(cx - STAIR_START, cx + STAIR_START, 0.)
        # x-z 四边形挤出为三棱斜坡的封闭实体；镜像后修正面朝向。
        vertices = np.array([[STAIR_START, 0, bottom], [STAIR_START + length, 0, bottom],
                             [STAIR_START + length, 0, top], [STAIR_START, 0, 0.],
                             [STAIR_START, sy, bottom], [STAIR_START + length, sy, bottom],
                             [STAIR_START + length, sy, top], [STAIR_START, sy, 0.]])
        faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                          [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                          [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
        for direction in (-1, 1):
            v = vertices.copy()
            v[:, 0] = cx + direction * v[:, 0]
            mesh = trimesh.Trimesh(v, faces.copy(), process=False)
            mesh.fix_normals()
            meshes.append(mesh)
        mirrored_interval(STAIR_START + length, cx, top)
    else:
        block(0., sx, 0.)
        if kind == "rubble":
            # generator 会在 cfg 上设置 seed；按行确定随机流，镜像使用同一组石块。
            rng = np.random.default_rng(np.random.SeedSequence([int(getattr(cfg, "seed", 42) or 0), level]))
            for ix, x in enumerate(np.arange(1.35, 5.0, .48)):
                for y in np.arange(.35 + .13 * (ix % 2), sy - .2, .52):
                    a, b = rng.uniform(.35, .65, 2)
                    rock_h = height * rng.uniform(.35, .70)
                    xx = x + rng.uniform(-.1, .1)
                    yy = float(np.clip(y + rng.uniform(-.1, .1), .6, sy - .6))
                    yaw = rng.uniform(-math.pi, math.pi)
                    layers = [(a, b, rock_h, rock_h / 2, yaw, 0., 0., 0.)]
                    if rng.random() < .25 + .35 * level / 5:
                        pitch = rng.uniform(-.15, .15) * level / 5
                        cap = height - rock_h
                        half_z = (cap * math.cos(pitch) + a * .65 * abs(math.sin(pitch))) / 2
                        layers.append((a * .65, b * .65, cap, height - half_z,
                                       yaw + .6, pitch, .04, -.03))
                    for aa, bb, hh, z, angle, pitch, dx, dy in layers:
                        source = trimesh.creation.box(extents=(aa, bb, hh))
                        transform = trimesh.transformations.euler_matrix(0., pitch, angle)
                        transform[:3, 3] = (cx + xx + dx, yy + dy, z)
                        source.apply_transform(transform)
                        meshes.append(source)
                        reflection = np.eye(4)
                        reflection[0, 0], reflection[0, 3] = -1., sx
                        mirror = source.copy()
                        mirror.apply_transform(reflection)
                        meshes.append(mirror)
    origin = np.asarray((cx, cy, spawn_height), dtype=np.float64)
    return meshes, origin


def mix_commands(values: torch.Tensor, stairs: torch.Tensor, *, limits=COMMAND_LIMITS[-1],
                 stop_probability=.1, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """非停止样本按 40/20/20/20 混合，楼梯按 80/20 直行/调整。

    values 为已按当前课程范围采样的 [N,3]。独立停止覆盖发生在混合之后，
    楼梯低速幅值 .20..min(.60, 当前上限)，不会越过初始 ±.25 限幅。
    第二个返回值是实际直行遍历掩码，用于保持整回合和中性统计。
    """
    if values.ndim != 2 or values.shape[1] != 3 or stairs.shape != values.shape[:1]:
        raise ValueError("命令形状必须为 [N,3]，楼梯掩码必须为 [N]")
    if not 0 <= stop_probability <= 1:
        raise ValueError("停止比例必须在 [0,1]")
    bound = values.new_tensor(limits)
    if bound.shape != (3,) or not torch.isfinite(bound).all() or (bound <= 0).any():
        raise ValueError("命令限幅必须是三个有限正数")
    mode = torch.rand(len(values), device=values.device, generator=generator)
    result = values.clone()
    result[mode < .4, 1:] = 0.
    lateral = (mode >= .4) & (mode < .6)
    result[lateral, 0] = 0.
    result[lateral, 2] = 0.
    result[(mode >= .6) & (mode < .8), :2] = 0.
    result[stairs] = values[stairs]
    traverse = stairs & (mode < .8)
    maximum = min(.6, float(bound[0]))
    minimum = min(.2, maximum)
    magnitude = minimum + (maximum - minimum) * torch.rand(len(values), device=values.device, generator=generator)
    result[traverse, 0] = values[traverse, 0].sign() * magnitude[traverse]
    result[traverse, 1:] = 0.
    stopped = torch.rand(len(values), device=values.device, generator=generator) < stop_probability
    result[stopped] = 0.
    return result.clamp(-bound, bound), traverse & ~stopped


def sample_mixed_commands(count: int, stairs: torch.Tensor, limits: Sequence[float], *,
                          stop_probability=.1, generator=None):
    """采样当前点足课程范围，楼梯遍历低速命令严格遵守上下界。"""
    bound = torch.as_tensor(limits, dtype=torch.float32, device=stairs.device)
    if bound.shape != (3,) or not torch.isfinite(bound).all() or (bound <= 0).any():
        raise ValueError("命令限幅必须是三个有限正数")
    values = (2 * torch.rand(count, 3, device=stairs.device, generator=generator) - 1) * bound
    return mix_commands(values, stairs, limits=limits, stop_probability=stop_probability, generator=generator)


def stable_course_pass(offset, gravity, levels, kinds, foot_bottom, contact, vertical_speed):
    """当前帧真实落地判定；调用方再要求连续稳定若干策略步。

    offset/foot_bottom 都相对出生原点；下阶 landing 是负的 5h。至少两只
    接触脚必须落在最终踏面高度上，悬空、倾斜或下落穿线都不算通过。
    """
    h = offset.new_tensor(STEP_HEIGHTS)[levels]
    up, down = kinds == COURSE_KINDS.index("up"), kinds == COURSE_KINDS.index("down")
    slope = kinds == COURSE_KINDS.index("slope")
    landing = torch.where(up, NUM_STEPS * h, torch.where(down, -NUM_STEPS * h, torch.zeros_like(h)))
    slope_top = 3 * torch.tan(torch.deg2rad(3 + 12 * levels.to(offset.dtype) / 5))
    landing = torch.where(slope, slope_top, landing)
    stairs = up | down
    distance = torch.where(stairs, STAIR_PASS_DISTANCE, COURSE_PASS_DISTANCE)
    crossed = (offset[:, 0].abs() > distance) & (offset[:, 1].abs() < EXIT_DISTANCE)
    flat = kinds == COURSE_KINDS.index("flat")
    crossed = torch.where(flat, offset[:, :2].abs().amax(-1) > COURSE_PASS_DISTANCE, crossed)
    landed = (contact & ((foot_bottom - landing[:, None]).abs() < .06)).sum(-1) >= 2
    # 碎石通过线位于碎石区外，因此落地仍以平面高度校验。
    body = offset[:, 2] - landing
    upright = (-gravity[:, 2] > math.cos(math.radians(25))) & (gravity[:, :2].norm(dim=-1) < .43)
    return crossed & landed & upright & (body > .20) & (body < .60) & (vertical_speed.abs() < .35)


class FullCourseHistory:
    """按列保存64回合窗口，按六种地形保存分项统计，CPU/GPU 共用实现。

    caps/trials/counts 与主训练既有三个 env 属性直接共享张量。其余持久状态
    为 stats、command_stage、iteration、velocity_mae；均包含于 state_dict。
    """

    def __init__(self, column_kinds, rows=6, device="cpu", *, caps=None, trials=None, counts=None):
        self.column_kinds = torch.as_tensor(column_kinds, dtype=torch.long, device=device)
        cols = len(self.column_kinds)
        if rows != len(STEP_HEIGHTS) or cols < 1:
            raise ValueError("full-v3 必须使用六等级和非空地形列")
        self.rows = rows
        self.caps = torch.zeros(cols, dtype=torch.long, device=device) if caps is None else caps
        self.trials = torch.zeros(cols, rows, HISTORY_WINDOW, device=device) if trials is None else trials
        self.counts = torch.zeros(cols, rows, dtype=torch.long, device=device) if counts is None else counts
        if self.caps.shape != (cols,) or self.trials.shape != (cols, rows, HISTORY_WINDOW) or self.counts.shape != (cols, rows):
            raise ValueError("已有课程状态与 full-v3 的列/行/64回合窗口不兼容")
        self.stats = torch.zeros(len(COURSE_KINDS), rows, len(STAT_FIELDS), dtype=torch.long, device=device)
        self.command_stage = 0
        self.iteration = 0
        self.velocity_mae = torch.full((3,), float("inf"), device=device)

    @property
    def limits(self):
        return COMMAND_LIMITS[self.command_stage]

    @torch.no_grad()
    def record(self, columns, levels, success, eligible, fallen, stuck):
        """每次只提交已结束回合；中性回合不进入成功率分母，也不使地形降级。"""
        passed = success & eligible
        failed = eligible & ~success
        values = torch.stack((eligible, passed, failed, fallen, stuck, ~eligible,
                              torch.ones_like(eligible)), -1).long()
        groups = self.column_kinds[columns] * self.rows + levels
        self.stats.view(-1, len(STAT_FIELDS)).index_add_(0, groups, values)
        trial_groups = columns[eligible] * self.rows + levels[eligible]
        outcomes = success[eligible].to(self.trials.dtype)
        # 同组大量并行结束时保留最后64个，避免重复索引写入的不确定性。
        for group in trial_groups.unique().tolist():
            col, row = divmod(group, self.rows)
            results = outcomes[trial_groups == group]
            old = int(self.counts[col, row])
            kept = min(len(results), HISTORY_WINDOW)
            slots = (torch.arange(len(results) - kept, len(results), device=results.device) + old) % HISTORY_WINDOW
            self.trials[col, row, slots] = results[-kept:]
            self.counts[col, row] += len(results)
            if (row == int(self.caps[col]) and row < self.rows - 1
                    and int(self.counts[col, row]) >= HISTORY_WINDOW
                    and self.iteration >= MIN_ITERATIONS[row + 1]
                    and self.trials[col, row].mean() >= .60):
                self.caps[col] += 1

    def recent(self, kind: int, level: int):
        selected = self.column_kinds == kind
        counts = self.counts[selected, level].clamp_max(HISTORY_WINDOW)
        count = int(counts.sum())
        passed = self.trials[selected, level].sum()
        return count, float(passed / max(count, 1))

    def update_commands(self, iteration: int, velocity_mae):
        """训练轮数、上下阶真实通过率和实际速度误差三者同时达标才扩速。"""
        self.iteration = max(0, int(iteration))
        error = torch.as_tensor(velocity_mae, device=self.velocity_mae.device, dtype=self.velocity_mae.dtype)
        if error.shape != (3,):
            raise ValueError("velocity_mae 必须为 [vx, vy, wz] 三项绝对误差")
        self.velocity_mae.copy_(error)
        stage = self.command_stage
        if stage == len(COMMAND_LIMITS) - 1 or self.iteration < MIN_ITERATIONS[stage + 1]:
            return stage
        accurate = torch.isfinite(error).all() & (error >= 0).all() & (error < error.new_tensor((.18, .18, .30))).all()
        quality = [self.recent(COURSE_KINDS.index(kind), stage) for kind in ("up", "down")]
        if bool(accurate) and all(count >= HISTORY_WINDOW and rate >= .60 for count, rate in quality):
            self.command_stage += 1
        return self.command_stage

    def state_dict(self):
        return {"version": 3, "column_kinds": self.column_kinds.detach().cpu().clone(),
                "caps": self.caps.detach().cpu().clone(), "trials": self.trials.detach().cpu().clone(),
                "counts": self.counts.detach().cpu().clone(), "stats": self.stats.detach().cpu().clone(),
                "command_stage": self.command_stage, "iteration": self.iteration,
                "velocity_mae": self.velocity_mae.detach().cpu().clone()}

    def load_state_dict(self, state):
        if state.get("version") != 3 or not torch.equal(torch.as_tensor(state["column_kinds"]).cpu(), self.column_kinds.cpu()):
            raise ValueError("full-v3 课程版本或地形列映射不匹配")
        for key in ("caps", "trials", "counts", "stats", "velocity_mae"):
            target = getattr(self, key)
            value = torch.as_tensor(state[key], device=target.device, dtype=target.dtype)
            if target.shape != value.shape:
                raise ValueError(f"课程状态 {key} 形状不匹配")
        stage, iteration = int(state["command_stage"]), int(state["iteration"])
        if not 0 <= stage < len(COMMAND_LIMITS) or iteration < 0:
            raise ValueError("课程阶段或时钟无效")
        if ((torch.as_tensor(state["caps"]) < 0) | (torch.as_tensor(state["caps"]) >= self.rows)).any():
            raise ValueError("地形难度上限无效")
        for key in ("caps", "trials", "counts", "stats", "velocity_mae"):
            getattr(self, key).copy_(torch.as_tensor(state[key]))
        self.command_stage, self.iteration = stage, iteration

    def summary(self):
        result = {"revision": 3, "iteration": self.iteration, "command_stage": self.command_stage,
                  "command_limits": list(self.limits),
                  "velocity_mae": [float(v) if math.isfinite(float(v)) else None for v in self.velocity_mae],
                  "column_kinds": [COURSE_KINDS[k] for k in self.column_kinds.tolist()],
                  "caps": self.caps.tolist(), "stat_fields": list(STAT_FIELDS), "terrain": {}}
        for kind, name in enumerate(COURSE_KINDS):
            result["terrain"][name] = {"by_level": self.stats[kind].tolist(),
                                      "recent_trials": [self.recent(kind, row)[0] for row in range(self.rows)],
                                      "recent_success": [self.recent(kind, row)[1] for row in range(self.rows)]}
        return result
