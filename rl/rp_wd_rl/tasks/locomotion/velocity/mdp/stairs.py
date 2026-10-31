"""奖励专用真值射线与轮端事件；不向部署Actor增加地形输入。"""

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.warp import raycast_mesh
from rp_wd_rl.assets.rp_wd_contract import LEG_NAMES, WHEEL_RADIUS
from rp_wd_rl.stair_logic import StairEvents


def stair_state(env):
    if not hasattr(env, "_odin_stair_events"):
        env._odin_stair_events = StairEvents(env.num_envs, env.device)
    state = env._odin_stair_events
    if state.step == env.common_step_counter:
        return state
    robot = env.scene["robot"]
    if not hasattr(state, "wheel_ids"):
        state.wheel_ids, _ = robot.find_bodies([f"{leg}_FOOT_LINK" for leg in LEG_NAMES], preserve_order=True)
        state.contact_ids, _ = env.scene["contact_forces"].find_bodies(
            [f"{leg}_FOOT_LINK" for leg in LEG_NAMES], preserve_order=True)
        if len(state.wheel_ids) != 4 or len(state.contact_ids) != 4:
            raise RuntimeError("台阶奖励必须解析到4个独立轮端。")
    positions = robot.data.body_pos_w[:, state.wheel_ids]
    offsets = torch.tensor([0., .06, .12, .20, .30, .42], device=env.device)
    starts = positions[:, :, None, :].repeat(1, 1, len(offsets), 1)
    starts[..., 0] += offsets
    starts[..., 2] += 2.
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    scanner = env.scene["height_scanner"]
    mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    hit = raycast_mesh(starts.flatten(1, 2), directions.flatten(1, 2), mesh, max_dist=5.)[0]
    heights = hit[..., 2].reshape(env.num_envs, 4, -1)
    ground = heights[..., 0]
    # 最近的更高踏面；不把第二级高度作为当前轮端的抬升目标。
    rise = heights[..., 1:] - ground[..., None]
    higher = (rise > .008) & (rise <= .16) & torch.isfinite(rise)
    index = higher.float().argmax(dim=-1)
    ahead = torch.gather(heights[..., 1:], -1, index[..., None]).squeeze(-1)
    ahead = torch.where(higher.any(dim=-1), ahead, ground)
    forces = env.scene["contact_forces"].data.net_forces_w_history
    normal = forces[:, :, state.contact_ids, 2].clamp_min(0).amax(dim=1)
    command = env.command_manager.get_term("base_velocity")
    from .odin_course import odin_kind_mask, odin_failure
    body_force = forces[:, :, command.nonfoot_ids].norm(dim=-1).amax(dim=(1, 2))
    outputs = state.update(
        ground=ground, ahead=ahead, wheel_x=positions[..., 0], bottom=positions[..., 2]-WHEEL_RADIUS,
        vertical_speed=robot.data.body_lin_vel_w[:, state.wheel_ids, 2], contact=normal > 8.,
        forward_speed=robot.data.root_lin_vel_w[:, 0], active=odin_kind_mask(env, "stairs_up"),
        safe=(-robot.data.projected_gravity_b[:, 2] > .9063) & (body_force < 20.),
        alive=~odin_failure(env), impact_ok=(normal < 350.) & (body_force[:, None] < 20.),
    )
    state.step = env.common_step_counter
    for name, value in outputs.items():
        command.metrics[f"stair_{name}"] = value
    return state


def reset_stair_events(env, env_ids):
    if hasattr(env, "_odin_stair_events"):
        env._odin_stair_events.reset(env_ids)


def stair_clearance(env):
    return stair_state(env).outputs["clearance"]


def stair_crossing(env):
    return stair_state(env).outputs["crossing"]


def stair_landing(env):
    return stair_state(env).outputs["landing"]


def stair_stage_complete(env):
    return stair_state(env).outputs["stage"]
