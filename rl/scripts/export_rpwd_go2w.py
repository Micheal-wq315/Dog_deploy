"""导出245维盲策略；RP_wd生成独立ROS2配置，保留原部署配置。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
import yaml
from rpwd_go2w_supermimic.core.policy import load_actor
from rpwd_go2w_supermimic.core.command_curriculum import STAGES
from rpwd_go2w_supermimic.robot.config import REPO_ROOT, RP_LEGS, RP_KINDS, hardware_contract


class DeploymentActor(nn.Module):
    """部署last_action是规范顺序；策略历史中的该项使用原环境动作顺序。"""
    def __init__(self, actor, order):
        super().__init__()
        self.actor = actor
        self.register_buffer('order', torch.tensor(order, dtype=torch.long))
        self.h = actor.history

    def forward(self, obs):
        start, end = 30 * self.h, 46 * self.h
        actions = obs[:, start:end].reshape(-1, self.h, 16).index_select(-1, self.order)
        return self.actor(torch.cat((obs[:, :start], actions.flatten(1), obs[:, end:]), -1))


def deployment_config(checkpoint):
    c = hardware_contract()
    config = yaml.safe_load((REPO_ROOT/'src/inference/robots/wheel_quad/configs/default.yaml').read_text())
    p = config['inference_node']['ros__parameters']
    physical_names = [f'{leg}_{kind}_JOINT' for leg in RP_LEGS for kind in RP_KINDS]
    physical_names += [f'{leg}_FOOT_JOINT' for leg in RP_LEGS]
    stage = checkpoint.get('curriculum_state', {}).get('stage', 0)
    _, vx, vy, wz = STAGES[stage]
    p.update(model_names=['policy_supermimic_rpwd.onnx'],
             obs_layouts=['ang_vel:3, gravity_b:3, dof_pos:12, dof_vel:12, last_action:16, cmd_vel:3'],
             frame_stacks=[checkpoint['history']], obs_stack_orders=['obs_major'],
             obs_scales_dof_vel=.05, clip_actions=1., clip_observations=1e9,
             action_scales=[.25,.6,.8]*4+[35.]*4,
             wheel_joint_indices=[12,13,14,15],
             usd2urdf=[c.URDF_JOINT_ORDER.index(n) for n in physical_names],
             clip_cmd=[*vx,*vy,*wz], act_alpha=1.)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    actor=load_actor(c,'cpu')
    robot=c.get('robot_id')
    if robot not in ('rpwd','go2w'):raise ValueError('只导出带机器人标识的新适配版检查点')
    model=DeploymentActor(actor,c['env_action_order']).eval()
    out=a.output or Path(os.environ['RPWD_GO2W_DATA'])/'exported'/f"{robot}_{c['step']}_{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True,exist_ok=False)
    name='policy_supermimic_rpwd.onnx' if robot=='rpwd' else 'policy_supermimic_go2w.onnx'
    path=out/name
    sample=torch.zeros(1,49*c['history'])
    torch.onnx.export(model,sample,str(path),input_names=['obs'],output_names=['actions'],
                      opset_version=17,dynamo=False,external_data=False)
    import onnxruntime as ort
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    generator=torch.Generator().manual_seed(7)
    error=0.
    for _ in range(16):
        x=torch.randn(sample.shape,generator=generator)
        with torch.inference_mode():expected=model(x).numpy()
        actual=session.run(None,{'obs':x.numpy()})[0]
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
        error=max(error,float(np.abs(actual-expected).max()))
    if robot=='rpwd':
        (out/'supermimic_rpwd.yaml').write_text(yaml.safe_dump(deployment_config(c),allow_unicode=True,sort_keys=False))
    (out/'contract.json').write_text(json.dumps(dict(robot_id=robot,robot_signature=c['robot_signature'],
        source_checkpoint=str(a.checkpoint.resolve()),step=c['step'],input=49*c['history'],output=16,
        history='term-major oldest-to-newest',last_action='canonical pre-scale; ONNX内部重排至训练环境顺序',
        onnx_max_abs_error=error,policy_dt=c['dt'],hardware_activation='尚未执行真机部署'),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out),'onnx_max_abs_error':error},ensure_ascii=False))

if __name__=='__main__':main()
