"""根据独立MuJoCo测试给出基础上阶验收结果，不以训练回报代替。"""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.summary.read_text())
if r.get('robot_id')!='rpwd_pointfoot':raise ValueError('验收结果不是大狗点足版')
checks={};groups={}
for lane in ('flat','stairs4','stairs8','stairs12','ramp','rubble','mixed'):
 ts=[x for x in r['trials'] if x['lane']==lane]
 groups[lane]=dict(passed=sum(bool(x['success']) for x in ts),trials=len(ts),reasons=[x['reason'] for x in ts])
 if lane in ('flat','stairs4','stairs8'):
  checks[lane]=len(ts)>=3 and all(x['success'] and x['velocity_mae'][0]<=.08 and x['max_tilt']<=35 for x in ts)
out=dict(robot_id='rpwd_pointfoot',model_step=r['model_step'],basic_stair_acceptance=all(checks.values()),checks=checks,groups=groups,
 note='基础验收要求平地及4/8cm各至少3种子全部完成、前向误差≤0.08m/s、倾角≤35度；12cm及碎石另报。有限测试不代表真机鲁棒性。')
a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
