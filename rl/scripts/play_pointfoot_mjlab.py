"""点足官方MuJoCo查看器；方向键平移、Q/E转向，松键归零。"""
from pathlib import Path
import os
from pointfoot_mujoco import main
base=Path(os.environ['POINTFOOT_MJLAB_DATA'])
def latest():
 candidates=[p for d in base.iterdir() if d.is_dir() and d.name not in ('validation','cache','native_tests') for p in d.glob('model_*.pt')]
 if not candidates:raise FileNotFoundError('还没有正式点足模型，请显式指定--checkpoint')
 return max(candidates,key=lambda p:(int(p.stem.split('_')[-1]),p.stat().st_mtime_ns))
if __name__=='__main__':main(checkpoint_finder=latest,output_base=base,expected_robot='rpwd_pointfoot',policy_backend='rpwd_pointfoot_supermimic')
