"""共享原生MuJoCo查看器与测试场，严格按机器人类型选择检查点。"""
import argparse
import os
from pathlib import Path
import sys
import torch
from go2w_mujoco import main

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--robot', choices=('rpwd', 'go2w'), default='rpwd')
a, remaining = p.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
base = Path(os.environ['RPWD_GO2W_DATA'])

def find():
    candidates = []
    for run in base.iterdir():
        if run.is_dir() and run.name not in ('validation', 'cache', 'native_tests'):
            candidates.extend(run.glob('model_*.pt'))
    for path in sorted(candidates, key=lambda x:(int(x.stem.split('_')[-1]), x.stat().st_mtime_ns), reverse=True):
        c = torch.load(path, map_location='cpu', weights_only=True)
        if c.get('robot_id') == a.robot:
            return path
    raise FileNotFoundError('尚无该机器人正式训练模型，请先训练或显式指定--checkpoint')

if __name__ == '__main__':
    main(checkpoint_finder=find, output_base=base, expected_robot=a.robot, policy_backend="rpwd_go2w_supermimic")
