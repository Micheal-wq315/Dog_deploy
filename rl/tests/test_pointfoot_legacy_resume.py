"""旧点足任务恢复保存等级时，空回合reset不应降级或记录失败。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch


def test_empty_episode_reset_preserves_restored_level():
    path = Path(__file__).parents[1] / 'rp_wd_rl/tasks/locomotion/velocity/mdp/pointfoot_stairs.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'pointfoot_terrain_curriculum')
    scope = {'torch': torch, 'ManagerBasedRLEnv': object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
    terrain = SimpleNamespace(terrain_levels=torch.tensor([3, 4]))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), device='cpu', episode_length_buf=torch.zeros(2),
                          _pointfoot_course_caps=torch.tensor([5, 5, 5, 5]))
    scope['pointfoot_terrain_curriculum'](env, [0, 1])
    assert terrain.terrain_levels.tolist() == [3, 4]
