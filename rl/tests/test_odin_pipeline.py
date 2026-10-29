"""用替代训练/评测入口验证长期流程的分段续训与失败停止，不启动GPU。"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, fail_flat=False):
        root = Path(self.folder.name)
        scripts = root / "scripts"
        scripts.mkdir()
        source = Path(__file__).resolve().parents[1] / "scripts"
        for name in ("train_odin_pipeline.sh", "qualify_odin_teacher.py"):
            shutil.copy(source / name, scripts / name)
        (scripts / "_common.sh").write_text('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
        train = '''#!/usr/bin/env python3
import json,sys
from pathlib import Path
args=sys.argv[1:]
def arg(name): return args[args.index(name)+1]
root=Path(__file__).resolve().parents[1]
with (root/'calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
kind='teacher' if 'Teacher' in arg('--task') else 'student'
run=root/'logs'/'rsl_rl'/('rp_wd_odin_'+kind)/('stamp_'+arg('--run_name'))
run.mkdir(parents=True)
target=int(arg('--max_iterations'))
checkpoint=run/f'model_{target-1}.pt'
checkpoint.write_text('仅用于流程验证')
(run/'training_result.json').write_text(json.dumps({'completed_iterations':target,'checkpoint':str(checkpoint)}))
'''
        gui = '''#!/usr/bin/env python3
import json,sys,os
from pathlib import Path
args=sys.argv[1:]; out=Path(args[args.index('--save-dir')+1]);out.mkdir(parents=True)
(out/'config.json').write_text(json.dumps({'completed':True}))
names=('flat_reference','step_up_06cm','step_down_08cm','low_box_12cm')
report={name:dict(first_attempts=2,first_completed_attempts=2,first_low_contact_completed_attempts=2) for name in names}
# 12cm不在本轮基础训练分布，失败应只作诊断。
report['low_box_12cm'].update(first_completed_attempts=0,first_low_contact_completed_attempts=0)
if os.environ.get('FAIL_FLAT')=='1': report['flat_reference'].update(first_completed_attempts=0)
(out/'terrain_summary.json').write_text(json.dumps(report))
(out/'summary.json').write_text(json.dumps([{'case':'flat_reference','vx':.30}]))
'''
        for name, body in (("train_odin_depth.sh", train), ("gui_test.sh", gui)):
            path = scripts / name
            path.write_text(body)
            path.chmod(0o755)
        result = subprocess.run(["bash", str(scripts / "train_odin_pipeline.sh"), "--num_envs", "256"],
                                env={**os.environ, "RPWD_ODIN_PIPELINE_INHIBITED": "1",
                                     "FAIL_FLAT": str(int(fail_flat))},
                                capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
        return result, calls

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_long_pipeline_resumes_and_distills_only_after_basic_pass(self):
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        targets = [int(args[args.index("--max_iterations") + 1]) for args in calls]
        self.assertEqual(targets, [2000, 6000, 12000, 18000, 2000])
        self.assertNotIn("--resume", calls[0])
        for args, previous in zip(calls[1:4], targets):
            self.assertIn("--resume", args)
            self.assertEqual(args[args.index("--checkpoint") + 1], f"model_{previous-1}.pt")
        teacher = calls[-1][calls[-1].index("--teacher_checkpoint") + 1]
        self.assertTrue(teacher.endswith("model_17999.pt"))
        self.assertEqual(targets[-2] + targets[-1], 20000)

    def test_failed_flat_gate_prevents_further_training(self):
        result, calls = self.run_pipeline(fail_flat=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
