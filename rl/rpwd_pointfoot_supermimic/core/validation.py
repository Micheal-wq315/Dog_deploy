"""在已落盘的检查点上运行独立CPU无噪声测试，防止只看探索回报。"""
import os,subprocess,sys
from pathlib import Path

def launch_cpu_check(checkpoint,output):
 script=Path(__file__).resolve().parents[2]/'scripts/play_pointfoot_mjlab.py'
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 with output.with_suffix('.log').open('w') as log:
  return subprocess.Popen([sys.executable,str(script),'--mode','evaluate','--checkpoint',str(checkpoint),
   '--lanes','flat','stairs4','stairs8','--seeds','0','--speed','.2','--seconds','50','--output',str(output)],
   env=env,stdout=log,stderr=subprocess.STDOUT)
