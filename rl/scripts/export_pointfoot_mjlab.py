"""导出完整235→12点足控制器，保留IK参考与限速；不覆盖轮足ROS配置。"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from rpwd_pointfoot_supermimic.core.policy import load_actor,FRAME_DIM,TERMS

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=load_actor(c);a.output.mkdir(parents=True,exist_ok=False)
 x=torch.zeros(1,FRAME_DIM*c['history']);x[0,-1]=1
 path=a.output/'大狗点足版.onnx'
 torch.onnx.export(model,x,str(path),input_names=['obs'],output_names=['actions'],opset_version=17,dynamo=False,external_data=False)
 import onnxruntime as ort
 sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);maximum=0.
 gen=torch.Generator().manual_seed(4)
 for _ in range(16):
  x=torch.randn(x.shape,generator=gen)*.2
  with torch.inference_mode():expected=model(x).numpy()
  actual=sess.run(None,{'obs':x.numpy()})[0]
  np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5);maximum=max(maximum,float(abs(actual-expected).max()))
 contract=dict(display_name='大狗点足版',robot_id='rpwd_pointfoot',robot_signature=c['robot_signature'],step=c['step'],input=FRAME_DIM*c['history'],output=12,
  terms=list(TERMS),history=c['history'],history_order='各项分别旧→新，首帧复制',last_action='规范关节顺序，实际限速后动作',clock='sin/cos，1.5Hz，每20ms推进',
  action_scales=[.25,.6,.8]*4,default_q=[0.,.72,-1.41]*4,kp=100.,kd=3.,effort=42.,onnx_max_abs_error=maximum,hardware_activation='未进行真机部署')
 (a.output/'contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n');print(json.dumps(contract,ensure_ascii=False))
if __name__=='__main__':main()
