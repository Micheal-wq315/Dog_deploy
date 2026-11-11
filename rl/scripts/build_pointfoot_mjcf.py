"""从点足URDF确定性生成MuJoCo资产；保留逐连杆惯量、固定脚端与原始关节轴。"""
from pathlib import Path
import math
import os
import xml.etree.ElementTree as E
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf'
OUT = ROOT / 'description/RP_wd_pointfoot/mjcf'
LEGS = dict(zip(('LF', 'RF', 'LB', 'RB'), ('FL', 'FR', 'RL', 'RR')))

def quaternion(rpy):
    r,p,y = [float(x)/2 for x in rpy.split()]
    cr,cp,cy,sr,sp,sy = math.cos(r),math.cos(p),math.cos(y),math.sin(r),math.sin(p),math.sin(y)
    return ' '.join(map(str,(cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy)))

def origin(node):
    n=node.find('origin')
    return dict(pos=n.get('xyz','0 0 0'),quat=quaternion(n.get('rpy','0 0 0'))) if n is not None else {}

def convert_dae(path, target):
    ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
    root=E.parse(path).getroot();vertices=[];faces=[]
    matrices=root.findall('.//c:visual_scene/c:node/c:matrix',ns)
    if len(matrices)!=1:raise ValueError('必须显式处理多个DAE节点变换')
    transform=np.fromstring(matrices[0].text,sep=' ').reshape(4,4)
    for mesh in root.findall('.//c:mesh',ns):
        sources={s.get('id'):np.fromstring(s.find('c:float_array',ns).text,sep=' ') for s in mesh.findall('c:source',ns)}
        refs={v.get('id'):v.find("c:input[@semantic='POSITION']",ns).get('source')[1:] for v in mesh.findall('c:vertices',ns)}
        for tri in mesh.findall('c:triangles',ns):
            inputs=tri.findall('c:input',ns);stride=max(int(i.get('offset','0')) for i in inputs)+1
            position=next(i for i in inputs if i.get('semantic')=='VERTEX')
            xyz=sources[refs[position.get('source')[1:]]].reshape(-1,3)
            xyz=(np.c_[xyz,np.ones(len(xyz))]@transform.T)[:,:3]
            index=np.fromstring(tri.find('c:p',ns).text,sep=' ',dtype=int).reshape(-1,stride)[:,int(position.get('offset','0'))].reshape(-1,3)
            faces.extend(index+len(vertices)+1);vertices.extend(xyz)
    with target.open('w') as f:
        f.write('# 源Go2 DAE顶点及场景变换；许可见上级LICENSE_GO2_ASSET\n')
        for v in vertices:f.write('v '+' '.join(f'{x:.9g}' for x in v)+'\n')
        for face in faces:f.write('f '+' '.join(map(str,face))+'\n')

def build():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True)
    source=E.parse(URDF).getroot()
    for p in (URDF.parent/'meshes').glob('*.dae'):convert_dae(p,OUT/'assets'/p.with_suffix('.obj').name)
    root=E.Element('mujoco',model='RP_wd_pointfoot_supermimic')
    E.SubElement(root,'compiler',angle='radian',autolimits='true',fusestatic='false',inertiafromgeom='false')
    asset=E.SubElement(root,'asset');world=E.SubElement(root,'worldbody')
    links={n.get('name'):n for n in source.findall('link')};children={}
    for j in source.findall('joint'):children.setdefault(j.find('parent').get('link'),[]).append(j)
    cache={}
    def body(name,parent,attributes):
        b=E.SubElement(parent,'body',name=name,**attributes)
        link=links[name];inertia=link.find('inertial')
        i=inertia.find('inertia');data=origin(inertia);data.pop('quat',None)
        if inertia.find('origin').get('rpy','0 0 0')!='0 0 0':raise ValueError('须旋转惯性张量')
        E.SubElement(b,'inertial',mass=inertia.find('mass').get('value'),fullinertia=' '.join(i.get(k) for k in ('ixx','iyy','izz','ixy','ixz','iyz')),**data)
        for tag in ('visual','collision'):
            for k,g in enumerate(link.findall(tag)):
                geom=g.find('geometry');attrs=origin(g)
                attrs.update(group='2' if tag=='visual' else '3',contype='0' if tag=='visual' else '2',conaffinity='0' if tag=='visual' else '1')
                foot=name.endswith('FOOT_LINK')
                if tag=='collision':attrs.update(name=(LEGS[name[:2]]+'_foot_collision') if foot else name+f'_collision_{k}',friction='.8 .02 .001',condim='3' if foot else '1',priority='1' if foot else '0')
                n=geom[0]
                if n.tag=='mesh':
                    path=(URDF.parent/n.get('filename')).resolve()
                    if path.suffix=='.dae':path=OUT/'assets'/path.with_suffix('.obj').name
                    key=str(path)
                    if key not in cache:
                        cache[key]='mesh_'+str(len(cache));E.SubElement(asset,'mesh',name=cache[key],file=os.path.relpath(path,OUT),scale=n.get('scale','1 1 1'))
                    attrs.update(type='mesh',mesh=cache[key],rgba='.18 .18 .2 1' if foot else '.8 .8 .82 1')
                elif n.tag=='sphere':attrs.update(type='sphere',size=n.get('radius'))
                elif n.tag=='cylinder':attrs.update(type='cylinder',size=f"{n.get('radius')} {float(n.get('length'))/2}")
                elif n.tag=='box':attrs.update(type='box',size=' '.join(str(float(x)/2) for x in n.get('size').split()))
                else:raise ValueError(n.tag)
                E.SubElement(b,'geom',**attrs)
        if name=='base_link':E.SubElement(b,'freejoint',name='floating_base')
        for j in children.get(name,[]):
            child=body(j.find('child').get('link'),b,origin(j))
            if j.get('type')!='fixed':
                leg,kind,_=j.get('name').split('_');kind={'ABAD':'hip','HIP':'thigh','KENN':'calf'}[kind]
                lim=j.find('limit')
                E.SubElement(child,'joint',name=f'{LEGS[leg]}_{kind}_joint',type='hinge',axis=j.find('axis').get('xyz'),range=lim.get('lower')+' '+lim.get('upper'),limited='true')
        return b
    body('base_link',world,{'pos':'0 0 .36'})
    E.indent(root)
    path=OUT/'RP_wd_pointfoot.xml';E.ElementTree(root).write(path,encoding='unicode')
    print(path)

if __name__=='__main__':build()
