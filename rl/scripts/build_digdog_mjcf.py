"""把 StackForce DigDogv1 的 Y-up/CAD 坐标 MJCF 转为框架使用的 Z-up 机器人。

转换内容：
1. 在 worldbody 外套一个带 freejoint 的 base_link 根节点，并以四元数
   (0.5,0.5,0.5,0.5) 旋转旧模型：旧 X=左/右、Y=上、Z=前 -> 新 X=前、Y=左、Z=上。
2. 旧 base_link 改名 body_link，作为根节点的刚性孩子。
3. 腿/关节/脚端统一改为 FL/FR/RL/RR + hip/thigh/calf 命名。
4. 根节点默认高度 0.195m；配合 rpwd_digdog_supermimic 的默认站姿，四脚在平地附近。
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BODY_MAP = {
    'FR_HAA': 'FR_hip', 'FR_HFE': 'FR_thigh', 'FR_KFE': 'FR_foot_link',
    'FL_HAA': 'FL_hip', 'FL_HFE': 'FL_thigh', 'FL_KFE': 'FL_foot_link',
    'BR_HAA': 'RR_hip', 'BR_HFE': 'RR_thigh', 'BR_KFE': 'RR_foot_link',
    'BL_HAA': 'RL_hip', 'BL_FHE': 'RL_thigh', 'BL_KFE': 'RL_foot_link',
}
JOINT_MAP = {
    'FR_Roll_joint_joint': 'FR_hip_joint', 'FR_HAA_joint': 'FR_thigh_joint', 'FR_KFE_joint': 'FR_calf_joint',
    'FL_HAA_joint': 'FL_hip_joint', 'FL_HFE_joint': 'FL_thigh_joint', 'FL_KFE_joint': 'FL_calf_joint',
    'BR_HAA_joint': 'RR_hip_joint', 'BR_HFE_joint': 'RR_thigh_joint', 'BR_KFE_joint': 'RR_calf_joint',
    'BL_HAA_joint': 'RL_hip_joint', 'BL_FHE_joint': 'RL_thigh_joint', 'BL_KFE_joint': 'RL_calf_joint',
}
PREFIX_MAP = {old: new.replace('_link', '') for old, new in BODY_MAP.items()}
WRAPPER_QUAT = '0.5 0.5 0.5 0.5'
DEFAULT_HEIGHT = 0.195


def main():
    if len(sys.argv) != 3:
        raise SystemExit('用法: build_digdog_mjcf.py <源robot.xml> <输出robot.xml>')
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    tree = ET.parse(src)
    root = tree.getroot()

    for elem in root.iter():
        name = elem.get('name')
        if name in BODY_MAP:
            elem.set('name', BODY_MAP[name])
        elif name in JOINT_MAP:
            elem.set('name', JOINT_MAP[name])
        elif name:
            for old, new in PREFIX_MAP.items():
                if name.startswith(old + '_'):
                    elem.set('name', new + name[len(old):])
                    break
        joint = elem.get('joint')
        if joint in JOINT_MAP:
            elem.set('joint', JOINT_MAP[joint])
        body = elem.get('body')
        if body == 'base_link':
            elem.set('body', 'body_link')
        if elem.tag == 'joint' and elem.get('type') == 'hinge':
            elem.set('armature', '0.01')

    wb = root.find('worldbody')
    old_base = wb.find("body[@name='base_link']")
    if old_base is None:
        raise SystemExit('未找到旧 base_link')
    for free in old_base.findall('freejoint'):
        old_base.remove(free)
    old_base.set('name', 'body_link')
    old_base.set('pos', '0 0 0')
    old_base.set('quat', WRAPPER_QUAT)
    wb.remove(old_base)

    new_root = ET.Element('body', {'name': 'base_link', 'pos': f'0 0 {DEFAULT_HEIGHT}'})
    new_root.append(ET.Element('freejoint', {'name': 'root_freejoint'}))
    new_root.append(old_base)
    wb.append(new_root)

    ET.indent(tree, space='  ')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding='utf-8', xml_declaration=True)
    print(f'已写出 {dst}')


if __name__ == '__main__':
    main()
