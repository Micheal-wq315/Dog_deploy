"""Shared static height atlas and privileged, differentiable yaw-frame scans."""
import numpy as np
import torch
from .bridge import Go2WBridge

SCAN_DIM = 21 * 13

class TerrainField:
  def __init__(self, heights, origin, resolution=.04):
    self.h, self.origin, self.res = heights, origin, resolution
    x,y = torch.meshgrid(torch.linspace(-1,1,21,device=heights.device),
                         torch.linspace(-.6,.6,13,device=heights.device), indexing="ij")
    self.offsets = torch.stack((x.flatten(),y.flatten()),-1)

  def sample(self, xy):
    # Clamp *integer* neighbour indices too. NaN survives float clamp and
    # casts to INT64_MIN; on CUDA that would poison the entire device context.
    finite = torch.isfinite(xy).all(-1)
    clean = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
    gx = ((clean[...,0]-self.origin[0])/self.res).clamp(0,self.h.shape[1]-1)
    gy = ((clean[...,1]-self.origin[1])/self.res).clamp(0,self.h.shape[0]-1)
    x = gx.floor().long().clamp(0,self.h.shape[1]-2)
    y = gy.floor().long().clamp(0,self.h.shape[0]-2)
    a,b = gx-x,gy-y
    value = ((1-a)*self.h[y,x]+a*self.h[y,x+1])*(1-b)+((1-a)*self.h[y+1,x]+a*self.h[y+1,x+1])*b
    # Do not disguise corrupt states as valid ground: the learner will discard
    # this update, while real invalid environments are reset before querying.
    return torch.where(finite, value, torch.full_like(value, float("nan")))

  def scan(self, state, pose):
    c,s = pose[:,2:3].cos(),pose[:,2:3].sin()
    dx,dy = self.offsets[:,0],self.offsets[:,1]
    xy = pose[:,None,:2]+torch.stack((c*dx-s*dy,s*dx+c*dy),-1)
    return (state[:,:1]+self.sample(pose[:,:2])[:,None]-self.sample(xy)).clamp(-1.5,1.5)

  @classmethod
  def from_env(cls, env, resolution=.04):
    """Build the static height atlas from an mjlab environment."""
    return cls.from_model(env.sim.mj_model,
                          set(env.scene["robot"].indexing.geom_ids.cpu().tolist()),
                          resolution, device=env.device)

  @classmethod
  def from_model(cls, model, robot_ids, resolution=.04, bounds=None, device='cpu'):
    """Rasterize exact oriented boxes in small bounding rectangles, no per-pixel rays.

    Shared CPU build is ~6 MB at 4 cm for a 48 x 48 m atlas. No robot geometry.
    Only static box courses are supported; planes are treated as z=0. Pass
    ``bounds=((x0,y0),(x1,y1))`` for native test parks whose flat ground is a
    plane outside the box bounding rectangle.
    """
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    corners = np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
    boxes, cylinders = [], []
    for i in range(model.ngeom):
      if i in robot_ids or not model.geom_contype[i]:
        continue
      if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
        continue
      center, rot, size = data.geom_xpos[i],data.geom_xmat[i].reshape(3,3),model.geom_size[i]
      if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # MuJoCo cylinder: local z axis, size=(radius, half_length, 0).
        axis = rot[:, 2].copy()
        radius, half = float(size[0]), float(size[1])
        extent = np.abs(axis) * half + radius
        cylinders.append((center.copy(), axis, radius, half, center - extent, center + extent))
        continue
      if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"Unsupported terrain geometry: {model.geom(i).name}")
      points = (corners*size) @ rot.T+center
      boxes.append((center.copy(),rot.copy(),size.copy(),points.min(0),points.max(0)))
    if bounds is None:
      lows = [b[3] for b in boxes] + [c[4] for c in cylinders]
      highs = [b[4] for b in boxes] + [c[5] for c in cylinders]
      if not lows:
        raise ValueError("Terrain field needs box/cylinder geometry or explicit bounds")
      lo=np.min(lows,axis=0)[:2]-resolution
      hi=np.max(highs,axis=0)[:2]+resolution
    else:
      lo=np.asarray(bounds[0],dtype=np.float64)
      hi=np.asarray(bounds[1],dtype=np.float64)
    shape=np.ceil((hi-lo)/resolution).astype(int)+1
    heights=np.zeros((shape[1],shape[0]),dtype=np.float32)
    for center,rot,size,low,high in boxes:
      start=np.maximum(0,np.floor((low[:2]-lo)/resolution).astype(int))
      end=np.minimum(shape,np.ceil((high[:2]-lo)/resolution).astype(int)+1)
      if (end<=start).any():
        continue
      xs=lo[0]+np.arange(start[0],end[0])*resolution
      ys=lo[1]+np.arange(start[1],end[1])*resolution
      x,y=np.meshgrid(xs,ys)
      # Intersect vertical line p=(x,y,z) with all three oriented-box slabs.
      lower=np.full_like(x,-np.inf); upper=np.full_like(x,np.inf); valid=np.ones_like(x,dtype=bool)
      for k in range(3):
        base=rot[0,k]*(x-center[0])+rot[1,k]*(y-center[1])-rot[2,k]*center[2]
        slope=rot[2,k]
        if abs(slope)<1e-10:
          valid &= np.abs(base)<=size[k]+1e-9
        else:
          z1=(-size[k]-base)/slope; z2=(size[k]-base)/slope
          lower=np.maximum(lower,np.minimum(z1,z2)); upper=np.minimum(upper,np.maximum(z1,z2))
      surface=np.where(valid & (lower<=upper),upper,0)
      patch=heights[start[1]:end[1],start[0]:end[0]]
      np.maximum(patch,surface,out=patch)
    for center,axis,radius,half,low,high in cylinders:
      start=np.maximum(0,np.floor((low[:2]-lo)/resolution).astype(int))
      end=np.minimum(shape,np.ceil((high[:2]-lo)/resolution).astype(int)+1)
      if (end<=start).any():
        continue
      xs=lo[0]+np.arange(start[0],end[0])*resolution
      ys=lo[1]+np.arange(start[1],end[1])*resolution
      x,y=np.meshgrid(xs,ys)
      dx, dy = x - center[0], y - center[1]
      dot_xy = dx * axis[0] + dy * axis[1]
      a = 1. - axis[2] ** 2
      b = -2. * dot_xy * axis[2]
      c = dx * dx + dy * dy - dot_xy * dot_xy - radius * radius
      if abs(a) < 1e-9:
        continue
      disc = b * b - 4. * a * c
      valid = disc >= 0.
      root = np.sqrt(np.maximum(disc, 0.))
      u = np.maximum((-b + root) / (2. * a), (-b - root) / (2. * a))
      t = dot_xy + u * axis[2]
      surface = np.where(valid & (np.abs(t) <= half), center[2] + u, 0.)
      patch=heights[start[1]:end[1],start[0]:end[0]]
      np.maximum(patch,surface,out=patch)
    return cls(torch.as_tensor(heights,device=device),tuple(lo),resolution)


def advance_pose(pose, delta):
  c,s=pose[:,2].cos(),pose[:,2].sin()
  return pose+torch.stack((c*delta[:,0]-s*delta[:,1],s*delta[:,0]+c*delta[:,1],delta[:,2]),-1)

class TerrainBridge(Go2WBridge):
  def __init__(self, env, field):
    super().__init__(env)
    self.field=field

  def state(self):
    state=super().state()
    state[:,0]-=self.field.sample(self.robot.data.root_link_pos_w[:,:2])
    return state
