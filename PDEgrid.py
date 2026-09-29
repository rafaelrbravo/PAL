from numba import njit,int32,float32,types
from numba.experimental import jitclass
from numba.extending import overload
import numpy as np


# Boundary values may be None, a scalar, or an array over the corresponding face.
def _BCValue(bc,i): pass
@overload(_BCValue)
def _BCValueOverload(bc,i):
    if isinstance(bc,types.NoneType): return lambda bc,i: 0.0
    if isinstance(bc,types.Number): return lambda bc,i: bc
    if isinstance(bc,types.Array): return lambda bc,i: bc.flat[i]

def _FlatOptional(a,size): pass
@overload(_FlatOptional)
def _FlatOptionalOverload(a,size):
    if isinstance(a,types.NoneType): return lambda a,size: np.empty(0,dtype=np.float32)
    if isinstance(a,types.Array): return lambda a,size: a.reshape(size)

def _BCValid(bc,size): pass
@overload(_BCValid)
def _BCValidOverload(bc,size):
    if isinstance(bc,types.NoneType): return lambda bc,size: True
    if isinstance(bc,types.Number): return lambda bc,size: np.isfinite(bc)
    if isinstance(bc,types.Array):
        def impl(bc,size):
            if bc.size!=size: return False
            for v in bc.flat:
                if not np.isfinite(v): return False
            return True
        return impl


specPDE=[
    ('_dimensions',int32[:]),
    ('_wrap',int32[:]),
    ('_field',float32[:]),
    ('_deltas',float32[:]),
    ('_length',int32),
    ('_NULL',int32),
    ('_dx',float32),
    ('_dy',float32),
    ('_dz',float32),
    ('_dt',float32),
    ('_dtdxHalf',float32),
    ('_dtdxSq',float32),
    ('_dtdyHalf',float32),
    ('_dtdySq',float32),
    ('_dtdzHalf',float32),
    ('_dtdzSq',float32),
    ('_adiScratch',float32[:]),
    ('_adiScratch2',float32[:]),
    ('_adiCp',float32[:]),
    ('_adiDp',float32[:]),
    ('_adiY',float32[:]),
    ('_adiZ',float32[:]),

]

@jitclass(specPDE)
class PDEgrid(object):
    def __init__(self,dimensions):
        if len(dimensions)<1 or len(dimensions)>3: raise ValueError("dimensions must have length 1, 2, or 3")
        for dim in dimensions:
            if not np.isfinite(dim) or dim==0 or dim!=np.floor(dim) or dim<-2147483647 or dim>2147483647: raise ValueError("dimensions must be nonzero integers between -2147483647 and 2147483647")
        self._dimensions=np.array(dimensions,dtype=int32)
        self._wrap=np.zeros(len(dimensions),dtype=int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._NULL=-1
        self._length=np.prod(self._dimensions)
        self._field=np.zeros(self._length,dtype=float32)
        self._deltas=np.zeros(self._length,dtype=float32)
        self._dx=1
        self._dy=1
        self._dz=1
        self._dtdxHalf=0.5
        self._dtdxSq=1
        self._dtdyHalf=0.5
        self._dtdySq=1
        self._dtdzHalf=0.5
        self._dtdzSq=1
        self._dt=1
        self._adiScratch=np.zeros(self._length,dtype=float32)
        self._adiScratch2=np.zeros(self._length,dtype=float32)
        maxDim=np.max(self._dimensions)
        self._adiCp=np.zeros(maxDim,dtype=float32)
        self._adiDp=np.zeros(maxDim,dtype=float32)
        self._adiY=np.zeros(maxDim,dtype=float32)
        self._adiZ=np.zeros(maxDim,dtype=float32)

    def __getitem__(self,index): return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self):
        if len(self._dimensions)<2: raise ValueError("Ydim requires a 2D or 3D grid")
        return self.Ydim_()
    def Ydim_(self): return self._dimensions[1]
    def Zdim(self):
        if len(self._dimensions)<3: raise ValueError("Zdim requires a 3D grid")
        return self.Zdim_()
    def Zdim_(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if not np.isfinite(i) or i!=np.floor(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=self._length: raise IndexError("index out of bounds")
        return self.ItoX_(i)
    def ItoX_(self,i):
        if len(self._dimensions)==1: return int32(i)
        if len(self._dimensions)==2: return int32(int32(i)/self._dimensions[1])
        return int32(int32(i)/(self._dimensions[1]*self._dimensions[2]))

    def ItoY(self,i):
        if len(self._dimensions)<2: raise ValueError("ItoY requires a 2D or 3D grid")
        if not np.isfinite(i) or i!=np.floor(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=self._length: raise IndexError("index out of bounds")
        return self.ItoY_(i)
    def ItoY_(self,i):
        if len(self._dimensions)==2: return int32(int32(i)%self._dimensions[1])
        return int32((int32(i)/self._dimensions[2])%self._dimensions[1])

    def ItoZ(self,i):
        if len(self._dimensions)!=3: raise ValueError("ItoZ requires a 3D grid")
        if not np.isfinite(i) or i!=np.floor(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=self._length: raise IndexError("index out of bounds")
        return self.ItoZ_(i)
    def ItoZ_(self,i): return int32(int32(i)%self._dimensions[2])

    def ToI(self,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==1:
            if x is None or y is not None or z is not None: raise ValueError("ToI on a 1D PDEgrid requires x only")
            if not np.isfinite(x) or x!=np.floor(x): raise ValueError("coordinates must be finite integers")
            if x<0 or x>=self._dimensions[0]: raise IndexError("coordinate out of bounds")
            return self.ToI_(x)
        if dim==2:
            if x is None or y is None or z is not None: raise ValueError("ToI on a 2D PDEgrid requires x and y")
            if not np.isfinite(x) or not np.isfinite(y) or x!=np.floor(x) or y!=np.floor(y): raise ValueError("coordinates must be finite integers")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise IndexError("coordinates out of bounds")
            return self.ToI_(x,y)
        if x is None or y is None or z is None: raise ValueError("ToI on a 3D PDEgrid requires x, y, and z")
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or x!=np.floor(x) or y!=np.floor(y) or z!=np.floor(z): raise ValueError("coordinates must be finite integers")
        if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise IndexError("coordinates out of bounds")
        return self.ToI_(x,y,z)
    def ToI_(self,x=-1,y=-1,z=-1):
        if len(self._dimensions)==1: return int32(x)
        if len(self._dimensions)==2: return int32(x)*self._dimensions[1]+int32(y)
        return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

    def Set(self,value,x,y=None,z=None):
        if not np.isfinite(value): raise ValueError("value must be finite")
        self.SetI_(value,self.ToI(x,y,z))
    def Set_(self,value,x,y=-1,z=-1): self.SetI_(value,self.ToI_(x,y,z))
    def SetI(self,value,i):
        if not np.isfinite(value): raise ValueError("value must be finite")
        if not np.isfinite(i) or i!=np.floor(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=self._length: raise IndexError("index out of bounds")
        self.SetI_(value,int32(i))
    def SetI_(self,value,i): self._field[i]=value

    def Add(self,value,x,y=None,z=None):
        if not np.isfinite(value): raise ValueError("value must be finite")
        self.AddI_(value,self.ToI(x,y,z))
    def Add_(self,value,x,y=-1,z=-1): self.AddI_(value,self.ToI_(x,y,z))
    def AddI(self,value,i):
        if not np.isfinite(value): raise ValueError("value must be finite")
        if not np.isfinite(i) or i!=np.floor(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=self._length: raise IndexError("index out of bounds")
        self.AddI_(value,int32(i))
    def AddI_(self,value,i): self._deltas[i]+=value

    def Get(self,x=None,y=None,z=None): return self.GetI(self.ToI(x,y,z))
    def Get_(self,x=-1,y=-1,z=-1): return self.GetI_(self.ToI_(x,y,z))
    def GetI(self,i):
        if not np.isfinite(i) or i!=np.floor(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=self._length: raise IndexError("index out of bounds")
        return self.GetI_(int32(i))
    def GetI_(self,i): return self._field[i]

    def Update(self):
        for i in range(self._length):
            self._field[i]+=self._deltas[i]
            self._deltas[i]=0

    def InWrap(self,idx,dimension):
        if not np.isfinite(dimension) or dimension!=np.floor(dimension): raise ValueError("dimension must be a finite integer")
        if dimension<0 or dimension>=len(self._dimensions): raise IndexError("dimension out of bounds")
        if not np.isfinite(idx) or idx!=np.floor(idx): raise ValueError("index must be a finite integer")
        return self.InWrap_(int32(idx),int32(dimension))
    def InWrap_(self,idx,dimension):
        if idx>=0 and idx<self._dimensions[dimension]: return idx
        if self._wrap[dimension]: return int32(idx%self._dimensions[dimension])
        return self._NULL

    def _RecalcSteps(self):
        self._dtdxHalf=(self._dt/self._dx)/2
        self._dtdxSq=self._dt/self._dx**2
        self._dtdyHalf=(self._dt/self._dy)/2
        self._dtdySq=self._dt/self._dy**2
        self._dtdzHalf=(self._dt/self._dz)/2
        self._dtdzSq=self._dt/self._dz**2

    def GetVoxelVol(self):
        if len(self._dimensions)==1: return self._dx
        if len(self._dimensions)==2: return self._dx*self._dy
        if len(self._dimensions)==3: return self._dx*self._dy*self._dz

    def SetTimeSpaceStep(self,dt,dx,dy=None,dz=None):
        dim=len(self._dimensions)
        if not np.isfinite(dt) or dt<=0: raise ValueError("dt must be finite and positive")
        if not np.isfinite(dx) or dx<=0: raise ValueError("dx must be finite and positive")
        dyVal=1.0; dzVal=1.0
        if dim==1:
            if dy is not None or dz is not None: raise ValueError("1D SetTimeSpaceStep requires only dt and dx")
        elif dim==2:
            if dy is None: raise ValueError("2D SetTimeSpaceStep requires dy")
            dyVal=dy
            if not np.isfinite(dyVal) or dyVal<=0: raise ValueError("dy must be finite and positive")
            if dz is not None: raise ValueError("2D SetTimeSpaceStep does not accept dz")
        else:
            if dy is None or dz is None: raise ValueError("3D SetTimeSpaceStep requires dy and dz")
            dyVal=dy; dzVal=dz
            if not np.isfinite(dyVal) or dyVal<=0: raise ValueError("dy must be finite and positive")
            if not np.isfinite(dzVal) or dzVal<=0: raise ValueError("dz must be finite and positive")
        self.SetTimeSpaceStep_(dt,dx,dyVal,dzVal)
    def SetTimeSpaceStep_(self,dt,dx,dy=1.0,dz=1.0):
        self._dt=dt
        self._dx=dx
        if len(self._dimensions)>=2: self._dy=dy
        if len(self._dimensions)==3: self._dz=dz
        self._RecalcSteps()

    def Dx(self):
        return self._dx
        
    def Dy(self):
        if len(self._dimensions)<2: raise ValueError("Dy requires a 2D or 3D grid")
        return self._dy
    
    def Dz(self):
        if len(self._dimensions)!=3: raise ValueError("Dz requires a 3D grid")
        return self._dz

    def Dt(self):
        return self._dt

    def GetFieldCopy1D(self): return np.copy(self._field)
    def GetFieldCopy2D(self):
        if len(self._dimensions)!=2: raise ValueError("GetFieldCopy2D requires a 2D grid")
        out=np.zeros((self.Xdim(),self.Ydim_()),dtype=np.float32)
        for x in range(self.Xdim()):
            for y in range(self.Ydim_()):
                out[x,y]=self.Get_(x,y)
        return out
    def GetFieldCopy3D(self):
        if len(self._dimensions)!=3: raise ValueError("GetFieldCopy3D requires a 3D grid")
        out=np.zeros((self.Xdim(),self.Ydim_(),self.Zdim_()),dtype=np.float32)
        for x in range(self.Xdim()):
            for y in range(self.Ydim_()):
                for z in range(self.Zdim_()):
                    out[x,y,z]=self.Get_(x,y,z)
        return out

    def Clear(self,val=0.0):
        if not np.isfinite(val): raise ValueError("val must be finite")
        for i in range(len(self)):
            self.SetI_(val,i)
            self._deltas[i]=0
        
    def _CheckBCs(self,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        dim=len(self._dimensions)
        if self._wrap[0] and (xMinBC is not None or xMaxBC is not None): raise ValueError("X boundary conditions cannot be specified on a wrapped axis")
        if not _BCValid(xMinBC,1 if dim==1 else (self.Ydim_() if dim==2 else self.Ydim_()*self.Zdim_())) or not _BCValid(xMaxBC,1 if dim==1 else (self.Ydim_() if dim==2 else self.Ydim_()*self.Zdim_())): raise ValueError("X boundary conditions must be finite scalars or arrays matching the X face")
        if dim<2:
            if yMinBC is not None or yMaxBC is not None or zMinBC is not None or zMaxBC is not None: raise ValueError("1D grids only have X boundary conditions")
            return
        if self._wrap[1] and (yMinBC is not None or yMaxBC is not None): raise ValueError("Y boundary conditions cannot be specified on a wrapped axis")
        ySize=self.Xdim() if dim==2 else self.Xdim()*self.Zdim_()
        if not _BCValid(yMinBC,ySize) or not _BCValid(yMaxBC,ySize): raise ValueError("Y boundary conditions must be finite scalars or arrays matching the Y face")
        if dim<3:
            if zMinBC is not None or zMaxBC is not None: raise ValueError("2D grids do not have Z boundary conditions")
            return
        if self._wrap[2] and (zMinBC is not None or zMaxBC is not None): raise ValueError("Z boundary conditions cannot be specified on a wrapped axis")
        zSize=self.Xdim()*self.Ydim_()
        if not _BCValid(zMinBC,zSize) or not _BCValid(zMaxBC,zSize): raise ValueError("Z boundary conditions must be finite scalars or arrays matching the Z face")

    def Advection(self,vx,vy=0.0,vz=0.0,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        if not np.isfinite(vx) or not np.isfinite(vy) or not np.isfinite(vz): raise ValueError("advection velocities must be finite")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        dim=len(self._dimensions)
        if dim==1:
            if vy!=0 or vz!=0: raise ValueError("1D Advection only accepts vx")
            cfl=self._dt*abs(vx)/self._dx
        elif dim==2:
            if vz!=0: raise ValueError("2D Advection only accepts vx and vy")
            cfl=self._dt*(abs(vx)/self._dx+abs(vy)/self._dy)
        else: cfl=self._dt*(abs(vx)/self._dx+abs(vy)/self._dy+abs(vz)/self._dz)
        if cfl>1: raise ValueError("advection CFL condition violated: dt*(|vx|/dx + |vy|/dy + |vz|/dz) must be <= 1")
        self.Advection_(vx,vy,vz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
    def Advection_(self,vx,vy=0.0,vz=0.0,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        if len(self._dimensions)==1: self._Advection1D(vx,xMinBC,xMaxBC)
        elif len(self._dimensions)==2: self._Advection2D(vx,vy,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._Advection3D(vx,vy,vz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def _Advection1D(self,vx,xMinBC,xMaxBC):
        X=self.Xdim(); f=self._field; d=self._deltas; a=self._dt/self._dx; wrap=self._wrap[0]!=0
        for x in range(X):
            c=f[x]; vp=vx if (x+1<X or wrap or xMaxBC is not None) else 0.0; vm=vx if (x>0 or wrap or xMinBC is not None) else 0.0; q=0.0
            if vp<0:
                n=f[x+1] if x+1<X else (f[0] if wrap else _BCValue(xMaxBC,0)); q-=vp*a*(n-c)
            if vm>0:
                n=f[x-1] if x>0 else (f[X-1] if wrap else _BCValue(xMinBC,0)); q+=vm*a*(n-c)
            d[x]+=q-c*a*(vp-vm)

    def _Advection2D(self,vx,vy,xMinBC,xMaxBC,yMinBC,yMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); f=self._field; d=self._deltas; ax=self._dt/self._dx; ay=self._dt/self._dy; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0
        for x in range(X):
            base=x*Y
            for y in range(Y):
                i=base+y; c=f[i]; vxp=vx if (x+1<X or wx or xMaxBC is not None) else 0.0; vxm=vx if (x>0 or wx or xMinBC is not None) else 0.0; vyp=vy if (y+1<Y or wy or yMaxBC is not None) else 0.0; vym=vy if (y>0 or wy or yMinBC is not None) else 0.0; q=0.0
                if vxp<0: q-=vxp*ax*((f[i+Y] if x+1<X else (f[y] if wx else _BCValue(xMaxBC,y)))-c)
                if vxm>0: q+=vxm*ax*((f[i-Y] if x>0 else (f[(X-1)*Y+y] if wx else _BCValue(xMinBC,y)))-c)
                if vyp<0: q-=vyp*ay*((f[i+1] if y+1<Y else (f[base] if wy else _BCValue(yMaxBC,x)))-c)
                if vym>0: q+=vym*ay*((f[i-1] if y>0 else (f[base+Y-1] if wy else _BCValue(yMinBC,x)))-c)
                d[i]+=q-c*(ax*(vxp-vxm)+ay*(vyp-vym))

    def _Advection3D(self,vx,vy,vz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z; f=self._field; d=self._deltas; ax=self._dt/self._dx; ay=self._dt/self._dy; az=self._dt/self._dz; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0; wz=self._wrap[2]!=0
        for x in range(X):
            xb=x*YZ
            for y in range(Y):
                base=xb+y*Z
                for z in range(Z):
                    i=base+z; c=f[i]; vxp=vx if (x+1<X or wx or xMaxBC is not None) else 0.0; vxm=vx if (x>0 or wx or xMinBC is not None) else 0.0; vyp=vy if (y+1<Y or wy or yMaxBC is not None) else 0.0; vym=vy if (y>0 or wy or yMinBC is not None) else 0.0; vzp=vz if (z+1<Z or wz or zMaxBC is not None) else 0.0; vzm=vz if (z>0 or wz or zMinBC is not None) else 0.0; q=0.0
                    if vxp<0: q-=vxp*ax*((f[i+YZ] if x+1<X else (f[y*Z+z] if wx else _BCValue(xMaxBC,y*Z+z)))-c)
                    if vxm>0: q+=vxm*ax*((f[i-YZ] if x>0 else (f[(X-1)*YZ+y*Z+z] if wx else _BCValue(xMinBC,y*Z+z)))-c)
                    if vyp<0: q-=vyp*ay*((f[i+Z] if y+1<Y else (f[xb+z] if wy else _BCValue(yMaxBC,x*Z+z)))-c)
                    if vym>0: q+=vym*ay*((f[i-Z] if y>0 else (f[xb+(Y-1)*Z+z] if wy else _BCValue(yMinBC,x*Z+z)))-c)
                    if vzp<0: q-=vzp*az*((f[i+1] if z+1<Z else (f[base] if wz else _BCValue(zMaxBC,x*Y+y)))-c)
                    if vzm>0: q+=vzm*az*((f[i-1] if z>0 else (f[base+Z-1] if wz else _BCValue(zMinBC,x*Y+y)))-c)
                    d[i]+=q-c*(ax*(vxp-vxm)+ay*(vyp-vym)+az*(vzp-vzm))

    def AdvectionInterfaces(self,xVels,yVels=None,zVels=None,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        dim=len(self._dimensions); self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        if xVels.ndim!=dim: raise ValueError("xVels shape must match grid dimensions")
        for axis in range(dim):
            if xVels.shape[axis]!=self._dimensions[axis]: raise ValueError("xVels shape must match grid dimensions")
        for v in xVels.flat:
            if not np.isfinite(v): raise ValueError("advection velocities must be finite")
        if dim==1:
            if yVels is not None or zVels is not None: raise ValueError("1D AdvectionInterfaces takes only xVels")
        elif dim==2:
            if yVels is None or zVels is not None: raise ValueError("2D AdvectionInterfaces requires xVels and yVels")
            if yVels.ndim!=dim: raise ValueError("yVels shape must match grid dimensions")
            for axis in range(dim):
                if yVels.shape[axis]!=self._dimensions[axis]: raise ValueError("yVels shape must match grid dimensions")
            for v in yVels.flat:
                if not np.isfinite(v): raise ValueError("advection velocities must be finite")
        else:
            if yVels is None or zVels is None: raise ValueError("3D AdvectionInterfaces requires xVels, yVels, and zVels")
            if yVels.ndim!=dim or zVels.ndim!=dim: raise ValueError("velocity shapes must match grid dimensions")
            for axis in range(dim):
                if yVels.shape[axis]!=self._dimensions[axis] or zVels.shape[axis]!=self._dimensions[axis]: raise ValueError("velocity shapes must match grid dimensions")
            for v in yVels.flat:
                if not np.isfinite(v): raise ValueError("advection velocities must be finite")
            for v in zVels.flat:
                if not np.isfinite(v): raise ValueError("advection velocities must be finite")
        xv=xVels.reshape(self._length); yv=_FlatOptional(yVels,self._length); zv=_FlatOptional(zVels,self._length)
        if dim==1: self._CheckAdvectionInterfacesCFL1D(xv,xMinBC,xMaxBC)
        elif dim==2: self._CheckAdvectionInterfacesCFL2D(xv,yv,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._CheckAdvectionInterfacesCFL3D(xv,yv,zv,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        self.AdvectionInterfaces_(xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def AdvectionInterfaces_(self,xVels,yVels=None,zVels=None,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        dim=len(self._dimensions); xv=xVels.reshape(self._length); yv=_FlatOptional(yVels,self._length); zv=_FlatOptional(zVels,self._length)
        if dim==1: self._Advection1Dfv(xv,xMinBC,xMaxBC)
        elif dim==2: self._Advection2Dfv(xv,yv,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._Advection3Dfv(xv,yv,zv,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def _CheckAdvectionInterfacesCFL1D(self,xVels,xMinBC,xMaxBC):
        for x in range(self.Xdim()):
            out=0.0
            if x<self.Xdim()-1 or self._wrap[0] or xMaxBC is not None: out+=max(xVels[x],0.0)/self._dx
            if x>0 or self._wrap[0]: out+=max(-xVels[self.InWrap_(x-1,0)],0.0)/self._dx
            elif xMinBC is not None: out+=max(-xVels[x],0.0)/self._dx
            if self._dt*out>1: raise ValueError("advection CFL condition violated")

    def _CheckAdvectionInterfacesCFL2D(self,xVels,yVels,xMinBC,xMaxBC,yMinBC,yMaxBC):
        for x in range(self.Xdim()):
            for y in range(self.Ydim_()):
                i=self.ToI_(x,y); out=0.0
                if x<self.Xdim()-1 or self._wrap[0] or xMaxBC is not None: out+=max(xVels[i],0.0)/self._dx
                if x>0 or self._wrap[0]: out+=max(-xVels[self.ToI_(self.InWrap_(x-1,0),y)],0.0)/self._dx
                elif xMinBC is not None: out+=max(-xVels[i],0.0)/self._dx
                if y<self.Ydim_()-1 or self._wrap[1] or yMaxBC is not None: out+=max(yVels[i],0.0)/self._dy
                if y>0 or self._wrap[1]: out+=max(-yVels[self.ToI_(x,self.InWrap_(y-1,1))],0.0)/self._dy
                elif yMinBC is not None: out+=max(-yVels[i],0.0)/self._dy
                if self._dt*out>1: raise ValueError("advection CFL condition violated")

    def _CheckAdvectionInterfacesCFL3D(self,xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        for x in range(self.Xdim()):
            for y in range(self.Ydim_()):
                for z in range(self.Zdim_()):
                    i=self.ToI_(x,y,z); out=0.0
                    if x<self.Xdim()-1 or self._wrap[0] or xMaxBC is not None: out+=max(xVels[i],0.0)/self._dx
                    if x>0 or self._wrap[0]: out+=max(-xVels[self.ToI_(self.InWrap_(x-1,0),y,z)],0.0)/self._dx
                    elif xMinBC is not None: out+=max(-xVels[i],0.0)/self._dx
                    if y<self.Ydim_()-1 or self._wrap[1] or yMaxBC is not None: out+=max(yVels[i],0.0)/self._dy
                    if y>0 or self._wrap[1]: out+=max(-yVels[self.ToI_(x,self.InWrap_(y-1,1),z)],0.0)/self._dy
                    elif yMinBC is not None: out+=max(-yVels[i],0.0)/self._dy
                    if z<self.Zdim_()-1 or self._wrap[2] or zMaxBC is not None: out+=max(zVels[i],0.0)/self._dz
                    if z>0 or self._wrap[2]: out+=max(-zVels[self.ToI_(x,y,self.InWrap_(z-1,2))],0.0)/self._dz
                    elif zMinBC is not None: out+=max(-zVels[i],0.0)/self._dz
                    if self._dt*out>1: raise ValueError("advection CFL condition violated")

    def _Advection1Dfv(self,xVels,xMinBC,xMaxBC):
        X=self.Xdim(); f=self._field; d=self._deltas; a=self._dt/self._dx; wrap=self._wrap[0]!=0
        for x in range(X):
            c=f[x]; vp=xVels[x] if (x+1<X or wrap or xMaxBC is not None) else 0.0; vm=xVels[x-1] if x>0 else (xVels[X-1] if wrap else (xVels[x] if xMinBC is not None else 0.0)); q=0.0
            if vp<0: q-=vp*a*((f[x+1] if x+1<X else (f[0] if wrap else _BCValue(xMaxBC,0)))-c)
            if vm>0: q+=vm*a*((f[x-1] if x>0 else (f[X-1] if wrap else _BCValue(xMinBC,0)))-c)
            d[x]+=q-c*a*(vp-vm)

    def _Advection2Dfv(self,xVels,yVels,xMinBC,xMaxBC,yMinBC,yMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); f=self._field; d=self._deltas; ax=self._dt/self._dx; ay=self._dt/self._dy; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0
        for x in range(X):
            base=x*Y
            for y in range(Y):
                i=base+y; c=f[i]; vxp=xVels[i] if (x+1<X or wx or xMaxBC is not None) else 0.0; vxm=xVels[i-Y] if x>0 else (xVels[(X-1)*Y+y] if wx else (xVels[i] if xMinBC is not None else 0.0)); vyp=yVels[i] if (y+1<Y or wy or yMaxBC is not None) else 0.0; vym=yVels[i-1] if y>0 else (yVels[base+Y-1] if wy else (yVels[i] if yMinBC is not None else 0.0)); q=0.0
                if vxp<0: q-=vxp*ax*((f[i+Y] if x+1<X else (f[y] if wx else _BCValue(xMaxBC,y)))-c)
                if vxm>0: q+=vxm*ax*((f[i-Y] if x>0 else (f[(X-1)*Y+y] if wx else _BCValue(xMinBC,y)))-c)
                if vyp<0: q-=vyp*ay*((f[i+1] if y+1<Y else (f[base] if wy else _BCValue(yMaxBC,x)))-c)
                if vym>0: q+=vym*ay*((f[i-1] if y>0 else (f[base+Y-1] if wy else _BCValue(yMinBC,x)))-c)
                d[i]+=q-c*(ax*(vxp-vxm)+ay*(vyp-vym))

    def _Advection3Dfv(self,xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z; f=self._field; d=self._deltas; ax=self._dt/self._dx; ay=self._dt/self._dy; az=self._dt/self._dz; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0; wz=self._wrap[2]!=0
        for x in range(X):
            xb=x*YZ
            for y in range(Y):
                base=xb+y*Z
                for z in range(Z):
                    i=base+z; c=f[i]; vxp=xVels[i] if (x+1<X or wx or xMaxBC is not None) else 0.0; vxm=xVels[i-YZ] if x>0 else (xVels[(X-1)*YZ+y*Z+z] if wx else (xVels[i] if xMinBC is not None else 0.0)); vyp=yVels[i] if (y+1<Y or wy or yMaxBC is not None) else 0.0; vym=yVels[i-Z] if y>0 else (yVels[xb+(Y-1)*Z+z] if wy else (yVels[i] if yMinBC is not None else 0.0)); vzp=zVels[i] if (z+1<Z or wz or zMaxBC is not None) else 0.0; vzm=zVels[i-1] if z>0 else (zVels[base+Z-1] if wz else (zVels[i] if zMinBC is not None else 0.0)); q=0.0
                    if vxp<0: q-=vxp*ax*((f[i+YZ] if x+1<X else (f[y*Z+z] if wx else _BCValue(xMaxBC,y*Z+z)))-c)
                    if vxm>0: q+=vxm*ax*((f[i-YZ] if x>0 else (f[(X-1)*YZ+y*Z+z] if wx else _BCValue(xMinBC,y*Z+z)))-c)
                    if vyp<0: q-=vyp*ay*((f[i+Z] if y+1<Y else (f[xb+z] if wy else _BCValue(yMaxBC,x*Z+z)))-c)
                    if vym>0: q+=vym*ay*((f[i-Z] if y>0 else (f[xb+(Y-1)*Z+z] if wy else _BCValue(yMinBC,x*Z+z)))-c)
                    if vzp<0: q-=vzp*az*((f[i+1] if z+1<Z else (f[base] if wz else _BCValue(zMaxBC,x*Y+y)))-c)
                    if vzm>0: q+=vzm*az*((f[i-1] if z>0 else (f[base+Z-1] if wz else _BCValue(zMinBC,x*Y+y)))-c)
                    d[i]+=q-c*(ax*(vxp-vxm)+ay*(vyp-vym)+az*(vzp-vzm))

    def Diffusion(self,rateConstant,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("rateConstant must be finite and nonnegative")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        dim=len(self._dimensions); out=2.0/(self._dx*self._dx)
        if dim>1: out+=2.0/(self._dy*self._dy)
        if dim>2: out+=2.0/(self._dz*self._dz)
        if rateConstant*self._dt*out>1: raise ValueError("diffusion stability condition violated")
        self.Diffusion_(rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
    def Diffusion_(self,rateConstant,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        if len(self._dimensions)==1: self._Diffusion1D(rateConstant,xMinBC,xMaxBC)
        elif len(self._dimensions)==2: self._Diffusion2D(rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._Diffusion3D(rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
    def _Diffusion1D(self,rateConstant,xMinBC,xMaxBC):
        X=self.Xdim(); f=self._field; d=self._deltas; scale=rateConstant*self._dtdxSq; wrap=self._wrap[0]!=0
        for x in range(X):
            c=f[x]; v=0.0
            if x+1<X: v+=f[x+1]-c
            elif wrap: v+=f[0]-c
            elif xMaxBC is not None: v+=_BCValue(xMaxBC,0)-c
            if x>0: v+=f[x-1]-c
            elif wrap: v+=f[X-1]-c
            elif xMinBC is not None: v+=_BCValue(xMinBC,0)-c
            d[x]+=v*scale

    def _Diffusion2D(self,rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); f=self._field; d=self._deltas; sx=rateConstant*self._dtdxSq; sy=rateConstant*self._dtdySq; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0
        for x in range(X):
            base=x*Y
            for y in range(Y):
                i=base+y; c=f[i]; v=0.0
                if x+1<X: v+=(f[i+Y]-c)*sx
                elif wx: v+=(f[y]-c)*sx
                elif xMaxBC is not None: v+=(_BCValue(xMaxBC,y)-c)*sx
                if x>0: v+=(f[i-Y]-c)*sx
                elif wx: v+=(f[(X-1)*Y+y]-c)*sx
                elif xMinBC is not None: v+=(_BCValue(xMinBC,y)-c)*sx
                if y+1<Y: v+=(f[i+1]-c)*sy
                elif wy: v+=(f[base]-c)*sy
                elif yMaxBC is not None: v+=(_BCValue(yMaxBC,x)-c)*sy
                if y>0: v+=(f[i-1]-c)*sy
                elif wy: v+=(f[base+Y-1]-c)*sy
                elif yMinBC is not None: v+=(_BCValue(yMinBC,x)-c)*sy
                d[i]+=v

    def _Diffusion3D(self,rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z; f=self._field; d=self._deltas; sx=rateConstant*self._dtdxSq; sy=rateConstant*self._dtdySq; sz=rateConstant*self._dtdzSq; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0; wz=self._wrap[2]!=0
        for x in range(X):
            xb=x*YZ
            for y in range(Y):
                base=xb+y*Z
                for z in range(Z):
                    i=base+z; c=f[i]; v=0.0; xbi=y*Z+z; ybi=x*Z+z; zbi=x*Y+y
                    if x+1<X: v+=(f[i+YZ]-c)*sx
                    elif wx: v+=(f[y*Z+z]-c)*sx
                    elif xMaxBC is not None: v+=(_BCValue(xMaxBC,xbi)-c)*sx
                    if x>0: v+=(f[i-YZ]-c)*sx
                    elif wx: v+=(f[(X-1)*YZ+y*Z+z]-c)*sx
                    elif xMinBC is not None: v+=(_BCValue(xMinBC,xbi)-c)*sx
                    if y+1<Y: v+=(f[i+Z]-c)*sy
                    elif wy: v+=(f[xb+z]-c)*sy
                    elif yMaxBC is not None: v+=(_BCValue(yMaxBC,ybi)-c)*sy
                    if y>0: v+=(f[i-Z]-c)*sy
                    elif wy: v+=(f[xb+(Y-1)*Z+z]-c)*sy
                    elif yMinBC is not None: v+=(_BCValue(yMinBC,ybi)-c)*sy
                    if z+1<Z: v+=(f[i+1]-c)*sz
                    elif wz: v+=(f[base]-c)*sz
                    elif zMaxBC is not None: v+=(_BCValue(zMaxBC,zbi)-c)*sz
                    if z>0: v+=(f[i-1]-c)*sz
                    elif wz: v+=(f[base+Z-1]-c)*sz
                    elif zMinBC is not None: v+=(_BCValue(zMinBC,zbi)-c)*sz
                    d[i]+=v

    def DiffusionADI(self,rateConstant,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("rateConstant must be finite and nonnegative")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        self.DiffusionADI_(rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
    def DiffusionADI_(self,rateConstant,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        if len(self._dimensions)==1: self._DiffusionADI1D(rateConstant,xMinBC,xMaxBC)
        elif len(self._dimensions)==2: self._DiffusionADI2D(rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._DiffusionADI3D(rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def _ADIPrepareLine(self,n,r,wrap,minSet,maxSet):
        # Precompute the Thomas coefficients once; every line on this axis has the same matrix.
        if n<=1: return
        if wrap and n==2: return
        off=-r/2
        if wrap:
            diag=1+r; gamma=-diag; alpha=off; beta=off
            self._adiCp[0]=off/(diag-gamma)
            for i in range(1,n):
                b=diag-(alpha*beta)/gamma if i==n-1 else diag
                denom=b-off*self._adiCp[i-1]
                self._adiCp[i]=off/denom if i<n-1 else 0.0
            # Sherman-Morrison correction vector; also identical for every line.
            self._adiDp[0]=gamma/(diag-gamma)
            for i in range(1,n):
                b=diag-(alpha*beta)/gamma if i==n-1 else diag
                d=alpha if i==n-1 else 0.0
                denom=b-off*self._adiCp[i-1]
                self._adiDp[i]=(d-off*self._adiDp[i-1])/denom
            self._adiZ[n-1]=self._adiDp[n-1]
            for i in range(n-2,-1,-1): self._adiZ[i]=self._adiDp[i]-self._adiCp[i]*self._adiZ[i+1]
        else:
            diag=1+r if minSet else 1+r/2
            self._adiCp[0]=off/diag
            for i in range(1,n):
                diag=(1+r if maxSet else 1+r/2) if i==n-1 else 1+r
                denom=diag-off*self._adiCp[i-1]
                self._adiCp[i]=off/denom if i<n-1 else 0.0

    def _ADISolvePrepared(self,n,r,rhs,out,wrap,minSet,maxSet,minVal,maxVal,addBC):
        if n==1:
            if wrap: out[0]=rhs[0]; return
            diag=1.0; d=rhs[0]
            if minSet: diag+=r/2; d+=r*minVal/2 if addBC else 0.0
            if maxSet: diag+=r/2; d+=r*maxVal/2 if addBC else 0.0
            out[0]=d/diag; return
        if wrap and n==2:
            diag=1+r; off=-r; det=diag*diag-off*off
            out[0]=(diag*rhs[0]-off*rhs[1])/det; out[1]=(diag*rhs[1]-off*rhs[0])/det; return
        off=-r/2
        d=rhs[0]+(r*minVal/2 if minSet and addBC and not wrap else 0.0)
        if wrap: firstDiag=2*(1+r)
        else: firstDiag=1+r if minSet else 1+r/2
        self._adiDp[0]=d/firstDiag
        for i in range(1,n):
            d=rhs[i]+(r*maxVal/2 if i==n-1 and maxSet and addBC and not wrap else 0.0)
            if wrap:
                diag=(1+r)+(off*off)/(1+r) if i==n-1 else 1+r
            else:
                diag=(1+r if maxSet else 1+r/2) if i==n-1 else 1+r
            denom=diag-off*self._adiCp[i-1]
            self._adiDp[i]=(d-off*self._adiDp[i-1])/denom
        out[n-1]=self._adiDp[n-1]
        for i in range(n-2,-1,-1): out[i]=self._adiDp[i]-self._adiCp[i]*out[i+1]
        if wrap:
            diag=1+r; vEnd=off/(-diag)
            factor=(out[0]+out[n-1]*vEnd)/(1+self._adiZ[0]+self._adiZ[n-1]*vEnd)
            for i in range(n): out[i]-=factor*self._adiZ[i]

    def _DiffusionADI1D(self,rateConstant,xMinBC,xMaxBC):
        n=self.Xdim(); r=rateConstant*self._dtdxSq
        for x in range(n):
            c=self._field[x]; delta=self._Delta1DBC(c,x+1,xMinBC,xMaxBC)+self._Delta1DBC(c,x-1,xMinBC,xMaxBC); self._adiScratch[x]=c+rateConstant*delta/2
        minSet=xMinBC is not None; maxSet=xMaxBC is not None; minVal=_BCValue(xMinBC,0) if minSet else 0.0; maxVal=_BCValue(xMaxBC,0) if maxSet else 0.0
        self._ADIPrepareLine(n,r,self._wrap[0]!=0,minSet,maxSet); self._ADISolvePrepared(n,r,self._adiScratch,self._adiY,self._wrap[0]!=0,minSet,maxSet,minVal,maxVal,True)
        for x in range(n): self._deltas[x]+=self._adiY[x]-self._field[x]

    def _DiffusionADI2D(self,rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); rx=rateConstant*self._dtdxSq; ry=rateConstant*self._dtdySq
        f=self._field; scratch=self._adiScratch; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0
        self._ADIPrepareLine(X,rx,wx,xMinBC is not None,xMaxBC is not None)
        for y in range(Y):
            # First PR half-step: explicit Y, implicit X.
            for x in range(X):
                i=x*Y+y; c=f[i]; v=0.0
                if y+1<Y: v+=f[i+1]-c
                elif wy: v+=f[i-(Y-1)]-c
                elif yMaxBC is not None: v+=_BCValue(yMaxBC,x)-c
                if y>0: v+=f[i-1]-c
                elif wy: v+=f[i+(Y-1)]-c
                elif yMinBC is not None: v+=_BCValue(yMinBC,x)-c
                self._adiDp[x]=c+ry*v/2
            minSet=xMinBC is not None; maxSet=xMaxBC is not None
            minVal=_BCValue(xMinBC,y) if minSet else 0.0; maxVal=_BCValue(xMaxBC,y) if maxSet else 0.0
            self._ADISolvePrepared(X,rx,self._adiDp,self._adiY,wx,minSet,maxSet,minVal,maxVal,True)
            for x in range(X): scratch[x*Y+y]=self._adiY[x]
        self._ADIPrepareLine(Y,ry,wy,yMinBC is not None,yMaxBC is not None)
        for x in range(X):
            base=x*Y
            # Second PR half-step: explicit X, implicit Y.
            for y in range(Y):
                i=base+y; c=scratch[i]; v=0.0
                if x+1<X: v+=scratch[i+Y]-c
                elif wx: v+=scratch[y]-c
                elif xMaxBC is not None: v+=_BCValue(xMaxBC,y)-c
                if x>0: v+=scratch[i-Y]-c
                elif wx: v+=scratch[(X-1)*Y+y]-c
                elif xMinBC is not None: v+=_BCValue(xMinBC,y)-c
                self._adiDp[y]=c+rx*v/2
            minSet=yMinBC is not None; maxSet=yMaxBC is not None
            minVal=_BCValue(yMinBC,x) if minSet else 0.0; maxVal=_BCValue(yMaxBC,x) if maxSet else 0.0
            self._ADISolvePrepared(Y,ry,self._adiDp,self._adiY,wy,minSet,maxSet,minVal,maxVal,True)
            for y in range(Y):
                i=base+y; self._deltas[i]+=self._adiY[y]-f[i]

    def _DiffusionADI3D(self,rateConstant,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z
        rx=rateConstant*self._dtdxSq; ry=rateConstant*self._dtdySq; rz=rateConstant*self._dtdzSq
        f=self._field; s1=self._adiScratch; s2=self._adiScratch2
        wx=self._wrap[0]!=0; wy=self._wrap[1]!=0; wz=self._wrap[2]!=0
        # Douglas-Gunn increment RHS q = dt*D*(Lx+Ly+Lz)u, then three directional solves.
        self._ADIPrepareLine(X,rx,wx,xMinBC is not None,xMaxBC is not None)
        for y in range(Y):
            yoff=y*Z
            for z in range(Z):
                for x in range(X):
                    i=x*YZ+yoff+z; c=f[i]; v=0.0
                    if x+1<X: v+=(f[i+YZ]-c)*self._dtdxSq
                    elif wx: v+=(f[yoff+z]-c)*self._dtdxSq
                    elif xMaxBC is not None: v+=(_BCValue(xMaxBC,yoff+z)-c)*self._dtdxSq
                    if x>0: v+=(f[i-YZ]-c)*self._dtdxSq
                    elif wx: v+=(f[(X-1)*YZ+yoff+z]-c)*self._dtdxSq
                    elif xMinBC is not None: v+=(_BCValue(xMinBC,yoff+z)-c)*self._dtdxSq
                    if y+1<Y: v+=(f[i+Z]-c)*self._dtdySq
                    elif wy: v+=(f[x*YZ+z]-c)*self._dtdySq
                    elif yMaxBC is not None: v+=(_BCValue(yMaxBC,x*Z+z)-c)*self._dtdySq
                    if y>0: v+=(f[i-Z]-c)*self._dtdySq
                    elif wy: v+=(f[x*YZ+(Y-1)*Z+z]-c)*self._dtdySq
                    elif yMinBC is not None: v+=(_BCValue(yMinBC,x*Z+z)-c)*self._dtdySq
                    if z+1<Z: v+=(f[i+1]-c)*self._dtdzSq
                    elif wz: v+=(f[x*YZ+yoff]-c)*self._dtdzSq
                    elif zMaxBC is not None: v+=(_BCValue(zMaxBC,x*Y+y)-c)*self._dtdzSq
                    if z>0: v+=(f[i-1]-c)*self._dtdzSq
                    elif wz: v+=(f[x*YZ+yoff+Z-1]-c)*self._dtdzSq
                    elif zMinBC is not None: v+=(_BCValue(zMinBC,x*Y+y)-c)*self._dtdzSq
                    self._adiDp[x]=rateConstant*v
                bi=yoff+z; minSet=xMinBC is not None; maxSet=xMaxBC is not None
                minVal=_BCValue(xMinBC,bi) if minSet else 0.0; maxVal=_BCValue(xMaxBC,bi) if maxSet else 0.0
                self._ADISolvePrepared(X,rx,self._adiDp,self._adiY,wx,minSet,maxSet,minVal,maxVal,False)
                for x in range(X): s1[x*YZ+yoff+z]=self._adiY[x]
        self._ADIPrepareLine(Y,ry,wy,yMinBC is not None,yMaxBC is not None)
        for x in range(X):
            xoff=x*YZ
            for z in range(Z):
                for y in range(Y): self._adiDp[y]=s1[xoff+y*Z+z]
                bi=x*Z+z; minSet=yMinBC is not None; maxSet=yMaxBC is not None
                minVal=_BCValue(yMinBC,bi) if minSet else 0.0; maxVal=_BCValue(yMaxBC,bi) if maxSet else 0.0
                self._ADISolvePrepared(Y,ry,self._adiDp,self._adiY,wy,minSet,maxSet,minVal,maxVal,False)
                for y in range(Y): s2[xoff+y*Z+z]=self._adiY[y]
        self._ADIPrepareLine(Z,rz,wz,zMinBC is not None,zMaxBC is not None)
        for x in range(X):
            xoff=x*YZ
            for y in range(Y):
                base=xoff+y*Z
                for z in range(Z): self._adiDp[z]=s2[base+z]
                bi=x*Y+y; minSet=zMinBC is not None; maxSet=zMaxBC is not None
                minVal=_BCValue(zMinBC,bi) if minSet else 0.0; maxVal=_BCValue(zMaxBC,bi) if maxSet else 0.0
                self._ADISolvePrepared(Z,rz,self._adiDp,self._adiY,wz,minSet,maxSet,minVal,maxVal,False)
                for z in range(Z): self._deltas[base+z]+=self._adiY[z]

    def DiffusionRadial(self,rateConstant):
        if len(self._dimensions)!=1: raise ValueError("DiffusionRadial requires a 1D grid")
        if self._wrap[0]: raise ValueError("DiffusionRadial does not support wrapping")
        if self._length<2: raise ValueError("DiffusionRadial requires at least 2 grid points")
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("rateConstant must be finite and nonnegative")
        if 6.0*rateConstant*self._dt/(self._dx*self._dx)>1: raise ValueError("spherical diffusion stability condition violated")
        self.DiffusionRadial_(rateConstant)
    def DiffusionRadial_(self,rateConstant):
        X=self.Xdim(); f=self._field; d=self._deltas; scale=3.0*rateConstant*self._dt/(self._dx*self._dx)
        d[0]+=scale*(f[1]-f[0])
        for i in range(1,X-1):
            rp=i+0.5; rm=i-0.5; denom=rp*rp*rp-rm*rm*rm
            d[i]+=scale*(rp*rp*(f[i+1]-f[i])-rm*rm*(f[i]-f[i-1]))/denom
        i=X-1; rp=i+0.5; rm=i-0.5; denom=rp*rp*rp-rm*rm*rm
        d[i]+=-scale*rm*rm*(f[i]-f[i-1])/denom

    def DiffusionMask(self,rateConstant,mask):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("rateConstant must be finite and nonnegative")
        if mask.ndim!=len(self._dimensions): raise ValueError("mask shape must match grid dimensions")
        for axis in range(len(self._dimensions)):
            if mask.shape[axis]!=self._dimensions[axis]: raise ValueError("mask shape must match grid dimensions")
        if not np.all(np.isfinite(mask)): raise ValueError("mask values must be finite")
        dim=len(self._dimensions); out=2.0/(self._dx*self._dx)
        if dim>1: out+=2.0/(self._dy*self._dy)
        if dim>2: out+=2.0/(self._dz*self._dz)
        if rateConstant*self._dt*out>1: raise ValueError("diffusion stability condition violated")
        self.DiffusionMask_(rateConstant,mask)
    def DiffusionMask_(self,rateConstant,mask):
        mask=mask.reshape(self._length)
        if len(self._dimensions)==1: self._Diffusion1DMask(rateConstant,mask)
        elif len(self._dimensions)==2: self._Diffusion2DMask(rateConstant,mask)
        else: self._Diffusion3DMask(rateConstant,mask)

    def _Diffusion1DMask(self,rateConstant,mask):
        scale=rateConstant*self._dtdxSq; X=self.Xdim()
        for x in range(X):
            if mask[x]<0: continue
            c=self._field[x]; d=0.0
            xp=x+1
            if xp<X:
                if mask[xp]>=0: d+=self._field[xp]-c
            elif self._wrap[0] and mask[0]>=0: d+=self._field[0]-c
            xm=x-1
            if xm>=0:
                if mask[xm]>=0: d+=self._field[xm]-c
            elif self._wrap[0] and mask[X-1]>=0: d+=self._field[X-1]-c
            self._deltas[x]+=d*scale

    def _Diffusion2DMask(self,rateConstant,mask):
        sx=rateConstant*self._dtdxSq; sy=rateConstant*self._dtdySq; X=self.Xdim(); Y=self.Ydim_()
        for x in range(X):
            for y in range(Y):
                i=x*Y+y
                if mask[i]<0: continue
                c=self._field[i]; d=0.0
                if x+1<X:
                    ip=i+Y
                    if mask[ip]>=0: d+=(self._field[ip]-c)*sx
                elif self._wrap[0]:
                    ip=y
                    if mask[ip]>=0: d+=(self._field[ip]-c)*sx
                if x>0:
                    im=i-Y
                    if mask[im]>=0: d+=(self._field[im]-c)*sx
                elif self._wrap[0]:
                    im=(X-1)*Y+y
                    if mask[im]>=0: d+=(self._field[im]-c)*sx
                if y+1<Y:
                    ip=i+1
                    if mask[ip]>=0: d+=(self._field[ip]-c)*sy
                elif self._wrap[1]:
                    ip=x*Y
                    if mask[ip]>=0: d+=(self._field[ip]-c)*sy
                if y>0:
                    im=i-1
                    if mask[im]>=0: d+=(self._field[im]-c)*sy
                elif self._wrap[1]:
                    im=x*Y+Y-1
                    if mask[im]>=0: d+=(self._field[im]-c)*sy
                self._deltas[i]+=d

    def _Diffusion3DMask(self,rateConstant,mask):
        sx=rateConstant*self._dtdxSq; sy=rateConstant*self._dtdySq; sz=rateConstant*self._dtdzSq
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    i=x*YZ+y*Z+z
                    if mask[i]<0: continue
                    c=self._field[i]; d=0.0
                    if x+1<X:
                        ip=i+YZ
                        if mask[ip]>=0: d+=(self._field[ip]-c)*sx
                    elif self._wrap[0]:
                        ip=y*Z+z
                        if mask[ip]>=0: d+=(self._field[ip]-c)*sx
                    if x>0:
                        im=i-YZ
                        if mask[im]>=0: d+=(self._field[im]-c)*sx
                    elif self._wrap[0]:
                        im=(X-1)*YZ+y*Z+z
                        if mask[im]>=0: d+=(self._field[im]-c)*sx
                    if y+1<Y:
                        ip=i+Z
                        if mask[ip]>=0: d+=(self._field[ip]-c)*sy
                    elif self._wrap[1]:
                        ip=x*YZ+z
                        if mask[ip]>=0: d+=(self._field[ip]-c)*sy
                    if y>0:
                        im=i-Z
                        if mask[im]>=0: d+=(self._field[im]-c)*sy
                    elif self._wrap[1]:
                        im=x*YZ+(Y-1)*Z+z
                        if mask[im]>=0: d+=(self._field[im]-c)*sy
                    if z+1<Z:
                        ip=i+1
                        if mask[ip]>=0: d+=(self._field[ip]-c)*sz
                    elif self._wrap[2]:
                        ip=x*YZ+y*Z
                        if mask[ip]>=0: d+=(self._field[ip]-c)*sz
                    if z>0:
                        im=i-1
                        if mask[im]>=0: d+=(self._field[im]-c)*sz
                    elif self._wrap[2]:
                        im=x*YZ+y*Z+Z-1
                        if mask[im]>=0: d+=(self._field[im]-c)*sz
                    self._deltas[i]+=d

    def DiffusionField(self,rateConstants,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        dim=len(self._dimensions); self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        if rateConstants.ndim!=dim: raise ValueError("rateConstants shape must match grid dimensions")
        for axis in range(dim):
            if rateConstants.shape[axis]!=self._dimensions[axis]: raise ValueError("rateConstants shape must match grid dimensions")
        for d in rateConstants.flat:
            if not np.isfinite(d): raise ValueError("rateConstants must be finite")
        self._CheckDiffusionFieldStability(rateConstants.reshape(self._length),xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        self.DiffusionField_(rateConstants,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
    def _CheckDiffusionFieldStability(self,rates,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        dim=len(self._dimensions)
        for x in range(self.Xdim()):
            for y in range(self.Ydim_() if dim>1 else 1):
                for z in range(self.Zdim_() if dim>2 else 1):
                    i=self.ToI_(x,y,z) if dim==3 else (self.ToI_(x,y) if dim==2 else x)
                    if rates[i]<0: continue
                    out=0.0; xp=self.InWrap_(x+1,0); xm=self.InWrap_(x-1,0)
                    if xp!=self._NULL: out+=self._DiscFaceRate(rates[i],rates[self.ToI_(xp,y,z) if dim==3 else (self.ToI_(xp,y) if dim==2 else xp)])/(self._dx*self._dx)
                    elif xMaxBC is not None: out+=rates[i]/(self._dx*self._dx)
                    if xm!=self._NULL: out+=self._DiscFaceRate(rates[i],rates[self.ToI_(xm,y,z) if dim==3 else (self.ToI_(xm,y) if dim==2 else xm)])/(self._dx*self._dx)
                    elif xMinBC is not None: out+=rates[i]/(self._dx*self._dx)
                    if dim>1:
                        yp=self.InWrap_(y+1,1); ym=self.InWrap_(y-1,1)
                        if yp!=self._NULL: out+=self._DiscFaceRate(rates[i],rates[self.ToI_(x,yp,z) if dim==3 else self.ToI_(x,yp)])/(self._dy*self._dy)
                        elif yMaxBC is not None: out+=rates[i]/(self._dy*self._dy)
                        if ym!=self._NULL: out+=self._DiscFaceRate(rates[i],rates[self.ToI_(x,ym,z) if dim==3 else self.ToI_(x,ym)])/(self._dy*self._dy)
                        elif yMinBC is not None: out+=rates[i]/(self._dy*self._dy)
                    if dim>2:
                        zp=self.InWrap_(z+1,2); zm=self.InWrap_(z-1,2)
                        if zp!=self._NULL: out+=self._DiscFaceRate(rates[i],rates[self.ToI_(x,y,zp)])/(self._dz*self._dz)
                        elif zMaxBC is not None: out+=rates[i]/(self._dz*self._dz)
                        if zm!=self._NULL: out+=self._DiscFaceRate(rates[i],rates[self.ToI_(x,y,zm)])/(self._dz*self._dz)
                        elif zMinBC is not None: out+=rates[i]/(self._dz*self._dz)
                    if self._dt*out>1: raise ValueError("diffusion stability condition violated")

    def DiffusionField_(self,rateConstants,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        rates=rateConstants.reshape(self._length)
        if len(self._dimensions)==1: self._Diffusion1Ddisc(rates,xMinBC,xMaxBC)
        elif len(self._dimensions)==2: self._Diffusion2Ddisc(rates,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._Diffusion3Ddisc(rates,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def _DiscFaceRate(self,d0,d1):
        if d0<=0 or d1<=0: return 0.0
        return 2*d0*d1/(d0+d1)

    def _Diffusion1Ddisc(self,rates,xMinBC,xMaxBC):
        X=self.Xdim(); f=self._field; d=self._deltas; s=self._dtdxSq; wrap=self._wrap[0]!=0
        for i in range(X):
            di=rates[i]
            if di<0: continue
            c=f[i]; v=0.0
            if i+1<X:
                dj=rates[i+1]
                if di>0 and dj>0: v+=(f[i+1]-c)*(2.0*di*dj/(di+dj))*s
            elif wrap:
                dj=rates[0]
                if di>0 and dj>0: v+=(f[0]-c)*(2.0*di*dj/(di+dj))*s
            elif xMaxBC is not None: v+=(_BCValue(xMaxBC,0)-c)*di*s
            if i>0:
                dj=rates[i-1]
                if di>0 and dj>0: v+=(f[i-1]-c)*(2.0*di*dj/(di+dj))*s
            elif wrap:
                dj=rates[X-1]
                if di>0 and dj>0: v+=(f[X-1]-c)*(2.0*di*dj/(di+dj))*s
            elif xMinBC is not None: v+=(_BCValue(xMinBC,0)-c)*di*s
            d[i]+=v

    def _Diffusion2Ddisc(self,rates,xMinBC,xMaxBC,yMinBC,yMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); f=self._field; d=self._deltas; sx=self._dtdxSq; sy=self._dtdySq; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0
        for x in range(X):
            base=x*Y
            for y in range(Y):
                i=base+y; di=rates[i]
                if di<0: continue
                c=f[i]; v=0.0
                if x+1<X: ip=i+Y; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                elif wx: ip=y; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                elif xMaxBC is not None: v+=(_BCValue(xMaxBC,y)-c)*di*sx
                if x>0: im=i-Y; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                elif wx: im=(X-1)*Y+y; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                elif xMinBC is not None: v+=(_BCValue(xMinBC,y)-c)*di*sx
                if y+1<Y: ip=i+1; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                elif wy: ip=base; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                elif yMaxBC is not None: v+=(_BCValue(yMaxBC,x)-c)*di*sy
                if y>0: im=i-1; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                elif wy: im=base+Y-1; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                elif yMinBC is not None: v+=(_BCValue(yMinBC,x)-c)*di*sy
                d[i]+=v

    def _Diffusion3Ddisc(self,rates,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z; f=self._field; d=self._deltas; sx=self._dtdxSq; sy=self._dtdySq; sz=self._dtdzSq; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0; wz=self._wrap[2]!=0
        for x in range(X):
            xb=x*YZ
            for y in range(Y):
                base=xb+y*Z
                for z in range(Z):
                    i=base+z; di=rates[i]
                    if di<0: continue
                    c=f[i]; v=0.0; xbi=y*Z+z; ybi=x*Z+z; zbi=x*Y+y
                    if x+1<X: ip=i+YZ; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                    elif wx: ip=y*Z+z; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                    elif xMaxBC is not None: v+=(_BCValue(xMaxBC,xbi)-c)*di*sx
                    if x>0: im=i-YZ; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                    elif wx: im=(X-1)*YZ+y*Z+z; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sx if di>0 and dj>0 else 0.0
                    elif xMinBC is not None: v+=(_BCValue(xMinBC,xbi)-c)*di*sx
                    if y+1<Y: ip=i+Z; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                    elif wy: ip=xb+z; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                    elif yMaxBC is not None: v+=(_BCValue(yMaxBC,ybi)-c)*di*sy
                    if y>0: im=i-Z; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                    elif wy: im=xb+(Y-1)*Z+z; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sy if di>0 and dj>0 else 0.0
                    elif yMinBC is not None: v+=(_BCValue(yMinBC,ybi)-c)*di*sy
                    if z+1<Z: ip=i+1; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sz if di>0 and dj>0 else 0.0
                    elif wz: ip=base; dj=rates[ip]; v+=(f[ip]-c)*(2.0*di*dj/(di+dj))*sz if di>0 and dj>0 else 0.0
                    elif zMaxBC is not None: v+=(_BCValue(zMaxBC,zbi)-c)*di*sz
                    if z>0: im=i-1; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sz if di>0 and dj>0 else 0.0
                    elif wz: im=base+Z-1; dj=rates[im]; v+=(f[im]-c)*(2.0*di*dj/(di+dj))*sz if di>0 and dj>0 else 0.0
                    elif zMinBC is not None: v+=(_BCValue(zMinBC,zbi)-c)*di*sz
                    d[i]+=v

    def DiffusionInterfaces(self,rateConstantsX,rateConstantsY=None,rateConstantsZ=None,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        dim=len(self._dimensions); self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        if rateConstantsX.ndim!=dim: raise ValueError("rateConstantsX shape must match grid dimensions")
        for axis in range(dim):
            if rateConstantsX.shape[axis]!=self._dimensions[axis]: raise ValueError("rateConstantsX shape must match grid dimensions")
        for d in rateConstantsX.flat:
            if not np.isfinite(d) or d<0: raise ValueError("rate constants must be finite and nonnegative")
        if dim==1:
            if rateConstantsY is not None or rateConstantsZ is not None: raise ValueError("1D DiffusionInterfaces takes only rateConstantsX")
        elif dim==2:
            if rateConstantsY is None or rateConstantsZ is not None: raise ValueError("2D DiffusionInterfaces requires rateConstantsX and rateConstantsY")
            if rateConstantsY.ndim!=dim: raise ValueError("rateConstantsY shape must match grid dimensions")
            for axis in range(dim):
                if rateConstantsY.shape[axis]!=self._dimensions[axis]: raise ValueError("rateConstantsY shape must match grid dimensions")
            for d in rateConstantsY.flat:
                if not np.isfinite(d) or d<0: raise ValueError("rate constants must be finite and nonnegative")
        else:
            if rateConstantsY is None or rateConstantsZ is None: raise ValueError("3D DiffusionInterfaces requires rateConstantsX, rateConstantsY, and rateConstantsZ")
            if rateConstantsY.ndim!=dim or rateConstantsZ.ndim!=dim: raise ValueError("rate constant shapes must match grid dimensions")
            for axis in range(dim):
                if rateConstantsY.shape[axis]!=self._dimensions[axis] or rateConstantsZ.shape[axis]!=self._dimensions[axis]: raise ValueError("rate constant shapes must match grid dimensions")
            for d in rateConstantsY.flat:
                if not np.isfinite(d) or d<0: raise ValueError("rate constants must be finite and nonnegative")
            for d in rateConstantsZ.flat:
                if not np.isfinite(d) or d<0: raise ValueError("rate constants must be finite and nonnegative")
        rx=rateConstantsX.reshape(self._length); ry=_FlatOptional(rateConstantsY,self._length); rz=_FlatOptional(rateConstantsZ,self._length)
        self._CheckDiffusionInterfacesStability(rx,ry,rz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        self.DiffusionInterfaces_(rateConstantsX,rateConstantsY,rateConstantsZ,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def _CheckDiffusionInterfacesStability(self,rx,ry,rz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        dim=len(self._dimensions)
        for x in range(self.Xdim()):
            for y in range(self.Ydim_() if dim>1 else 1):
                for z in range(self.Zdim_() if dim>2 else 1):
                    i=self.ToI_(x,y,z) if dim==3 else (self.ToI_(x,y) if dim==2 else x); out=0.0
                    if x<self.Xdim()-1 or self._wrap[0] or xMaxBC is not None: out+=rx[i]/(self._dx*self._dx)
                    if x>0 or self._wrap[0]:
                        im=self.ToI_(self.InWrap_(x-1,0),y,z) if dim==3 else (self.ToI_(self.InWrap_(x-1,0),y) if dim==2 else self.InWrap_(x-1,0)); out+=rx[im]/(self._dx*self._dx)
                    elif xMinBC is not None: out+=rx[i]/(self._dx*self._dx)
                    if dim>1:
                        if y<self.Ydim_()-1 or self._wrap[1] or yMaxBC is not None: out+=ry[i]/(self._dy*self._dy)
                        if y>0 or self._wrap[1]:
                            im=self.ToI_(x,self.InWrap_(y-1,1),z) if dim==3 else self.ToI_(x,self.InWrap_(y-1,1)); out+=ry[im]/(self._dy*self._dy)
                        elif yMinBC is not None: out+=ry[i]/(self._dy*self._dy)
                    if dim>2:
                        if z<self.Zdim_()-1 or self._wrap[2] or zMaxBC is not None: out+=rz[i]/(self._dz*self._dz)
                        if z>0 or self._wrap[2]: out+=rz[self.ToI_(x,y,self.InWrap_(z-1,2))]/(self._dz*self._dz)
                        elif zMinBC is not None: out+=rz[i]/(self._dz*self._dz)
                    if self._dt*out>1: raise ValueError("diffusion stability condition violated")

    def DiffusionInterfaces_(self,rateConstantsX,rateConstantsY=None,rateConstantsZ=None,xMinBC=None,xMaxBC=None,yMinBC=None,yMaxBC=None,zMinBC=None,zMaxBC=None):
        dim=len(self._dimensions); rx=rateConstantsX.reshape(self._length); ry=_FlatOptional(rateConstantsY,self._length); rz=_FlatOptional(rateConstantsZ,self._length)
        if dim==1: self._Diffusion1Dfv(rx,xMinBC,xMaxBC)
        elif dim==2: self._Diffusion2Dfv(rx,ry,xMinBC,xMaxBC,yMinBC,yMaxBC)
        else: self._Diffusion3Dfv(rx,ry,rz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)

    def _Diffusion1Dfv(self,rx,xMinBC,xMaxBC):
        X=self.Xdim(); f=self._field; d=self._deltas; s=self._dtdxSq; wrap=self._wrap[0]!=0
        for i in range(X):
            c=f[i]; v=0.0
            if i+1<X: v+=(f[i+1]-c)*rx[i]*s
            elif wrap: v+=(f[0]-c)*rx[i]*s
            elif xMaxBC is not None: v+=(_BCValue(xMaxBC,0)-c)*rx[i]*s
            if i>0: v+=(f[i-1]-c)*rx[i-1]*s
            elif wrap: v+=(f[X-1]-c)*rx[X-1]*s
            elif xMinBC is not None: v+=(_BCValue(xMinBC,0)-c)*rx[i]*s
            d[i]+=v

    def _Diffusion2Dfv(self,rx,ry,xMinBC,xMaxBC,yMinBC,yMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); f=self._field; d=self._deltas; sx=self._dtdxSq; sy=self._dtdySq; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0
        for x in range(X):
            base=x*Y
            for y in range(Y):
                i=base+y; c=f[i]; v=0.0
                if x+1<X: v+=(f[i+Y]-c)*rx[i]*sx
                elif wx: v+=(f[y]-c)*rx[i]*sx
                elif xMaxBC is not None: v+=(_BCValue(xMaxBC,y)-c)*rx[i]*sx
                if x>0: v+=(f[i-Y]-c)*rx[i-Y]*sx
                elif wx: im=(X-1)*Y+y; v+=(f[im]-c)*rx[im]*sx
                elif xMinBC is not None: v+=(_BCValue(xMinBC,y)-c)*rx[i]*sx
                if y+1<Y: v+=(f[i+1]-c)*ry[i]*sy
                elif wy: v+=(f[base]-c)*ry[i]*sy
                elif yMaxBC is not None: v+=(_BCValue(yMaxBC,x)-c)*ry[i]*sy
                if y>0: v+=(f[i-1]-c)*ry[i-1]*sy
                elif wy: im=base+Y-1; v+=(f[im]-c)*ry[im]*sy
                elif yMinBC is not None: v+=(_BCValue(yMinBC,x)-c)*ry[i]*sy
                d[i]+=v

    def _Diffusion3Dfv(self,rx,ry,rz,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC):
        X=self.Xdim(); Y=self.Ydim_(); Z=self.Zdim_(); YZ=Y*Z; f=self._field; d=self._deltas; sx=self._dtdxSq; sy=self._dtdySq; sz=self._dtdzSq; wx=self._wrap[0]!=0; wy=self._wrap[1]!=0; wz=self._wrap[2]!=0
        for x in range(X):
            xb=x*YZ
            for y in range(Y):
                base=xb+y*Z
                for z in range(Z):
                    i=base+z; c=f[i]; v=0.0; xbi=y*Z+z; ybi=x*Z+z; zbi=x*Y+y
                    if x+1<X: v+=(f[i+YZ]-c)*rx[i]*sx
                    elif wx: v+=(f[y*Z+z]-c)*rx[i]*sx
                    elif xMaxBC is not None: v+=(_BCValue(xMaxBC,xbi)-c)*rx[i]*sx
                    if x>0: v+=(f[i-YZ]-c)*rx[i-YZ]*sx
                    elif wx: im=(X-1)*YZ+y*Z+z; v+=(f[im]-c)*rx[im]*sx
                    elif xMinBC is not None: v+=(_BCValue(xMinBC,xbi)-c)*rx[i]*sx
                    if y+1<Y: v+=(f[i+Z]-c)*ry[i]*sy
                    elif wy: v+=(f[xb+z]-c)*ry[i]*sy
                    elif yMaxBC is not None: v+=(_BCValue(yMaxBC,ybi)-c)*ry[i]*sy
                    if y>0: v+=(f[i-Z]-c)*ry[i-Z]*sy
                    elif wy: im=xb+(Y-1)*Z+z; v+=(f[im]-c)*ry[im]*sy
                    elif yMinBC is not None: v+=(_BCValue(yMinBC,ybi)-c)*ry[i]*sy
                    if z+1<Z: v+=(f[i+1]-c)*rz[i]*sz
                    elif wz: v+=(f[base]-c)*rz[i]*sz
                    elif zMaxBC is not None: v+=(_BCValue(zMaxBC,zbi)-c)*rz[i]*sz
                    if z>0: v+=(f[i-1]-c)*rz[i-1]*sz
                    elif wz: im=base+Z-1; v+=(f[im]-c)*rz[im]*sz
                    elif zMinBC is not None: v+=(_BCValue(zMinBC,zbi)-c)*rz[i]*sz
                    d[i]+=v

    def _Delta1DBC(self,centerVal,x,xMinBC,xMaxBC):
        xw=self.InWrap_(x,0)
        if xw!=self._NULL: return (self._field[xw]-centerVal)*self._dtdxSq
        if x<0 and xMinBC is not None: return (_BCValue(xMinBC,0)-centerVal)*self._dtdxSq
        if x>=self.Xdim() and xMaxBC is not None: return (_BCValue(xMaxBC,0)-centerVal)*self._dtdxSq
        return 0.0

    def _DeltaX2DBC(self,centerVal,x,y,xMinBC,xMaxBC):
        xw=self.InWrap_(x,0)
        if xw!=self._NULL: return (self.Get_(xw,y)-centerVal)*self._dtdxSq
        if x<0 and xMinBC is not None: return (_BCValue(xMinBC,y)-centerVal)*self._dtdxSq
        if x>=self.Xdim() and xMaxBC is not None: return (_BCValue(xMaxBC,y)-centerVal)*self._dtdxSq
        return 0.0

    def _DeltaY2DBC(self,centerVal,x,y,yMinBC,yMaxBC):
        yw=self.InWrap_(y,1)
        if yw!=self._NULL: return (self.Get_(x,yw)-centerVal)*self._dtdySq
        if y<0 and yMinBC is not None: return (_BCValue(yMinBC,x)-centerVal)*self._dtdySq
        if y>=self.Ydim_() and yMaxBC is not None: return (_BCValue(yMaxBC,x)-centerVal)*self._dtdySq
        return 0.0

    def _DeltaX3DBC(self,centerVal,x,y,z,xMinBC,xMaxBC):
        xw=self.InWrap_(x,0)
        if xw!=self._NULL: return (self.Get_(xw,y,z)-centerVal)*self._dtdxSq
        bi=y*self.Zdim_()+z
        if x<0 and xMinBC is not None: return (_BCValue(xMinBC,bi)-centerVal)*self._dtdxSq
        if x>=self.Xdim() and xMaxBC is not None: return (_BCValue(xMaxBC,bi)-centerVal)*self._dtdxSq
        return 0.0

    def _DeltaY3DBC(self,centerVal,x,y,z,yMinBC,yMaxBC):
        yw=self.InWrap_(y,1)
        if yw!=self._NULL: return (self.Get_(x,yw,z)-centerVal)*self._dtdySq
        bi=x*self.Zdim_()+z
        if y<0 and yMinBC is not None: return (_BCValue(yMinBC,bi)-centerVal)*self._dtdySq
        if y>=self.Ydim_() and yMaxBC is not None: return (_BCValue(yMaxBC,bi)-centerVal)*self._dtdySq
        return 0.0

    def _DeltaZ3DBC(self,centerVal,x,y,z,zMinBC,zMaxBC):
        zw=self.InWrap_(z,2)
        if zw!=self._NULL: return (self.Get_(x,y,zw)-centerVal)*self._dtdzSq
        bi=x*self.Ydim_()+y
        if z<0 and zMinBC is not None: return (_BCValue(zMinBC,bi)-centerVal)*self._dtdzSq
        if z>=self.Zdim_() and zMaxBC is not None: return (_BCValue(zMaxBC,bi)-centerVal)*self._dtdzSq
        return 0.0

