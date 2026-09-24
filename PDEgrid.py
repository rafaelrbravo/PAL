from numba import njit,int32,float32
from numba.experimental import jitclass
import numpy as np

#TODO: setup arbitrary BCs for diffusion!

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

]

@jitclass(specPDE)
class PDEgrid(object):
    def __init__(self,dimensions):
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

    def __getitem__(self,index): return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self): return self._dimensions[1]
    def Zdim(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if len(self._dimensions)==2: return self.ItoX2D(i)
        else: return self.ItoX3D(i)

    def ItoX2D(self,i): return int32(int32(i) / self._dimensions[1])
    def ItoX3D(self,i): return int32(int32(i)/(self._dimensions[1]*self._dimensions[2]))

    def ItoY(self,i):
        if len(self._dimensions)==2: return self.ItoY2D(i)
        else: return self.ItoY3D(i)

    def ItoY2D(self,i): return int32(int32(i) % self._dimensions[1])
    def ItoY3D(self,i): return int32((int32(i)/self._dimensions[2])%self._dimensions[1])

    def ItoZ3D(self,i): return int32(int32(i)%self._dimensions[2])

    def ItoZ(self,i): return int32(int32(i)%self._dimensions[2])

    def XYtoI(self, x, y): return int32(x)*self._dimensions[1]+int32(y)

    def XYZtoI(self, x, y, z): return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

    def Set(self,i,value):
        self._field[i]=value

    def Set1D(self,x,value):
        self._field[x]=value

    def Set2D(self,x,y,value):
        self._field[self.XYtoI(x,y)]=value

    def Set3D(self,x,y,z,value):
        self._field[self.XYZtoI(x,y,z)]=value
    
    def Add(self,i,value):
        self._deltas[i]+=value
    
    def Add1D(self,x,value):
        self._deltas[x]+=value

    def Add2D(self,x,y,value):
        self._deltas[self.XYtoI(x,y)]+=value

    def Add3D(self,x,y,z,value):
        self._deltas[self.XYZtoI(x,y,z)]+=value

    def Get(self,i):
        return self._field[i]

    def Get1D(self,x):
        return self._field[x]

    def Get2D(self,x,y):
        return self._field[self.XYtoI(x,y)]

    def Get3D(self,x,y,z):
        return self._field[self.XYZtoI(x,y,z)]

    def Update(self):
        for i in range(self._length):
            self._field[i]+=self._deltas[i]
            self._deltas[i]=0

    def InWrap(self,idx, dimension):
            if idx>=0 and idx<self._dimensions[dimension]: return idx
            if self._wrap[dimension]: return int32(idx%self._dimensions[dimension])
            return self._NULL

#    def _RecalcSteps(self):
#        self._dxdtHalf=(self._dx/self._dt)/2
#        self._dxSqdt=self._dx**2/self._dt
#        self._dydtHalf=(self._dy/self._dt)/2
#        self._dySqdt=self._dy**2/self._dt
#        self._dzdtHalf=(self._dz/self._dt)/2
#        self._dzSqdt=self._dz**2/self._dt
    def _RecalcSteps(self):
        self._dtdxHalf=(self._dt/self._dt)/2
        self._dtdxSq=self._dt/self._dx**2
        self._dtdyHalf=(self._dt/self._dy)/2
        self._dtdySq=self._dt/self._dy**2
        self._dtdzHalf=(self._dt/self._dz)/2
        self._dtdzSq=self._dt/self._dz**2

    def GetVoxelVol(self):
        if len(self._dimensions)==1: return self._dx
        if len(self._dimensions)==2: return self._dx*self._dy
        if len(self._dimensions)==3: return self._dx*self._dy*self._dz

    def SetDxDyDzDt(self,dx,dy,dz,dt):
        self._dx=dx
        self._dy=dy
        self._dz=dz
        self._dt=dt
        self._RecalcSteps()
    def SetDxDt(self,dx,dt):
        self._dx=dx
        self._dt=dt
        self._RecalcSteps()
    def SetDxDyDt(self,dx,dy,dt):
        self._dx=dx
        self._dy=dy
        self._dt=dt
        self._RecalcSteps()
    def SetSpaceStep(self,val):
        self._dx=val
        self._dy=val
        self._dz=val
        self._RecalcSteps()

    def SetDx(self,val):
        self._dx=val
        self._RecalcSteps()

    def SetDy(self,val):
        self._dy=val
        self._RecalcSteps()

    def SetDz(self,val):
        self._dz=val
        self._RecalcSteps()

    def SetDt(self,val):
        self._dt=val
        self._RecalcSteps()

    def Dx(self):
        return self._dx
        
    def Dy(self):
        return self._dy
    
    def Dz(self):
        return self._dz

    def Dt(self):
        return self._dt

    def GetFieldCopy(self): return np.copy(self._field)#.reshape(np.copy(self._dimensions))
    def GetFieldCopy1D(self): return self.GetFieldCopy()
    def GetFieldCopy2D(self):
        out=np.zeros((self.Xdim(),self.Ydim()))
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                out[x,y]=self.Get2D(x,y)
        return out
    def GetFieldCopy3D(self):
        out=np.zeros((self.Xdim(),self.Ydim(),self.Zdim()))
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                for z in range(self.Zdim()):
                    out[x,y,z]=self.Get3D(x,y,z)
        return out

    def Clear(self):
        for i in range(len(self)):
            self.Set(i,0)
        for i in range(len(self)):
            self._deltas[i]=0
        
    def Diffusion(self,rateConstant):
        if len(self._dimensions)==1:
            self._Diffusion1D(rateConstant)
        elif len(self._dimensions)==2:
            self._Diffusion2D(rateConstant)
        elif len(self._dimensions)==3:
            self._Diffusion3D(rateConstant)

    def _Diffusion1D(self,rateConstant):
        for x in range(self._length):
            self._DiffusionStencil1D(rateConstant,x)

    def _Diffusion2D(self,rateConstant):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                i=self.XYtoI(x,y)
                self._DiffusionStencil2D(rateConstant,i,x,y)

    def _Diffusion3D(self,rateConstant):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                for z in range(self.Zdim()):
                    i=self.XYZtoI(x,y,z)
                    self._DiffusionStencil3D(rateConstant,i,x,y,z)


#    def DiffusionSpherical1Dfv(self,rateConstant):
#        self._deltas[0]+=rateConstant*6*(self._field[1]-self._field[0])*self._dtdxSq
#        for r in range(1,self.Xdim()-1):
#            xp1=(self._field[r+1]-self._field[r])*(r+0.5)**2
#            xm1=(self._field[r]-self._field[r-1])*(r-0.5)**2
#            self._deltas[r]+=(rateConstant*(xp1-xm1))/(r**2)*self._dtdxSq
#        xm1=(self._field[self.Xdim()-1]-self._field[self.Xdim()-2])*(r-0.5)**2
#        self._deltas[self.Xdim()-1]+=(rateConstant*(-xm1))/(self.Xdim()-1**2)*self._dtdxSq


#    def _DiffusionSpherical1DfvTerm(self,rateConstant,vi,ai,cm,cp,dr,dt):
#        return ((rateConstant*ai*dt)/(vi*dr))*((cp-cm))
#
#
#    def DiffusionSpherical1Dfv(self,rateConstant):
#        v0=(4/3)*np.pi*0.5**3
#        a0=4*np.pi*0.5**2
#        self._deltas[0]+=self._DiffusionSpherical1DfvTerm(rateConstant,v0,a0,self._field[0],self._field[1],self._dx,self._dt)
#        for i in range(1,self.Xdim()-1):
#            vi=(4/3)*np.pi*((i+0.5)**3-(i-0.5)**3)
#            ap=4*np.pi*(i+0.5)**2
#            am=4*np.pi*(i-0.5)**2
#            self._deltas[i]+=self._DiffusionSpherical1DfvTerm(rateConstant,vi,ap,self._field[i],self._field[i+1],self._dx,self._dt)-self._DiffusionSpherical1DfvTerm(rateConstant,vi,am,self._field[i-1],self._field[i],self._dx,self._dt)
#        vf=(4/3)*np.pi*((self.Xdim()-0.5)**3-(self.Xdim()-1.5)**3)
#        af=4*np.pi*(self.Xdim()-1.5)**3
#        self._deltas[self.Xdim()-1]+=-self._DiffusionSpherical1DfvTerm(rateConstant,vf,af,self._field[self.Xdim()-2],self._field[self.Xdim()-1],self._dx,self._dt)

    def DiffusionSpherical1Dfv(self,rateConstant):
        scalingFactor=(rateConstant*self._dt)/self._dx
        r=self._dx*0.5
        self._deltas[0]+=(scalingFactor/((1/3)*r**3))*(r**2*(self._field[1]-self._field[0]))
        for i in range(1,self.Xdim()-1):
            rp=(i+0.5)*self._dx
            rm=(i-0.5)*self._dx
            self._deltas[i]+=(scalingFactor/((1/3)*(rp**3-rm**3)))*(rp**2*(self._field[i+1]-self._field[i])-rm**2*(self._field[i]-self._field[i-1]))
        r=self._dx*((self.Xdim()-1))
        rp=(self.Xdim()-0.5)*self._dx
        rm=(self.Xdim()-1.5)*self._dx
        self._deltas[self.Xdim()-1]+=(scalingFactor/((1/3)*(rp**3-rm**3)))*(-rm**2*(self._field[self.Xdim()-1]-self._field[self.Xdim()-2]))

#ORIGINAL PDE
#    def DiffusionSpherical1D(self,rateConstant):
#        self._deltas[0]+=(6*rateConstant*(self._field[1]-self._field[0]))*self._dtdxSq
#        for x in range(1,self.Xdim()-1):
#            lap=(self._field[x+1]-2*self._field[x]+self._field[x-1])*self._dtdxSq
#            deriv=(self._dt/self._dx)*(self._field[x+1]-self._field[x-1])*self._dtdxHalf
#            self._deltas[x]+=rateConstant*(lap+((rateConstant*2)/x)*deriv)
#        lap=(self._field[self.Xdim()-2]-self._field[self.Xdim()-1])*self._dtdxSq
#        deriv=(self._field[self.Xdim()-2]-self._field[self.Xdim()-1])*self._dtdxHalf
#        self._deltas[self.Xdim()-1]+=rateConstant*(lap+((rateConstant*2)/(self.Xdim()-1))*deriv)

#MIGRATING PDE
#    def DiffusionSpherical1D(self,rateConstant):
#        self._deltas[0]+=(6*rateConstant*(self._field[1]-self._field[0]))*self._dtdxSq
#        for x in range(1,self.Xdim()-1):
#            lap=(self._field[x+1]-2*self._field[x]+self._field[x-1])*self._dtdxSq
#            deriv=(self._field[x+1]-self._field[x-1])*self._dtdxHalf
#            self._deltas[x]+=rateConstant*(lap)+rateConstant*(2/(x*self._dx))*deriv
#        lap=(self._field[self.Xdim()-2]-self._field[self.Xdim()-1])*self._dtdxSq
#        deriv=(self._field[self.Xdim()-2]-self._field[self.Xdim()-1])*self._dtdxHalf
#        self._deltas[self.Xdim()-1]+=rateConstant*lap+((rateConstant*2)/(self.Xdim()-1))*deriv

#ATTEMPT AT MAKING WELL-FOUNDED PDE
#    def DiffusionSpherical1D(self,rateConstant):
#        self._deltas[0]+=(6*rateConstant*(self._field[1]-self._field[0]))*self._dtdxSq
#        for x in range(1,self.Xdim()-1):
#            lap=(self._field[x+1]-2*self._field[x]+self._field[x-1])*self._dtdxSq
#            deriv=(self._field[x+1]-self._field[x-1])*self._dtdxHalf
#            self._deltas[x]+=rateConstant*lap+((2*rateConstant)/(x*self._dx))*deriv
#        lap=self._field[self.Xdim()-2]-self._field[self.Xdim()-1]*self._dtdxSq
#        deriv=self._field[self.Xdim()-1]-self._field[self.Xdim()-2]*self._dtdxHalf
#        self._deltas[self.Xdim()-1]+=rateConstant*lap+((2*rateConstant)/(x*self._dx))*deriv

## FROM ARTICLE, DOESN'T MAKE SENSE
#    def DiffusionSpherical1D(self,rateConstant):
#        self._deltas[0]+=(6*rateConstant*(self._field[1]-self._field[0]))*self._dtdxSq
#        for x in range(1,self.Xdim()-1):
#            lap=(self._field[x+1]-2*self._field[x]+self._field[x-1])
#            self._deltas[x]+=(rateConstant/x)*(lap)*self._dtdxSq
#        lap=(self._field[self.Xdim()-2]-self._field[self.Xdim()-1])
#        self._deltas[self.Xdim()-1]+=(rateConstant/(self.Xdim()-1))*(lap)*self._dtdxSq

    def DiffusionBc(self,rateConstant,bc):
        if len(self._dimensions)==1:
            self._Diffusion1Dbc(rateConstant,bc)
        elif len(self._dimensions)==2:
            self._Diffusion2Dbc(rateConstant,bc)
        elif len(self._dimensions)==3:
            self._Diffusion3Dbc(rateConstant,bc)

    def _Diffusion1Dbc(self,rateConstant,bc):
        for x in range(self._length):
            self._DiffusionStencil1Dbc(rateConstant,x,bc)

    def _Diffusion2Dbc(self,rateConstant,bc):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                i=self.XYtoI(x,y)
                self._DiffusionStencil2Dbc(rateConstant,i,x,y,bc)

    def _Diffusion3Dbc(self,rateConstant,bc):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                for z in range(self.Zdim()):
                    i=self.XYZtoI(x,y,z)
                    self._DiffusionStencil3Dbc(rateConstant,i,x,y,z,bc)

    def DiffusionDisc(self,rateConstants):
        if len(self._dimensions)==1:
            self._Diffusion1Ddisc(rateConstants)
        elif len(self._dimensions)==2:
            self._Diffusion2Ddisc(rateConstants)
        elif len(self._dimensions)==3:
            self._Diffusion3Ddisc(rateConstants)

    def _Diffusion1Ddisc(self,rateConstants):
        for x in range(self._length):
            if(rateConstants[x]>=0):
                self._DiffusionStencil1Dbc(rateConstants[x],x,rateConstants)
                self._TaxisStencilOtherField1Dbc(rateConstants,x,rateConstants)

    def _Diffusion2Ddisc(self,rateConstants):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                if(rateConstants[x,y]>=0):
                    i=self.XYtoI(x,y)
                    self._DiffusionStencil2Dbc(rateConstants[x,y],i,x,y,rateConstants)
                    self._TaxisStencilOtherField2Dbc(rateConstants,i,x,y,rateConstants)

    def _Diffusion3Ddisc(self,rateConstants):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                for z in range(self.Ydim()):
                    if(rateConstants[x,y,z]>=0):
                        i=self.XYZtoI(x,y,z)
                        self._DiffusionStencil3Dbc(rateConstants[x,y,z],i,x,y,z,rateConstants)
                        self._TaxisStencil3DOtherFieldbc(rateConstants,i,x,y,z,rateConstants)

    def _DiffusionStencil1D(self,rateConstant,x):
            centerVal=self._field[x]
            deltaSum=self._Delta1D(centerVal,x+1)
            deltaSum+=self._Delta1D(centerVal,x-1)
            self._deltas[x]+=deltaSum*rateConstant

    def _DiffusionStencil1Dbc(self,rateConstant,x,bc):
        centerVal=self._field[x]
        deltaSum=self._Delta1Dbc(centerVal,x+1,bc)
        deltaSum+=self._Delta1Dbc(centerVal,x-1,bc)
        self._deltas[x]+=deltaSum*rateConstant

    def _DiffusionStencil2D(self,rateConstant,i,x,y):
        centerVal=self._field[i]
        deltaSum=self._DeltaX2D(centerVal,x+1,y)
        deltaSum+=self._DeltaX2D(centerVal,x-1,y)
        deltaSum+=self._DeltaY2D(centerVal,x,y+1)
        deltaSum+=self._DeltaY2D(centerVal,x,y-1)
        self._deltas[i]+=deltaSum*rateConstant

    def _DiffusionStencil2Dbc(self,rateConstant,i,x,y,bc):
        centerVal=self._field[i]
        deltaSum=self._DeltaX2Dbc(centerVal,x+1,y,bc)
        deltaSum+=self._DeltaX2Dbc(centerVal,x-1,y,bc)
        deltaSum+=self._DeltaY2Dbc(centerVal,x,y+1,bc)
        deltaSum+=self._DeltaY2Dbc(centerVal,x,y-1,bc)
        self._deltas[i]+=deltaSum*rateConstant

    def _DiffusionStencil3D(self,rateConstant,i,x,y,z):
            centerVal=self._field[i]
            deltaSum=self._DeltaX3D(centerVal,x+1,y,z)
            deltaSum+=self._DeltaX3D(centerVal,x-1,y,z)
            deltaSum+=self._DeltaY3D(centerVal,x,y+1,z)
            deltaSum+=self._DeltaY3D(centerVal,x,y-1,z)
            deltaSum+=self._DeltaZ3D(centerVal,x,y,z+1)
            deltaSum+=self._DeltaZ3D(centerVal,x,y,z-1)
            self._deltas[i]+=deltaSum*rateConstant

    def _DiffusionStencil3Dbc(self,rateConstant,i,x,y,z,bc):
            centerVal=self._field[i]
            deltaSum=self._DeltaX3Dbc(centerVal,x+1,y,z,bc)
            deltaSum+=self._DeltaX3Dbc(centerVal,x-1,y,z,bc)
            deltaSum+=self._DeltaY3Dbc(centerVal,x,y+1,z,bc)
            deltaSum+=self._DeltaY3Dbc(centerVal,x,y-1,z,bc)
            deltaSum+=self._DeltaZ3Dbc(centerVal,x,y,z+1,bc)
            deltaSum+=self._DeltaZ3Dbc(centerVal,x,y,z-1,bc)
            self._deltas[i]+=deltaSum*rateConstant

    def Diffusion1Dfv(self,rateConstantsX):
        for x in range(self._length):
            centerVal=self._field[x]
            deltaSum=self._Delta1D(centerVal,x+1)*rateConstantsX[x]
            deltaSum+=self._Delta1D(centerVal,x-1)*rateConstantsX[x-1]
            self._deltas[x]+=deltaSum

    def Diffusion2Dfv(self,rateConstantsX,rateConstantsY):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                i=self.XYtoI(x,y)
                centerVal=self._field[i]
                deltaSum=self._DeltaX2D(centerVal,x+1,y)*rateConstantsX[x,y]
                deltaSum+=self._DeltaX2D(centerVal,x-1,y)*rateConstantsX[x-1,y]
                deltaSum+=self._DeltaY2D(centerVal,x,y+1)*rateConstantsY[x,y]
                deltaSum+=self._DeltaY2D(centerVal,x,y-1)*rateConstantsY[x,y-1]
                self._deltas[i]+=deltaSum


    def Diffusion3Dfv(self,rateConstantsX,rateConstantsY,rateConstantsZ):
        for x in range(self.Xdim()):
            for y in range(self.Ydim()):
                for z in range(self.Zdim()):
                    i=self.XYZtoI(x,y,z)
                    centerVal=self._field[i]
                    deltaSum=self._DeltaX3D(centerVal,x+1,y,z)*rateConstantsX[x,y,z]
                    deltaSum+=self._DeltaX3D(centerVal,x-1,y,z)*rateConstantsX[x-1,y,z]
                    deltaSum+=self._DeltaY3D(centerVal,x,y+1,z)*rateConstantsY[x,y,z]
                    deltaSum+=self._DeltaY3D(centerVal,x,y-1,z)*rateConstantsY[x,y-1,z]
                    deltaSum+=self._DeltaZ3D(centerVal,x,y,z+1)*rateConstantsZ[x,y,z]
                    deltaSum+=self._DeltaZ3D(centerVal,x,y,z-1)*rateConstantsZ[x,y,z-1]
                    self._deltas[i]+=deltaSum

    def _TaxisStencil1D(self,rateConstant,x):
        self._deltas[x]+=self._GradX1D(x)*rateConstant

    def _TaxisStencilOtherField1D(self,field,x):
        self._deltas[x]+=self._GradX1D(x)*self._GradOtherX1D(field,x)

    def _TaxisStencilOtherField1Dbc(self,field,x,bc):
        self._deltas[x]+=self._GradX1Dbc(x,bc)*self._GradOtherX1Dbc(field,x,bc)

    def _TaxisStencilOtherField2D(self,field,i,x,y):
        self._deltas[i]+=self._GradX2D(x,y)*self._GradOtherX2D(field,x,y)+self._GradY2D(x,y)*self._GradOtherY2D(field,x,y)

    def _TaxisStencilOtherField2Dbc(self,field,i,x,y,bc):
        self._deltas[i]+=self._GradX2Dbc(x,y,bc)*self._GradOtherX2Dbc(field,x,y,bc)+self._GradY2Dbc(x,y,bc)*self._GradOtherY2Dbc(field,x,y,bc)

    def _TaxisStencilOtherField3D(self,field,i,x,y,z):
        self._deltas[i]+=self._GradX3D(x,y,z)*self._GradOtherX3D(field,x,y,z)+self._GradY3D(x,y,z)*self._GradOtherY3D(field,x,y,z)+self._GradZ3D(x,y,z)*self._GradOtherZ3D(field,x,y,z)

    def _TaxisStencil3DOtherFieldbc(self,field,i,x,y,z,bc):
        self._deltas[i]+=self._GradX3Dbc(x,y,z,bc)*self._GradOtherX3Dbc(field,x,y,z,bc)+self._GradY3Dbc(x,y,z,bc)*self._GradOtherY3Dbc(field,x,y,z,bc)+self._GradZ3Dbc(x,y,z,bc)*self._GradOtherZ3Dbc(field,x,y,z,bc)

    def _InWrap1Dbc(self,x,bc):
            if x>=0 and x<self._dimensions[0]: 
                if bc[x]<0:return self._NULL
                return x
            if self._wrap[0]: 
                x=int32(x%self._dimensions[0])
                if bc[x]<0:return self._NULL
                return x
            return self._NULL

    def _InWrap2DXbc(self,x,y,bc):
            if x>=0 and x<self._dimensions[0]: 
                if bc[x,y]<0:return self._NULL
                return x
            if self._wrap[0]: 
                x=int32(x%self._dimensions[0])
                if bc[x,y]<0:return self._NULL
                return x
            return self._NULL

    def _InWrap2DYbc(self,x,y,bc):
            if y>=0 and y<self._dimensions[1]: 
                if bc[x,y]<0:return self._NULL
                return y
            if self._wrap[1]: 
                y=int32(y%self._dimensions[1])
                if bc[x,y]<0:return self._NULL
                return y
            return self._NULL

    def _InWrap3DXbc(self,x,y,z,bc):
            if x>=0 and x<self._dimensions[0]: 
                if bc[x,y,z]<0:return self._NULL
                return x
            if self._wrap[0]: 
                x=int32(x%self._dimensions[0])
                if bc[x,y,z]<0:return self._NULL
                return x
            return self._NULL

    def _InWrap3DYbc(self,x,y,z,bc):
            if y>=0 and y<self._dimensions[1]: 
                if bc[x,y,z]<0:return self._NULL
                return y
            if self._wrap[1]: 
                y=int32(y%self._dimensions[1])
                if bc[x,y,z]<0:return self._NULL
                return y
            return self._NULL

    def _InWrap3DZbc(self,x,y,z,bc):
            if z>=0 and z<self._dimensions[2]: 
                if bc[x,y,z]<0:return self._NULL
                return z
            if self._wrap[2]: 
                z=int32(z%self._dimensions[2])
                if bc[x,y,z]<0:return self._NULL
                return z
            return self._NULL

    def _Delta1D(self,centerVal,x):
        x=self.InWrap(x,0)
        if x==self._NULL:return 0.0
        return (self._field[x]-centerVal)*self._dtdxSq

    def _Delta1Dbc(self,centerVal,x,bc):
        x=self._InWrap1Dbc(x,bc)
        if x==self._NULL:return 0.0
        return (self._field[x]-centerVal)*self._dtdxSq

    def _DeltaX2D(self,centerVal,x,y):
        x=self.InWrap(x,0)
        if x==self._NULL:return 0.0
        return (self.Get2D(x,y)-centerVal)*self._dtdxSq
    
    def _DeltaX2Dbc(self,centerVal,x,y,bc):
        x=self._InWrap2DXbc(x,y,bc)
        if x==self._NULL:return 0.0
        return (self.Get2D(x,y)-centerVal)*self._dtdxSq

    def _DeltaY2D(self,centerVal,x,y):
        y=self.InWrap(y,1)
        if y==self._NULL:return 0.0
        return (self.Get2D(x,y)-centerVal)*self._dtdySq

    def _DeltaY2Dbc(self,centerVal,x,y,bc):
        y=self._InWrap2DYbc(x,y,bc)
        if y==self._NULL:return 0.0
        return (self.Get2D(x,y)-centerVal)*self._dtdySq

    def _DeltaX3D(self,centerVal,x,y,z):
        x=self.InWrap(x,0)
        if x==self._NULL:return 0.0
        return (self.Get3D(x,y,z)-centerVal)*self._dtdxSq

    def _DeltaX3Dbc(self,centerVal,x,y,z,bc):
        x=self._InWrap3DXbc(x,y,z,bc)
        if x==self._NULL:return 0.0
        return (self.Get3D(x,y,z)-centerVal)*self._dtdxSq

    def _DeltaY3D(self,centerVal,x,y,z):
        y=self.InWrap(y,1)
        if y==self._NULL:return 0.0
        return (self.Get3D(x,y,z)-centerVal)*self._dtdySq

    def _DeltaY3Dbc(self,centerVal,x,y,z,bc):
        y=self._InWrap3DYbc(x,y,z,bc)
        if y==self._NULL:return 0.0
        return (self.Get3D(x,y,z)-centerVal)*self._dtdySq

    def _DeltaZ3D(self,centerVal,x,y,z):
        z=self.InWrap(z,2)
        if z==self._NULL:return 0.0
        return (self.Get3D(x,y,z)-centerVal)*self._dtdzSq

    def _DeltaZ3Dbc(self,centerVal,x,y,z,bc):
        z=self._InWrap3DZbc(x,y,z,bc)
        if z==self._NULL:return 0.0
        return (self.Get3D(x,y,z)-centerVal)*self._dtdzSq

    def _GradX1D(self,x):
        xPlus1idx=self.InWrap(x+1,0)
        xMinus1idx=self.InWrap(x-1,0)
        xPlus1=self.Get1D(xPlus1idx) if xPlus1idx!=self._NULL else self.Get1D(x)
        xMinus1=self.Get1D(xMinus1idx) if xMinus1idx!=self._NULL else self.Get1D(x)
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradX1Dbc(self,x,bc):
        xPlus1idx=self._InWrap1Dbc(x+1,bc)
        xMinus1idx=self._InWrap1Dbc(x-1,bc)
        xPlus1=self.Get1D(xPlus1idx) if xPlus1idx!=self._NULL else self.Get1D(x)
        xMinus1=self.Get1D(xMinus1idx) if xMinus1idx!=self._NULL else self.Get1D(x)
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradX2D(self,x,y):
        xPlus1idx=self.InWrap(x+1,0)
        xMinus1idx=self.InWrap(x-1,0)
        xPlus1=self.Get2D(xPlus1idx,y) if xPlus1idx!=self._NULL else self.Get2D(x,y)
        xMinus1=self.Get2D(xMinus1idx,y) if xMinus1idx!=self._NULL else self.Get2D(x,y)
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradX2Dbc(self,x,y,bc):
        xPlus1idx=self._InWrap2DXbc(x+1,y,bc)
        xMinus1idx=self._InWrap2DXbc(x-1,y,bc)
        xPlus1=self.Get2D(xPlus1idx,y) if xPlus1idx!=self._NULL else self.Get2D(x,y)
        xMinus1=self.Get2D(xMinus1idx,y) if xMinus1idx!=self._NULL else self.Get2D(x,y)
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradY2D(self,x,y):
        yPlus1idx=self.InWrap(y+1,1)
        yMinus1idx=self.InWrap(y-1,1)
        yPlus1=self.Get2D(x,yPlus1idx) if yPlus1idx!=self._NULL else self.Get2D(x,y)
        yMinus1=self.Get2D(x,yMinus1idx) if yMinus1idx!=self._NULL else self.Get2D(x,y)
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradY2Dbc(self,x,y,bc):
        yPlus1idx=self._InWrap2DYbc(x,y+1,bc)
        yMinus1idx=self._InWrap2DYbc(x,y-1,bc)
        yPlus1=self.Get2D(x,yPlus1idx) if yPlus1idx!=self._NULL else self.Get2D(x,y)
        yMinus1=self.Get2D(x,yMinus1idx) if yMinus1idx!=self._NULL else self.Get2D(x,y)
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradX3D(self,x,y,z):
        xPlus1idx=self.InWrap(x+1,0)
        xMinus1idx=self.InWrap(x-1,0)
        xPlus1=self.Get3D(xPlus1idx,y,z) if xPlus1idx!=self._NULL else self.Get3D(x,y,z)
        xMinus1=self.Get3D(xMinus1idx,y,z) if xMinus1idx!=self._NULL else self.Get3D(x,y,z)
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradX3Dbc(self,x,y,z,bc):
        xPlus1idx=self._InWrap3DXbc(x+1,y,z,bc)
        xMinus1idx=self._InWrap3DXbc(x-1,y,z,bc)
        xPlus1=self.Get3D(xPlus1idx,y,z) if xPlus1idx!=self._NULL else self.Get3D(x,y,z)
        xMinus1=self.Get3D(xMinus1idx,y,z) if xMinus1idx!=self._NULL else self.Get3D(x,y,z)
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradY3D(self,x,y,z):
        yPlus1idx=self.InWrap(y+1,1)
        yMinus1idx=self.InWrap(y-1,1)
        yPlus1=self.Get3D(x,yPlus1idx,z) if yPlus1idx!=self._NULL else self.Get3D(x,y,z)
        yMinus1=self.Get3D(x,yMinus1idx,z) if yMinus1idx!=self._NULL else self.Get3D(x,y,z)
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradY3Dbc(self,x,y,z,bc):
        yPlus1idx=self._InWrap3DYbc(x,y+1,z,bc)
        yMinus1idx=self._InWrap3DYbc(x,y-1,z,bc)
        yPlus1=self.Get3D(x,yPlus1idx,z) if yPlus1idx!=self._NULL else self.Get3D(x,y,z)
        yMinus1=self.Get3D(x,yMinus1idx,z) if yMinus1idx!=self._NULL else self.Get3D(x,y,z)
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradZ3D(self,x,y,z):
        zPlus1idx=self.InWrap(z+1,2)
        zMinus1idx=self.InWrap(z-1,2)
        zPlus1=self.Get3D(x,y,zPlus1idx) if zPlus1idx!=self._NULL else self.Get3D(x,y,z)
        zMinus1=self.Get3D(x,y,zMinus1idx) if zMinus1idx!=self._NULL else self.Get3D(x,y,z)
        return (zPlus1-zMinus1)*self._dtdzHalf

    def _GradZ3Dbc(self,x,y,z,bc):
        zPlus1idx=self._InWrap3DZbc(x,y,z+1,bc)
        zMinus1idx=self._InWrap3DZbc(x,y,z-1,bc)
        zPlus1=self.Get3D(x,y,zPlus1idx) if zPlus1idx!=self._NULL else self.Get3D(x,y,z)
        zMinus1=self.Get3D(x,y,zMinus1idx) if zMinus1idx!=self._NULL else self.Get3D(x,y,z)
        return (zPlus1-zMinus1)*self._dtdzHalf

    def _GradOtherX1D(self,field,x):
        xPlus1idx=self.InWrap(x+1,0)
        xMinus1idx=self.InWrap(x-1,0)
        xPlus1=field[xPlus1idx] if xPlus1idx!=self._NULL else field[x]
        xMinus1=field[xMinus1idx] if xMinus1idx!=self._NULL else field[x]
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradOtherX1Dbc(self,field,x,bc):
        xPlus1idx=self._InWrap1Dbc(x+1,bc)
        xMinus1idx=self._InWrap1Dbc(x-1,bc)
        xPlus1=field[xPlus1idx] if xPlus1idx!=self._NULL else field[x]
        xMinus1=field[xMinus1idx] if xMinus1idx!=self._NULL else field[x]
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradOtherX2D(self,field,x,y):
        xPlus1idx=self.InWrap(x+1,0)
        xMinus1idx=self.InWrap(x-1,0)
        xPlus1=field[xPlus1idx,y] if xPlus1idx!=self._NULL else field[x,y]
        xMinus1=field[xMinus1idx,y] if xMinus1idx!=self._NULL else field[x,y]
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradOtherX2Dbc(self,field,x,y,bc):
        xPlus1idx=self._InWrap2DXbc(x+1,y,bc)
        xMinus1idx=self._InWrap2DXbc(x-1,y,bc)
        xPlus1=field[xPlus1idx,y] if xPlus1idx!=self._NULL else field[x,y]
        xMinus1=field[xMinus1idx,y] if xMinus1idx!=self._NULL else field[x,y]
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradOtherY2D(self,field,x,y):
        yPlus1idx=self.InWrap(y+1,1)
        yMinus1idx=self.InWrap(y-1,1)
        yPlus1=field[x,yPlus1idx] if yPlus1idx!=self._NULL else field[x,y]
        yMinus1=field[x,yMinus1idx] if yMinus1idx!=self._NULL else field[x,y]
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradOtherY2Dbc(self,field,x,y,bc):
        yPlus1idx=self._InWrap2DYbc(x,y+1,bc)
        yMinus1idx=self._InWrap2DYbc(x,y-1,bc)
        yPlus1=field[x,yPlus1idx] if yPlus1idx!=self._NULL else field[x,y]
        yMinus1=field[x,yMinus1idx] if yMinus1idx!=self._NULL else field[x,y]
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradOtherX3D(self,field,x,y,z):
        xPlus1idx=self.InWrap(x+1,0)
        xMinus1idx=self.InWrap(x-1,0)
        xPlus1=field[xPlus1idx,y,z] if xPlus1idx!=self._NULL else field[x,y,z]
        xMinus1=field[xMinus1idx,y,z] if xMinus1idx!=self._NULL else field[x,y,z]
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradOtherX3Dbc(self,field,x,y,z,bc):
        xPlus1idx=self._InWrap3DXbc(x+1,y,z,bc)
        xMinus1idx=self._InWrap3DXbc(x-1,y,z,bc)
        xPlus1=field[xPlus1idx,y,z] if xPlus1idx!=self._NULL else field[x,y,z]
        xMinus1=field[xMinus1idx,y,z] if xMinus1idx!=self._NULL else field[x,y,z]
        return (xPlus1-xMinus1)*self._dtdxHalf

    def _GradOtherY3D(self,field,x,y,z):
        yPlus1idx=self.InWrap(y+1,1)
        yMinus1idx=self.InWrap(y-1,1)
        yPlus1=field[x,yPlus1idx,z] if yPlus1idx!=self._NULL else field[x,y,z]
        yMinus1=field[x,yMinus1idx,z] if yMinus1idx!=self._NULL else field[x,y,z]
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradOtherY3Dbc(self,field,x,y,z,bc):
        yPlus1idx=self._InWrap3DYbc(x,y+1,z,bc)
        yMinus1idx=self._InWrap3DYbc(x,y-1,z,bc)
        yPlus1=field[x,yPlus1idx,z] if yPlus1idx!=self._NULL else field[x,y,z]
        yMinus1=field[x,yMinus1idx,z] if yMinus1idx!=self._NULL else field[x,y,z]
        return (yPlus1-yMinus1)*self._dtdyHalf

    def _GradOtherZ3D(self,field,x,y,z):
        zPlus1idx=self.InWrap(z+1,2)
        zMinus1idx=self.InWrap(z-1,2)
        zPlus1=field[x,y,zPlus1idx] if zPlus1idx!=self._NULL else field[x,y,z]
        zMinus1=field[x,y,zMinus1idx] if zMinus1idx!=self._NULL else field[x,y,z]
        return (zPlus1-zMinus1)*self._dtdzHalf

    def _GradOtherZ3Dbc(self,field,x,y,z,bc):
        zPlus1idx=self._InWrap3DZbc(x,y,z+1,bc)
        zMinus1idx=self._InWrap3DZbc(x,y,z-1,bc)
        zPlus1=field[x,y,zPlus1idx] if zPlus1idx!=self._NULL else field[x,y,z]
        zMinus1=field[x,y,zMinus1idx] if zMinus1idx!=self._NULL else field[x,y,z]
        return (zPlus1-zMinus1)*self._dtdzHalf

#    def _Taxis1D(self,field):
#        for x in range(len(self.Xdim())):
#            self._TaxisStencil1D(field,x)
#
#    def Taxis1Dgrad(self,gradX):
#        for x in range(len(self.Xdim())):
#            self.Add1D(x,self._GradX1D(x)*gradX[x])
#
#    def _Taxis2D(self,field):
#        for x in range(len(self.Xdim())):
#            for y in range(len(self.Ydim())):
#                i=self.XYtoI(x,y)
#                self._TaxisStencil2D(field,i,x,y)
#
#    def Taxis2Dgrad(self,gradX,gradY):
#        for x in range(len(self.Xdim())):
#            for y in range(len(self.Ydim())):
#                self.Add2D(x,y,self._GradX2D(x,y)*gradX[x,y]+self._GradY2D(x,y)*gradY[x,y])
#
#    def _Taxis3D(self,field):
#        for x in range(len(self.Xdim())):
#            for y in range(len(self.Ydim())):
#                for z in range(len(self.Zdim())):
#                    i=self.XYZtoI(x,y,z)
#                    self._TaxisStencil3D(field,i,x,y,z)
#
#
#    def Taxis3Dgrad(self,gradX,gradY,gradZ):
#        for x in range(len(self.Xdim())):
#            for y in range(len(self.Ydim())):
#                for z in range(len(self.Zdim())):
#                    self.Add3D(x,y,z,self._GradX3D(x,y,z)*gradX[x,y,z]+self._GradY3D(x,y,z)*gradY[x,y,z]+self._GradZ3D(x,y,z)*gradZ[x,y,z])
#
#    def GradToField1DX(self,fieldIn,fieldOut):
#        for x in range(len(self.Xdim)):
#            fieldOut[x]=self._GradX1Dother(fieldIn,x)
#
#    def GradToField2DX(self,fieldIn,fieldOut):
#        for x in range(len(self.Xdim)):
#            for y in range(len(self.Ydim)):
#                fieldOut[x,y]=self._GradX2Dother(fieldIn,x,y)
#
#    def GradToField2DY(self,fieldIn,fieldOut):
#        for x in range(len(self.Xdim)):
#            for y in range(len(self.Ydim)):
#                fieldOut[x,y]=self._GradY2Dother(fieldIn,x,y)
#
#    def GradToField3DX(self,fieldIn,fieldOut):
#        for x in range(len(self.Xdim)):
#            for y in range(len(self.Ydim)):
#                for z in range(len(self.Ydim)):
#                    fieldOut[x,y,z]=self._GradX3Dother(fieldIn,x,y,z)
#
#    def GradToField3DY(self,fieldIn,fieldOut):
#        for x in range(len(self.Xdim)):
#            for y in range(len(self.Ydim)):
#                for z in range(len(self.Ydim)):
#                    fieldOut[x,y,z]=self._GradY3Dother(fieldIn,x,y,z)
#
#    def GradToField3DZ(self,fieldIn,fieldOut):
#        for x in range(len(self.Xdim)):
#            for y in range(len(self.Ydim)):
#                for z in range(len(self.Ydim)):
#                    fieldOut[x,y,z]=self._GradZ3Dother(fieldIn,x,y,z)



#    def GetColors(self,RedFunction,GreenFunction,BlueFunction):
#        out=np.zeros((self._length,3))
#        for i in range(self._length):
#            out[i,0]=RedFunction(self,i)*256
#            out[i,1]=GreenFunction(self,i)*256
#            out[i,1]=BlueFunction(self,i)*256
#        return out

        


#    def DiffusionZeroFlux(self):
#        if len(self._dimensions)==1:



    
#test=PDEgrid((3,))
#test.Add(0,0.5)
#test.DiffusionZeroFlux()
#test.Update()