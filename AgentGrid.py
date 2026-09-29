import numba as nb
from numba import int32,float32
from numba.experimental import jitclass
import numpy as np
"""
base version:
support 1D grids
support off-lattice
"""
"""
expanded version:
support 2D and 3D grids
support on-lattice
support stackable
"""
from QueryList import QueryList

specAgentGrid=[('_dimensions',int32[:]),
               ('_wrap',int32[:]),
               ('_isStackable',int32),
               ('_NULL',int32),
               ('_ALIVE',int32),
               ('_NEXT',int32),
               ('_PREV',int32),
               ('_LOC',int32),
               ('_X',int32),
               ('_Y',int32),
               ('_Z',int32),
               ('_N_BUILTIN_PROPS',int32),
               ('_length',int32),
               ('_mem',float32[:,:]),
               ('_loc',int32[:]),
               ('_counts',int32[:]),
               ('_dead',int32[:]),
               ('_aliveIDs',int32[:]),
               ('_alivePos',int32[:]),
               ('_maxPop',int32),
               ('_totalProps',int32),
               ('_grid',int32[:]),
               ('_nDead',int32),
               ('_pop',int32),
               ]


@jitclass(specAgentGrid)
class AgentGrid(object):
# dimension sizes as a tuple, number of properties for each agent as an integer, whether agents can stack
    def __init__(self,dimensions,numAgentProps,isStackable):
        if len(dimensions)>3: raise Exception("AgentGrid supports at most 3 dimensions")
        if numAgentProps<0 or numAgentProps>2147483647: raise Exception(f"numAgentProps out of int32 range:{numAgentProps}")
        if numAgentProps!=int32(numAgentProps): raise Exception(f"numAgentProps must be an integer:{numAgentProps}")
        if isStackable!=0 and isStackable!=1: raise Exception(f"isStackable must be boolean isStackable:{isStackable}")
        dimensionsRaw=np.array(dimensions)
        length=1
        for i in range(len(dimensionsRaw)):
            dim=dimensionsRaw[i]
            if not np.isfinite(dim): raise Exception(f"dimension size must be finite:{dim}")
            if dim==0: raise Exception("dimension sizes cannot be 0")
            if dim<=-2147483648 or dim>2147483647: raise Exception(f"dimension size out of int32 range:{dim}")
            if dim!=int32(dim): raise Exception(f"dimension sizes must be integers dimension:{dim}")
            length*=abs(dim)
            if length>2147483647: raise Exception(f"grid length exceeds int32 range:{length}")
        self._dimensions=np.array(dimensions,dtype=np.int32)
        self._wrap=np.zeros(len(dimensions),dtype=np.int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._isStackable=int32(isStackable)
        self._NULL=int32(-1)
        self._ALIVE=int32(0)
        propIter=1
        if self._isStackable:
            self._NEXT=propIter
            self._PREV=propIter+1
            propIter+=2
        else:
            self._NEXT=-1
            self._PREV=-1
        if len(self._dimensions)>0:
            self._LOC=-1
            self._X=propIter
            propIter+=1
        if len(self._dimensions)>1:
            self._Y=propIter
            propIter += 1
        else: self._Y=-1
        if len(self._dimensions)>2:
            self._Z=propIter
            propIter += 1
        else: self._Z=-1
        if len(self._dimensions)>0: self._length=int32(np.prod(self._dimensions))
        else: self._length=int32(0)
        self._N_BUILTIN_PROPS=propIter
        self._maxPop=int32(1000)
        self._totalProps= numAgentProps + self._N_BUILTIN_PROPS
        self._mem=np.zeros((self._maxPop, self._totalProps), dtype=np.float32)
        self._loc=np.full(self._maxPop,self._NULL,dtype=np.int32)
        if self._isStackable:
            self._counts=np.zeros((self._length), dtype=np.int32)
        else:
            self._counts=np.zeros(0, dtype=np.int32)
        self._dead=np.zeros(self._maxPop, dtype=np.int32)
        self._FillArr(self._dead, self._NULL)
        self._aliveIDs=np.zeros(self._maxPop,dtype=np.int32)
        self._alivePos=np.full(self._maxPop,self._NULL,dtype=np.int32)
        if len(self._dimensions)>0: 
            self._grid=np.zeros(int32(self._length),dtype=np.int32)
            self._FillArr(self._grid, self._NULL)
        if len(self._dimensions)==0:
            self._grid=np.zeros(0,dtype=np.int32)
            self._LOC=-1
            self._X=-1
        self._nDead=0
        self._pop=0

    def _SetupMem(self,maxPop,copyMem):
        self._maxPop=maxPop
        prevMem=self._mem
        self._mem=np.zeros((self._maxPop, self._totalProps), dtype=float32)
        prevLoc=self._loc
        self._loc=np.full(self._maxPop,self._NULL,dtype=int32)
        prevDead=self._dead
        self._dead=np.full(self._maxPop,self._NULL, dtype=int32)
        prevAliveIDs=self._aliveIDs
        prevAlivePos=self._alivePos
        self._aliveIDs=np.zeros(self._maxPop,dtype=int32)
        self._alivePos=np.full(self._maxPop,self._NULL,dtype=int32)
        if copyMem:
            n=self._pop+self._nDead
            self._mem[:n,:]=prevMem[:n,:]
            self._loc[:n]=prevLoc[:n]
            self._alivePos[:n]=prevAlivePos[:n]
            self._dead[:self._nDead]=prevDead[:self._nDead]
            self._aliveIDs[:self._pop]=prevAliveIDs[:self._pop]


    def _FillArr(self,arr,val):
        for i in range(len(arr)):
            arr[i]=val

#returns the spatial value, with wraparound applied if it is allowed
    def InWrapSQ(self,value,dimension):
        if dimension!=int32(dimension) or dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        if not np.isfinite(value): raise Exception(f"square coordinate must be finite value:{value}")
        if value!=int32(value): raise Exception(f"square coordinate must be an integer value:{value}")
        return self.InWrapSQ_(value,dimension)

    def InWrapSQ_(self,value,dimension):
        if value>=0 and value<self._dimensions[dimension]: return int32(value)
        if self._wrap[dimension]: return int32(value%self._dimensions[dimension])
        return self._NULL

    def InWrap(self,value,dimension):
        if dimension!=int32(dimension) or dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        if not np.isfinite(value): raise Exception(f"coordinate must be finite value:{value}")
        return self.InWrap_(value,dimension)

    def InWrap_(self,value,dimension):
        if value>=0 and value<self._dimensions[dimension]: return value
        if self._wrap[dimension]: return value%self._dimensions[dimension]
        return self._NULL

    def _InWrapCrash(self,value, dimension):
        if value>=0 and value<self._dimensions[dimension]: return value
        if self._wrap[dimension]: return int32(value%self._dimensions[dimension])
        raise Exception("Can't go to location without wrap around!")

#returns the shortest distance between two location values with wraparound applied if it is allowed
    def DispWrap(self,x1,x2,dimension):
        if dimension!=int32(dimension) or dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        if not np.isfinite(x1) or not np.isfinite(x2): raise Exception(f"coordinates must be finite x1:{x1} x2:{x2}")
        if x1<0 or x1>=self._dimensions[dimension] or x2<0 or x2>=self._dimensions[dimension]: raise Exception(f"coordinates out of bounds x1:{x1} x2:{x2}")
        return self.DispWrap_(x1,x2,dimension)

    def DispWrap_(self,x1,x2,dimension):
        dim=self._dimensions[dimension]
        if abs(x2-x1)*2>dim and self._wrap[dimension]:
            if x2>x1:return (x1+dim)-x2
            else:return x1-(x2+dim)
        return x2-x1

    def _SetAgentPositionSQ(self, idx):
        loc=self._loc[idx]
        if len(self._dimensions)==1:
            self._mem[idx,self._X]=loc+0.5
        elif len(self._dimensions)==2:
            self._mem[idx,self._X]=self.ItoX_(loc)+0.5
            self._mem[idx,self._Y]=self.ItoY_(loc)+0.5
        elif len(self._dimensions)==3:
            self._mem[idx,self._X]=self.ItoX_(loc)+0.5
            self._mem[idx,self._Y]=self.ItoY_(loc)+0.5
            self._mem[idx,self._Z]=self.ItoZ_(loc)+0.5

    def _SetAgentPosition1D(self,idx,x):
        self._mem[idx,self._X]=x

    def _SetAgentPosition2D(self,idx,x,y):
        self._mem[idx, self._X] = x
        self._mem[idx, self._Y] = y

    def _SetAgentPosition3D(self,idx,x,y,z):
        self._mem[idx, self._X] = x
        self._mem[idx, self._Y] = y
        self._mem[idx, self._Z] = z

#creates a nonspatial agent, returns agent idx
    def NewAgent(self,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0:
            if x is not None or y is not None or z is not None: raise Exception("NewAgent on a 0D AgentGrid takes no coordinates")
            return self.NewAgent_()
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("NewAgent on a 1D AgentGrid requires x only")
            if not np.isfinite(x): raise Exception(f"coordinate must be finite x:{x}")
            xw=self.InWrap_(x,0)
            if xw==self._NULL: raise Exception(f"attempting to add agent out of bounds x:{x}")
            if not self._isStackable and self._grid[int32(xw)]!=self._NULL: raise Exception("stacking not allowed!")
            return self.NewAgent_(x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("NewAgent on a 2D AgentGrid requires x and y")
            if not np.isfinite(x) or not np.isfinite(y): raise Exception(f"coordinates must be finite x:{x} y:{y}")
            xw=self.InWrap_(x,0);yw=self.InWrap_(y,1)
            if xw==self._NULL or yw==self._NULL: raise Exception(f"attempting to add agent out of bounds x:{x} y:{y}")
            if not self._isStackable and self._grid[self.ToI_(int32(xw),int32(yw))]!=self._NULL: raise Exception("stacking not allowed!")
            return self.NewAgent_(x,y)
        if x is None or y is None or z is None: raise Exception("NewAgent on a 3D AgentGrid requires x, y, and z")
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z): raise Exception(f"coordinates must be finite x:{x} y:{y} z:{z}")
        xw=self.InWrap_(x,0);yw=self.InWrap_(y,1);zw=self.InWrap_(z,2)
        if xw==self._NULL or yw==self._NULL or zw==self._NULL: raise Exception(f"attempting to add agent out of bounds x:{x} y:{y} z:{z}")
        if not self._isStackable and self._grid[self.ToI_(int32(xw),int32(yw),int32(zw))]!=self._NULL: raise Exception("stacking not allowed!")
        return self.NewAgent_(x,y,z)

    def NewAgent_(self,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==0:return self.NewAgent0D_()
        if dim==1:
            xw=self.InWrap_(x,0);newID=self.NewAgent0D_();self._PutAgent(newID,int32(xw));self._SetAgentPosition1D(newID,xw);return newID
        if dim==2:
            xw=self.InWrap_(x,0);yw=self.InWrap_(y,1);newID=self.NewAgent0D_();self._PutAgent(newID,self.ToI_(int32(xw),int32(yw)));self._SetAgentPosition2D(newID,xw,yw);return newID
        xw=self.InWrap_(x,0);yw=self.InWrap_(y,1);zw=self.InWrap_(z,2);newID=self.NewAgent0D_();self._PutAgent(newID,self.ToI_(int32(xw),int32(yw),int32(zw)));self._SetAgentPosition3D(newID,xw,yw,zw);return newID

    def NewAgent0D_(self):
        newID=self._pop
        if self._nDead>0:
            newID=self._dead[self._nDead-1]
            self._nDead-=1
        elif self._maxPop==newID:
            self._SetupMem(self._maxPop*2,1)
        self._aliveIDs[self._pop]=newID
        self._alivePos[newID]=self._pop
        self._pop+=1
        self._mem[newID,self._ALIVE]=1
        return newID

#create a new agent at specified location index, returns agent idx
    def NewAgentSQ(self,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("NewAgentSQ requires a spatial AgentGrid")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("NewAgentSQ on a 1D AgentGrid requires x only")
            if x!=int32(x): raise Exception(f"square coordinate must be an integer x:{x}")
            xw=self.InWrapSQ_(int32(x),0)
            if xw==self._NULL: raise Exception(f"attempting to add agent out of bounds x:{x}")
            if not self._isStackable and self._grid[xw]!=self._NULL: raise Exception("stacking not allowed!")
            return self.NewAgentSQ_(x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("NewAgentSQ on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"square coordinates must be integers x:{x} y:{y}")
            xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1)
            if xw==self._NULL or yw==self._NULL: raise Exception(f"attempting to add agent out of bounds x:{x} y:{y}")
            if not self._isStackable and self._grid[self.ToI_(xw,yw)]!=self._NULL: raise Exception("stacking not allowed!")
            return self.NewAgentSQ_(x,y)
        if x is None or y is None or z is None: raise Exception("NewAgentSQ on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"square coordinates must be integers x:{x} y:{y} z:{z}")
        xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1);zw=self.InWrapSQ_(int32(z),2)
        if xw==self._NULL or yw==self._NULL or zw==self._NULL: raise Exception(f"attempting to add agent out of bounds x:{x} y:{y} z:{z}")
        if not self._isStackable and self._grid[self.ToI_(xw,yw,zw)]!=self._NULL: raise Exception("stacking not allowed!")
        return self.NewAgentSQ_(x,y,z)

    def NewAgentSQ_(self,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:
            xw=self.InWrapSQ_(int32(x),0);newID=self.NewAgent0D_();self._PutAgent(newID,xw);self._SetAgentPosition1D(newID,xw+0.5);return newID
        if dim==2:
            xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1);newID=self.NewAgent0D_();self._PutAgent(newID,xw*self._dimensions[1]+yw);self._SetAgentPosition2D(newID,xw+0.5,yw+0.5);return newID
        xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1);zw=self.InWrapSQ_(int32(z),2);newID=self.NewAgent0D_();self._PutAgent(newID,(xw*self._dimensions[1]+yw)*self._dimensions[2]+zw);self._SetAgentPosition3D(newID,xw+0.5,yw+0.5,zw+0.5);return newID

    def NewAgentI(self,i):
        if len(self._dimensions)==0: raise Exception("NewAgentI requires a spatial AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"attempting to add agent out of bounds i:{i}")
        if not self._isStackable and self._grid[int32(i)]!=self._NULL: raise Exception("stacking not allowed!")
        return self.NewAgentI_(i)

    def NewAgentI_(self, i):
        newID=self.NewAgent0D_()
        self._PutAgent(newID,i)
        self._SetAgentPositionSQ(newID)
        return newID

#create a new agent at specified integer coordinates (at center of square), returns agent idx



#create a new off lattice agent at exact coordinates



#remove agent from grid
    def Dispose(self,idx):
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.Dispose_(idx)

    def Dispose_(self,idx):
        if len(self._dimensions)!=0: self._PopAgent(idx)
        self._mem[idx,self._ALIVE]=0
        self._dead[self._nDead]=idx
        self._nDead+=1
        pos=self._alivePos[idx]
        lastID=self._aliveIDs[self._pop-1]
        self._aliveIDs[pos]=lastID
        self._alivePos[lastID]=pos
        self._alivePos[idx]=self._NULL
        self._pop-=1
        if self._NEXT != -1:
            self._mem[idx, self._NEXT] = self._NULL
        if self._PREV != -1:
            self._mem[idx, self._PREV] = self._NULL
        if len(self._dimensions)!=0:
            self._loc[idx]=self._NULL

#map index to x coordinate
    def ItoX(self,i):
        if len(self._dimensions)!=2 and len(self._dimensions)!=3: raise Exception("ItoX requires a 2D or 3D AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoX_(i)

    def ItoX_(self,i):
        if len(self._dimensions)==2:return int32(int32(i)//self._dimensions[1])
        return int32(int32(i)//(self._dimensions[1]*self._dimensions[2]))

#map index to y coordinate
    def ItoY(self,i):
        if len(self._dimensions)!=2 and len(self._dimensions)!=3: raise Exception("ItoY requires a 2D or 3D AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoY_(i)

    def ItoY_(self,i):
        if len(self._dimensions)==2:return int32(int32(i)%self._dimensions[1])
        return int32((int32(i)//self._dimensions[2])%self._dimensions[1])

    def ItoZ(self,i):
        if len(self._dimensions)!=3: raise Exception("ItoZ requires a 3D AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoZ_(i)

    def ItoZ_(self,i): return int32(int32(i)%self._dimensions[2])

#map coordinates to indices
    def ToI(self,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("ToI requires a spatial AgentGrid")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("ToI on a 1D AgentGrid requires x only")
            if x!=int32(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
            return self.ToI_(x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("ToI on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"coordinates must be integers x:{x} y:{y}")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
            return self.ToI_(x,y)
        if x is None or y is None or z is None: raise Exception("ToI on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"coordinates must be integers x:{x} y:{y} z:{z}")
        if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.ToI_(x,y,z)

    def ToI_(self,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:return int32(x)
        if dim==2:return int32(x)*self._dimensions[1]+int32(y)
        return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

#returns whether agent is alive
    def Alive(self,idx):
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        return self.Alive_(idx)

    def Alive_(self,idx): return self._mem[idx,self._ALIVE]

#returns x,y,z coordinate of agent
    def X(self,idx):
        if len(self._dimensions)<1: raise Exception("X requires a spatial AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.X_(idx)

    def X_(self,idx): return self._mem[idx,self._X]

    def Y(self,idx):
        if len(self._dimensions)<2: raise Exception("Y requires a 2D or 3D AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.Y_(idx)

    def Y_(self,idx): return self._mem[idx,self._Y]

    def Z(self,idx):
        if len(self._dimensions)<3: raise Exception("Z requires a 3D AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.Z_(idx)

    def Z_(self,idx): return self._mem[idx,self._Z]

#returns x,y,z square coordinate of agent
    def XSQ(self,idx):
        if len(self._dimensions)<1: raise Exception("XSQ requires a spatial AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.XSQ_(idx)

    def XSQ_(self,idx): return int32(self._mem[idx,self._X])

    def YSQ(self,idx):
        if len(self._dimensions)<2: raise Exception("YSQ requires a 2D or 3D AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.YSQ_(idx)

    def YSQ_(self,idx): return int32(self._mem[idx,self._Y])

    def ZSQ(self,idx):
        if len(self._dimensions)<3: raise Exception("ZSQ requires a 3D AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.ZSQ_(idx)

    def ZSQ_(self,idx): return int32(self._mem[idx,self._Z])

#returns square index of agent
    def I(self,idx):
        if len(self._dimensions)<1: raise Exception("I requires a spatial AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        return self.I_(idx)

    def I_(self,idx): return int32(self._loc[idx])

#returns x,y,z dimension length of the grid
    def Xdim(self):
        if len(self._dimensions)<1: raise Exception("Xdim requires a spatial AgentGrid")
        return self.Xdim_()

    def Xdim_(self): return self._dimensions[0]
    def Ydim(self):
        if len(self._dimensions)<2: raise Exception("Ydim requires a 2D or 3D AgentGrid")
        return self.Ydim_()

    def Ydim_(self): return self._dimensions[1]
    def Zdim(self):
        if len(self._dimensions)<3: raise Exception("Zdim requires a 3D AgentGrid")
        return self.Zdim_()

    def Zdim_(self): return self._dimensions[2]

    def __len__(self): return self._length

#returns property value for agent
    def GetP(self,idx,propId):
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        if propId!=int32(propId) or propId<0 or propId>=self._totalProps-self._N_BUILTIN_PROPS: raise Exception(f"invalid property id:{propId}")
        return self.GetP_(idx,propId)

    def GetP_(self,idx,propId):
        propId=int32(propId)
        return self._mem[idx,propId+self._N_BUILTIN_PROPS]

#set property value for agent
    def SetP(self,idx,propId,value):
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        if propId!=int32(propId) or propId<0 or propId>=self._totalProps-self._N_BUILTIN_PROPS: raise Exception(f"invalid property id:{propId}")
        return self.SetP_(idx,propId,value)

    def SetP_(self,idx,propId,value):
        propId=int32(propId)
        self._mem[idx,propId+self._N_BUILTIN_PROPS]=value

    def _PutAgent(self,idx,x):
        prev = self._grid[x]
        if self._isStackable:
            self._counts[x]+=1
            if prev!=self._NULL:
                self._mem[prev,self._PREV]=idx
            self._mem[idx,self._PREV]=self._NULL
            self._mem[idx,self._NEXT]=self._grid[x]
            self._loc[idx]=x
            self._grid[x]=idx
        else:
            self._grid[x]=idx
            self._loc[idx]=x

    def _PopAgent(self,idx):
        pos = int32(self._loc[idx])
        if self._isStackable:
            self._counts[pos]-=1
            next=int32(self._mem[idx,self._NEXT])
            prev=int32(self._mem[idx,self._PREV])
            if self._grid[pos]==idx:
                self._grid[pos]=next
            if next!=self._NULL:
                self._mem[next,self._PREV]=prev
            if prev!=self._NULL:
                self._mem[prev,self._NEXT]=next
        else:
            self._grid[pos]=self._NULL

    def _MoveAgent(self, idx, x):
        if x!=self._loc[idx]:
            self._PopAgent(idx)
            self._PutAgent(idx, x)

#move agent to center of square at specified index
    def MoveSQ(self,idx,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("MoveSQ requires a spatial AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("MoveSQ on a 1D AgentGrid requires x only")
            if x!=int32(x): raise Exception(f"square coordinate must be an integer x:{x}")
            xw=self.InWrapSQ_(int32(x),0)
            if xw==self._NULL: raise Exception(f"attempting to move out of bounds x:{x}")
            if not self._isStackable and self._grid[xw]!=self._NULL and self._grid[xw]!=idx: raise Exception("stacking not allowed!")
            return self.MoveSQ_(idx,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("MoveSQ on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"square coordinates must be integers x:{x} y:{y}")
            xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1)
            if xw==self._NULL or yw==self._NULL: raise Exception(f"attempting to move out of bounds x:{x}, y:{y}")
            loc=self.ToI_(xw,yw)
            if not self._isStackable and self._grid[loc]!=self._NULL and self._grid[loc]!=idx: raise Exception("stacking not allowed!")
            return self.MoveSQ_(idx,x,y)
        if x is None or y is None or z is None: raise Exception("MoveSQ on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"square coordinates must be integers x:{x} y:{y} z:{z}")
        xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1);zw=self.InWrapSQ_(int32(z),2)
        if xw==self._NULL or yw==self._NULL or zw==self._NULL: raise Exception(f"attempting to move out of bounds x:{x}, y:{y}, z:{z}")
        loc=self.ToI_(xw,yw,zw)
        if not self._isStackable and self._grid[loc]!=self._NULL and self._grid[loc]!=idx: raise Exception("stacking not allowed!")
        return self.MoveSQ_(idx,x,y,z)

    def MoveSQ_(self,idx,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:
            xw=self.InWrapSQ_(int32(x),0);self._MoveAgent(idx,xw);self._SetAgentPosition1D(idx,xw+0.5);return
        if dim==2:
            xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1);self._MoveAgent(idx,xw*self._dimensions[1]+yw);self._SetAgentPosition2D(idx,xw+0.5,yw+0.5);return
        xw=self.InWrapSQ_(int32(x),0);yw=self.InWrapSQ_(int32(y),1);zw=self.InWrapSQ_(int32(z),2);self._MoveAgent(idx,(xw*self._dimensions[1]+yw)*self._dimensions[2]+zw);self._SetAgentPosition3D(idx,xw+0.5,yw+0.5,zw+0.5)

    def MoveI(self,idx,i):
        if len(self._dimensions)==0: raise Exception("MoveI requires a spatial AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"attempting to move to index that is out of bounds i:{i}")
        if not self._isStackable and self._grid[int32(i)]!=self._NULL and self._grid[int32(i)]!=idx: raise Exception("stacking not allowed!")
        return self.MoveI_(idx,i)

    def MoveI_(self, idx, i):
        self._MoveAgent(idx,int32(i))
        self._SetAgentPositionSQ(idx)

#move agent to center of coordinates, will apply wrap around to coordinates if enabled



#move agent off-lattice
    def Move(self,idx,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("Move requires a spatial AgentGrid")
        if idx!=int32(idx) or idx<0 or idx>=self._pop+self._nDead: raise Exception(f"invalid agent id:{idx}")
        if self._mem[idx,self._ALIVE]==0: raise Exception(f"agent is not alive id:{idx}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("Move on a 1D AgentGrid requires x only")
            if not np.isfinite(x): raise Exception(f"coordinate must be finite x:{x}")
            xw=self.InWrap_(x,0)
            if xw==self._NULL: raise Exception(f"attempting to move out of bounds x:{x}")
            loc=int32(xw)
            if not self._isStackable and self._grid[loc]!=self._NULL and self._grid[loc]!=idx: raise Exception("stacking not allowed!")
            return self.Move_(idx,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("Move on a 2D AgentGrid requires x and y")
            if not np.isfinite(x) or not np.isfinite(y): raise Exception(f"coordinates must be finite x:{x} y:{y}")
            xw=self.InWrap_(x,0);yw=self.InWrap_(y,1)
            if xw==self._NULL or yw==self._NULL: raise Exception(f"attempting to move out of bounds x:{x}, y:{y}")
            loc=self.ToI_(int32(xw),int32(yw))
            if not self._isStackable and self._grid[loc]!=self._NULL and self._grid[loc]!=idx: raise Exception("stacking not allowed!")
            return self.Move_(idx,x,y)
        if x is None or y is None or z is None: raise Exception("Move on a 3D AgentGrid requires x, y, and z")
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z): raise Exception(f"coordinates must be finite x:{x} y:{y} z:{z}")
        xw=self.InWrap_(x,0);yw=self.InWrap_(y,1);zw=self.InWrap_(z,2)
        if xw==self._NULL or yw==self._NULL or zw==self._NULL: raise Exception(f"attempting to move out of bounds x:{x}, y:{y}, z:{z}")
        loc=self.ToI_(int32(xw),int32(yw),int32(zw))
        if not self._isStackable and self._grid[loc]!=self._NULL and self._grid[loc]!=idx: raise Exception("stacking not allowed!")
        return self.Move_(idx,x,y,z)

    def Move_(self,idx,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:
            xw=self.InWrap_(x,0);self._MoveAgent(idx,int32(xw));self._SetAgentPosition1D(idx,xw);return
        if dim==2:
            xw=self.InWrap_(x,0);yw=self.InWrap_(y,1);self._MoveAgent(idx,self.ToI_(int32(xw),int32(yw)));self._SetAgentPosition2D(idx,xw,yw);return
        xw=self.InWrap_(x,0);yw=self.InWrap_(y,1);zw=self.InWrap_(z,2);self._MoveAgent(idx,self.ToI_(int32(xw),int32(yw),int32(zw)));self._SetAgentPosition3D(idx,xw,yw,zw)




#returns the most recent agent that moved to specified location
    def GetLastI(self,i):
        if len(self._dimensions)==0: raise Exception("GetLastI requires a spatial AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.GetLastI_(i)

    def GetLastI_(self,i): return self._grid[int32(i)]

    def GetLast(self,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("GetLast requires a spatial AgentGrid")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("GetLast on a 1D AgentGrid requires x only")
            if x!=int32(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
            return self.GetLast_(x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("GetLast on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"coordinates must be integers x:{x} y:{y}")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
            return self.GetLast_(x,y)
        if x is None or y is None or z is None: raise Exception("GetLast on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"coordinates must be integers x:{x} y:{y} z:{z}")
        if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.GetLast_(x,y,z)

    def GetLast_(self,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:return self._grid[int32(x)]
        if dim==2:return self._grid[self.ToI_(x,y)]
        return self._grid[self.ToI_(x,y,z)]

#returns number of agents at specified location
    def CountI(self,i):
        if len(self._dimensions)==0: raise Exception("CountI requires a spatial AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.CountI_(i)

    def CountI_(self,i):
        i=int32(i)
        if self._isStackable:return self._counts[i]
        return int32(self._grid[i]!=self._NULL)

    def Count(self,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("Count requires a spatial AgentGrid")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("Count on a 1D AgentGrid requires x only")
            if x!=int32(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
            return self.Count_(x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("Count on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"coordinates must be integers x:{x} y:{y}")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
            return self.Count_(x,y)
        if x is None or y is None or z is None: raise Exception("Count on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"coordinates must be integers x:{x} y:{y} z:{z}")
        if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.Count_(x,y,z)

    def Count_(self,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:return self.CountI_(x)
        if dim==2:return self.CountI_(self.ToI_(x,y))
        return self.CountI_(self.ToI_(x,y,z))



#adds all agents at specified location to search list
    def AddI(self,searchListOut,i):
        if len(self._dimensions)==0: raise Exception("AddI requires a spatial AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.AddI_(searchListOut,i)

    def AddI_(self,searchListOut,i):
        i=int32(i);curr=self._grid[i]
        if curr==self._NULL:return searchListOut
        addN=self._counts[i] if self._isStackable else 1
        need=searchListOut._length+addN
        if need>len(searchListOut._mem):
            newLen=len(searchListOut._mem)*2
            if newLen<need:newLen=need
            searchListOut._SetupArr(newLen,1)
        if self._isStackable:
            while curr!=self._NULL:
                searchListOut._mem[searchListOut._length]=curr;searchListOut._length+=1
                curr=int32(self._mem[curr,self._NEXT])
        else:
            searchListOut._mem[searchListOut._length]=curr;searchListOut._length+=1
        return searchListOut

#clears searchlist, adds all agents at specified location to search list
    def GetI(self,searchListOut,i):
        if len(self._dimensions)==0: raise Exception("GetI requires a spatial AgentGrid")
        if i!=int32(i) or i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.GetI_(searchListOut,i)

    def GetI_(self,searchListOut,i):
        searchListOut.Clear()
        return self.AddI_(searchListOut,i)

    def Add(self,searchListOut,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("Add requires a spatial AgentGrid")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("Add on a 1D AgentGrid requires x only")
            if x!=int32(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
            return self.Add_(searchListOut,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("Add on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"coordinates must be integers x:{x} y:{y}")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
            return self.Add_(searchListOut,x,y)
        if x is None or y is None or z is None: raise Exception("Add on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"coordinates must be integers x:{x} y:{y} z:{z}")
        if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.Add_(searchListOut,x,y,z)

    def Add_(self,searchListOut,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        if dim==1:return self.AddI_(searchListOut,x)
        if dim==2:return self.AddI_(searchListOut,self.ToI_(x,y))
        return self.AddI_(searchListOut,self.ToI_(x,y,z))


    def Get(self,searchListOut,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("Get requires a spatial AgentGrid")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("Get on a 1D AgentGrid requires x only")
            if x!=int32(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
            return self.Get_(searchListOut,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("Get on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception(f"coordinates must be integers x:{x} y:{y}")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
            return self.Get_(searchListOut,x,y)
        if x is None or y is None or z is None: raise Exception("Get on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception(f"coordinates must be integers x:{x} y:{y} z:{z}")
        if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.Get_(searchListOut,x,y,z)

    def Get_(self,searchListOut,x=-1,y=-1,z=-1):
        searchListOut.Clear();return self.Add_(searchListOut,x,y,z)






    def AddBox(self,searchListOut,x1,x2,y1=None,y2=None,z1=None,z2=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("AddBox requires a spatial AgentGrid")
        if dim==1:
            if y1 is not None or y2 is not None or z1 is not None or z2 is not None: raise Exception("AddBox on a 1D AgentGrid requires x1 and x2 only")
            if x1!=int32(x1) or x2!=int32(x2): raise Exception(f"box bounds must be integers x1:{x1} x2:{x2}")
            if x2<x1: raise Exception(f"box upper bound must be >= lower bound x1:{x1} x2:{x2}")
            return self.AddBox_(searchListOut,x1,x2)
        if dim==2:
            if y1 is None or y2 is None or z1 is not None or z2 is not None: raise Exception("AddBox on a 2D AgentGrid requires x1, x2, y1, and y2")
            if x1!=int32(x1) or x2!=int32(x2) or y1!=int32(y1) or y2!=int32(y2): raise Exception("box bounds must be integers")
            if x2<x1 or y2<y1: raise Exception("box upper bounds must be >= lower bounds")
            return self.AddBox_(searchListOut,x1,x2,y1,y2)
        if y1 is None or y2 is None or z1 is None or z2 is None: raise Exception("AddBox on a 3D AgentGrid requires x1, x2, y1, y2, z1, and z2")
        if x1!=int32(x1) or x2!=int32(x2) or y1!=int32(y1) or y2!=int32(y2) or z1!=int32(z1) or z2!=int32(z2): raise Exception("box bounds must be integers")
        if x2<x1 or y2<y1 or z2<z1: raise Exception("box upper bounds must be >= lower bounds")
        return self.AddBox_(searchListOut,x1,x2,y1,y2,z1,z2)

    def AddBox_(self,searchListOut,x1,x2,y1=-1,y2=-1,z1=-1,z2=-1):
        dim=len(self._dimensions)
        if dim==1:
            for x in range(x1,x2):
                xw=self.InWrapSQ_(x,0)
                if xw!=self._NULL:self.AddI_(searchListOut,xw)
            return searchListOut
        yd=self._dimensions[1]
        if dim==2:
            for x in range(x1,x2):
                xw=self.InWrapSQ_(x,0)
                if xw!=self._NULL:
                    row=xw*yd
                    for y in range(y1,y2):
                        yw=self.InWrapSQ_(y,1)
                        if yw!=self._NULL:self.AddI_(searchListOut,row+yw)
            return searchListOut
        zd=self._dimensions[2]
        for x in range(x1,x2):
            xw=self.InWrapSQ_(x,0)
            if xw!=self._NULL:
                plane=xw*yd*zd
                for y in range(y1,y2):
                    yw=self.InWrapSQ_(y,1)
                    if yw!=self._NULL:
                        row=plane+yw*zd
                        for z in range(z1,z2):
                            zw=self.InWrapSQ_(z,2)
                            if zw!=self._NULL:self.AddI_(searchListOut,row+zw)
        return searchListOut


    def GetBox(self,searchListOut,x1,x2,y1=None,y2=None,z1=None,z2=None):
        searchListOut.Clear();return self.AddBox(searchListOut,x1,x2,y1,y2,z1,z2)

    def GetBox_(self,searchListOut,x1,x2,y1=-1,y2=-1,z1=-1,z2=-1):
        searchListOut.Clear();return self.AddBox_(searchListOut,x1,x2,y1,y2,z1,z2)






    def _InclusionCriteria(self,searchListOut,loc,inclusionCriteria):
        if inclusionCriteria<0:searchListOut.Add(loc)
        elif inclusionCriteria==0:
            if self._grid[loc]==self._NULL:searchListOut.Add(loc)
        elif inclusionCriteria>0:
            if self._grid[loc]!=self._NULL:searchListOut.Add(loc)
        return searchListOut

#put location indicies that are valid into searchlist
#inclusion criteria: -1: ignore, 0: emptyOnly, 1: occupiedOnly
    def MapHood(self,searchListOut,inclusionCriteria,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("MapHood requires a spatial AgentGrid")
        if searchListOut._hoodDim!=dim: raise Exception(f"MapHood requires a neighborhood matching AgentGrid dimension:{dim} neighborhood dimension:{searchListOut._hoodDim}")
        if inclusionCriteria!=-1 and inclusionCriteria!=0 and inclusionCriteria!=1: raise Exception(f"invalid inclusion criteria:{inclusionCriteria}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("MapHood on a 1D AgentGrid requires x only")
            if x!=int32(x): raise Exception(f"center must be an integer x:{x}")
            return self.MapHood_(searchListOut,inclusionCriteria,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("MapHood on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception("center coordinates must be integers")
            return self.MapHood_(searchListOut,inclusionCriteria,x,y)
        if x is None or y is None or z is None: raise Exception("MapHood on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception("center coordinates must be integers")
        return self.MapHood_(searchListOut,inclusionCriteria,x,y,z)

    def MapHood_(self,searchListOut,inclusionCriteria,x=-1,y=-1,z=-1):
        dim=len(self._dimensions);searchListOut.Clear()
        # Linear-offset fast path for neighborhoods wholly inside the domain.
        interior=(x+searchListOut._hoodMin[0]>=0 and x+searchListOut._hoodMax[0]<self._dimensions[0])
        if dim>1:interior=interior and y+searchListOut._hoodMin[1]>=0 and y+searchListOut._hoodMax[1]<self._dimensions[1]
        if dim>2:interior=interior and z+searchListOut._hoodMin[2]>=0 and z+searchListOut._hoodMax[2]<self._dimensions[2]
        if interior:
            yd=self._dimensions[1] if dim>1 else 1;zd=self._dimensions[2] if dim>2 else 1
            if searchListOut._hoodGridY!=yd or searchListOut._hoodGridZ!=zd:
                if dim==1:
                    for k in range(searchListOut._hoodLen):searchListOut._hoodLinear[k]=searchListOut._hood[k]
                elif dim==2:
                    for k in range(searchListOut._hoodLen):
                        j=k*2;searchListOut._hoodLinear[k]=searchListOut._hood[j]*yd+searchListOut._hood[j+1]
                else:
                    for k in range(searchListOut._hoodLen):
                        j=k*3;searchListOut._hoodLinear[k]=(searchListOut._hood[j]*yd+searchListOut._hood[j+1])*zd+searchListOut._hood[j+2]
                searchListOut._hoodGridY=yd;searchListOut._hoodGridZ=zd
            center=x if dim==1 else x*yd+y if dim==2 else (x*yd+y)*zd+z
            need=searchListOut._length+searchListOut._hoodLen
            if need>len(searchListOut._mem):
                newLen=len(searchListOut._mem)*2
                if newLen<need:newLen=need
                searchListOut._SetupArr(newLen,1)
            if inclusionCriteria<0:
                for k in range(searchListOut._hoodLen):
                    searchListOut._mem[searchListOut._length]=center+searchListOut._hoodLinear[k];searchListOut._length+=1
            elif inclusionCriteria==0:
                for k in range(searchListOut._hoodLen):
                    loc=center+searchListOut._hoodLinear[k]
                    if self._grid[loc]==self._NULL:searchListOut._mem[searchListOut._length]=loc;searchListOut._length+=1
            else:
                for k in range(searchListOut._hoodLen):
                    loc=center+searchListOut._hoodLinear[k]
                    if self._grid[loc]!=self._NULL:searchListOut._mem[searchListOut._length]=loc;searchListOut._length+=1
            return searchListOut
        # Boundary/wrap fallback.
        for i in range(searchListOut._hoodLen):
            newX=self.InWrapSQ_(searchListOut._hood[i*dim]+x,0)
            if dim==1:
                if newX!=self._NULL:self._InclusionCriteria(searchListOut,newX,inclusionCriteria)
            elif dim==2:
                newY=self.InWrapSQ_(searchListOut._hood[i*dim+1]+y,1)
                if newX!=self._NULL and newY!=self._NULL:self._InclusionCriteria(searchListOut,self.ToI_(newX,newY),inclusionCriteria)
            else:
                newY=self.InWrapSQ_(searchListOut._hood[i*dim+1]+y,1);newZ=self.InWrapSQ_(searchListOut._hood[i*dim+2]+z,2)
                if newX!=self._NULL and newY!=self._NULL and newZ!=self._NULL:self._InclusionCriteria(searchListOut,self.ToI_(newX,newY,newZ),inclusionCriteria)
        return searchListOut




#put all agents in neighborhood centered around center coordinates into searchlist
    def AddHood(self,searchListOut,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("AddHood requires a spatial AgentGrid")
        if searchListOut._hoodDim!=dim: raise Exception(f"AddHood requires a neighborhood matching AgentGrid dimension:{dim} neighborhood dimension:{searchListOut._hoodDim}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("AddHood on a 1D AgentGrid requires x only")
            if x!=int32(x): raise Exception(f"center must be an integer x:{x}")
            return self.AddHood_(searchListOut,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("AddHood on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception("center coordinates must be integers")
            return self.AddHood_(searchListOut,x,y)
        if x is None or y is None or z is None: raise Exception("AddHood on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception("center coordinates must be integers")
        return self.AddHood_(searchListOut,x,y,z)

    def AddHood_(self,searchListOut,x=-1,y=-1,z=-1):
        dim=len(self._dimensions)
        interior=(x+searchListOut._hoodMin[0]>=0 and x+searchListOut._hoodMax[0]<self._dimensions[0])
        if dim>1:interior=interior and y+searchListOut._hoodMin[1]>=0 and y+searchListOut._hoodMax[1]<self._dimensions[1]
        if dim>2:interior=interior and z+searchListOut._hoodMin[2]>=0 and z+searchListOut._hoodMax[2]<self._dimensions[2]
        if interior:
            yd=self._dimensions[1] if dim>1 else 1;zd=self._dimensions[2] if dim>2 else 1
            if searchListOut._hoodGridY!=yd or searchListOut._hoodGridZ!=zd:
                if dim==1:
                    for k in range(searchListOut._hoodLen):searchListOut._hoodLinear[k]=searchListOut._hood[k]
                elif dim==2:
                    for k in range(searchListOut._hoodLen):
                        j=k*2;searchListOut._hoodLinear[k]=searchListOut._hood[j]*yd+searchListOut._hood[j+1]
                else:
                    for k in range(searchListOut._hoodLen):
                        j=k*3;searchListOut._hoodLinear[k]=(searchListOut._hood[j]*yd+searchListOut._hood[j+1])*zd+searchListOut._hood[j+2]
                searchListOut._hoodGridY=yd;searchListOut._hoodGridZ=zd
            center=x if dim==1 else x*yd+y if dim==2 else (x*yd+y)*zd+z
            for k in range(searchListOut._hoodLen):self.AddI_(searchListOut,center+searchListOut._hoodLinear[k])
            return searchListOut
        if dim==1:
            for i in range(searchListOut._hoodLen):
                xw=self.InWrapSQ_(x+searchListOut._hood[i],0)
                if xw!=self._NULL:self.AddI_(searchListOut,xw)
        elif dim==2:
            yd=self._dimensions[1]
            for i in range(searchListOut._hoodLen):
                j=i*2;xw=self.InWrapSQ_(x+searchListOut._hood[j],0);yw=self.InWrapSQ_(y+searchListOut._hood[j+1],1)
                if xw!=self._NULL and yw!=self._NULL:self.AddI_(searchListOut,xw*yd+yw)
        else:
            yd=self._dimensions[1];zd=self._dimensions[2]
            for i in range(searchListOut._hoodLen):
                j=i*3;xw=self.InWrapSQ_(x+searchListOut._hood[j],0);yw=self.InWrapSQ_(y+searchListOut._hood[j+1],1);zw=self.InWrapSQ_(z+searchListOut._hood[j+2],2)
                if xw!=self._NULL and yw!=self._NULL and zw!=self._NULL:self.AddI_(searchListOut,(xw*yd+yw)*zd+zw)
        return searchListOut


    def GetHood(self,searchListOut,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("GetHood requires a spatial AgentGrid")
        if searchListOut._hoodDim!=dim: raise Exception(f"GetHood requires a neighborhood matching AgentGrid dimension:{dim} neighborhood dimension:{searchListOut._hoodDim}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("GetHood on a 1D AgentGrid requires x only")
            if x!=int32(x): raise Exception(f"center must be an integer x:{x}")
            return self.GetHood_(searchListOut,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("GetHood on a 2D AgentGrid requires x and y")
            if x!=int32(x) or y!=int32(y): raise Exception("center coordinates must be integers")
            return self.GetHood_(searchListOut,x,y)
        if x is None or y is None or z is None: raise Exception("GetHood on a 3D AgentGrid requires x, y, and z")
        if x!=int32(x) or y!=int32(y) or z!=int32(z): raise Exception("center coordinates must be integers")
        return self.GetHood_(searchListOut,x,y,z)

    def GetHood_(self,searchListOut,x=-1,y=-1,z=-1):
        searchListOut.Clear();return self.AddHood_(searchListOut,x,y,z)






#adds all agents within radius of center coordinates to searchlist
    def AddInRadius(self,searchListOut,radius,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("AddInRadius requires a spatial AgentGrid")
        if not np.isfinite(radius) or radius<0: raise Exception(f"radius must be finite and 0 or positive radius:{radius}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("AddInRadius on a 1D AgentGrid requires x only")
            if not np.isfinite(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"center out of bounds x:{x}")
            return self.AddInRadius_(searchListOut,radius,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("AddInRadius on a 2D AgentGrid requires x and y")
            if not np.isfinite(x) or not np.isfinite(y) or x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"center out of bounds x:{x} y:{y}")
            return self.AddInRadius_(searchListOut,radius,x,y)
        if x is None or y is None or z is None: raise Exception("AddInRadius on a 3D AgentGrid requires x, y, and z")
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"center out of bounds x:{x} y:{y} z:{z}")
        return self.AddInRadius_(searchListOut,radius,x,y,z)

    def AddInRadius_(self,searchListOut,radius,x=-1,y=-1,z=-1):
        dim=len(self._dimensions);startX=int32(np.floor(x-radius));endX=int32(np.floor(x+radius))+1
        if self._wrap[0] and endX-startX>self._dimensions[0]:startX=0;endX=self._dimensions[0]
        if dim==1:
            for xi in range(startX,endX):
                xw=self.InWrapSQ_(xi,0)
                if xw!=self._NULL:
                    curr=self.GetLastI_(xw)
                    if not self._isStackable and curr!=self._NULL:
                        xComp=self.DispWrap_(x,self._mem[curr,self._X],0)
                        if np.abs(xComp)<=radius:searchListOut.Add(curr)
                    else:
                        while curr!=self._NULL:
                            xComp=self.DispWrap_(x,self._mem[curr,self._X],0)
                            if np.abs(xComp)<=radius:searchListOut.Add(curr)
                            curr=int32(self._mem[curr,self._NEXT])
            return searchListOut
        startY=int32(np.floor(y-radius));endY=int32(np.floor(y+radius))+1
        if self._wrap[1] and endY-startY>self._dimensions[1]:startY=0;endY=self._dimensions[1]
        radSq=radius*radius
        if dim==2:
            for xi in range(startX,endX):
                xw=self.InWrapSQ_(xi,0)
                if xw!=self._NULL:
                    for yi in range(startY,endY):
                        yw=self.InWrapSQ_(yi,1)
                        if yw!=self._NULL:
                            curr=self.GetLast_(xw,yw)
                            if not self._isStackable and curr!=self._NULL:
                                xComp=self.DispWrap_(x,self._mem[curr,self._X],0);yComp=self.DispWrap_(y,self._mem[curr,self._Y],1)
                                if xComp*xComp+yComp*yComp<=radSq:searchListOut.Add(curr)
                            else:
                                while curr!=self._NULL:
                                    xComp=self.DispWrap_(x,self._mem[curr,self._X],0);yComp=self.DispWrap_(y,self._mem[curr,self._Y],1)
                                    if xComp*xComp+yComp*yComp<=radSq:searchListOut.Add(curr)
                                    curr=int32(self._mem[curr,self._NEXT])
            return searchListOut
        startZ=int32(np.floor(z-radius));endZ=int32(np.floor(z+radius))+1
        if self._wrap[2] and endZ-startZ>self._dimensions[2]:startZ=0;endZ=self._dimensions[2]
        for xi in range(startX,endX):
            xw=self.InWrapSQ_(xi,0)
            if xw!=self._NULL:
                for yi in range(startY,endY):
                    yw=self.InWrapSQ_(yi,1)
                    if yw!=self._NULL:
                        for zi in range(startZ,endZ):
                            zw=self.InWrapSQ_(zi,2)
                            if zw!=self._NULL:
                                curr=self.GetLast_(xw,yw,zw)
                                if not self._isStackable and curr!=self._NULL:
                                    xComp=self.DispWrap_(x,self._mem[curr,self._X],0);yComp=self.DispWrap_(y,self._mem[curr,self._Y],1);zComp=self.DispWrap_(z,self._mem[curr,self._Z],2)
                                    if xComp*xComp+yComp*yComp+zComp*zComp<=radSq:searchListOut.Add(curr)
                                else:
                                    while curr!=self._NULL:
                                        xComp=self.DispWrap_(x,self._mem[curr,self._X],0);yComp=self.DispWrap_(y,self._mem[curr,self._Y],1);zComp=self.DispWrap_(z,self._mem[curr,self._Z],2)
                                        if xComp*xComp+yComp*yComp+zComp*zComp<=radSq:searchListOut.Add(curr)
                                        curr=int32(self._mem[curr,self._NEXT])
        return searchListOut


    def GetInRadius(self,searchListOut,radius,x=None,y=None,z=None):
        dim=len(self._dimensions)
        if dim==0: raise Exception("GetInRadius requires a spatial AgentGrid")
        if not np.isfinite(radius) or radius<0: raise Exception(f"radius must be finite and 0 or positive radius:{radius}")
        if dim==1:
            if x is None or y is not None or z is not None: raise Exception("GetInRadius on a 1D AgentGrid requires x only")
            if not np.isfinite(x) or x<0 or x>=self._dimensions[0]: raise Exception(f"center out of bounds x:{x}")
            return self.GetInRadius_(searchListOut,radius,x)
        if dim==2:
            if x is None or y is None or z is not None: raise Exception("GetInRadius on a 2D AgentGrid requires x and y")
            if not np.isfinite(x) or not np.isfinite(y) or x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"center out of bounds x:{x} y:{y}")
            return self.GetInRadius_(searchListOut,radius,x,y)
        if x is None or y is None or z is None: raise Exception("GetInRadius on a 3D AgentGrid requires x, y, and z")
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"center out of bounds x:{x} y:{y} z:{z}")
        return self.GetInRadius_(searchListOut,radius,x,y,z)

    def GetInRadius_(self,searchListOut,radius,x=-1,y=-1,z=-1):
        searchListOut.Clear();return self.AddInRadius_(searchListOut,radius,x,y,z)






#returns total live agent population
    def Pop(self): return self._pop

#returns an array of all live agent idxs
    def All(self,shuffle):
        out=self._aliveIDs[:self._pop].copy()
        if shuffle:np.random.shuffle(out)
        return out

#clear searchlist, put all live agents into searchlist
    def AllToList(self,searchListOut,shuffle):
        if len(searchListOut._mem)<self._pop: searchListOut._SetupArr(self._pop,0)
        for j in range(self._pop): searchListOut._mem[j]=self._aliveIDs[j]
        searchListOut._length=self._pop
        if shuffle:searchListOut.Shuffle()
        return searchListOut

