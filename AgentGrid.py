import numba as nb
from numba import njit
from numba import int32,float32,boolean
from numba.experimental import jitclass
import numpy as np
import time
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

specQueryList=[('_mem', int32[:]),
                ('_hood',int32[:]),
                ('_hoodDim',int32),
                ('_hoodLen',int32),
                ('_length',int32)]

#used to collect agents from spatial queries
@jitclass(specQueryList)
class QueryList(object):
    def __init__(self):
        self._length=0
        self._hoodDim=0
        self._hoodLen=0
        self._SetupArr(100,0)
        self._hood=np.zeros(0,dtype=int32)

    def _SetupArr(self,length,copy):
        newIds=np.full(int32(length),-1,dtype=int32)
        if copy!=0:
            for i in range(self._length):
                newIds[i]=self._mem[i]
        self._mem=newIds


    def __len__(self):
        return self._length

    def __getitem__(self,index):
        return self._mem[index]

#add a single agent id to the searchlist
    def Add(self,idx):
        currLen=len(self._mem)
        if currLen==self._length: self._SetupArr(currLen*2,1)
        self._mem[self._length]=idx
        self._length+=1
        return self


    def Clear(self):
        self._length=0
        return self

#randomizes the search list order
    def Shuffle(self):
        for i in range(self._length-1):
            j=np.random.randint(i,self._length)
            temp=self._mem[i]
            self._mem[i]=self._mem[j]
            self._mem[j]=temp
        return self

#pulls a random id from the searchlist
    def Random(self):
        if self._length==0:return -1
        return self._mem[np.random.randint(self._length)]

#attaches a neighborhood to the searchlist to allow gathering agent ids within neighborhood
    def SetHood(self, dimension, neighborhood):
        self._hoodDim=dimension
        self._hood=np.array(neighborhood,dtype=int32)
        self._hoodLen=len(neighborhood)//self._hoodDim
        return self



#internal mapping: X:-1,Y:-2,Z:-3,I:-4
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
               ('_dead',int32[:]),
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
            self._LOC=propIter
            self._X=propIter+1
            propIter+=2
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
#        self._propertyMap=np.zeros(propIter-1,dtype=int32)
#        for i in range(propIter-1):
#            if i==0:self._propertyMap[i]=self._LOC
#            if i==1:self._propertyMap[i]=self._X
#            if i==2:self._propertyMap[i]=self._Y
#            if i==3:self._propertyMap[i]=self._Z
        self._N_BUILTIN_PROPS=propIter
        self._maxPop=int32(10)
        self._totalProps= numAgentProps + self._N_BUILTIN_PROPS
        self._mem=np.zeros((self._maxPop, self._totalProps), dtype=np.float32)
        self._dead=np.zeros(self._maxPop, dtype=np.int32)
        self._FillArr(self._dead, self._NULL)
        if len(self._dimensions)>0: 
            self._grid=np.zeros(int32(self._length),dtype=np.int32)
            self._FillArr(self._grid, self._NULL)
        if len(self._dimensions)==0:
            self._grid=np.zeros(0,dtype=np.int32)
            self._LOC=-1
            self._X=-1
        self._nDead=0
        self._pop=0
        self._SetupMem(self._maxPop,False)

    def _SetupMem(self,maxPop,copyMem):
        self._maxPop=maxPop
        prevMem=self._mem
        self._mem=np.zeros((self._maxPop, self._totalProps), dtype=float32)
        prevDead=self._dead
        self._dead=np.full(self._maxPop,self._NULL, dtype=int32)
        if copyMem:
            for i in range(self._pop+self._nDead):
                for j in range(self._totalProps):
                    self._mem[i,j]=prevMem[i,j]
            for i in range(self._nDead):
                self._dead[i]=prevDead[i]


    def _FillArr(self,arr,val):
        for i in range(len(arr)):
            arr[i]=val

#returns the spatial value, with wraparound applied if it is allowed
    def InWrapSQ(self,value, dimension):
        if value>=0 and value<self._dimensions[dimension]: return value
        if self._wrap[dimension]: return int32(value%self._dimensions[dimension])
        return self._NULL

    def InWrap(self,value, dimension):
        if value>=0 and value<self._dimensions[dimension]: return value
        if self._wrap[dimension]: return value%self._dimensions[dimension]
        return self._NULL

    def _InWrapCrash(self,value, dimension):
        if value>=0 and value<self._dimensions[dimension]: return value
        if self._wrap[dimension]: return int32(value%self._dimensions[dimension])
        raise Exception("Can't go to location without wrap around!")

#returns the shortest distance between two location values with wraparound applied if it is allowed
    def DispWrap(self, x1, x2, dimension):
        dim=self._dimensions[dimension]
        if abs(x2-x1)*2>dim and self._wrap[dimension]:
            if x2>x1:return (x1+dim)-x2
            else: return x1-(x2+dim)
        return x2-x1

    def _SetAgentPositionSQ(self, idx):
        loc=self._mem[idx,self._LOC]
        if len(self._dimensions)==1:
            self._mem[idx,self._X]=loc+0.5
        elif len(self._dimensions)==2:
            self._mem[idx,self._X]=self.ItoX2D(loc)+0.5
            self._mem[idx,self._Y]=self.ItoY2D(loc)+0.5
        elif len(self._dimensions)==3:
            self._mem[idx,self._X]=self.ItoX3D(loc)+0.5
            self._mem[idx,self._Y]=self.ItoY3D(loc)+0.5
            self._mem[idx,self._Z]=self.ItoZ3D(loc)+0.5

    def _SetAgentPosition1D(self,idx,x):
        self._mem[idx,self._X]=x

    def _SetAgentPosition2D(self,idx,x,y):
        self._mem[idx, self._X] = x
        self._mem[idx, self._Y] = y

    def _SetAgentPosition3D(self,idx,x,y,z):
        self._mem[idx, self._X] = x
        self._mem[idx, self._Y] = y
        self._mem[idx, self._Z] = z

    def _NewAgent(self):
        newID=self._pop
        if self._nDead>0:
            newID=self._dead[self._nDead-1]
            self._nDead-=1
        elif self._maxPop==newID:
            self._SetupMem(self._maxPop*2,1)
        self._pop += 1
        self._mem[newID,self._ALIVE]=1
        return newID

#creates a nonspatial agent, returns agent idx
    def NewAgent(self):
        if len(self._dimensions)!=0: raise Exception("Can't create new nonspatial agent on spatial domain!")
        return self._NewAgent()

#create a new agent at specified location index, returns agent idx
    def NewAgentSQ(self, i):
        if i<0 or i>=self._length:raise Exception(f"attempting to add agent out of bounds x:{i}")
        newID=self._NewAgent()
        self._PutAgent(newID,i)
        self._SetAgentPositionSQ(newID)
        return newID

#create a new agent at specified integer coordinates (at center of square), returns agent idx
    def NewAgentSQ1D(self, x):
        xw=self.InWrapSQ(int32(x),0)
        if xw==self._NULL:raise Exception(f"attempting to add agent out of bounds x:{x}")
        return self.NewAgentSQ(xw)

    def NewAgentSQ2D(self,x,y):
        xw=self.InWrapSQ(int32(x),0)
        yw=self.InWrapSQ(int32(y),1)
        if xw==self._NULL or yw==self._NULL:raise Exception(f"attempting to add agent out of bounds x:{x} y:{y}")
        loc=self.XYtoI(int32(xw),int32(yw))
        return self.NewAgentSQ(loc)

    def NewAgentSQ3D(self,x,y,z):
        xw=self.InWrapSQ(int32(x),0)
        yw=self.InWrapSQ(int32(y),1)
        zw=self.InWrapSQ(int32(z),2)
        if xw==self._NULL or yw==self._NULL or zw==self._NULL:raise Exception(f"attempting to add agent out of bounds x:{x} y:{y} z:{z}")
        loc=self.XYZtoI(int32(xw),int32(yw),int32(zw))
        return self.NewAgentSQ(loc)

#create a new off lattice agent at exact coordinates
    def NewAgent1D(self,x):
        xw=self.InWrap(x,0)
        if xw==self._NULL:raise Exception(f"attempting to add agent out of bounds x:{x}")
        newID=self._NewAgent()
        self._PutAgent(newID,int32(xw))
        self._SetAgentPosition1D(newID,xw)
        return newID

    def NewAgent2D(self,x,y):
        xw=self.InWrap(x,0)
        yw=self.InWrap(y,1)
        if xw==self._NULL or yw==self._NULL:raise Exception(f"attempting to add agent out of bounds x:{x} y:{y}")
        newID=self._NewAgent()
        self._PutAgent(newID,self.XYtoI(int32(xw),int32(yw)))
        self._SetAgentPosition2D(newID,xw,yw)
        return newID

    def NewAgent3D(self,x,y,z):
        xw=self.InWrap(x,0)
        yw=self.InWrap(y,1)
        zw=self.InWrap(z,2)
        if xw==self._NULL or yw==self._NULL or zw==self._NULL:raise Exception(f"attempting to add agent out of bounds x:{x} y:{y} z:{z}")
        newID=self._NewAgent()
        self._PutAgent(newID,self.XYZtoI(int32(xw),int32(yw),int32(zw)))
        self._SetAgentPosition3D(newID,xw,yw,zw)
        return newID

#remove agent from grid
    def Dispose(self,idx):
        if len(self._dimensions)!=0: self._PopAgent(idx)
        self._mem[idx,self._ALIVE]=0
        self._dead[self._nDead]=idx
        self._nDead+=1
        self._pop-=1
        if self._NEXT != -1:
            self._mem[idx, self._NEXT] = self._NULL
        if self._PREV != -1:
            self._mem[idx, self._PREV] = self._NULL
        if self._LOC != -1:
            self._mem[idx, self._LOC] = self._NULL

#map index to x coordinate
    def ItoX(self,i):
        if len(self._dimensions)==2: return self.ItoX2D(i)
        else: return self.ItoX3D(i)

    def ItoX2D(self,i): return int32(int32(i) / self._dimensions[1])
    def ItoX3D(self,i): return int32(int32(i)/(self._dimensions[1]*self._dimensions[2]))

#map index to y coordinate
    def ItoY(self,i):
        if len(self._dimensions)==2: return self.ItoY2D(i)
        else: return self.ItoY3D(i)

    def ItoY2D(self,i): return int32(int32(i) % self._dimensions[1])
    def ItoY3D(self,i): return int32((int32(i)/self._dimensions[2])%self._dimensions[1])

    def ItoZ3D(self,i): return int32(int32(i)%self._dimensions[2])

    def ItoZ(self,i): return int32(int32(i)%self._dimensions[2])

#map coordinates to indices
    def XYtoI(self, x, y): return int32(x)*self._dimensions[1]+int32(y)

    def XYZtoI(self, x, y, z): return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

#returns whether agent is alive
    def Alive(self,idx):return self._mem[idx,self._ALIVE]

#returns x,y,z coordinate of agent
    def X(self,idx): return self._mem[idx,self._X]

    def Y(self,idx): return self._mem[idx,self._Y]

    def Z(self,idx): return self._mem[idx,self._Z]

#returns x,y,z square coordinate of agent
    def XSQ(self,idx): return int32(self._mem[idx,self._X])

    def YSQ(self,idx): return int32(self._mem[idx,self._Y])

    def ZSQ(self,idx): return int32(self._mem[idx,self._Z])

#returns square index of agent
    def I(self,idx): return int32(self._mem[idx,self._LOC])

#returns x,y,z dimension length of the grid
    def Xdim(self): return self._dimensions[0]
    def Ydim(self): return self._dimensions[1]
    def Zdim(self): return self._dimensions[2]

    def __len__(self): return self._length

#returns property value for agent
    def GetP(self,idx,propId):
        propId=int32(propId)
        if propId<0:
            raise Exception("propIds must be 0 or positive")
        return self._mem[idx,propId+self._N_BUILTIN_PROPS]

#set property value for agent
    def SetP(self,idx,propId,value):
        propId=int32(propId)
        if propId<0:
            raise Exception("propIds must be 0 or positive")
        self._mem[idx,propId+self._N_BUILTIN_PROPS]=value

    def _PutAgent(self,idx,x):
        prev = self._grid[x]
        if self._isStackable:
            if prev!=self._NULL:
                self._mem[prev,self._PREV]=idx
            self._mem[idx,self._PREV]=self._NULL
            self._mem[idx,self._NEXT]=self._grid[x]
            self._mem[idx,self._LOC]=x
            self._grid[x]=idx
        else:
            if prev!=self._NULL:
                raise Exception("stacking not allowed!")
            self._grid[x]=idx
            self._mem[idx,self._LOC]=x

    def _PopAgent(self,idx):
        pos = int32(self._mem[idx, self._LOC])
        if self._isStackable:
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
        if x!=self._mem[idx,self._LOC]:
            self._PopAgent(idx)
            self._PutAgent(idx, x)

#move agent to center of square at specified index
    def MoveSQ(self, idx, i):
        if i<0 or i>=self._length:
            raise Exception(f"attempting to move to index that is out of bounds i:{i}")
        self._MoveAgent(idx,int32(i))
        self._SetAgentPositionSQ(idx)

#move agent to center of coordinates, will apply wrap around to coordinates if enabled
    def MoveSQ1D(self, idx, x):
        x2=self.InWrapSQ(int32(x),0)
        if x2==self._NULL:raise Exception(f"attempting to move out of bounds x:{x}")
        self._MoveAgent(idx,x2)
        self._SetAgentPositionSQ(idx)

    def MoveSQ2D(self, idx, x, y):
        x2=self.InWrapSQ(int32(x),0)
        y2=self.InWrapSQ(int32(y),1)
        if x2==self._NULL or y2==self._NULL:raise Exception(f"attempting to move out of bounds x:{x}, y:{y}")
        i=self.XYtoI(int32(x2),int32(y2))
        self._MoveAgent(idx,i)
        self._SetAgentPositionSQ(idx)

    def MoveSQ3D(self, idx, x, y, z):
        x2=self.InWrapSQ(int32(x),0)
        y2=self.InWrapSQ(int32(y),1)
        z2=self.InWrapSQ(int32(z),2)
        if x2==self._NULL or y2==self._NULL or z2==self._NULL:raise Exception(f"attempting to move out of bounds x:{x}, y:{y}, z:{z}")
        i=self.XYZtoI(int32(x2),int32(y2),int32(z2))
        self._MoveAgent(idx,i)
        self._SetAgentPositionSQ(idx)

#move agent off-lattice
    def Move1D(self, idx, x):
        x2=self.InWrap(x,0)
        if x2==self._NULL:raise Exception(f"attempting to move out of bounds x:{x}")
        self._MoveAgent(idx,int32(x2))
        self._SetAgentPosition1D(idx,x2)

    def Move2D(self, idx, x, y):
        x2=self.InWrap(x,0)
        y2=self.InWrap(y,1)
        if x2==self._NULL or y2==self._NULL:raise Exception(f"attempting to move out of bounds x:{x}, y:{y}")
        self._MoveAgent(idx,self.XYtoI(int32(x2),int32(y2)))
        self._SetAgentPosition2D(idx,x2,y2)

    def Move3D(self, idx, x, y, z):
        x2=self.InWrap(x,0)
        y2=self.InWrap(y,1)
        z2=self.InWrap(z,2)
        if x2==self._NULL or y2==self._NULL or z2==self._NULL:raise Exception(f"attempting to move out of bounds x:{x}, y:{y}, z:{z}")
        self._MoveAgent(idx,self.XYZtoI(int32(x2),int32(y2),int32(z2)))
        self._SetAgentPosition3D(idx,x2,y2,z2)

#returns the most recent agent that moved to specified location
    def GetLastI(self,x): return self._grid[int32(x)]

    def GetLast1D(self,x): return self._grid[int32(x)]

    def GetLast2D(self,x,y): return self._grid[self.XYtoI(x,y)]

    def GetLast3D(self,x,y,z): return self._grid[self.XYZtoI(x,y,z)]

#adds all agents at specified location to search list
    def AddI(self, searchListOut, x):
        curr=self.GetLastI(int32(x))
        if self._isStackable:
            while curr!=self._NULL:
                searchListOut.Add(curr)
                curr=int32(self._mem[curr,self._NEXT])
        elif curr!=self._NULL: searchListOut.Add(curr)
        return searchListOut

#clears searchlist, adds all agents at specified location to search list
    def GetI(self,searchListOut,x):
        searchListOut.Clear()
        return self.AddI(searchListOut,x)

    def Add1D(self, searchListOut, x):
        return self.AddI(searchListOut, x)

    def Get1D(self,searchListOut,x):
        searchListOut.Clear()
        return self.Add1D(searchListOut,x)

    def Add2D(self, searchListOut, x, y):
        return self.AddI(searchListOut, self.XYtoI(x, y))

    def Get2D(self,searchListOut,x,y):
        searchListOut.Clear()
        return self.Add2D(searchListOut,x,y)

    def Add3D(self, searchListOut, x, y, z):
        return self.AddI(searchListOut, self.XYZtoI(x, y, z))

    def Get3D(self,searchListOut,x,y,z):
        searchListOut.Clear()
        return self.Add3D(searchListOut,x,y,z)

    def AddBox1D(self, searchListOut, x1, x2):
        for i in range(x1,x2): 
            xw=self.InWrapSQ(i,0)
            if xw!=self._NULL: self.AddI(searchListOut, xw)
        return searchListOut

    def GetBox1D(self, searchListOut, x1, x2):
        searchListOut.Clear()
        return self.AddBox1D(searchListOut,x1,x2)

    def AddBox2D(self, searchListOut, x1, x2, y1, y2):
        for x in range(x1,x2):
            xw=self.InWrapSQ(x,0)
            if xw!=self._NULL:
                for y in range(y1,y2):
                    yw=self.InWrapSQ(y,1)
                    if yw!=self._NULL:
                        self.AddI(searchListOut, self.XYtoI(xw, yw))
        return searchListOut

    def GetBox2D(self, searchListOut, x1, x2,y1,y2):
        searchListOut.Clear()
        return self.AddBox2D(searchListOut,x1,x2,y1,y2)

    def AddBox3D(self, searchListOut, x1, x2, y1, y2, z1, z2):
        for x in range(x1,x2):
            xw=self.InWrapSQ(x,0)
            if xw!=self._NULL:
                for y in range(y1,y2):
                    yw=self.InWrapSQ(y,1)
                    if yw!=self._NULL:
                        for z in range(z1, z2):
                            zw=self.InWrapSQ(z,2)
                            if zw!=self._NULL:
                                self.AddI(searchListOut, self.XYZtoI(xw, yw, zw))
        return searchListOut

    def GetBox3D(self, searchListOut, x1, x2,y1,y2,z1,z2):
        searchListOut.Clear()
        return self.AddBox3D(searchListOut,x1,x2,y1,y2,z1,z2)

    def _InclusionCriteria(self,searchListOut,loc,inclusionCriteria):
        if inclusionCriteria<0: searchListOut.Add(loc)
        elif inclusionCriteria==0:
            if self._grid[loc]==self._NULL: searchListOut.Add(loc)
        elif inclusionCriteria>0:
            if self._grid[loc]!=self._NULL: searchListOut.Add(loc)
        return searchListOut

#put location indicies that are valid into searchlist
#inclusion criteria: -1: ignore, 0: emptyOnly, 1: occupiedOnly
    def MapHood1D(self, searchListOut, xCenter,inclusionCriteria):
        searchListOut.Clear()
        for i in range(searchListOut._hoodLen):
            loc=int32(self.InWrapSQ(searchListOut._hood[i] + xCenter,0))
            if loc!=self._NULL:
                self._InclusionCriteria(searchListOut,loc,inclusionCriteria)
        return searchListOut

    def MapHood2D(self, searchListOut, xCenter, yCenter,inclusionCriteria):
        searchListOut.Clear()
        for i in range(searchListOut._hoodLen):
            newX=self.InWrapSQ(searchListOut._hood[i * 2] + xCenter,0)
            newY=self.InWrapSQ(searchListOut._hood[i * 2 + 1] + yCenter,1)
            if newX!=self._NULL and newY!=self._NULL:
                loc=self.XYtoI(newX, newY)
                self._InclusionCriteria(searchListOut,loc,inclusionCriteria)
        return searchListOut

    def MapHood3D(self, searchListOut, xCenter, yCenter, zCenter,inclusionCriteria):
        searchListOut.Clear()
        for i in range(searchListOut._hoodLen):
            newX=self.InWrapSQ(searchListOut._hood[i * 3] + xCenter,0)
            newY=self.InWrapSQ(searchListOut._hood[i * 3 + 1] + yCenter,1)
            newZ=self.InWrapSQ(searchListOut._hood[i * 3 + 2] + zCenter,2)
            if newX!=self._NULL and newY!=self._NULL and newZ!=self._NULL:
                loc=self.XYZtoI(newX, newY, newZ)
                self._InclusionCriteria(searchListOut,loc,inclusionCriteria)
        return searchListOut

#put all agents in neighborhood centered around center coordinates into searchlist
    def AddHood1D(self, searchListOut, xCenter):
        for i in range(searchListOut._hoodLen):
            x=self.InWrapSQ(xCenter + searchListOut._hood[i],0)
            if x!=self._NULL: self.AddI(searchListOut, x)
        return searchListOut

    def GetHood1D(self,searchListOut,xCenter):
        searchListOut.Clear()
        return self.AddHood1D(searchListOut, xCenter)

    def AddHood2D(self, searchListOut, xCenter, yCenter):
        for i in range(searchListOut._hoodLen):
            x=self.InWrapSQ(xCenter + searchListOut._hood[i * 2],0)
            y=self.InWrapSQ(yCenter + searchListOut._hood[i * 2 + 1],1)
            if x!=self._NULL and y!=self._NULL: self.AddI(searchListOut, self.XYtoI(x,y))
        return searchListOut

    def GetHood2D(self, searchListOut, xCenter, yCenter):
        searchListOut.Clear()
        return self.AddHood2D(searchListOut, xCenter, yCenter)

    def AddHood3D(self, searchListOut, xCenter, yCenter, zCenter):
        for i in range(searchListOut._hoodLen):
            x=self.InWrapSQ(xCenter + searchListOut._hood[i * 3],0)
            y=self.InWrapSQ(yCenter + searchListOut._hood[i * 3 + 1],1)
            z=self.InWrapSQ(zCenter + searchListOut._hood[i * 3 + 2],2)
            if x!=self._NULL and y!=self._NULL and z!=self._NULL: self.AddI(searchListOut, self.XYZtoI(x,y,z))
        return searchListOut

    def GetHood3D(self, searchListOut, xCenter, yCenter, zCenter):
        searchListOut.Clear()
        return self.AddHood3D(searchListOut, xCenter, yCenter, zCenter)

#adds all agents within radius of center coordinates to searchlist
    def AddInRadius1D(self, searchListOut, radius, xCenter):
        endX=int32(xCenter+radius+1)
        startX=int32(xCenter-radius)
        for x in range(startX,endX):
            xw=self.InWrapSQ(x,0)
            if xw!=self._NULL:
                curr = self.GetLastI(xw)
                if not self._isStackable and curr!=self._NULL: 
                    xComp=self.DispWrap(xCenter,self._mem[curr,self._X],0)
                    if np.abs(xComp)<=radius: searchListOut.Add(curr)
                else:
                    while curr != self._NULL:
                        xComp=self.DispWrap(xCenter,self._mem[curr,self._X],0)
                        if np.abs(xComp)<=radius: searchListOut.Add(curr)
                        curr = int32(self._mem[curr, self._NEXT])
        return searchListOut

    def GetInRadius1D(self,searchListOut, radius, xCenter):
        searchListOut.Clear()
        return self.AddInRadius1D(searchListOut,radius,xCenter)

    def AddInRadius2D(self, searchListOut, radius, xCenter, yCenter):
        endX=int32(xCenter+radius+1)
        startX=int32(xCenter-radius)
        endY=int32(yCenter+radius+1)
        startY=int32(yCenter-radius)
        radSq=radius*radius
        for x in range(startX,endX):
            xw=self.InWrapSQ(x,0)
            if xw!=self._NULL:
                for y in range(startY, endY):
                    yw=self.InWrapSQ(y,1)
                    if yw!=self._NULL:
                        curr = self.GetLast2D(int32(xw),int32(yw))
                        if not self._isStackable and curr!=self._NULL: 
                            xComp=self.DispWrap(xCenter,self._mem[curr,self._X],0)
                            yComp=self.DispWrap(yCenter,self._mem[curr,self._Y],1)
                            if (xComp*xComp+yComp*yComp)<=radSq: searchListOut.Add(curr)
                        else:
                            while curr != self._NULL:
                                xComp=self.DispWrap(xCenter,self._mem[curr,self._X],0)
                                yComp=self.DispWrap(yCenter,self._mem[curr,self._Y],1)
                                if (xComp*xComp+yComp*yComp)<=radSq: searchListOut.Add(curr)
                                curr = int32(self._mem[curr, self._NEXT])
        return searchListOut

    def GetInRadius2D(self,searchListOut, radius, xCenter, yCenter):
        searchListOut.Clear()
        return self.AddInRadius2D(searchListOut,radius,xCenter,yCenter)

    def AddInRadius3D(self, searchListOut, radius, xCenter, yCenter, zCenter):
        endX = int32(xCenter + radius + 1)
        startX = int32(xCenter - radius)
        endY = int32(yCenter + radius + 1)
        startY = int32(yCenter - radius)
        endZ = int32(zCenter + radius + 1)
        startZ = int32(zCenter - radius)
        radSq = radius * radius
        for x in range(startX, endX):
            xWrap = self.InWrapSQ(x, 0)
            if xWrap != self._NULL:
                for y in range(startY, endY):
                    yWrap = self.InWrapSQ(y, 1)
                    if yWrap != self._NULL:
                        for z in range(startZ, endZ):
                            zWrap = self.InWrapSQ(z, 2)
                            if zWrap != self._NULL:
                                curr = self.GetLast3D(xWrap, yWrap, zWrap)
                                if not self._isStackable and curr!=self._NULL: 
                                    xComp = self.DispWrap(xCenter, self._mem[curr, self._X], 0)
                                    yComp = self.DispWrap(yCenter, self._mem[curr, self._Y], 1)
                                    zComp = self.DispWrap(zCenter, self._mem[curr, self._Z], 2)
                                    if (xComp * xComp + yComp * yComp + zComp * zComp) <= radSq: searchListOut.Add(curr)
                                else:
                                    while curr != self._NULL:
                                        xComp = self.DispWrap(xCenter, self._mem[curr, self._X], 0)
                                        yComp = self.DispWrap(yCenter, self._mem[curr, self._Y], 1)
                                        zComp = self.DispWrap(zCenter, self._mem[curr, self._Z], 2)
                                        if (xComp * xComp + yComp * yComp + zComp * zComp) <= radSq: searchListOut.Add(curr)
                                        curr = int32(self._mem[curr, self._NEXT])
        return searchListOut

    def GetInRadius3D(self,searchListOut, radius, xCenter, yCenter, zCenter):
        searchListOut.Clear()
        return self.AddInRadius3D(searchListOut,radius,xCenter,yCenter,zCenter)

#returns total live agent population
    def Pop(self):
        return self._pop

#returns an array of all live agent idxs
    def All(self,shuffle):
        out=np.zeros(self._pop,dtype=int32)
        j=0
        for i in range(self._pop+self._nDead):
            if self._mem[i,self._ALIVE]!=0:
                out[j]=i
                j+=1
        if shuffle: np.random.shuffle(out)
        return out

#clear searchlist, put all live agents into searchlist
    def AllToList(self,searchListOut,shuffle):
        searchListOut.Clear()
        for i in range(self._pop+self._nDead):
            if self._mem[i,self._ALIVE]:
                searchListOut.Add(i)
        if shuffle:searchListOut.Shuffle()
        return searchListOut
    
    #-1:X, -2:Y, -3:Z, -4:I
# returns an array containing all requested properties for all agents
    def AllProperties(self,requestedProperties,shuffle):
        requestedProperties=np.array(requestedProperties,dtype=int32)
        out=np.zeros((len(requestedProperties),self._pop),dtype=float32)
        for i in range(len(requestedProperties)):
            p=requestedProperties[i]
            if p<0:
                if p==-1:requestedProperties[i]=self._X
                elif p==-2:requestedProperties[i]=self._Y
                elif p==-3:requestedProperties[i]=self._Z
                elif p==-4:requestedProperties[i]=self._LOC
                else:raise Exception("negative property ids can only be used with the following mapping: -1:X, -2:Y, -3:Z, -4:I")
            else:requestedProperties[i]=requestedProperties[i]+self._N_BUILTIN_PROPS
        agents=self.All(shuffle)
        for i in range(len(agents)):
            for j in range(len(requestedProperties)):
                out[j,i]=self._mem[agents[i],requestedProperties[j]]
        return out

# returns an array containing all requested properties from searchlist
    def QueryListProperties(self,searchList,requestedProperties):
        requestedProperties=np.array(requestedProperties,dtype=int32)
        out=np.zeros((len(requestedProperties),len(searchList)),dtype=float32)
        for i in range(len(requestedProperties)):
            p=requestedProperties[i]
            if p<0:
                if p==-1:requestedProperties[i]=self._X
                elif p==-2:requestedProperties[i]=self._Y
                elif p==-3:requestedProperties[i]=self._Z
                elif p==-4:requestedProperties[i]=self._LOC
                else:raise Exception("negative property ids can only be used with the following mapping: -1:X, -2:Y, -3:Z, -4:I")
            else:requestedProperties[i]=requestedProperties[i]+self._N_BUILTIN_PROPS
        for i in range(len(searchList)):
            for j in range(len(requestedProperties)):
                out[j,i]=self._mem[searchList[i],requestedProperties[j]]
        return out

