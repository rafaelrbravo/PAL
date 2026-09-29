from numba import int32
from numba.experimental import jitclass
import numpy as np

specQueryList=[('_mem', int32[:]),
                ('_hood',int32[:]),
                ('_hoodLinear',int32[:]),
                ('_hoodGridY',int32),
                ('_hoodGridZ',int32),
                ('_hoodMin',int32[:]),
                ('_hoodMax',int32[:]),
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
        self._hoodLinear=np.zeros(0,dtype=int32)
        self._hoodGridY=-1;self._hoodGridZ=-1
        self._hoodMin=np.zeros(3,dtype=int32);self._hoodMax=np.zeros(3,dtype=int32)

    def _SetupArr(self,length,copy):
        newIds=np.full(int32(length),-1,dtype=int32)
        if copy!=0:
            for i in range(self._length):
                newIds[i]=self._mem[i]
        self._mem=newIds


    def __len__(self):
        return self._length

    def __getitem__(self,index):
        if index!=int32(index) or index<0 or index>=self._length: raise Exception(f"QueryList index out of bounds index:{index}")
        return self._mem[index]

#add a single agent id to the querylist
    def Add(self,idx):
        currLen=len(self._mem)
        if currLen==self._length: self._SetupArr(currLen*2,1)
        self._mem[self._length]=idx
        self._length+=1
        return self


    def Clear(self):
        self._length=0
        return self

#randomizes the query list order
    def Shuffle(self):
        for i in range(self._length-1):
            j=np.random.randint(i,self._length)
            temp=self._mem[i]
            self._mem[i]=self._mem[j]
            self._mem[j]=temp
        return self

#pulls a random id from the querylist
    def Random(self):
        if self._length==0:return -1
        return self._mem[np.random.randint(self._length)]

#attaches a neighborhood to the querylist to allow gathering agent ids within neighborhood
    def SetHood(self, dimension, neighborhood):
        if dimension<1 or dimension>3: raise Exception(f"neighborhood dimension must be 1, 2, or 3 dimension:{dimension}")
        if dimension!=int32(dimension): raise Exception(f"neighborhood dimension must be an integer dimension:{dimension}")
        if len(neighborhood)%dimension!=0: raise Exception(f"neighborhood length must be divisible by dimension length:{len(neighborhood)} dimension:{dimension}")
        for i in range(len(neighborhood)):
            if neighborhood[i]!=int32(neighborhood[i]): raise Exception(f"neighborhood offsets must be integers value:{neighborhood[i]}")
        return self.SetHood_(dimension,neighborhood)

    def SetHood_(self, dimension, neighborhood):
        self._hoodDim=dimension
        self._hood=np.empty(len(neighborhood),dtype=int32)
        for i in range(len(neighborhood)): self._hood[i]=neighborhood[i]
        self._hoodLen=len(neighborhood)//self._hoodDim
        self._hoodLinear=np.zeros(self._hoodLen,dtype=int32)
        self._hoodGridY=-1;self._hoodGridZ=-1
        for d in range(3): self._hoodMin[d]=0;self._hoodMax[d]=0
        if self._hoodLen>0:
            for d in range(self._hoodDim):
                self._hoodMin[d]=self._hood[d];self._hoodMax[d]=self._hood[d]
            for i in range(1,self._hoodLen):
                for d in range(self._hoodDim):
                    v=self._hood[i*self._hoodDim+d]
                    if v<self._hoodMin[d]:self._hoodMin[d]=v
                    if v>self._hoodMax[d]:self._hoodMax[d]=v
        return self



