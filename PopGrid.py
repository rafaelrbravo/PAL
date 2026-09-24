from numba import njit,int32,int64,float32,float64
from numba.experimental import jitclass
from scipy.stats import binom
import numpy as np

#TODO: setup arbitrary BCs for diffusion!

specMultinomial=[
    ('_pop',int64),
    ('_prob',float64)
]
@jitclass(specMultinomial)
class MultinomialCalc(object):
    def Binomial(self,pop,prob):
        return np.random.binomial(pop,prob)

    def __init__(self):
        self._pop=0
        self._prob=0
    
    def Setup(self,pop):
        self._pop=pop
        self._prob=1

    def Sample(self,prob):
        if self._prob<0:raise Exception(f"negative probability {self._prob}")
        if self._pop==0 or self._prob==0: return 0
        if self._prob-prob<=0:
            ret=self._pop
            self._pop=0
            self._prob-=prob
            return ret
        ret=self.Binomial(self._pop,prob/self._prob)
        self._prob-=prob
        self._pop-=ret
        return ret

specPopGrid=[
    ('_dimensions',int32[:]),
    ('_wrap',int32[:]),
    ('_field',int64[:]),
    ('_deltas',int64[:]),
    ('_length',int64),
    ('_NULL',int32),
]

@jitclass(specPopGrid)
class PopGrid(object):
    def __init__(self,dimensions):
        self._dimensions=np.array(dimensions,dtype=int32)
        self._wrap=np.zeros(len(dimensions),dtype=int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._NULL=-1
        self._length=int64(np.prod(self._dimensions))
        self._field=np.zeros(self._length,dtype=int64)
        self._deltas=np.zeros(self._length,dtype=int64)

    def __getitem__(self,index): return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self): return self._dimensions[1]
    def Zdim(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if len(self._dimensions)==2: return self.ItoX2D(i)
        else: return self.ItoX3D(i)

    def ItoX2D(self,i): return int32(int32(i) // self._dimensions[1])
    def ItoX3D(self,i): return int32(int32(i)//(self._dimensions[1]*self._dimensions[2]))

    def ItoY(self,i):
        if len(self._dimensions)==2: return self.ItoY2D(i)
        else: return self.ItoY3D(i)

    def ItoY2D(self,i): return int32(int32(i) % self._dimensions[1])
    def ItoY3D(self,i): return int32((int32(i)//self._dimensions[2])%self._dimensions[1])

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
    def InWrap(self,value, dimension):
            if value>=0 and value<self._dimensions[dimension]: return value
            if self._wrap[dimension]: return int32(value%self._dimensions[dimension])
            return self._NULL
    def GetFieldCopy(self):
        return np.copy(self._field)
#mn=MultinomialCalc()
#rng=np.random.default_rng()
#np.random.binomial
#
#@njit
#def BinTest(pop,prob,coinFlip):
##    if coinFlip:
##        total=0
##        for i in range(pop):
##            if np.random.uniform()<prob:
##                total+=1
##        return total
##    else:
#    return rng.binomial(pop,prob)
##        return np.random.Generator.binomial(pop,prob)
#
#POP=10000
#PROB=0.5
#
#for i in range(10): BinTest(POP,PROB,0)
#t0=time.time_ns()
#for i in range(100): BinTest(POP,PROB,0)
#print(time.time_ns()-t0)
#binom


