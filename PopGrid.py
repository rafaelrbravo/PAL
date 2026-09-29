from numba import int8,int16,int32,int64,float64
from numba.experimental import jitclass
import numpy as np
from typing import Protocol

specMultinomial=[('_pop',int64),('_prob',float64)]
@jitclass(specMultinomial)
class MultinomialCalc(object):
    def __init__(self):
        self._pop=0
        self._prob=0

    def Binomial(self,pop,prob):
        if not np.isfinite(pop) or pop<0 or pop>9223372036854775807 or pop!=int(pop): raise Exception(f"pop must be a nonnegative int64 pop:{pop}")
        if not np.isfinite(prob) or prob<0 or prob>1: raise Exception(f"prob must be between 0 and 1 prob:{prob}")
        return self.Binomial_(pop,prob)

    def Binomial_(self,pop,prob):
        return np.random.binomial(pop,prob)

    def Setup(self,pop):
        if not np.isfinite(pop) or pop<0 or pop>9223372036854775807 or pop!=int(pop): raise Exception(f"pop must be a nonnegative int64 pop:{pop}")
        self.Setup_(pop)

    def Setup_(self,pop):
        self._pop=pop
        self._prob=1

    def Sample(self,prob):
        if not np.isfinite(prob) or prob<0 or prob>self._prob+1e-12: raise Exception(f"prob must be between 0 and remaining probability {self._prob} prob:{prob}")
        return self.Sample_(prob)

    def Sample_(self,prob):
        if self._pop==0 or prob==0:return 0
        if self._prob-prob<=0:
            ret=self._pop
            self._pop=0
            self._prob=0
            return ret
        ret=self.Binomial_(self._pop,prob/self._prob)
        self._prob-=prob
        self._pop-=ret
        return ret

# Capacity-specialized PopGrid implementations. PopGrid(...) below is the public factory.

specPopGrid8=[('_dimensions',int32[:]),('_wrap',int32[:]),('_field',int8[:]),('_deltas',int8[:]),('_length',int64),('_NULL',int32),('_capacity',int64)]
@jitclass(specPopGrid8)
class _PopGrid8(object):
    def __init__(self,dimensions,capacity):
        self._capacity=int64(capacity)
        if len(dimensions)<1 or len(dimensions)>3: raise Exception("PopGrid requires 1 to 3 dimensions")
        length=1
        for dim in dimensions:
            if not np.isfinite(dim) or dim!=int(dim): raise Exception(f"dimension sizes must be integers dimension:{dim}")
            if dim==0: raise Exception("dimension sizes cannot be 0")
            if dim<=-2147483648 or dim>2147483647: raise Exception(f"dimension size exceeds int32 range dimension:{dim}")
            length*=abs(dim)
            if length>2147483647: raise Exception(f"grid length exceeds int32 index range length:{length}")
        self._dimensions=np.array(dimensions,dtype=int32)
        self._wrap=np.zeros(len(dimensions),dtype=int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._NULL=-1
        self._length=int64(length)
        self._field=np.zeros(self._length,dtype=np.int8)
        self._deltas=np.zeros(self._length,dtype=np.int8)

    def __getitem__(self,index):
        if index<0 or index>=self._length: raise Exception(f"index out of bounds i:{index}")
        return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self):
        if len(self._dimensions)<2: raise Exception("Ydim requires a 2D or 3D PopGrid")
        return self.Ydim_()
    def Ydim_(self): return self._dimensions[1]
    def Zdim(self):
        if len(self._dimensions)<3: raise Exception("Zdim requires a 3D PopGrid")
        return self.Zdim_()
    def Zdim_(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoX_(i)
    def ItoX_(self,i):
        if len(self._dimensions)==1:return int32(i)
        if len(self._dimensions)==2:return int32(int32(i)//self._dimensions[1])
        return int32(int32(i)//(self._dimensions[1]*self._dimensions[2]))
    def ItoY(self,i):
        if len(self._dimensions)<2: raise Exception("ItoY requires a 2D or 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoY_(i)
    def ItoY_(self,i):
        if len(self._dimensions)==2:return int32(int32(i)%self._dimensions[1])
        return int32((int32(i)//self._dimensions[2])%self._dimensions[1])
    def ItoZ(self,i):
        if len(self._dimensions)!=3: raise Exception("ItoZ requires a 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoZ_(i)
    def ItoZ_(self,i): return int32(int32(i)%self._dimensions[2])

    def ToI(self,x,y=None,z=None):
        if not np.isfinite(x) or x!=int(x): raise Exception(f"x must be an integer x:{x}")
        if y is not None and (not np.isfinite(y) or y!=int(y)): raise Exception(f"y must be an integer y:{y}")
        if z is not None and (not np.isfinite(z) or z!=int(z)): raise Exception(f"z must be an integer z:{z}")
        if len(self._dimensions)==1:
            if y is not None or z is not None: raise Exception("1D PopGrid requires x only")
            if x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
        elif len(self._dimensions)==2:
            if y is None or z is not None: raise Exception("2D PopGrid requires x,y")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
        else:
            if y is None or z is None: raise Exception("3D PopGrid requires x,y,z")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.ToI_(x,y,z)
    def ToI_(self,x,y=None,z=None):
        if y is None:return int32(x)
        if z is None:return int32(x)*self._dimensions[1]+int32(y)
        return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

    def SetI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,i)
    def SetI_(self,value,i): self._field[i]=value
    def Set(self,value,x,y=None,z=None):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,self.ToI(x,y,z))
    def Set_(self,value,x,y=None,z=None): self._field[self.ToI_(x,y,z)]=value

    def AddI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>127 or value<-128 or value>0 and self._deltas[i]>127-value or value<0 and self._deltas[i]<-128-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def AddI_(self,value,i): self._deltas[i]+=value
    def Add(self,value,x,y=None,z=None):
        i=self.ToI(x,y,z)
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>127 or value<-128 or value>0 and self._deltas[i]>127-value or value<0 and self._deltas[i]<-128-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def Add_(self,value,x,y=None,z=None): self._deltas[self.ToI_(x,y,z)]+=value

    def GetI(self,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.GetI_(i)
    def GetI_(self,i): return self._field[i]
    def Get(self,x,y=None,z=None): return self.GetI_(self.ToI(x,y,z))
    def Get_(self,x,y=None,z=None): return self._field[self.ToI_(x,y,z)]

    def Update(self):
        for i in range(self._length):
            if self._deltas[i]>0 and self._field[i]>self._capacity-self._deltas[i]: raise Exception(f"population exceeds capacity {self._capacity} at i:{i}")
            if self._deltas[i]<0 and self._deltas[i]<-self._field[i]: raise Exception(f"population cannot become negative at i:{i}")
        self.Update_()
    def Update_(self):
        for i in range(self._length):
            self._field[i]+=self._deltas[i]
            self._deltas[i]=0

    def InWrap(self,value,dimension):
        if dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        return self.InWrap_(value,dimension)
    def InWrap_(self,value,dimension):
        if value>=0 and value<self._dimensions[dimension]:return value
        if self._wrap[dimension]:return int32(value%self._dimensions[dimension])
        return self._NULL

    def GetFieldCopy(self): return np.copy(self._field)

    def Clear(self,value=0):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.Clear_(value)
    def Clear_(self,value=0):
        for i in range(self._length):
            self._field[i]=value
            self._deltas[i]=0


specPopGrid16=[('_dimensions',int32[:]),('_wrap',int32[:]),('_field',int16[:]),('_deltas',int16[:]),('_length',int64),('_NULL',int32),('_capacity',int64)]
@jitclass(specPopGrid16)
class _PopGrid16(object):
    def __init__(self,dimensions,capacity):
        self._capacity=int64(capacity)
        if len(dimensions)<1 or len(dimensions)>3: raise Exception("PopGrid requires 1 to 3 dimensions")
        length=1
        for dim in dimensions:
            if not np.isfinite(dim) or dim!=int(dim): raise Exception(f"dimension sizes must be integers dimension:{dim}")
            if dim==0: raise Exception("dimension sizes cannot be 0")
            if dim<=-2147483648 or dim>2147483647: raise Exception(f"dimension size exceeds int32 range dimension:{dim}")
            length*=abs(dim)
            if length>2147483647: raise Exception(f"grid length exceeds int32 index range length:{length}")
        self._dimensions=np.array(dimensions,dtype=int32)
        self._wrap=np.zeros(len(dimensions),dtype=int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._NULL=-1
        self._length=int64(length)
        self._field=np.zeros(self._length,dtype=np.int16)
        self._deltas=np.zeros(self._length,dtype=np.int16)

    def __getitem__(self,index):
        if index<0 or index>=self._length: raise Exception(f"index out of bounds i:{index}")
        return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self):
        if len(self._dimensions)<2: raise Exception("Ydim requires a 2D or 3D PopGrid")
        return self.Ydim_()
    def Ydim_(self): return self._dimensions[1]
    def Zdim(self):
        if len(self._dimensions)<3: raise Exception("Zdim requires a 3D PopGrid")
        return self.Zdim_()
    def Zdim_(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoX_(i)
    def ItoX_(self,i):
        if len(self._dimensions)==1:return int32(i)
        if len(self._dimensions)==2:return int32(int32(i)//self._dimensions[1])
        return int32(int32(i)//(self._dimensions[1]*self._dimensions[2]))
    def ItoY(self,i):
        if len(self._dimensions)<2: raise Exception("ItoY requires a 2D or 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoY_(i)
    def ItoY_(self,i):
        if len(self._dimensions)==2:return int32(int32(i)%self._dimensions[1])
        return int32((int32(i)//self._dimensions[2])%self._dimensions[1])
    def ItoZ(self,i):
        if len(self._dimensions)!=3: raise Exception("ItoZ requires a 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoZ_(i)
    def ItoZ_(self,i): return int32(int32(i)%self._dimensions[2])

    def ToI(self,x,y=None,z=None):
        if not np.isfinite(x) or x!=int(x): raise Exception(f"x must be an integer x:{x}")
        if y is not None and (not np.isfinite(y) or y!=int(y)): raise Exception(f"y must be an integer y:{y}")
        if z is not None and (not np.isfinite(z) or z!=int(z)): raise Exception(f"z must be an integer z:{z}")
        if len(self._dimensions)==1:
            if y is not None or z is not None: raise Exception("1D PopGrid requires x only")
            if x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
        elif len(self._dimensions)==2:
            if y is None or z is not None: raise Exception("2D PopGrid requires x,y")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
        else:
            if y is None or z is None: raise Exception("3D PopGrid requires x,y,z")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.ToI_(x,y,z)
    def ToI_(self,x,y=None,z=None):
        if y is None:return int32(x)
        if z is None:return int32(x)*self._dimensions[1]+int32(y)
        return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

    def SetI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,i)
    def SetI_(self,value,i): self._field[i]=value
    def Set(self,value,x,y=None,z=None):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,self.ToI(x,y,z))
    def Set_(self,value,x,y=None,z=None): self._field[self.ToI_(x,y,z)]=value

    def AddI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>32767 or value<-32768 or value>0 and self._deltas[i]>32767-value or value<0 and self._deltas[i]<-32768-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def AddI_(self,value,i): self._deltas[i]+=value
    def Add(self,value,x,y=None,z=None):
        i=self.ToI(x,y,z)
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>32767 or value<-32768 or value>0 and self._deltas[i]>32767-value or value<0 and self._deltas[i]<-32768-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def Add_(self,value,x,y=None,z=None): self._deltas[self.ToI_(x,y,z)]+=value

    def GetI(self,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.GetI_(i)
    def GetI_(self,i): return self._field[i]
    def Get(self,x,y=None,z=None): return self.GetI_(self.ToI(x,y,z))
    def Get_(self,x,y=None,z=None): return self._field[self.ToI_(x,y,z)]

    def Update(self):
        for i in range(self._length):
            if self._deltas[i]>0 and self._field[i]>self._capacity-self._deltas[i]: raise Exception(f"population exceeds capacity {self._capacity} at i:{i}")
            if self._deltas[i]<0 and self._deltas[i]<-self._field[i]: raise Exception(f"population cannot become negative at i:{i}")
        self.Update_()
    def Update_(self):
        for i in range(self._length):
            self._field[i]+=self._deltas[i]
            self._deltas[i]=0

    def InWrap(self,value,dimension):
        if dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        return self.InWrap_(value,dimension)
    def InWrap_(self,value,dimension):
        if value>=0 and value<self._dimensions[dimension]:return value
        if self._wrap[dimension]:return int32(value%self._dimensions[dimension])
        return self._NULL

    def GetFieldCopy(self): return np.copy(self._field)

    def Clear(self,value=0):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.Clear_(value)
    def Clear_(self,value=0):
        for i in range(self._length):
            self._field[i]=value
            self._deltas[i]=0


specPopGrid32=[('_dimensions',int32[:]),('_wrap',int32[:]),('_field',int32[:]),('_deltas',int32[:]),('_length',int64),('_NULL',int32),('_capacity',int64)]
@jitclass(specPopGrid32)
class _PopGrid32(object):
    def __init__(self,dimensions,capacity):
        self._capacity=int64(capacity)
        if len(dimensions)<1 or len(dimensions)>3: raise Exception("PopGrid requires 1 to 3 dimensions")
        length=1
        for dim in dimensions:
            if not np.isfinite(dim) or dim!=int(dim): raise Exception(f"dimension sizes must be integers dimension:{dim}")
            if dim==0: raise Exception("dimension sizes cannot be 0")
            if dim<=-2147483648 or dim>2147483647: raise Exception(f"dimension size exceeds int32 range dimension:{dim}")
            length*=abs(dim)
            if length>2147483647: raise Exception(f"grid length exceeds int32 index range length:{length}")
        self._dimensions=np.array(dimensions,dtype=int32)
        self._wrap=np.zeros(len(dimensions),dtype=int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._NULL=-1
        self._length=int64(length)
        self._field=np.zeros(self._length,dtype=np.int32)
        self._deltas=np.zeros(self._length,dtype=np.int32)

    def __getitem__(self,index):
        if index<0 or index>=self._length: raise Exception(f"index out of bounds i:{index}")
        return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self):
        if len(self._dimensions)<2: raise Exception("Ydim requires a 2D or 3D PopGrid")
        return self.Ydim_()
    def Ydim_(self): return self._dimensions[1]
    def Zdim(self):
        if len(self._dimensions)<3: raise Exception("Zdim requires a 3D PopGrid")
        return self.Zdim_()
    def Zdim_(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoX_(i)
    def ItoX_(self,i):
        if len(self._dimensions)==1:return int32(i)
        if len(self._dimensions)==2:return int32(int32(i)//self._dimensions[1])
        return int32(int32(i)//(self._dimensions[1]*self._dimensions[2]))
    def ItoY(self,i):
        if len(self._dimensions)<2: raise Exception("ItoY requires a 2D or 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoY_(i)
    def ItoY_(self,i):
        if len(self._dimensions)==2:return int32(int32(i)%self._dimensions[1])
        return int32((int32(i)//self._dimensions[2])%self._dimensions[1])
    def ItoZ(self,i):
        if len(self._dimensions)!=3: raise Exception("ItoZ requires a 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoZ_(i)
    def ItoZ_(self,i): return int32(int32(i)%self._dimensions[2])

    def ToI(self,x,y=None,z=None):
        if not np.isfinite(x) or x!=int(x): raise Exception(f"x must be an integer x:{x}")
        if y is not None and (not np.isfinite(y) or y!=int(y)): raise Exception(f"y must be an integer y:{y}")
        if z is not None and (not np.isfinite(z) or z!=int(z)): raise Exception(f"z must be an integer z:{z}")
        if len(self._dimensions)==1:
            if y is not None or z is not None: raise Exception("1D PopGrid requires x only")
            if x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
        elif len(self._dimensions)==2:
            if y is None or z is not None: raise Exception("2D PopGrid requires x,y")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
        else:
            if y is None or z is None: raise Exception("3D PopGrid requires x,y,z")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.ToI_(x,y,z)
    def ToI_(self,x,y=None,z=None):
        if y is None:return int32(x)
        if z is None:return int32(x)*self._dimensions[1]+int32(y)
        return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

    def SetI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,i)
    def SetI_(self,value,i): self._field[i]=value
    def Set(self,value,x,y=None,z=None):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,self.ToI(x,y,z))
    def Set_(self,value,x,y=None,z=None): self._field[self.ToI_(x,y,z)]=value

    def AddI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>2147483647 or value<-2147483648 or value>0 and self._deltas[i]>2147483647-value or value<0 and self._deltas[i]<-2147483648-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def AddI_(self,value,i): self._deltas[i]+=value
    def Add(self,value,x,y=None,z=None):
        i=self.ToI(x,y,z)
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>2147483647 or value<-2147483648 or value>0 and self._deltas[i]>2147483647-value or value<0 and self._deltas[i]<-2147483648-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def Add_(self,value,x,y=None,z=None): self._deltas[self.ToI_(x,y,z)]+=value

    def GetI(self,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.GetI_(i)
    def GetI_(self,i): return self._field[i]
    def Get(self,x,y=None,z=None): return self.GetI_(self.ToI(x,y,z))
    def Get_(self,x,y=None,z=None): return self._field[self.ToI_(x,y,z)]

    def Update(self):
        for i in range(self._length):
            if self._deltas[i]>0 and self._field[i]>self._capacity-self._deltas[i]: raise Exception(f"population exceeds capacity {self._capacity} at i:{i}")
            if self._deltas[i]<0 and self._deltas[i]<-self._field[i]: raise Exception(f"population cannot become negative at i:{i}")
        self.Update_()
    def Update_(self):
        for i in range(self._length):
            self._field[i]+=self._deltas[i]
            self._deltas[i]=0

    def InWrap(self,value,dimension):
        if dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        return self.InWrap_(value,dimension)
    def InWrap_(self,value,dimension):
        if value>=0 and value<self._dimensions[dimension]:return value
        if self._wrap[dimension]:return int32(value%self._dimensions[dimension])
        return self._NULL

    def GetFieldCopy(self): return np.copy(self._field)

    def Clear(self,value=0):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.Clear_(value)
    def Clear_(self,value=0):
        for i in range(self._length):
            self._field[i]=value
            self._deltas[i]=0


specPopGrid64=[('_dimensions',int32[:]),('_wrap',int32[:]),('_field',int64[:]),('_deltas',int64[:]),('_length',int64),('_NULL',int32),('_capacity',int64)]
@jitclass(specPopGrid64)
class _PopGrid64(object):
    def __init__(self,dimensions,capacity):
        self._capacity=int64(capacity)
        if len(dimensions)<1 or len(dimensions)>3: raise Exception("PopGrid requires 1 to 3 dimensions")
        length=1
        for dim in dimensions:
            if not np.isfinite(dim) or dim!=int(dim): raise Exception(f"dimension sizes must be integers dimension:{dim}")
            if dim==0: raise Exception("dimension sizes cannot be 0")
            if dim<=-2147483648 or dim>2147483647: raise Exception(f"dimension size exceeds int32 range dimension:{dim}")
            length*=abs(dim)
            if length>2147483647: raise Exception(f"grid length exceeds int32 index range length:{length}")
        self._dimensions=np.array(dimensions,dtype=int32)
        self._wrap=np.zeros(len(dimensions),dtype=int32)
        for i in range(len(self._dimensions)):
            if self._dimensions[i]<0:
                self._dimensions[i]=-self._dimensions[i]
                self._wrap[i]=1
        self._NULL=-1
        self._length=int64(length)
        self._field=np.zeros(self._length,dtype=np.int64)
        self._deltas=np.zeros(self._length,dtype=np.int64)

    def __getitem__(self,index):
        if index<0 or index>=self._length: raise Exception(f"index out of bounds i:{index}")
        return self._field[index]

    def Xdim(self): return self._dimensions[0]
    def Ydim(self):
        if len(self._dimensions)<2: raise Exception("Ydim requires a 2D or 3D PopGrid")
        return self.Ydim_()
    def Ydim_(self): return self._dimensions[1]
    def Zdim(self):
        if len(self._dimensions)<3: raise Exception("Zdim requires a 3D PopGrid")
        return self.Zdim_()
    def Zdim_(self): return self._dimensions[2]
    def __len__(self): return self._length

    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoX_(i)
    def ItoX_(self,i):
        if len(self._dimensions)==1:return int32(i)
        if len(self._dimensions)==2:return int32(int32(i)//self._dimensions[1])
        return int32(int32(i)//(self._dimensions[1]*self._dimensions[2]))
    def ItoY(self,i):
        if len(self._dimensions)<2: raise Exception("ItoY requires a 2D or 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoY_(i)
    def ItoY_(self,i):
        if len(self._dimensions)==2:return int32(int32(i)%self._dimensions[1])
        return int32((int32(i)//self._dimensions[2])%self._dimensions[1])
    def ItoZ(self,i):
        if len(self._dimensions)!=3: raise Exception("ItoZ requires a 3D PopGrid")
        if not np.isfinite(i) or i!=int(i): raise Exception(f"index must be a finite integer i:{i}")
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.ItoZ_(i)
    def ItoZ_(self,i): return int32(int32(i)%self._dimensions[2])

    def ToI(self,x,y=None,z=None):
        if not np.isfinite(x) or x!=int(x): raise Exception(f"x must be an integer x:{x}")
        if y is not None and (not np.isfinite(y) or y!=int(y)): raise Exception(f"y must be an integer y:{y}")
        if z is not None and (not np.isfinite(z) or z!=int(z)): raise Exception(f"z must be an integer z:{z}")
        if len(self._dimensions)==1:
            if y is not None or z is not None: raise Exception("1D PopGrid requires x only")
            if x<0 or x>=self._dimensions[0]: raise Exception(f"coordinate out of bounds x:{x}")
        elif len(self._dimensions)==2:
            if y is None or z is not None: raise Exception("2D PopGrid requires x,y")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1]: raise Exception(f"coordinates out of bounds x:{x} y:{y}")
        else:
            if y is None or z is None: raise Exception("3D PopGrid requires x,y,z")
            if x<0 or x>=self._dimensions[0] or y<0 or y>=self._dimensions[1] or z<0 or z>=self._dimensions[2]: raise Exception(f"coordinates out of bounds x:{x} y:{y} z:{z}")
        return self.ToI_(x,y,z)
    def ToI_(self,x,y=None,z=None):
        if y is None:return int32(x)
        if z is None:return int32(x)*self._dimensions[1]+int32(y)
        return int32(x)*self._dimensions[1]*self._dimensions[2]+int32(y)*self._dimensions[2]+int32(z)

    def SetI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,i)
    def SetI_(self,value,i): self._field[i]=value
    def Set(self,value,x,y=None,z=None):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.SetI_(value,self.ToI(x,y,z))
    def Set_(self,value,x,y=None,z=None): self._field[self.ToI_(x,y,z)]=value

    def AddI(self,value,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>9223372036854775807 or value<-9223372036854775808 or value>0 and self._deltas[i]>9223372036854775807-value or value<0 and self._deltas[i]<-9223372036854775808-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def AddI_(self,value,i): self._deltas[i]+=value
    def Add(self,value,x,y=None,z=None):
        i=self.ToI(x,y,z)
        if not np.isfinite(value) or value<-9223372036854775808 or value>9223372036854775807 or value!=int(value): raise Exception(f"delta must be an int64 value:{value}")
        if value>9223372036854775807 or value<-9223372036854775808 or value>0 and self._deltas[i]>9223372036854775807-value or value<0 and self._deltas[i]<-9223372036854775808-value: raise Exception(f"delta overflow at i:{i}")
        self.AddI_(value,i)
    def Add_(self,value,x,y=None,z=None): self._deltas[self.ToI_(x,y,z)]+=value

    def GetI(self,i):
        if i<0 or i>=self._length: raise Exception(f"index out of bounds i:{i}")
        return self.GetI_(i)
    def GetI_(self,i): return self._field[i]
    def Get(self,x,y=None,z=None): return self.GetI_(self.ToI(x,y,z))
    def Get_(self,x,y=None,z=None): return self._field[self.ToI_(x,y,z)]

    def Update(self):
        for i in range(self._length):
            if self._deltas[i]>0 and self._field[i]>self._capacity-self._deltas[i]: raise Exception(f"population exceeds capacity {self._capacity} at i:{i}")
            if self._deltas[i]<0 and self._deltas[i]<-self._field[i]: raise Exception(f"population cannot become negative at i:{i}")
        self.Update_()
    def Update_(self):
        for i in range(self._length):
            self._field[i]+=self._deltas[i]
            self._deltas[i]=0

    def InWrap(self,value,dimension):
        if dimension<0 or dimension>=len(self._dimensions): raise Exception(f"invalid dimension:{dimension}")
        return self.InWrap_(value,dimension)
    def InWrap_(self,value,dimension):
        if value>=0 and value<self._dimensions[dimension]:return value
        if self._wrap[dimension]:return int32(value%self._dimensions[dimension])
        return self._NULL

    def GetFieldCopy(self): return np.copy(self._field)

    def Clear(self,value=0):
        if not np.isfinite(value) or value<0 or value>self._capacity or value!=int(value): raise Exception(f"population must be an integer between 0 and capacity {self._capacity} value:{value}")
        self.Clear_(value)
    def Clear_(self,value=0):
        for i in range(self._length):
            self._field[i]=value
            self._deltas[i]=0




class PopGridType(Protocol):
    def __len__(self): ...
    def __getitem__(self,i): ...
    def Xdim(self): ...
    def Ydim(self): ...
    def Ydim_(self): ...
    def Zdim(self): ...
    def Zdim_(self): ...
    def ItoX(self,i): ...
    def ItoX_(self,i): ...
    def ItoY(self,i): ...
    def ItoY_(self,i): ...
    def ItoZ(self,i): ...
    def ItoZ_(self,i): ...
    def ToI(self,x,y=None,z=None): ...
    def ToI_(self,x,y=None,z=None): ...
    def SetI(self,value,i): ...
    def SetI_(self,value,i): ...
    def Set(self,value,x,y=None,z=None): ...
    def Set_(self,value,x,y=None,z=None): ...
    def AddI(self,value,i): ...
    def AddI_(self,value,i): ...
    def Add(self,value,x,y=None,z=None): ...
    def Add_(self,value,x,y=None,z=None): ...
    def GetI(self,i): ...
    def GetI_(self,i): ...
    def Get(self,x,y=None,z=None): ...
    def Get_(self,x,y=None,z=None): ...
    def Update(self): ...
    def Update_(self): ...
    def InWrap(self,i,dim): ...
    def InWrap_(self,i,dim): ...
    def Clear(self,value=0): ...
    def Clear_(self,value=0): ...
    def GetFieldCopy(self): ...

def PopGrid(dimensions,capacity=None) -> PopGridType:
    """Create a PopGrid using the smallest signed storage type that can hold capacity.

    Negative dimensions retain the existing wrapping convention. If capacity is omitted,
    the grid uses int64 storage and the maximum int64 population capacity.
    """
    if capacity is None:
        capacity=9223372036854775807
    if not np.isfinite(capacity) or capacity<0 or capacity>9223372036854775807 or capacity!=int(capacity):
        raise Exception(f"capacity must be a nonnegative int64 capacity:{capacity}")
    capacity=int(capacity)
    if capacity<=127:return _PopGrid8(dimensions,capacity)
    if capacity<=32767:return _PopGrid16(dimensions,capacity)
    if capacity<=2147483647:return _PopGrid32(dimensions,capacity)
    return _PopGrid64(dimensions,capacity)
