import numpy as NP
from numba import njit,int32,float64,int64
from numba.experimental import jitclass
from numpy.random import rand
import time

mnSpec=[('_n',int64), 
          ('_p',float64), 
          ('_n_last',int64), 
          ('_n_prev',int64),
          ('_par',float64),
          ('_np',float64),
          ('_q',float64),
          ('_p_last',float64),
          ('_p_prev',float64),
          ('_m',int64),
          ('_nm',int64),
          ('_pq',float64),
          ('_rc',float64),
          ('_ss',float64),
          ('_xm',float64),
          ('_xl',float64),
          ('_xr',float64),
          ('_ll',float64),
          ('_lr',float64),
          ('_c',float64),
          ('_p1',float64),
          ('_p2',float64),
          ('_p3',float64),
          ('_p4',float64),
          ('_ch',float64),
          ('_br',float64),
          ('_bnr',float64),
          ('_bnpq',float64),
          ('_bb',float64),
          ('_ba',float64),
          ('_bc',float64),
          ('_balpha',float64),
          ('_bvr',float64),
          ('_burvr',float64),
          ('_bm',int64),
          ('_bn_last',int64),
          ('_bp_last',float64),
          ('_stirlingCorrection',float64[:])
          ]

@jitclass(mnSpec)
class Binomial(object):
    def __init__(self):
        self._n=0
        self._p=0.0
        self._n_last=-1
        self._n_prev=-1
        self._par=0.0
        self._np=0.0
        self._q=0.0
        self._p_last=-1.0
        self._p_prev=-1.0
        self._m=0
        self._nm=0
        self._pq=0.0
        self._rc=0.0
        self._ss=0.0
        self._xm=0.0
        self._xl=0.0
        self._xr=0.0
        self._ll=0.0
        self._lr=0.0
        self._c=0.0
        self._p1=0.0
        self._p2=0.0
        self._p3=0.0
        self._p4=0.0
        self._ch=0.0
        self._br=0.0
        self._bnr=0.0
        self._bnpq=0.0
        self._bb=0.0
        self._ba=0.0
        self._bc=0.0
        self._balpha=0.0
        self._bvr=0.0
        self._burvr=0.0
        self._bm=0
        self._bn_last=-1
        self._bp_last=-1.0
        self._stirlingCorrection=NP.array((0.0,
            8.106146679532726e-02, 4.134069595540929e-02,
            2.767792568499834e-02, 2.079067210376509e-02,
            1.664469118982119e-02, 1.387612882307075e-02,
            1.189670994589177e-02, 1.041126526197209e-02,
            9.255462182712733e-03, 8.330563433362871e-03,
            7.573675487951841e-03, 6.942840107209530e-03,
            6.408994188004207e-03, 5.951370112758848e-03,
            5.554733551962801e-03, 5.207655919609640e-03,
            4.901395948434738e-03, 4.629153749334029e-03,
            4.385560249232324e-03, 4.166319691996922e-03,
            3.967954218640860e-03, 3.787618068444430e-03,
            3.622960224683090e-03, 3.472021382978770e-03,
            3.333155636728090e-03, 3.204970228055040e-03,
            3.086278682608780e-03, 2.976063983550410e-03,
            2.873449362352470e-03, 2.777674929752690e-03),dtype=NP.float64) 

    def Sample(self,n,p):
        if not NP.isfinite(n) or n<0 or n>9223372036854775807 or n!=NP.floor(n): raise Exception(f"n must be an integer between 0 and 9223372036854775807 n:{n}")
        if not NP.isfinite(p) or p<0 or p>1: raise Exception(f"p must be between 0 and 1 p:{p}")
        return self.Sample_(n,p)

    def Sample_(self,n,p):
        if p==1:return n
        if p==0 or n==0:return 0
        if n<=3:return self._CoinFlip(n,p)
        q=1.0-p if p>0.5 else p
        mean=NP.float64(n)*q
        if mean<10.0:return NP.random.binomial(n,p)
        if n<=68719476736:
            ret=self._BTRD(n,q)
            return n-ret if p>0.5 else ret
        return self._ColtInt(n,p)

    def _StirlingCorrection(self,k):
        C1 =  8.33333333333333333e-02
        C3 = -2.77777777777777778e-03
        C5 =  7.93650793650793651e-04
        C7 = -5.95238095238095238e-04

        r=0.0
        rr=0.0

        if (k > 30):
            r = 1.0 / k
            rr = r * r
            return r*(C1 + rr*(C3 + rr*(C5 + rr*C7)))
        else:return self._stirlingCorrection[k]


    def _BTRDFC(self,k):
        if k<10:return self._stirlingCorrection[k+1]
        x=1.0/(k+1.0)
        x2=x*x
        return (1.0/12.0-(1.0/360.0-(1.0/1260.0)*x2)*x2)*x

    def _BTRD(self,n,p):
        if n!=self._bn_last or p!=self._bp_last:
            self._bn_last=n
            self._bp_last=p
            self._bm=NP.int64((NP.float64(n)+1.0)*p)
            self._br=p/(1.0-p)
            self._bnr=(NP.float64(n)+1.0)*self._br
            self._bnpq=NP.float64(n)*p*(1.0-p)
            s=NP.sqrt(self._bnpq)
            self._bb=1.15+2.53*s
            self._ba=-0.0873+0.0248*self._bb+0.01*p
            self._bc=NP.float64(n)*p+0.5
            self._balpha=(2.83+5.1/self._bb)*s
            self._bvr=0.92-4.2/self._bb
            self._burvr=0.86*self._bvr
        while True:
            v=rand()
            if v<=self._burvr:
                u=v/self._bvr-0.43
                return NP.int64(NP.floor((2.0*self._ba/(0.5-NP.abs(u))+self._bb)*u+self._bc))
            if v>=self._bvr:u=rand()-0.5
            else:
                u=v/self._bvr-0.93
                u=(-0.5 if u<0 else 0.5)-u
                v=rand()*self._bvr
            us=0.5-NP.abs(u)
            k=NP.int64(NP.floor((2.0*self._ba/us+self._bb)*u+self._bc))
            if k<0 or k>n:continue
            v=v*self._balpha/(self._ba/(us*us)+self._bb)
            km=NP.abs(k-self._bm)
            if km<=15:
                f=1.0
                if self._bm<k:
                    i=self._bm
                    while i!=k:
                        i+=1
                        f*=self._bnr/i-self._br
                elif self._bm>k:
                    i=k
                    while i!=self._bm:
                        i+=1
                        v*=self._bnr/i-self._br
                if v<=f:return k
                continue
            v=NP.log(v)
            rho=(km/self._bnpq)*(((km/3.0+0.625)*km+1.0/6.0)/self._bnpq+0.5)
            t=-km*km/(2.0*self._bnpq)
            if v<t-rho:return k
            if v>t+rho:continue
            nm=n-self._bm+1
            h=(self._bm+0.5)*NP.log((self._bm+1)/(self._br*nm))+self._BTRDFC(self._bm)+self._BTRDFC(n-self._bm)
            nk=n-k+1
            if v<=h+(NP.float64(n)+1.0)*NP.log(NP.float64(nm)/nk)+(k+0.5)*NP.log(NP.float64(nk)*self._br/(k+1))-self._BTRDFC(k)-self._BTRDFC(n-k):return k

    def _ColtInt(self,n, p) :
        i=0
        f=0.0
        if(n != self._n_last or p != self._p_last) :
            self._n_last = n
            self._p_last = p
            self._par = min(p, 1.0 - p)
            self._q = 1.0 - self._par
            self._np = NP.float64(n) * self._par
            if(self._np <= 0.0): return -1

            rm = self._np + self._par
            self._m = NP.int64(rm)
            self._pq = self._par / self._q
            self._rc = (NP.float64(n) + 1.0) * self._pq
            self._ss = self._np * self._q
            i = NP.int64(2.195 * NP.sqrt(self._ss) - 4.6 * self._q)
            self._xm = NP.float64(self._m) + 0.5
            self._xl = NP.float64(self._m - i)
            self._xr = NP.float64(NP.int64(self._m + i) + 1)
            f = (rm - self._xl) / (rm - self._xl * self._par)
            self._ll = f * (1.0 + 0.5 * f)
            f = (self._xr - rm) / (self._xr * self._q)
            self._lr = f * (1.0 + 0.5 * f)
            self._c = 0.134 + 20.5 / (15.3 + NP.float64(self._m))
            self._p1 = NP.float64(i) + 0.5
            self._p2 = self._p1 * (1.0 + self._c + self._c)
            self._p3 = self._p2 + self._c / self._ll
            self._p4 = self._p3 + self._c / self._lr

        K=0
        U=0.0
        while(1):
            V=0.0
            while(1):
                V = rand()
                U = rand() * self._p4
                if(U <= self._p1):
                    K = NP.int64(self._xm - U + self._p1 * V)
                    if p>0.5:return n-K
                    else:return K

                X=0.0
                if(U <= self._p2):
                    X = self._xl + (U - self._p1) / self._c
                    V = V * self._c + 1.0 - NP.abs(self._xm - X) / self._p1
                    if(V < 1.0):
                        K = NP.int64(X)
                        break
                elif(U <= self._p3):
                    X = self._xl + NP.log(V) / self._ll
                    if(X >= 0.0):
                        K = NP.int64(X)
                        V *= (U - self._p2) * self._ll
                        break
                else:
                    K = NP.int64(self._xr - NP.log(V) / self._lr)
                    if(K<=n):
                        V *= (U - self._p3) * self._lr
                        break

            Km = NP.abs(K - self._m)
            if Km > 20 and NP.float64(NP.int64(Km + Km) + 2) < self._ss:
                V = NP.log(V)
                T = NP.float64(NP.float64(-Km * Km) / (self._ss + self._ss))
                E = NP.float64(Km) / self._ss * ((NP.float64(Km) * (NP.float64(Km) * 0.3333333333333333 + 0.625) + 0.16666666666666666) / self._ss + 0.5)
                if(V <= T - E):
                    break

                if(V <= T + E):
                    if(n != self._n_prev or self._par != self._p_prev):
                        self._n_prev = n
                        self._p_prev = self._par
                        self._nm = n - self._m + 1
                        self._ch = self._xm * NP.log((NP.float64(self._m) + 1.0) / (self._pq * NP.float64(self._nm))) + self._StirlingCorrection(self._m + 1) + self._StirlingCorrection(self._nm)

                    nK = NP.int64(n - K + 1)
                    if(V <= self._ch + (NP.float64(n) + 1.0) * NP.log(NP.float64(self._nm) / NP.float64(nK)) + (NP.float64(K) + 0.5) * NP.log(NP.float64(nK) * self._pq / (NP.float64(K) + 1.0)) - self._StirlingCorrection(K + 1) - self._StirlingCorrection(nK)):
                        break
            else:
                f = 1.0
                if self._m < K:
                    i = self._m

                    while(i < K):
                        i+=1
                        f *= self._rc / NP.float64(i) - self._pq
                        if(f < V):
                            break
                else:
                    i = K

                    while(i < self._m):
                        i+=1
                        V *= self._rc / NP.float64(i) - self._pq
                        if(V > f):
                            break

                if(V <= f):
                    break

        if p > 0.5:return n - K
        else:return K

    def _CoinFlip(self,n,p):
            total=0
            for i in range(n):
                if rand()<p:
                    total+=1
            return total

    def _rk_binomial_inversion(self,n, p):
        q=0.0;qn=0.0;np=0.0;px=0.0;U=0.0
        X=0;bound=0

        q = 1.0 - p
        qn = NP.exp(n * NP.log(q))
        np = n*p
        bound = int(min(n, np + 10.0*NP.sqrt(np*q + 1)))

        X = 0
        px = qn

        U = rand()
        while (U > px):
            X+=1
            if (X > bound):
                X = 0
                px = qn
                U = rand()
            else:
                U -= px
                px  = ((n-X+1) * p * px)/(X*q)
        return X

    def SetupMulti(self,n):
        if not NP.isfinite(n) or n<0 or n>9223372036854775807 or n!=NP.floor(n): raise Exception(f"n must be an integer between 0 and 9223372036854775807 n:{n}")
        self.SetupMulti_(n)

    def SetupMulti_(self,n):
        self._n=n
        self._p=1

    def SampleMulti(self,p):
        if not NP.isfinite(p) or p<0 or p>self._p+1e-12: raise Exception(f"p must be between 0 and remaining probability {self._p} p:{p}")
        return self.SampleMulti_(p)

    def SampleMulti_(self,p):
        if self._n==0 or p==0:return 0
        if self._p-p<=0:
            nSelected=self._n
            self._n=0
            self._p=0
            return nSelected
        nSelected=self.Sample_(self._n,NP.float64(p)/self._p)
        self._p-=p
        self._n-=nSelected
        return nSelected


@njit
def Test(binomial:Binomial,n,p,repeats,coinFlip):
    for j in range(repeats):
        if coinFlip:
            total=0
            for i in range(n):
                if rand()<p:
                    total+=1
        else: binomial.Sample(n,p)


#bin=Binomial()
#N=50
#P=0.1
#Test(bin,N,P,100,0)
#t0=time.time_ns()
#Test(bin,N,P,10000,0)
#print(time.time_ns()-t0)