
from AgentGrid import AgentGrid,QueryList
import numpy as np
import PixWindow as PixWindow
from numba import njit

@njit
def Mutate(value,mutRate):
    return max(min(value+(np.random.random()-0.5)*2*mutRate,1.0),0.0)

@njit
def Step(g:AgentGrid,hood:QueryList,mutRate:float):
    for a in g.All(True): 
        g.MapHood2D(hood,g.XSQ(a),g.YSQ(a),0)
        iNew=hood.Random()
        if iNew>=0:
            aNew=g.NewAgentSQ(iNew)
            g.SetP(aNew,0,Mutate(g.GetP(a,0),mutRate))
            g.SetP(aNew,1,Mutate(g.GetP(a,1),mutRate))
            g.SetP(aNew,2,Mutate(g.GetP(a,2),mutRate))
        if np.random.random()<0.2:
            g.Dispose(a)

@njit
def Draw(g:AgentGrid,pix:np.array):
    for i in range(len(g)):
        a=g.GetLastI(i)
        if a!=-1:
            pix[i,0]=int(g.GetP(a,0)*255)
            pix[i,1]=int(g.GetP(a,1)*255)
            pix[i,2]=int(g.GetP(a,2)*255)

if __name__=="__main__":

    g=AgentGrid((500,500),3,False)
    a0=g.NewAgentSQ2D(250,250)
    g.SetP(a0,0,0.5)
    g.SetP(a0,1,0.5)
    g.SetP(a0,2,0.5)
    hood=QueryList()
    hood.SetHood(2,[1,0,-1,0,0,1,0,-1])
    pix,win=PixWindow.StartPixWindow(500,500,2)
    flatPix=np.reshape(pix,(-1,3))
    while(win.IsOpen()):
        Step(g,hood,0.01)
        Draw(g,flatPix)
