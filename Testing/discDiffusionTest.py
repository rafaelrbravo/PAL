#%%
import sys
sys.path.append('/users/rbravo/library/cloudstorage/box-box')
sys.path.append("C:/Users/Rafael/Box")
from PAL.PDEgrid import PDEgrid
import numpy as np
#%% 1D test
g1=PDEgrid((3,))
g2=PDEgrid((5,))
g3=PDEgrid((5,))
bc2=np.array([0,0.1,0.1,0,0])
bc3=np.array([-1,0.1,0.1,0.1,-1])
g1.Set(1,1)
g2.Set(2,1)
g3.Set(2,1)
for i in range(10):
    g1.Diffusion(0.1)
    g1.Update()
    g2.Diffusion1Dfv(bc2)
    g2.Update()
    g3._Diffusion1Ddisc(bc3)
    g3.Update()
print(bc2)
print(bc3)
print(g1.GetFieldCopy())
print(g2.GetFieldCopy())#[1:4])
print(g3.GetFieldCopy())#[1:4])
# %% 2D test
g1=PDEgrid((3,3))
g2=PDEgrid((5,5))
g3=PDEgrid((5,5))
bc2X=np.zeros((5,5))
bc2Y=np.zeros((5,5))
bc3=np.zeros((5,5))-1
for x in range(1,3):
    for y in range(1,4):
        bc2X[x,y]=0.1
for x in range(1,4):
    for y in range(1,3):
        bc2Y[x,y]=0.1
for x in range(1,4):
    for y in range(1,4):
        bc3[x,y]=0.1
g1.Set2D(1,1,1)
g2.Set2D(2,2,1)
g3.Set2D(2,2,1)
for i in range(2):
    g1.Diffusion(0.1)
    g1.Update()
    g2.Diffusion2Dfv(bc2X,bc2Y)
    g2.Update()
    g3._Diffusion2Ddisc(bc3)
    g3.Update()
print(bc2X)
print(bc2Y)
print(bc3)
print(g1.GetFieldCopy2D())
print(g2.GetFieldCopy2D()[1:4,1:4])
print(g3.GetFieldCopy2D()[1:4,1:4])
# %% 3D test
g1=PDEgrid((3,3,3))
g2=PDEgrid((5,5,5))
g3=PDEgrid((5,5,5))
bc2X=np.zeros((5,5,5))
bc2Y=np.zeros((5,5,5))
bc2Z=np.zeros((5,5,5))
bc3=np.zeros((5,5,5))-1
for x in range(1,3):
    for y in range(1,4):
        for z in range(1,4):
            bc2X[x,y,z]=0.1
for x in range(1,4):
    for y in range(1,3):
        for z in range(1,4):
            bc2Y[x,y,z]=0.1
for x in range(1,4):
    for y in range(1,4):
        for z in range(1,3):
            bc2Z[x,y,z]=0.1
for x in range(1,4):
    for y in range(1,4):
        for z in range(1,4):
            bc3[x,y,z]=0.1
g1.Set3D(1,1,1,1)
g2.Set3D(2,2,2,1)
g3.Set3D(2,2,2,1)
for i in range(3):
    g1.Diffusion(0.1)
    g1.Update()
    g2.Diffusion3Dfv(bc2X,bc2Y,bc2Z)
    g2.Update()
    g3._Diffusion3Ddisc(bc3)
    g3.Update()
print(bc2X)
print(bc2Y)
print(bc2Z)
print(bc3)
print(g1.GetFieldCopy3D())
print(g2.GetFieldCopy3D()[1:4,1:4,1:4])
print(g3.GetFieldCopy3D()[1:4,1:4,1:4])
## %% 3D test same boundaries
#g1=PDEgrid((3,3,3))
#g2=PDEgrid((3,3,3))
#g3=PDEgrid((3,3,3))
#bc2=np.zeros((3,3,3))+0.1
#bc3=np.zeros((3,3,3))+0.1
#g1.Set3D(1,1,1,1)
#g2.Set3D(1,1,1,1)
#g3.Set3D(1,1,1,1)
#for i in range(3):
#    g1.Diffusion(0.1)
#    g1.Update()
#    g2.DiffusionDisc3Dfv(bc3,bc3,bc3)
#    g2.Update()
#    g3.Diffusion3Dbc(bc3)
#    g3.Update()
#print(bc2)
#print(bc3)
#print(g1.GetFieldCopy3D())
#print(g2.GetFieldCopy3D())
#print(g3.GetFieldCopy3D())
## %%

# %%
