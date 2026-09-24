#%%
import sys
sys.path.append('/users/rbravo/library/cloudstorage/box-box')
sys.path.append("C:/Users/Rafael/Box")
from PAL.PDEgrid import PDEgrid
g1=PDEgrid((3,))
g2=PDEgrid((3,3))
g3=PDEgrid((3,3,3))
def ClearPDE(g):
    for i in range(len(g)):
        g.Set(i,0)
#%% 1D
ClearPDE(g1)
g1.Set1D(0,1)
g1.Set1D(1,2)
g1.Set1D(2,3)
print([g1._GradX1D(0),g1._GradX1D(1),g1._GradX1D(2)])
# %% 2D x
ClearPDE(g2)
g2.Set2D(0,0,1)
g2.Set2D(1,0,2)
g2.Set2D(2,0,3)
print([g2._GradX2D(0,0),g2._GradX2D(1,0),g2._GradX2D(2,0)])

#%% 2D y
ClearPDE(g2)
g2.Set2D(0,0,1)
g2.Set2D(0,1,2)
g2.Set2D(0,2,3)
print([g2._GradY2D(0,0),g2._GradY2D(0,1),g2._GradY2D(0,2)])
# %% 3D x
ClearPDE(g3)
g3.Set3D(0,0,0,1)
g3.Set3D(1,0,0,2)
g3.Set3D(2,0,0,3)
print([g3._GradX3D(0,0,0),g3._GradX3D(1,0,0),g3._GradX3D(2,0,0)])
# %% 3D y
ClearPDE(g3)
g3.Set3D(0,1,0,1)
g3.Set3D(0,2,0,2)
g3.Set3D(0,3,0,3)
print([g3._GradY3D(0,0,0),g3._GradY3D(0,1,0),g3._GradY3D(0,2,0)])
# %%
ClearPDE(g3)
g3.Set3D(0,0,0,1)
g3.Set3D(0,0,1,2)
g3.Set3D(0,0,2,3)
print([g3._GradZ3D(0,0,0),g3._GradZ3D(0,0,1),g3._GradZ3D(0,0,2)])
