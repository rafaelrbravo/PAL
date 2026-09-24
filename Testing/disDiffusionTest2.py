#%%
import sys
sys.path.append('/users/rbravo/library/cloudstorage/box-box')
sys.path.append("C:/Users/Rafael/Box")
from PAL.PDEgrid import PDEgrid
import numpy as np

#%%
LENGTH=32
dx=1/LENGTH
xs=np.arange(LENGTH)
ds=(np.sin((xs*np.pi)/(LENGTH-1)))/(10000*np.square(dx))
ds2=(np.sin(((xs+0.5)*np.pi)/(LENGTH-1)))/(10000*np.square(dx))
print(max(ds))
print(max(ds2))

print(ds)
print(ds2)
# %%
pde1=PDEgrid((LENGTH,))
pde2=PDEgrid((LENGTH,))
pde1.Set(LENGTH//2,1)
pde2.Set(LENGTH//2,1)
for t in range(100):
    pde1._Diffusion1Ddisc(ds)
    pde1.Update()
    pde2.Diffusion1Dfv(ds2)
    pde2.Update()

out=pde1.GetFieldCopy()
out2=pde2.GetFieldCopy()
print(out)
print(out2)
print(np.average(np.square(out-out2)))


# %%
