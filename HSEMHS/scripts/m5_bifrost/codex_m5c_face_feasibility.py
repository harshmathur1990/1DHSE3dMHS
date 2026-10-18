# Optional, time-limited face-data feasibility diagnostic, not an MHS solve.
# See CODEX_REVIEW_M5c_2026-10-18.md for the limitations and the inconclusive run.
import numpy as np
import h5py
from scipy.sparse import coo_matrix
from scipy.optimize import linprog
from time import monotonic

path='/Users/harshm/Documents/WorkRepo/MHS/IO/input-mhs-hse.nc'
with h5py.File(path,'r') as f:
    z=f['z'][::2].astype(float)
    x=f['x'][:].reshape(42,12).mean(axis=1)
    y=f['y'][:].reshape(42,12).mean(axis=1)
    B=[]
    for name in ('B_x','B_y','B_z'):
        # Same block averaging / vertical subsampling; use Float32 accumulation
        # as in the source's mean of Float32 blocks. Summation order may differ.
        a=f[name][:,:,::2]
        B.append(a.reshape(42,12,42,12,89).mean(axis=(1,3),dtype=np.float32).astype(float))
nx,ny,nz=B[0].shape
sig=np.sqrt((0.05*np.sqrt(sum(q*q for q in B)))**2+(5e-4)**2)
obs=[0.5*(B[0]+np.roll(B[0],-1,axis=0)),0.5*(B[1]+np.roll(B[1],-1,axis=1)),np.concatenate([B[2][:,:,:1],0.5*(B[2][:,:,:-1]+B[2][:,:,1:]),B[2][:,:,-1:]],axis=2)]
sf=[np.sqrt(sig*np.roll(sig,-1,axis=0)),np.sqrt(sig*np.roll(sig,-1,axis=1)),np.concatenate([sig[:,:,:1],np.sqrt(sig[:,:,:-1]*sig[:,:,1:]),sig[:,:,-1:]],axis=2)]
dx=float(np.mean(np.diff(x)));dy=float(np.mean(np.diff(y)))
zf=np.concatenate([[z[0]+0.5*(z[0]-z[1])],0.5*(z[:-1]+z[1:]),[z[-1]-0.5*(z[-2]-z[-1])]])
hz=(zf[:-1]-zf[1:])[None,None,:]
ids=np.arange(nx*ny*nz).reshape(nx,ny,nz)
n=ids.size
idz=np.arange(nx*ny*(nz+1)).reshape(nx,ny,nz+1)+2*n
cols=[ids,np.roll(ids,1,axis=0),ids+n,np.roll(ids+n,1,axis=1),idz[:,:,:-1],idz[:,:,1:]]
coeff=[sf[0]/dx,-np.roll(sf[0],1,axis=0)/dx,sf[1]/dy,-np.roll(sf[1],1,axis=1)/dy,sf[2][:,:,:-1]/hz,-sf[2][:,:,1:]/hz]
den=sum(abs(q) for q in coeff)
div=(obs[0]-np.roll(obs[0],1,axis=0))/dx+(obs[1]-np.roll(obs[1],1,axis=1))/dy+(obs[2][:,:,:-1]-obs[2][:,:,1:])/hz
A=coo_matrix((np.concatenate([(q/den).ravel() for q in coeff]),(np.tile(ids.ravel(),6),np.concatenate([q.ravel() for q in cols]))),shape=(n,3*n+nx*ny)).tocsc()
rhs=(-div/den).ravel()
print('necessary local max-sigma bound:',np.max(abs(rhs)),flush=True)
print('LP max-sigma feasibility: variables',A.shape[1],'equalities',A.shape[0],'nonzeros',A.nnz,flush=True)
tm=monotonic()
res=linprog(np.zeros(A.shape[1]),A_eq=A,b_eq=rhs,bounds=(-4.9,4.9),method='highs',options={'time_limit':60.0,'primal_feasibility_tolerance':1e-8})
print('LP status:',res.status,res.message,'seconds',monotonic()-tm,flush=True)
if res.success:
    print('max normalized equality error:',np.max(abs(A@res.x-rhs)),flush=True)
    print('max correction/sigma:',np.max(abs(res.x)),'fraction above 3 sigma:',np.mean(abs(res.x)>3),'mean squared correction:',np.mean(res.x**2),flush=True)
