"""Review diagnostic: eliminate face variables from centre maximum feasibility.

For uniform periodic x/y, Hx=(I+shift_x)/2 and Hy=(I+shift_y)/2.
Writing C=H B, q=Hy Dx Cx + Hx Dy Cy, divergence-free B implies
    hz[k] q[k] + hz[k+1] q[k+1] + 2 Hx Hy (Cz[k]-Cz[k+1]) = 0.
Additionally, Cx has zero alternating x sum and Cy zero alternating y sum
on even grids. These conditions are sufficient too: invert Hx,Hy on their
nonzero Fourier modes, then recover Bz by its divergence recurrence. On
horizontal Nyquist modes, the unobserved face component enforces divergence.
This script reconstructs B and checks both H B and div B directly.

Usage from HSEMHS/: python3 scripts/m5_bifrost/codex_m5e_centre_feasibility.py
    --obs out/m5d/observations.h5 --t 4.999 --time-limit 60 --out FILE.npz
Does not modify production solver code. No tail constraint is imposed.
"""
import argparse
import subprocess
import sys
import time
import h5py
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ap = argparse.ArgumentParser()
ap.add_argument('--obs', required=True)
ap.add_argument('--t', type=float, default=4.999)
ap.add_argument('--time-limit', type=float, default=60)
ap.add_argument('--out', required=True)
ap.add_argument('--check-only', action='store_true')
ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
args = ap.parse_args()
if not args.worker and not args.check_only:
    try:
        result = subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker'],
                                timeout=args.time_limit+20)
        raise SystemExit(result.returncode)
    except subprocess.TimeoutExpired:
        print('External wall limit reached; worker stopped, feasibility unresolved.', flush=True)
        raise SystemExit(124)
with h5py.File(args.obs) as f:
    obs = [f[f'centre_obs{c}'][()].T for c in (1, 2, 3)]
    sig = [f[f'centre_sigma{c}'][()].T for c in (1, 2, 3)]
    dx, dy, hz = [f[k][:] for k in ('dxc', 'dyc', 'hz')]
nx, ny, nz = obs[0].shape
assert np.all(dx == dx[0]) and np.all(dy == dy[0]), 'Requires uniform x/y'
n = nx*ny*nz
ids = np.arange(n).reshape(nx, ny, nz)
nr = nx*ny*(nz-1)
row = np.arange(nr)
rows, cols, vals = [], [], []
def add(index, coefficient):
    rows.append(row)
    cols.append(index.ravel())
    vals.append(np.broadcast_to(coefficient, (nx, ny, nz-1)).ravel())
for zoff in (0, 1):
    zsel = slice(zoff, nz-1+zoff)
    for xoff in (0, 1):
        for yoff in (0, 1):
            index = np.roll(np.roll(ids, xoff, 0), yoff, 1)[:, :, zsel]
            add(index, (-1)**xoff * .5*hz[zsel]/dx[0])
            add(index+n, (-1)**yoff * .5*hz[zsel]/dy[0])
            add(index+2*n, (-1)**zoff * .5)
L = sp.coo_matrix((np.concatenate(vals),
                  (np.concatenate(rows), np.concatenate(cols))), shape=(nr, 3*n)).tocsr()
# Necessary Nyquist-range constraints; these are essential for sufficiency.
nyq = []
if nx % 2 == 0:
    rr = np.broadcast_to(np.arange(ny*nz).reshape(1, ny, nz), (nx, ny, nz))
    vv = np.broadcast_to((-1.)**np.arange(nx)[:, None, None], (nx, ny, nz))
    nyq.append(sp.coo_matrix((vv.ravel(), (rr.ravel(), ids.ravel())), shape=(ny*nz, 3*n)).tocsr())
if ny % 2 == 0:
    rr = np.broadcast_to(np.arange(nx*nz).reshape(nx, 1, nz), (nx, ny, nz))
    vv = np.broadcast_to((-1.)**np.arange(ny)[None, :, None], (nx, ny, nz))
    nyq.append(sp.coo_matrix((vv.ravel(), (rr.ravel(), (ids+n).ravel())), shape=(nx*nz, 3*n)).tocsr())
L = sp.vstack([L]+nyq, format='csr')
y = np.concatenate([v.ravel() for v in obs])
s = np.concatenate([v.ravel() for v in sig])
A = (L @ sp.diags(s)).tocsr()
den = np.asarray(abs(A).sum(axis=1)).ravel()
rhs = -(L@y)/den
A = (sp.diags(1/den)@A).tocsr()
print('Reduced centre LP:', A.shape, 'nnz', A.nnz,
      'necessary local bound', max(abs(rhs)), flush=True)

def centre(B):
    return [.5*(B[0]+np.roll(B[0], 1, 0)), .5*(B[1]+np.roll(B[1], 1, 1)),
            .5*(B[2][:, :, :-1]+B[2][:, :, 1:])]
def divergence(B):
    return ((B[0]-np.roll(B[0], 1, 0))/dx[:, None, None]
            +(B[1]-np.roll(B[1], 1, 1))/dy[None, :, None]
            +(B[2][:, :, :-1]-B[2][:, :, 1:])/hz)
def reconstruct(C):
    cf = [np.fft.fftn(v, axes=(0, 1)) for v in C]
    ex = np.exp(-2j*np.pi*np.fft.fftfreq(nx))[:, None, None]
    ey = np.exp(-2j*np.pi*np.fft.fftfreq(ny))[None, :, None]
    hx, hy = .5*(1+ex), .5*(1+ey)
    dxsym, dysym = (1-ex)/dx[0], (1-ey)/dy[0]
    zx, zy = abs(hx)<1e-12, abs(hy)<1e-12
    bx = np.divide(cf[0], hx, out=np.zeros_like(cf[0]), where=~zx)
    by = np.divide(cf[1], hy, out=np.zeros_like(cf[1]), where=~zy)
    regular = np.broadcast_to(~(zx|zy), (nx, ny, 1))[:, :, 0]
    d = dxsym*bx+dysym*by
    bz = np.zeros((nx, ny, nz+1), dtype=complex)
    bz[:, :, 0] = cf[2][:, :, 0]-.5*hz[0]*d[:, :, 0]
    bz[:, :, 1:] = bz[:, :, :1]+np.cumsum(d*hz, axis=2)
    # On a horizontal Nyquist mode Hx or Hy is zero: its free face component
    # can fit divergence after Bz is chosen to reproduce all observed Cz.
    bz[~regular, 0] = 0
    for k in range(nz):
        bz[~regular, k+1] = 2*cf[2][~regular, k]-bz[~regular, k]
    target_d = (bz[:, :, 1:]-bz[:, :, :-1])/hz
    new_bx = np.divide(target_d-dysym*by, dxsym, out=np.zeros_like(bx), where=abs(dxsym)>1e-12/dx[0])
    bx = np.where(zx, new_bx, bx)
    new_by = np.divide(target_d-dxsym*bx, dysym, out=np.zeros_like(by), where=abs(dysym)>1e-12/dy[0])
    by = np.where(zy & ~zx, new_by, by)
    B = [np.fft.ifftn(v, axes=(0, 1)).real for v in (bx, by, bz)]
    return B

# Check elimination against an unrelated random curl field, including boundaries.
rng = np.random.default_rng(51)
ax, ay = [rng.normal(size=(nx, ny, nz+1)) for _ in range(2)]
az = rng.normal(size=(nx, ny, nz))
curl = [(az-np.roll(az, 1, 1))/dy[0]-(ay[:, :, :-1]-ay[:, :, 1:])/hz,
        (ax[:, :, :-1]-ax[:, :, 1:])/hz-(az-np.roll(az, 1, 0))/dx[0],
        (ay-np.roll(ay, 1, 0))/dx[0]-(ax-np.roll(ax, 1, 1))/dy[0]]
curl_C = np.concatenate([v.ravel() for v in centre(curl)])
print('Random-curl check: max |L H curl|', max(abs(L@curl_C)),
      'max |div curl|', np.max(abs(divergence(curl))), flush=True)
test_C = centre(curl)
test_B = reconstruct(test_C)
print('Random-curl reconstruction: max centre error',
      max(np.max(abs(a-b)) for a,b in zip(centre(test_B),test_C)),
      'max divergence', np.max(abs(divergence(test_B))), flush=True)
if args.check_only:
    raise SystemExit(0)
tm = time.time()
res = linprog(np.zeros(3*n), A_eq=A, b_eq=rhs, bounds=(-args.t, args.t),
              method='highs-ipm', options={'time_limit': args.time_limit,
              'primal_feasibility_tolerance': 1e-9, 'disp': True})
print('status', res.status, res.message, 'seconds', time.time()-tm, flush=True)
if not res.success:
    raise SystemExit(0)
C = [(y+s*res.x)[q*n:(q+1)*n].reshape(nx, ny, nz) for q in range(3)]
B = reconstruct(C)
pred = centre(B)
cor = np.concatenate([((v-o)/sg).ravel() for v, o, sg in zip(pred, obs, sig)])
print('Reconstructed field: max centre sigma', max(abs(cor)), 'tail', np.mean(abs(cor)>3),
      'max physical div', np.max(abs(divergence(B))),
      'max reconstruction error', max(np.max(abs(a-b)) for a,b in zip(pred,C)), flush=True)
np.savez(args.out, B_face=np.concatenate([v.ravel() for v in B]),
         c_centre=cor, centre_c=res.x, t=args.t, shape=(nx,ny,nz),
         max_div=np.max(abs(divergence(B))))
print('Saved', args.out, flush=True)
