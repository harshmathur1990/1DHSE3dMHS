#!/usr/bin/env python3
"""Independent check of a saved face field (any .npz with B_face in the flat face order of data_lp.py: x-faces,
y-faces, z-faces, each C-order (x, y, z)) against the shared observation artifact: divergence recomputed from the
metrics, face-model and centre-model residuals, maxima and tails (budget 1 % of each observation set), mean c².
usage: python3 verify_field.py OBS.h5 FIELD.npz [FIELD2.npz ...]"""
import sys, numpy as np, h5py
obs = sys.argv[1]
with h5py.File(obs, "r") as f:
    rd = lambda k: np.ascontiguousarray(f[k][()].T)
    fo = [rd(f"face_obs{c}") for c in (1, 2, 3)]; fs = [rd(f"face_sigma{c}") for c in (1, 2, 3)]
    co = [rd(f"centre_obs{c}") for c in (1, 2, 3)]; cs = [rd(f"centre_sigma{c}") for c in (1, 2, 3)]
    dxc = f["dxc"][()]; dyc = f["dyc"][()]; hz = f["hz"][()]
nx, ny, nz = co[0].shape; n = nx * ny * nz; nf = 3 * n + nx * ny
fobs = np.concatenate([v.ravel() for v in fo]); fsig = np.concatenate([v.ravel() for v in fs])
cobs = np.concatenate([v.ravel() for v in co]); csig = np.concatenate([v.ravel() for v in cs])
for path in sys.argv[2:]:
    Bf = np.load(path)["B_face"]; assert Bf.size == nf, f"{path}: {Bf.size} faces, expected {nf}"
    Bx, By, Bz = Bf[:n].reshape(nx, ny, nz), Bf[n:2 * n].reshape(nx, ny, nz), Bf[2 * n:].reshape(nx, ny, nz + 1)
    d = (Bx - np.roll(Bx, 1, 0)) / dxc[:, None, None] + (By - np.roll(By, 1, 1)) / dyc[None, :, None] + (Bz[:, :, :-1] - Bz[:, :, 1:]) / hz
    C = np.concatenate([(0.5 * (Bx + np.roll(Bx, 1, 0))).ravel(), (0.5 * (By + np.roll(By, 1, 1))).ravel(), (0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])).ravel()])
    cf = (Bf - fobs) / fsig; cc = (C - cobs) / csig
    Kf, Kc = int(0.01 * nf), int(0.01 * 3 * n)
    print(f"{path}: max|div B| {np.max(np.abs(d)):.2e} T/m (relative {np.max(np.abs(d)) / (np.median(np.abs(fobs)) / np.median(dxc)):.1e})\n"
          f"  FACE   max {np.max(np.abs(cf)):.6f}σ  >3σ {np.sum(np.abs(cf) > 3)} ({np.mean(np.abs(cf) > 3):.4%}; K {Kf})  #>5σ {np.sum(np.abs(cf) > 5)}  mean c² {np.mean(cf ** 2):.4f}\n"
          f"  CENTRE max {np.max(np.abs(cc)):.6f}σ  >3σ {np.sum(np.abs(cc) > 3)} ({np.mean(np.abs(cc) > 3):.4%}; K {Kc})  #>5σ {np.sum(np.abs(cc) > 5)}  mean c² {np.mean(cc ** 2):.4f}")
