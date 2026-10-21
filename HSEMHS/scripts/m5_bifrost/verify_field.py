#!/usr/bin/env python3
"""Independent check of a saved face field (any .npz with B_face in the flat face order of data_lp.py: x-faces,
y-faces, z-faces, each C-order (x, y, z)) against the shared observation artifact: divergence recomputed from the
metrics, face-model and centre-model residuals, maxima and tails (budget 1 % of each observation set), mean c².
Gates (M5g, Codex M5f review finding 6): every value finite and max |div B| ≤ DIV_TOL relative to median|y_f|/median(dx);
a field failing a gate is reported INVALID and the exit code is 1. Criteria lines (max ≤ 5σ, tail ≤ K) are printed only
for valid fields.
usage: python3 verify_field.py OBS.h5 FIELD.npz [FIELD2.npz ...]"""
import sys, numpy as np, h5py
obs = sys.argv[1]
DIV_TOL = 1e-9
ninvalid = 0
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
    finite = bool(np.all(np.isfinite(Bf)) and np.all(np.isfinite(cf)) and np.all(np.isfinite(cc)))
    div_rel = np.max(np.abs(d)) / (np.median(np.abs(fobs)) / np.median(dxc)) if finite else np.inf
    valid = finite and div_rel <= DIV_TOL
    print(f"{path}: max|div B| {np.max(np.abs(d)):.2e} T/m (relative {np.max(np.abs(d)) / (np.median(np.abs(fobs)) / np.median(dxc)):.1e})\n"
          f"  FACE   max {np.max(np.abs(cf)):.6f}σ  >3σ {np.sum(np.abs(cf) > 3)} ({np.mean(np.abs(cf) > 3):.4%}; K {Kf})  #>5σ {np.sum(np.abs(cf) > 5)}  mean c² {np.mean(cf ** 2):.4f}\n"
          f"  CENTRE max {np.max(np.abs(cc)):.6f}σ  >3σ {np.sum(np.abs(cc) > 3)} ({np.mean(np.abs(cc) > 3):.4%}; K {Kc})  #>5σ {np.sum(np.abs(cc) > 5)}  mean c² {np.mean(cc ** 2):.4f}")
    if not valid:
        ninvalid += 1
        print(f"  INVALID: finite {finite}, relative divergence {div_rel:.2e} (tolerance {DIV_TOL:.0e})")
    else:
        for name, c, K in (("FACE", cf, Kf), ("CENTRE", cc, Kc)):
            print(f"  {name}: max ≤ 5σ {np.max(np.abs(c)) <= 5}, tail > 3σ ≤ 1 % {np.sum(np.abs(c) > 3) <= K}")
sys.exit(1 if ninvalid else 0)
