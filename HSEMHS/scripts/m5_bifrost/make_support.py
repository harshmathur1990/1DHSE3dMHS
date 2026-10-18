#!/usr/bin/env python3
"""
Build the fixed support S for the simultaneous maximum/tail LP (Codex M5d review §3): the faces of a saved witness
with |c| > 3, topped up with the largest remaining |c| until |S| = K = floor(0.01 N) (or truncated to the K largest
if the witness has more than K). Output: a boolean .npy in the flat face order of data_lp.py (x-faces, y-faces,
z-faces; each C-order (x, y, z)).
usage: python3 make_support.py WITNESS.h5 OUT.npy [--no-fill]
"""
import sys, numpy as np, h5py
wf, out = sys.argv[1], sys.argv[2]; fill = "--no-fill" not in sys.argv
with h5py.File(wf, "r") as f:
    c = np.concatenate([np.ascontiguousarray(f[f"c{q}"][()].T).ravel() for q in (1, 2, 3)])
    p = f.attrs.get("p", None); t = f.attrs.get("t", None)
N = c.size; K = int(np.floor(0.01 * N)); a = np.abs(c)
above = a > 3
order = np.argsort(-a)
if above.sum() > K or fill:
    S = np.zeros(N, bool); S[order[:K]] = True
else:
    S = above
np.save(out, S)
print(f"witness p={p} t={t}: N={N}, K={K}, |c|>3 in witness: {int(above.sum())}; |S| = {int(S.sum())}; "
      f"min |c| inside S {a[S].min():.4f}, max |c| outside S {a[~S].max() if (~S).any() else 0:.4f}; saved {out}")
