#!/usr/bin/env python3
"""
NLTE hydrogen-ionization correction for the HSEMHS EOS, learned from Bifrost non-equilibrium
hydrogen (hion) simulations.

For every simulation cell
    D = log10(n_p / n_HI)_sim - log10(n_p / n_HI)_Saha(T, p_e,sim)
where the Saha ratio is Wittmann's pressure form with u_HI = 2, u_HII = 1, chi_H = 13.598434 eV
(HSEMHS converts this reference to its own partition functions exactly). D is the departure of
the simulation's hydrogen ionization from Saha at the simulation's own electron density, so
differences in abundances between Bifrost and Wittmann do not enter it; D = 0 is LTE.

D is binned in (log10 T, log10 P[Pa]); the median per bin is taken per run and the runs are
averaged with equal weight. A normalized Gaussian smoothing fills and smooths the grid, and the
correction is tapered to zero where the simulations have no data and above T = 10^4.6 K (fully
ionized hydrogen, where D carries no information on n_e). Output (HDF5, datasets on the grid
(log10T, log10P), C order): D (the tapered correction HSEMHS uses), D_raw (smoothed, untapered),
weight, count_<run>, D_<run>, sigma (half the 16-84 % range of D within a bin, pooled).

Usage (eagle9, `source ~/loadamd.sh`):
    ~/amdenv/bin/python3.12 make_nlte_correction.py OUT.h5
"""
import sys, time, datetime, hashlib, numpy as np, h5py
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
from helita.sim.bifrost import BifrostData

RUNS = {'en024048_hion': ([500, 600, 700], 2), 'ch012012_hion': ([900, 930, 960], 4)}   # snapshots, x/y stride
ROOT = '/mn/stornext/d21/RoCS/matsc/3d/run'
LT = np.round(np.arange(3.30, 5.0001, 0.01), 4)          # bin centres
LP = np.round(np.arange(-4.0, 5.6001, 0.02), 4)
SAHA_C, EION_H, BK_CGS = 9.0804625434325867, 13.598434, 1.3806488e-16
MIN_COUNT, SIGMA_BINS = 20, 2.0
POOL = ThreadPoolExecutor(8)

def read_snapshot(run, s, stride):
    fdir = f'{ROOT}/{run}'
    d = BifrostData(run, snap=s, fdir=fdir, verbose=False)
    nx, ny, nz = d.nx, d.ny, d.nz
    def var(fn, v):
        with open(f'{fdir}/{fn}', 'rb') as fh:
            fh.seek(v * nx * ny * nz * 4)
            a = np.fromfile(fh, dtype='<f4', count=nx * ny * nz).reshape((nx, ny, nz), order='F')
        return np.ascontiguousarray(a[::stride, ::stride]).ravel()
    hion = f'{run}.hion_{s:03d}.snap'
    jobs = {'T': (hion, 1), 'ne': (hion, 0), 'P': (f'{run}_{s:03d}.aux', 0)}
    jobs.update({f'n{i}': (hion, i + 1) for i in range(1, 7)})
    A = dict(zip(jobs, POOL.map(lambda k: var(*jobs[k]), jobs)))
    T = A['T'].astype(np.float64); P = A['P'].astype(np.float64) * 1e4          # Pa
    nhi = sum(A[f'n{i}'].astype(np.float64) for i in range(1, 6)); npr = A['n6'].astype(np.float64)
    pe = A['ne'].astype(np.float64) * BK_CGS * T                                    # cgs
    theta = 5040.0 / T
    lref = np.log10(0.5) + SAHA_C - theta * EION_H - np.log10(pe) - 2.5 * np.log10(theta)
    D = np.log10(npr / nhi) - lref
    ok = np.isfinite(D) & (nhi > 0) & (npr > 0)
    return np.log10(T[ok]), np.log10(P[ok]), D[ok]

def bin_stats(lT, lP, D):
    i = np.rint((lT - LT[0]) / 0.01).astype(np.int64); j = np.rint((lP - LP[0]) / 0.02).astype(np.int64)
    m = (i >= 0) & (i < len(LT)) & (j >= 0) & (j < len(LP))
    key = i[m] * len(LP) + j[m]; D = D[m]
    o = np.lexsort((D, key)); key = key[o]; D = D[o]
    u, st, cnt = np.unique(key, return_index=True, return_counts=True)
    q = lambda f: D[st + np.floor(f * (cnt - 1)).astype(np.int64)]
    med = np.full(len(LT) * len(LP), np.nan); p16 = med.copy(); p84 = med.copy(); n = np.zeros(len(LT) * len(LP), np.int64)
    med[u] = 0.5 * (q(0.5) + D[st + np.ceil(0.5 * (cnt - 1)).astype(np.int64)]); p16[u] = q(0.16); p84[u] = q(0.84); n[u] = cnt
    sh = (len(LT), len(LP))
    return med.reshape(sh), p16.reshape(sh), p84.reshape(sh), n.reshape(sh)

def smooth(val, w):
    num = gaussian_filter(np.where(w > 0, val * w, 0.0), SIGMA_BINS, mode='nearest')
    den = gaussian_filter(w, SIGMA_BINS, mode='nearest')
    return np.where(den > 1e-6, num / np.maximum(den, 1e-300), 0.0), den

def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

def main(out):
    t0 = time.time(); per = {}
    for run, (snaps, stride) in RUNS.items():
        parts = [read_snapshot(run, s, stride) for s in snaps]
        lT, lP, D = (np.concatenate([p[k] for p in parts]) for k in range(3))
        per[run] = bin_stats(lT, lP, D)
        print(f'[{time.time()-t0:5.0f}s] {run} snaps {snaps}: {len(D):,} cells, {np.count_nonzero(per[run][3] >= MIN_COUNT)} bins with >= {MIN_COUNT}', flush=True)
        for a, b in ((3.5, 3.7), (3.7, 3.9), (3.9, 4.0), (4.0, 4.2), (4.2, 4.4), (4.4, 4.6)):
            mm = (lT >= a) & (lT < b)
            print(f'     log T {a}-{b}: median D {np.median(D[mm]):+.3f}, 16-84% {np.percentile(D[mm],16):+.3f} {np.percentile(D[mm],84):+.3f}', flush=True)
        del parts, lT, lP, D
    meds = np.stack([per[r][0] for r in RUNS]); cov = np.stack([(per[r][3] >= MIN_COUNT).astype(float) for r in RUNS])
    sig = np.stack([0.5 * (per[r][2] - per[r][1]) for r in RUNS])
    w = cov.mean(0)                                                  # 0, 0.5 or 1: equal weight per run
    pooled = np.where(cov.sum(0) > 0, np.nansum(np.where(cov > 0, meds, 0), 0) / np.maximum(cov.sum(0), 1), 0.0)
    D_raw, wsm = smooth(pooled, w)
    sigma, _ = smooth(np.where(cov.sum(0) > 0, np.nansum(np.where(cov > 0, sig, 0), 0) / np.maximum(cov.sum(0), 1), 0.0), w)
    LTg = LT[:, None] * np.ones((1, len(LP)))
    weight = smoothstep(wsm, 0.15, 0.5) * (1 - smoothstep(LTg, 4.4, 4.6))
    D = weight * D_raw
    with h5py.File(out, 'w') as f:
        f['log10T'] = LT; f['log10P'] = LP
        f['D'] = D; f['D_raw'] = D_raw; f['weight'] = weight; f['sigma'] = sigma
        for r in RUNS:
            f[f'count_{r}'] = per[r][3]
            f[f'D_{r}'] = smooth(np.nan_to_num(per[r][0]), (per[r][3] >= MIN_COUNT).astype(float))[0]
        a = f.attrs
        a['definition'] = 'D = log10(n_p/n_HI)_sim - log10(n_p/n_HI)_Saha(T, p_e,sim); Saha: Wittmann pressure form'
        a['saha_reference_u_HI'] = 2.0; a['saha_reference_u_HII'] = 1.0; a['saha_reference_eion_eV'] = EION_H
        a['saha_reference_saha_c'] = SAHA_C
        a['runs'] = '; '.join(f'{r} snapshots {s} stride {st}' for r, (s, st) in RUNS.items())
        a['source'] = ROOT; a['min_count'] = MIN_COUNT; a['smoothing_sigma_bins'] = SIGMA_BINS
        a['taper'] = 'weight = smoothstep(smoothed coverage, 0.15, 0.5) * (1 - smoothstep(log10 T, 4.4, 4.6))'
        a['created'] = datetime.datetime.now().isoformat(timespec='seconds')
        a['script_sha256'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()
    print(f'[{time.time()-t0:5.0f}s] wrote {out}: grid {len(LT)} x {len(LP)}, weighted bins {np.count_nonzero(weight > 0.5)}', flush=True)
    for a, b in ((3.5, 3.7), (3.7, 3.9), (3.9, 4.0), (4.0, 4.2), (4.2, 4.4)):
        mm = (LT >= a) & (LT < b); dd = D[mm][weight[mm] > 0.5]
        print(f'     log T {a}-{b}: correction D in {np.percentile(dd,5):+.2f} .. {np.percentile(dd,95):+.2f} (5-95 % of covered bins)')

if __name__ == '__main__':
    main(sys.argv[1])
