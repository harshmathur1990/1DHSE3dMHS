"""Perturbed τ-grid inputs for the input-error test of the method paper (Sect. "Errors in the input").

usage: python make_perturbed_inputs.py INPUT.nc OUTDIR
Writes OUTDIR/input_<tag>_s<seed>.nc for tag in (small, large) and seeds 1-3, the layout of INPUT with T and B replaced:

  T' = T exp(σ_T ε_T),   B_i' = B_i (1 + σ_B ε_i)   (i = x, y, z, independent ε)

Each ε is a random field with unit standard deviation, Gaussian-correlated over 2 horizontal pixels (periodic, by FFT)
and 0.5 dex in log10 τ (by convolution along the axis, edges reflected), as a crude model of inversion errors that are
correlated over a resolution element and over the width of a response function. Amplitudes:
small σ_T = 0.02, σ_B = 0.10; large σ_T = 0.05, σ_B = 0.25. The truth of INPUT stays the reference. The 180° azimuth
ambiguity is assumed resolved.
"""
import os
import sys

import h5py
import numpy as np

AMP = {"small": (0.02, 0.10), "large": (0.05, 0.25)}
LH, LV = 2.0, 0.5                                     # correlation lengths: pixels, dex


def field(rng, nx, ny, lt):
    """Unit-variance correlated random field, h5py layout (x, y, τ)."""
    w = rng.standard_normal((nx, ny, len(lt)))
    kx = np.fft.fftfreq(nx)[:, None]; ky = np.fft.fftfreq(ny)[None, :]
    g = np.exp(-2 * (np.pi * LH) ** 2 * (kx ** 2 + ky ** 2))      # Gaussian of σ = LH pixels in Fourier space
    w = np.real(np.fft.ifft2(np.fft.fft2(w, axes=(0, 1)) * g[:, :, None], axes=(0, 1)))
    d = lt[1] - lt[0]; half = int(np.ceil(3 * LV / d))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * d / LV) ** 2); k /= k.sum()
    wp = np.concatenate([w[:, :, half:0:-1], w, w[:, :, -2:-half - 2:-1]], axis=2)
    w = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 2, wp)
    return (w - w.mean()) / w.std()


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(inp) as f:
        lt = f["log10tau500"][:]; T = f["temperature"][:]; B = {c: f[c][:] for c in ("B_x", "B_y", "B_z")}
        x = f["x"][:]; y = f["y"][:]; attrs = dict(f.attrs)
    nx, ny, _ = T.shape
    for tag, (sT, sB) in AMP.items():
        for seed in (1, 2, 3):
            rng = np.random.default_rng(1000 * seed + (1 if tag == "small" else 2))
            fn = os.path.join(outdir, f"input_{tag}_s{seed}.nc")
            with h5py.File(fn, "w") as f:
                f["log10tau500"] = lt
                f["x"] = x; f["x"].attrs["units"] = "m"
                f["y"] = y; f["y"].attrs["units"] = "m"
                eT = field(rng, nx, ny, lt)
                f["temperature"] = T * np.exp(sT * eT)
                for c in ("B_x", "B_y", "B_z"):
                    f[c] = B[c] * (1 + sB * field(rng, nx, ny, lt))
                for k, v in attrs.items():
                    f.attrs[k] = v
                f.attrs["perturbation"] = (f"T exp({sT} eps), B_i (1 + {sB} eps_i); eps unit-variance Gaussian-correlated over "
                                           f"{LH} px and {LV} dex; seed {seed}; from {os.path.abspath(inp)}")
                f.attrs["created_by"] = "reports/paper_252_2026-11-10/make_perturbed_inputs.py"
            print("wrote", fn, f"(rms dlnT {np.sqrt(np.mean((sT * eT) ** 2)):.3f})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
