"""
measure_pkmu.py
---------------
P(k, mu) and P_ell(k) of a real-space delta_F grid produced by
recon_paint_field.py, with the appropriate TSC window deconvolved.

The zero-displacement reference lies exactly on mesh nodes, while the
displaced field samples generic subcell positions.  Those cases do not have
the same TSC response.  Use ``--window node`` for an unjittered reference and
the default ``--window continuous`` for displaced or jittered fields.

TSC compensation for a grid-assigned field is the standard
    W(k) = prod_i [ sinc(pi k_i / (2 k_nyq_i)) ]^p ,   p = 3 for TSC
applied as P -> P / W^2.  Interlacing is not available here (we only have the
painted field, not the catalog), so modes near Nyquist stay aliased; keep the
fit to k well below Nyquist (the default kmax=0.5 is ~1/1.6 of Nyquist at
nmesh=512).

Usage
-----
  python measure_pkmu.py --files dF_ref_*.npy dF_recon_*.npy \
                         --nmu 5 --kmax 0.6 --out pkmu.npz
  python measure_pkmu.py --files dF_ref_*.npy dF_recon_*.npy \
                         --nmu 5 --kmax 0.5 --cache-each
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import rfftn

LBOX = 2000.0


def pkmu(field, lbox=LBOX, dk=0.005, kmax=0.6, nmu=5, los=2,
         compensate=True, window="continuous", workers=1):
    """Return k_cen, mu_cen, P(k,mu), and the mode counts."""
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError(f"pkmu requires a cubic 3D field, got {field.shape}; "
                         "this may be an MPI-local slab")
    n = field.shape[0]
    kf = 2.0 * np.pi / lbox
    k_nyq = np.pi * n / lbox

    dkc = rfftn(field, workers=int(workers))
    dkc /= n ** 3                      # sum-convention FFT -> Fourier coeffs
    p3d = (np.abs(dkc) ** 2).astype(np.float64) * lbox ** 3
    del dkc
    p3d[0, 0, 0] = 0.0

    kx = np.fft.fftfreq(n, d=1.0 / n) * kf
    kz = np.fft.rfftfreq(n, d=1.0 / n) * kf
    if compensate:
        if window == "continuous":
            # Ensemble TSC window for generic subcell particle positions.
            def w(kv):
                x = np.pi * kv / (2.0 * k_nyq)
                return np.sinc(x / np.pi) ** 3
            wx, wz = w(kx), w(kz)
        elif window == "node":
            # A particle exactly on every mesh node is a special case.  Its
            # discrete TSC stencil is [1/8, 3/4, 1/8], whose Fourier response
            # is 3/4 + cos(2 pi m/N)/4, not sinc(m/N)^3.
            mx = np.fft.fftfreq(n, d=1.0 / n)
            mz = np.fft.rfftfreq(n, d=1.0 / n)
            wx = 0.75 + 0.25 * np.cos(2.0 * np.pi * mx / n)
            wz = 0.75 + 0.25 * np.cos(2.0 * np.pi * mz / n)
        else:
            raise ValueError("window must be 'continuous' or 'node'")
        p3d /= (wx[:, None, None] * wx[None, :, None]
                * wz[None, None, :]) ** 2

    K = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2
                + kz[None, None, :] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        MU = np.abs(kz[None, None, :] / K)
    MU[0, 0, 0] = 0.0

    # rfft double-counts nothing at kz=0 and kz=Nyquist; weight the rest by 2
    wgt = np.full((1, 1, len(kz)), 2.0)
    wgt[0, 0, 0] = 1.0
    if n % 2 == 0:
        wgt[0, 0, -1] = 1.0
    wgt = np.broadcast_to(wgt, K.shape)

    k_edges = np.arange(0.0, kmax + dk, dk)
    mu_edges = np.linspace(0.0, 1.0, nmu + 1)
    ik = np.digitize(K.ravel(), k_edges) - 1
    im = np.clip(np.digitize(MU.ravel(), mu_edges) - 1, 0, nmu - 1)
    nk = len(k_edges) - 1
    ok = (ik >= 0) & (ik < nk)
    idx = (ik[ok] * nmu + im[ok]).astype(np.int64)
    ww = wgt.ravel()[ok]

    cnt = np.bincount(idx, weights=ww, minlength=nk * nmu)
    psum = np.bincount(idx, weights=ww * p3d.ravel()[ok], minlength=nk * nmu)
    ksum = np.bincount(idx, weights=ww * K.ravel()[ok], minlength=nk * nmu)
    msum = np.bincount(idx, weights=ww * MU.ravel()[ok], minlength=nk * nmu)
    cnt = cnt.reshape(nk, nmu)
    good = cnt > 0
    P = np.where(good, psum.reshape(nk, nmu) / np.maximum(cnt, 1), np.nan)
    kc = np.where(good, ksum.reshape(nk, nmu) / np.maximum(cnt, 1), np.nan)
    mc = np.where(good, msum.reshape(nk, nmu) / np.maximum(cnt, 1), np.nan)
    return kc, mc, P, cnt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--dk", type=float, default=0.005)
    p.add_argument("--kmax", type=float, default=0.6)
    p.add_argument("--nmu", type=int, default=5)
    p.add_argument("--ncpu", type=int, default=1,
                   help="FFT worker count; defaults to one for login-node safety")
    p.add_argument("--no-compensate", action="store_true")
    p.add_argument("--window", choices=("continuous", "node"),
                   default="continuous",
                   help="TSC transfer: generic subcell positions or the "
                        "special exact-mesh-node stencil")
    p.add_argument("--out", default=None)
    p.add_argument("--cache-each", action="store_true",
                   help="save one standard k, mu, P, N cache beside each input")
    a = p.parse_args()
    if a.out is None and not a.cache_each:
        p.error("specify --out or --cache-each; refusing to discard spectra")

    res = {}
    for fn in a.files:
        g = np.load(fn)
        kc, mc, P, cnt = pkmu(g, dk=a.dk, kmax=a.kmax, nmu=a.nmu,
                              compensate=not a.no_compensate,
                              window=a.window, workers=a.ncpu)
        key = Path(fn).stem
        res[key + "__k"] = kc
        res[key + "__mu"] = mc
        res[key + "__P"] = P
        res[key + "__N"] = cnt
        print(f"{key}: grid {g.shape[0]}^3  std={g.std():.5f}")
        if a.cache_each:
            source = Path(fn)
            window_tag = "" if a.window == "continuous" else f"_win{a.window}"
            compensate_tag = "_raw" if a.no_compensate else ""
            cache = source.parent / (f"{source.stem}.pkmu_dk{a.dk}_kmax"
                                     f"{a.kmax}_nmu{a.nmu}{window_tag}"
                                     f"{compensate_tag}.npz")
            np.savez(cache, k=kc, mu=mc, P=P, N=cnt,
                     source=np.array(str(source)),
                     window=np.array(a.window),
                     compensated=np.array(not a.no_compensate),
                     fft_workers=np.array(a.ncpu))
            print(f"saved -> {cache}")
        del g
    if a.out:
        np.savez(a.out, **res)
        print(f"saved -> {a.out}")
