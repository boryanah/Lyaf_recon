"""
validate_p3d.py
---------------
Brute-force reimplementation of the P3D estimator in numpy, run on a small
periodic box, and compared against `power_periodic` on the identical input.

Why bother: the PERIODIC+LYA combination needed four separate fixes (see
BUILD_NERSC.md), three of which were SILENT -- they produced plausible-looking
numbers rather than an error.  A normalization that is off by a constant is
exactly the kind of thing that survives every smoke test and then quietly
rescales a bias measurement.  This script closes that hole: it reproduces the
estimator from the source-level definition, so agreement means the kernel, the
mu convention, the pair-weighting AND the absolute normalization are all right.

What the estimator is (power_mod/power_counts.h::count_pairs + save_spectrum):

    P_l(k_i) = 2/norm * sum_{i<j, r<R0, tid_i != tid_j}
                   d_i d_j W(r;R0) L_l(mu) Kbar_l(k_i, r)
    norm     = (sum_w)^2 / V                        [PERIODIC power_norm]
    d        = wg - w,   mu = |dz| / r              [PERIODIC cleanup_l]
    Kbar_l   = [K_l(k_hi r) - K_l(k_lo r)] / [(k_hi r)^3 - (k_lo r)^3]
    K_l(x)   = 3 (-1)^(l/2) (2l+1) B_l(x)           [kernel_interp.h]
      B_0    = sin x - x cos x
      B_2    = x cos x - 4 sin x + 3 Si(x)
      B_4    = 1/2 [ (105/x - 2x) cos x + (22 - 105/x^2) sin x + 15 Si(x) ]

The factor 2 is the `one_grid` doubling: the C++ pair loop visits each
unordered pair once.

Kbar_0 is exactly <j_0(kr)> averaged over the bin with k^2 dk weight, which is
the check that the (2l+1)(-1)^(l/2) prefactor convention matches
P_l(k) = 4 pi (-i)^l int r^2 dr j_l(kr) xi_l(r).

Usage
-----
  python validate_p3d.py                       # ~30 s
  python validate_p3d.py --nsk 60 --npix 96    # tighter, slower
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import sici

HERE = Path(__file__).resolve().parent
BIN = HERE.parent.parent / "power_3d_code" / "for_Boryana" / "P3D"


def pair_weight(r, R0):
    """W(r;R0), the smooth truncation from power_counts.h::pair_weight."""
    x = r / R0
    w = np.ones_like(r)
    m = (r >= R0 / 2.) & (r < 3. * R0 / 4.)
    w[m] = 1. - 8. * (2 * x[m] - 1) ** 3 + 8. * (2 * x[m] - 1) ** 4
    m = (r >= 3. * R0 / 4.) & (r < R0)
    w[m] = -64. * (x[m] - 1) ** 3 - 128. * (x[m] - 1) ** 4
    w[r >= R0] = 0.
    return w


def K_ell(x, ell):
    """kernel_interp.h: 3 (-1)^(l/2) (2l+1) B_l(x), with the x<0.01 cutoff."""
    s, c = np.sin(x), np.cos(x)
    Si = sici(x)[0]
    if ell == 0:
        B = s - x * c
    elif ell == 2:
        B = x * c - 4 * s + 3 * Si
    elif ell == 4:
        with np.errstate(divide="ignore", invalid="ignore"):
            B = 0.5 * ((105. / x - 2. * x) * c
                       + (22. - 105. / x ** 2) * s + 15 * Si)
    else:
        raise ValueError(ell)
    out = B * (3. * (-1) ** (ell // 2) * (2 * ell + 1))
    return np.where(x < 0.01, 0., out)


def Kbar(r, klo, khi, ell):
    """Bin-averaged kernel, exactly as count_pairs accumulates it."""
    xlo, xhi = klo * r, khi * r
    num = K_ell(xhi, ell) - K_ell(xlo, ell)
    den = xhi ** 3 - xlo ** 3
    return num / den


def brute_force(pos, d, w, tid, L, R0, kbins, ells, chunk=256):
    """Direct O(N^2) evaluation with minimum-image periodic separations."""
    n = len(pos)
    out = np.zeros((len(kbins), len(ells)))
    klo = kbins[:, 0][:, None]
    khi = kbins[:, 1][:, None]
    sumw = w.sum()
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        dp = pos[i0:i1, None, :] - pos[None, :, :]
        dp -= L * np.round(dp / L)                     # minimum image
        r = np.sqrt((dp ** 2).sum(-1))
        # keep j > i only (each unordered pair once), inside R0, different tid
        jj = np.arange(n)[None, :] > np.arange(i0, i1)[:, None]
        keep = jj & (r < R0) & (r > 0) & (tid[i0:i1, None] != tid[None, :])
        if not keep.any():
            continue
        rr = r[keep]
        mu = np.abs(dp[..., 2][keep]) / rr
        wij = (d[i0:i1, None] * d[None, :])[keep] * pair_weight(rr, R0)
        for je, ell in enumerate(ells):
            Lg = {0: np.ones_like(mu),
                  2: 0.5 * (3 * mu ** 2 - 1),
                  4: (35 * mu ** 4 - 30 * mu ** 2 + 3) / 8.}[ell]
            out[:, je] += ((wij * Lg)[None, :] * Kbar(rr[None, :], klo, khi, ell)
                           ).sum(1)
    return 2.0 * out / (sumw ** 2 / L ** 3)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=float, default=500.0, help="test box [Mpc/h]")
    p.add_argument("--nsk", type=int, default=40, help="skewers")
    p.add_argument("--npix", type=int, default=64, help="pixels per skewer")
    p.add_argument("--R0", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fast", type=int, default=1)
    p.add_argument("--nside", type=int, default=32,
                   help="pair-search grid; the production default is too "
                        "fine for this deliberately tiny catalog")
    p.add_argument("--tmp", default="/pscratch/sd/b/boryanah/Lyaf_p3d/validate")
    a = p.parse_args()

    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # --- build a small periodic sample -------------------------------------
    cell = a.L / a.npix
    ix = rng.integers(0, a.npix, a.nsk)
    iy = rng.integers(0, a.npix, a.nsk)
    # make skewers distinct so tid really is one per sightline
    uniq = {}
    for k in range(a.nsk):
        while (ix[k], iy[k]) in uniq:
            ix[k], iy[k] = rng.integers(0, a.npix), rng.integers(0, a.npix)
        uniq[(ix[k], iy[k])] = k
    z = np.arange(a.npix) * cell
    pos = np.stack([np.repeat(ix * cell, a.npix),
                    np.repeat(iy * cell, a.npix),
                    np.tile(z, a.nsk)], axis=1)
    tid = np.repeat(np.arange(a.nsk), a.npix)
    d = rng.normal(0, 0.3, len(pos))
    d -= d.mean()
    w = np.ones(len(pos))

    # k-bins: a handful, wide enough that the brute force is not noise
    kb = np.array([[0.02, 0.05], [0.05, 0.08], [0.08, 0.11],
                   [0.11, 0.14], [0.14, 0.17], [0.17, 0.20]])
    kfile = tmp / "kbins.txt"
    np.savetxt(kfile, kb, fmt="%.8f\t%.8f")

    # --- write the input, with ghost corners pinning the period to L --------
    eps = 1e-6
    ghosts = np.array([[x, y, zz, 0.0, 0.0, -1 - i, 0] for i, (x, y, zz) in
                       enumerate((p_, q_, r_) for p_ in (0., a.L - eps)
                                 for q_ in (0., a.L - eps)
                                 for r_ in (0., a.L - eps))])
    rows = np.column_stack([pos, w, 1.0 + d, tid, np.zeros(len(pos))])
    infile = tmp / "validate_in.txt"
    np.savetxt(infile, np.vstack([rows, ghosts]),
               fmt="%.8f %.8f %.8f %.8f %.8f %d %d")
    print(f"  {len(pos)} particles, {a.nsk} skewers, L = {a.L}, R0 = {a.R0}")

    # --- run the C++ code ---------------------------------------------------
    cmd = [str(BIN / "power_periodic"), "-in", str(infile),
           "-binfile", str(kfile), "-output", str(tmp),
           "-out_string", "validate", "-max_l", "4", "-R0", str(a.R0),
           "-nthread", "8", "-nside", str(a.nside), "-nbin_RR", "100", "-max_l_RR", "4",
           "-fast", str(a.fast), "-perbox"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("P3D validation executable failed with code "
                           f"{r.returncode}:\n{r.stderr}")
    for line in r.stdout.splitlines():
        if any(s in line for s in ("periodic box-size", "Norm =",
                                   "particles in use", "accepted")):
            print("  [cpp]", line.strip())
    fn = tmp / f"validate_power_spectrum_n{len(kb)}_l4_R0{int(a.R0)}.txt"
    if not fn.exists():
        raise FileNotFoundError(f"P3D did not create expected output {fn}")
    cpp = np.loadtxt(fn)

    # --- brute force --------------------------------------------------------
    print("  brute force ...")
    ref = brute_force(pos, d, w, tid, a.L, a.R0, kb, [0, 2, 4])

    print(f"\n  {'k range':>14} | {'ell':>3} | {'C++':>12} | {'python':>12} "
          f"| {'ratio':>8}")
    print("  " + "-" * 62)
    worst = 0.
    for i in range(len(kb)):
        for j, ell in enumerate([0, 2, 4]):
            rat = cpp[i, j] / ref[i, j] if ref[i, j] != 0 else np.nan
            worst = max(worst, abs(rat - 1))
            print(f"  {kb[i,0]:.3f}-{kb[i,1]:.3f} | {ell:3d} | {cpp[i,j]:12.5f} "
                  f"| {ref[i,j]:12.5f} | {rat:8.4f}")
    print(f"\n  worst |ratio-1| = {worst:.2e}   "
          f"{'PASS' if worst < 2e-2 else 'FAIL'}")
