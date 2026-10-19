"""
robustness.py
-------------
Is the ~0.2-0.3% offset of alpha from unity a fitting systematic or the
physical nonlinear BAO shift?

The template carries damping only -- exp(-k^2(mu^2 Sig_par^2 + ...)/2) -- and
no shift term, so a genuine nonlinear shift lands in alpha by construction.
That is the standard way it is measured, but it means a fitting systematic and
a real shift look identical in a single fit.

They are separable by how they respond to analysis choices:

  * a BROADBAND / FITTING systematic moves with kmax, with the number of
    broadband terms, and with the number of mu wedges;
  * a PHYSICAL shift does not -- it is a property of the ruler, not of how
    much of the spectrum you look at.

So: refit the same 6-phase stacks over a grid of (kmin, kmax, nbb, nmu) and
watch alpha.  The P(k,mu) measurements are cached, so everything except a
change of nmu is a pure refit.

Reconstruction is the second lever: it should remove most of a physical
nonlinear shift, so alpha_post should sit closer to 1 than alpha_pre if the
offset is physical.

Usage
-----
  python robustness.py --pre 'dF_ref_*.npy' --post 'dF_recon_*lyadrv*.npy'
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fit_bao_pkmu import measure, bin_mask, PkmuModel, PkmuFitter

Z = 2.5


def stack_fit(files, cosmo, kmin, kmax, nbb, nmu, dk, kmax_meas, var="gauss"):
    """Fit the mean of `files` and return the result dict."""
    meas = [measure(f, dk, kmax_meas, nmu, verbose=False) for f in files]
    k0, mu0, P0, N0 = meas[0]
    for i, (ki, mui, Pi, Ni) in enumerate(meas[1:], start=1):
        if Pi.shape != P0.shape or not np.allclose(ki, k0, equal_nan=True) \
                or not np.allclose(mui, mu0, equal_nan=True) \
                or not np.array_equal(Ni, N0):
            raise ValueError(f"incompatible P(k,mu) grid in {files[i]}")
    good = bin_mask(k0, P0, N0, kmin, kmax)
    iw = np.broadcast_to(np.arange(P0.shape[1])[None, :], P0.shape)[good]
    Ps = np.array([m[2][good] for m in meas])
    Pmean = Ps.mean(axis=0)

    # Gaussian band-power variance.  Prefer this over the 6-realization
    # scatter: a per-bin variance estimated from n samples is itself noisy, and
    # weighting by 1/s^2 inflates chi2 by E[sigma^2/s^2] = (n-1)/(n-3) = 5/3 for
    # n = 6, while deflating the Delta chi2 = 1 errors by sqrt(5/3) = 1.29.
    var_g = 2.0 * Pmean ** 2 / np.maximum(N0[good], 1.0)
    if var == "scatter" and len(Ps) >= 3:
        var1 = np.maximum(np.var(Ps, axis=0, ddof=1), 1e-3 * var_g)
    else:
        var1 = var_g

    model = PkmuModel(cosmo, k0[good], mu0[good], iw, nmu, nbb=nbb)
    res = PkmuFitter(model, Pmean, len(Ps) / var1).fit()
    res['nbin'] = int(good.sum())
    return res


def main(a):
    from template_pk import class_pk
    cosmo = class_pk(z=Z)
    pre = sorted(glob.glob(a.pre))
    post = sorted(glob.glob(a.post))
    if not pre or not post:
        raise ValueError(f"empty input: {len(pre)} pre files, {len(post)} post files")
    if len(pre) != len(post):
        raise ValueError(f"pre/post file counts differ: {len(pre)} vs {len(post)}")
    phase = lambda name: re.search(r"ph\d+", Path(name).name).group(0)
    if [phase(x) for x in pre] != [phase(x) for x in post]:
        raise ValueError("pre/post phase lists do not match")
    print(f"template r_s = {cosmo.rs_drag:.3f} Mpc/h")
    print(f"pre : {len(pre)} files\npost: {len(post)} files\n")

    base = dict(kmin=a.kmin, kmax=a.kmax, nbb=a.nbb, nmu=a.nmu,
                dk=a.dk, kmax_meas=a.kmax_meas)

    # one axis varied at a time, around the baseline
    variations = []
    for v in a.scan_kmax:
        variations.append(("kmax", v, {**base, "kmax": v}))
    for v in a.scan_kmin:
        variations.append(("kmin", v, {**base, "kmin": v}))
    for v in a.scan_nbb:
        variations.append(("nbb", v, {**base, "nbb": v}))
    for v in a.scan_nmu:
        variations.append(("nmu", v, {**base, "nmu": v}))

    print(f"{'axis':6s} {'value':>6s} {'nbin':>5s} | "
          f"{'ap_pre':>8s} {'ap_post':>8s} | {'at_pre':>8s} {'at_post':>8s} | "
          f"{'gain_ap':>7s} {'gain_at':>7s} | {'c2_pre':>6s} {'c2_post':>7s}")
    print("-" * 108)
    rows = []
    for axis, val, kw in variations:
        rp = stack_fit(pre, cosmo, var=a.var, **kw)
        rq = stack_fit(post, cosmo, var=a.var, **kw)
        gp = rp['sigma_ap'] / rq['sigma_ap']
        gt = rp['sigma_at'] / rq['sigma_at']
        print(f"{axis:6s} {val:6.3g} {rp['nbin']:5d} | "
              f"{rp['ap']:8.4f} {rq['ap']:8.4f} | "
              f"{rp['at']:8.4f} {rq['at']:8.4f} | "
              f"{gp:7.2f} {gt:7.2f} | "
              f"{rp['chi2']/rp['ndof']:6.2f} {rq['chi2']/rq['ndof']:7.2f}",
              flush=True)
        rows.append(dict(axis=axis, value=val, pre=rp, post=rq,
                         gain_ap=gp, gain_at=gt))

    # ---- verdict ---------------------------------------------------------
    print()
    for tag, key in (("alpha_par ", 'ap'), ("alpha_perp", 'at')):
        for w, lab in (('pre', 'pre '), ('post', 'post')):
            v = np.array([r[w][key] for r in rows])
            e = np.array([r[w]['sigma_' + key] for r in rows])
            print(f"  {tag} {lab}: spread over the whole grid = "
                  f"{v.max()-v.min():.4f}   "
                  f"(median error {np.median(e):.4f})   "
                  f"mean offset from 1 = {v.mean()-1:+.4f}")
    print("\n  A spread comparable to or larger than the error means the offset "
          "is\n  ANALYSIS-DEPENDENT -> a fitting systematic.  A spread well below "
          "the\n  error, with a stable nonzero offset, points at a real shift.")

    np.savez(a.save,
             axis=np.array([r['axis'] for r in rows]),
             value=np.array([r['value'] for r in rows]),
             **{f"{w}_{k}": np.array([r[w][k] for r in rows])
                for w in ('pre', 'post')
                for k in ('ap', 'at', 'sigma_ap', 'sigma_at', 'beta',
                          'sig_par', 'sig_per', 'chi2')})
    print(f"\n  saved -> {a.save}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    D = "/pscratch/sd/b/boryanah/Lyaf_recon"
    p.add_argument("--pre", default=f"{D}/dF_ref_Model_1_LOSz_nmesh512_*ph00[0-5]*.npy")
    p.add_argument("--post", default=(f"{D}/dF_recon_Model_1_LOSz_nmesh512_"
                                      "R10.00_nr512_MG_bF-0.148066_bt0.957688_lyadrv_"
                                      "AbacusSummit_base_c000_ph00[0-5]_z2.500.npy"))
    p.add_argument("--kmin", type=float, default=0.02)
    p.add_argument("--kmax", type=float, default=0.40)
    p.add_argument("--nbb", type=int, default=3)
    p.add_argument("--nmu", type=int, default=5)
    p.add_argument("--dk", type=float, default=0.005)
    p.add_argument("--kmax_meas", type=float, default=0.5)
    p.add_argument("--scan_kmax", type=float, nargs="*",
                   default=[0.25, 0.30, 0.35, 0.40])
    p.add_argument("--scan_kmin", type=float, nargs="*", default=[0.015, 0.03])
    p.add_argument("--scan_nbb", type=int, nargs="*", default=[2, 4, 5])
    p.add_argument("--scan_nmu", type=int, nargs="*", default=[3])
    p.add_argument("--var", default="gauss", choices=["gauss", "scatter"],
                   help="band-power variance: Gaussian mode counting (default) "
                        "or the realization scatter, which inflates chi2 by "
                        "(n-1)/(n-3) when n is small")
    p.add_argument("--save", default=f"{D}/robustness_Model_1_LOSz_6phases.npz")
    main(p.parse_args())
