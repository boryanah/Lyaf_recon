"""
regime2_fit.py
--------------
Six-phase stacked fits of the sparse reconstruction regimes on ONE footing:

    pre        : dF_skref_f{frac}_..._seed1234   (sparse_skewers.py)
    Regime III : dF_skrec_f{frac}_..._seed1234   (sparse self-reconstruction)
    Regime II  : dF_exrec_f{frac}_..._halo{tag}  (regime2_ensemble.py)

Every state uses the fitter of check_sparse_stability.fit_stack: the mean of
six phases, Gaussian mode-counting weights (2P^2/N)/6, three broadband terms
per wedge, 0.02 < k < 0.35 h/Mpc, and a free-amplitude P_1D(k_par) alias
column whose shape is the phase average for THAT state.  Damping priors are
the production values (dense pre / dense post, width 1 Mpc/h); a prior-free
fit is repeated for every state because the precision gain must not rest on
the prior.  Per-phase fits give the paired responses alpha_post - alpha_pre
and their six-phase standard errors.

Usage:  python regime2_fit.py [--fracs 0.155 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "paper_prd"))

from fit_bao_pkmu import measure, bin_mask, PkmuModel, PkmuFitter  # noqa: E402
from template_pk import class_pk                                   # noqa: E402
from recon_paint_field import OUT_DIR                               # noqa: E402
from sparse_skewers import measure_p1d                              # noqa: E402
from regime2_ensemble import field_path, p1d_path, TRACERS, tracer_tag  # noqa: E402

PHASES = [f"{i:03d}" for i in range(6)]
PRE_PRIOR, POST_PRIOR = (4.61, 2.81, 1.0), (2.16, 1.36, 1.0)
KMIN, KMAX, NMU, NBB = 0.02, 0.35, 5, 3
SIMS = [f"AbacusSummit_base_c000_ph{p}" for p in PHASES]
RESPONSE = "bF-0.148066_bt0.957688"


def pre_path(frac, sim):
    return OUT_DIR / (f"dF_skref_f{frac:.3f}_Model_1_LOSz_nmesh512_seed1234_"
                      f"{sim}_z2.500.npy")


def self_path(frac, sim):
    return OUT_DIR / (f"dF_skrec_f{frac:.3f}_Model_1_LOSz_nmesh512_R10.00_nr512_"
                      f"MG_{RESPONSE}_seed1234_{sim}_z2.500.npy")


def p1d(path, window):
    """Phase-level P_1D, cached next to the field so it is measured once."""
    cache = path.with_name(path.stem + f".p1d_{window}.npz")
    if cache.exists():
        with np.load(cache) as d:
            return np.array(d["kz"]), np.array(d["p1d"])
    alt = p1d_path(path)                      # written by regime2_ensemble
    if window == "continuous" and alt.exists():
        with np.load(alt) as d:
            return np.array(d["kz"]), np.array(d["p1d"])
    kz, p = measure_p1d(np.load(path), window=window)
    np.savez(cache, kz=kz, p1d=p, window=np.array(window))
    return kz, p


def state(paths, window):
    spectra = [measure(p, 0.005, 0.5, NMU, verbose=False) for p in paths]
    shapes = [p1d(p, window) for p in paths]
    return spectra, (shapes[0][0], np.mean([s[1] for s in shapes], axis=0))


def fitter(spectra, alias, prior, index=None):
    k, mu, p0, n = spectra[0]
    good = bin_mask(k, p0, n, KMIN, KMAX)
    iw = np.broadcast_to(np.arange(k.shape[1])[None, :], k.shape)[good]
    powers = np.asarray([s[2][good] for s in spectra])
    mean = powers.mean(axis=0)
    var = 2.0 * mean ** 2 / np.maximum(n[good], 1.0)
    extra = np.interp(k[good] * mu[good], alias[0], alias[1])[:, None]
    model = PkmuModel(COSMO, k[good], mu[good], iw, NMU, nbb=NBB, extra_cols=extra)
    if index is None:                       # six-phase stack
        return PkmuFitter(model, mean, len(powers) / var, sig_prior=prior)
    return PkmuFitter(model, powers[index], 1.0 / var, sig_prior=prior)


def summarize(label, pre, post):
    out = {}
    for tag, prior_pre, prior_post in (("prior", PRE_PRIOR, POST_PRIOR),
                                       ("noprior", None, None)):
        a = fitter(pre[0], pre[1], prior_pre).fit()
        b = fitter(post[0], post[1], prior_post).fit()
        out[tag] = (a, b)
    # paired per-phase responses (central, prior) -- no profile errors needed
    d_ap, d_at = [], []
    for i in range(len(PHASES)):
        a = fitter(pre[0], pre[1], PRE_PRIOR, i).fit(compute_errors=False)
        b = fitter(post[0], post[1], POST_PRIOR, i).fit(compute_errors=False)
        d_ap.append(100 * (b["ap"] - a["ap"]))
        d_at.append(100 * (b["at"] - a["at"]))
    d_ap, d_at = np.array(d_ap), np.array(d_at)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    a, b = out["prior"]
    an, bn = out["noprior"]
    print(f"  {label:26s} post ap {b['ap']:.4f}+-{b['sigma_ap']:.4f} "
          f"at {b['at']:.4f}+-{b['sigma_at']:.4f} Sig {b['sig_par']:.2f}/{b['sig_per']:.2f} | "
          f"gain {a['sigma_ap']/b['sigma_ap']:.3f}/{a['sigma_at']/b['sigma_at']:.3f} "
          f"(no prior {an['sigma_ap']/bn['sigma_ap']:.3f}/{an['sigma_at']/bn['sigma_at']:.3f}) | "
          f"dAp {d_ap.mean():+.3f}+-{se(d_ap):.3f} dAt {d_at.mean():+.3f}+-{se(d_at):.3f} %",
          flush=True)
    return dict(pre=a, post=b, pre_noprior=an, post_noprior=bn,
                d_ap=d_ap, d_at=d_at)


def main(a):
    global COSMO
    COSMO = class_pk(z=2.5)
    results = {}
    for frac in a.fracs:
        pre = state([pre_path(frac, s) for s in SIMS], "node")
        a0 = fitter(pre[0], pre[1], PRE_PRIOR).fit()
        print(f"\nf={frac:.3f}  pre ap {a0['ap']:.4f}+-{a0['sigma_ap']:.4f} "
              f"at {a0['at']:.4f}+-{a0['sigma_at']:.4f} "
              f"Sig {a0['sig_par']:.2f}/{a0['sig_per']:.2f}", flush=True)
        rows = {"III_self": [self_path(frac, s) for s in SIMS]}
        for t in TRACERS:
            rows[f"II_{tracer_tag(t)}"] = [field_path(frac, t, s) for s in SIMS]
        for label, paths in rows.items():
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                print(f"  {label}: missing {len(missing)} fields, skipped")
                continue
            results[(frac, label)] = summarize(label, pre,
                                               state(paths, "continuous"))
    flat = {}
    for (frac, label), r in results.items():
        key = f"f{frac:.3f}_{label}"
        for s in ("pre", "post", "pre_noprior", "post_noprior"):
            for k in ("ap", "at", "sigma_ap", "sigma_at", "sig_par",
                      "sig_per", "beta", "chi2", "ndof"):
                flat[f"{key}_{s}_{k}"] = r[s][k]
        flat[f"{key}_dap_percent"] = r["d_ap"]
        flat[f"{key}_dat_percent"] = r["d_at"]
    np.savez(OUT_DIR / "regime2_regime3_sparse_fits.npz", **flat)
    print("saved", OUT_DIR / "regime2_regime3_sparse_fits.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.155, 0.5])
    main(p.parse_args())
