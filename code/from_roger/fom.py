#!/usr/bin/env python3
"""Figure of merit FoM = 1 / sqrt(det Cov(alpha_iso, alpha_AP)) before/after.

Paper precision setup (polynomial model, 0.02 <= k <= 0.40, six-box stack of
LOSz, Gaussian mode-counting weights, free damping).  The 2x2 covariance of
(alpha_par, alpha_perp) is 2 H^-1, with H the Hessian of the profile chi^2
(beta, Sigma_par, Sigma_perp re-optimized at every point) from a quadratic fit
on a 5x5 grid around the best fit.  For alpha_iso = apar^(1/3) aperp^(2/3) and
alpha_AP = apar / aperp the Jacobian at alpha = 1 has |det| = 1, so the FoM is
the same in (iso, AP) and (par, perp).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

CLEAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN))
sys.path.insert(0, str(CLEAN / "paper_prd"))


def profile_cov(fitter, best, sig):
    grid = np.linspace(-1.5, 1.5, 5)
    pts, vals = [], []
    for u in grid:
        for v in grid:
            ap, at = best[0] + u * sig[0], best[1] + v * sig[1]
            obj = lambda x: fitter._chi2([ap, at, *x])           # noqa: E731
            r = minimize(obj, best[2:], method="L-BFGS-B",
                         bounds=[(0, 5), (0, 20), (0, 20)])
            pts.append((ap - best[0], at - best[1]))
            vals.append(r.fun)
    x, y = np.array(pts).T
    a = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    c = np.linalg.lstsq(a, np.array(vals), rcond=None)[0]
    hess = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    return 2 * np.linalg.inv(hess)


def main(model_no=1, kmax=0.40):
    from fit_bao_pkmu import PkmuFitter, PkmuModel, bin_mask
    from referee_lcv_audit import source_path
    from template_pk import class_pk
    cosmo = class_pk(z=2.5)
    src = np.load(source_path(model_no, "z"))
    k, mu, n = src["k"], src["mu"], src["N"]
    iw = np.broadcast_to(np.arange(5)[None, :], k.shape)
    jac = np.array([[1 / 3, 2 / 3], [1.0, -1.0]])             # (iso, AP) from (par, perp)
    out = {}
    for st in ("pre", "post"):
        p = src[f"m{model_no}_{st}_raw_bandpower"]
        mean = p.mean(0)
        var = 2 * mean ** 2 / np.maximum(n, 1)
        good = bin_mask(k, mean, n, 0.02, kmax)
        model = PkmuModel(cosmo, k[good], mu[good], iw[good], 5, nbb=3)
        fitter = PkmuFitter(model, mean[good], len(p) / var[good])
        r = fitter.fit()
        best = np.array([r[x] for x in ("ap", "at", "beta", "sig_par", "sig_per")])
        cov = profile_cov(fitter, best, (r["sigma_ap"], r["sigma_at"]))
        cov_ia = jac @ cov @ jac.T
        out[st] = dict(cov=cov, cov_ia=cov_ia, fom=1 / np.sqrt(np.linalg.det(cov)),
                       rho=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        print(f"Model {model_no} {st:4s}: 100 sigma(par, perp) = "
              f"{100 * np.sqrt(cov[0, 0]):.3f}, {100 * np.sqrt(cov[1, 1]):.3f}  "
              f"rho = {out[st]['rho']:+.2f}  100 sigma(iso, AP) = "
              f"{100 * np.sqrt(cov_ia[0, 0]):.3f}, {100 * np.sqrt(cov_ia[1, 1]):.3f}  "
              f"FoM = {out[st]['fom']:.0f}", flush=True)
    g = out["post"]["fom"] / out["pre"]["fom"]
    gi = np.sqrt(out["pre"]["cov_ia"][0, 0] / out["post"]["cov_ia"][0, 0])
    ga = np.sqrt(out["pre"]["cov_ia"][1, 1] / out["post"]["cov_ia"][1, 1])
    print(f"Model {model_no}: FoM gain {g:.2f}; gains alpha_iso {gi:.2f}, alpha_AP {ga:.2f}")


if __name__ == "__main__":
    for m in (1, 2, 3, 4):
        main(m)
