#!/usr/bin/env python3
"""Configuration-space precision gains in the paper's own definition.

The paper defines the gain as G = sigma_pre / sigma_post, with sigma the
one-dimensional profile-likelihood interval (Delta chi^2 = 1, all other
nonlinear parameters re-optimized) of a fit to the six-box mean, each state
with its own variance (Sec. on the BAO model; Table "precision").  The
box-to-box scatter is not used for gains anywhere in the paper.

Here the same statistic is computed in configuration space, WITHOUT the
control variate, with a survey covariance as the error model: the six-box
mean raw xi(r_par, r_perp) of each state is fit with the paper's
configuration-space model and the eBOSS DR16 or DESI DR1 covariance, and the
profile-likelihood (MINOS) errors on alpha_par, alpha_perp, alpha_iso and
F_AP are compared before and after reconstruction.

Stages
  profile CONFIG COV MODEL LOS   dense fields (model_xi.py measurements)
  sparse  CONFIG COV SPARSE      sparse-sightline fields (sparse_xi.py)
  mc      CONFIG COV MODEL LOS SEED N
          noise Monte Carlo: Gaussian noise drawn from the covariance used in
          the fit is added to the six-box mean of each state (same draws for
          both states), and each realization is refit.  Its scatter must
          reproduce the likelihood errors if they are to be trusted.
  summary
CONFIG is a key of xi_survey_fit.CONFIGS (paper = the draft's setup).
Outputs: qso_desi_20260930/model_xi/profile/.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "paper_prd"))

import model_xi as mx                                       # noqa: E402

OUT = mx.OUT / "profile"
NL = ("bias_LYA", "beta_LYA", "sigmaNL_par", "sigmaNL_per")
# Covariance normalizations, relative to the matrix used for the weights
# (eBOSS / 20, or DR1 rescaled to the same mean variance): 1 = as used in the
# draft's fits, "pub" = the published survey matrix.
SCALES = (1.0, "pub")


def published_scale(fitter, cov):
    """Factor that turns the fit's covariance into the published matrix."""
    if cov == "eboss":
        return 20.0
    # make_fitter rescales DR1 to the mean variance of eBOSS / 20 over the fitted
    # bins; undo that.  (Not taken from fitter.icov, which also carries the
    # inflated variances of the bins removed for sparse data.)
    scan = mx.OUT / "scan"
    mask = fitter.mask
    dr1 = np.load(scan / "cov_dr1.npy")[np.ix_(mask, mask)]
    eboss20 = np.load(scan / "cov_eboss.npy")[np.ix_(mask, mask)]
    return float(np.mean(np.diag(dr1)) / np.mean(np.diag(eboss20)))


def objective_factory(fitter, data, scale):
    def objective(params):
        residual = fitter.predict(params) - data
        rhs = fitter.icov_b.T @ residual
        residual -= fitter.basis @ np.linalg.solve(fitter.normal, rhs)
        return float(residual @ fitter.icov @ residual) / scale
    return objective


def minuit(chi2, names, start, limits, steps):
    import iminuit
    m = iminuit.Minuit(lambda *x: chi2(dict(zip(names, x))), name=names, **start)
    for name in names:
        m.limits[name] = limits[name]
        m.errors[name] = steps[name]
    m.errordef = 1
    m.tol = 1e-6
    m.migrad(ncall=100000)
    return m


def fisher(fitter, theta, scale):
    """Linearized likelihood errors at theta: finite-difference Jacobian with
    the broadband projected out; parameters at a limit are held fixed.
    Returns 1-sigma errors on (ap, at, aiso, fap)."""
    h = {"ap": 2e-3, "at": 2e-3, "bias_LYA": 1e-3 * abs(theta["bias_LYA"]), "beta_LYA": 2e-2,
         "sigmaNL_par": .1, "sigmaNL_per": .1}
    cols = []
    for key in ("ap", "at") + NL:
        lo, hi = fitter.sample["limits"][key]
        if theta[key] - h[key] < lo or theta[key] + h[key] > hi:
            continue
        up = fitter.predict({**theta, key: theta[key] + h[key]})
        down = fitter.predict({**theta, key: theta[key] - h[key]})
        cols.append((up - down) / (2 * h[key]))
    jac = np.column_stack(cols)
    jac = jac - fitter.basis @ np.linalg.solve(fitter.normal, fitter.icov_b.T @ jac)
    cov = np.linalg.inv(jac.T @ fitter.icov @ jac / scale)[:2, :2]
    iso = np.array([1 / 3, 2 / 3])
    ap_ = np.array([1.0, -1.0])
    return np.sqrt([cov[0, 0], cov[1, 1], iso @ cov @ iso, ap_ @ cov @ ap_])


def profile(fitter, data, scale, start):
    """Best fit, MINOS errors in (ap, at) and (aiso, fap), and Fisher errors."""
    lim = dict(fitter.sample["limits"])
    step = {"ap": .01, "at": .01, "aiso": .01, "fap": .01, "bias_LYA": .002,
            "beta_LYA": .05, "sigmaNL_par": .3, "sigmaNL_per": .3}
    chi2 = objective_factory(fitter, data, scale)
    m = minuit(chi2, ("ap", "at") + NL, start, lim, step)
    if not m.fmin.is_valid:                    # poorly fitting data: default tolerance
        m.tol = 0.1
        m.migrad(ncall=100000)
    best = {key: m.values[key] for key in ("ap", "at") + NL}
    f = fisher(fitter, best, scale)
    out = {"chi2": m.fval, "valid": m.fmin.is_valid,
           "fisher_ap": f[0], "fisher_at": f[1], "fisher_aiso": f[2], "fisher_fap": f[3]}
    if not m.fmin.is_valid:
        for key in ("ap", "at") + NL:
            out[key] = m.values[key]
        for key in ("ap", "at", "aiso", "fap"):
            out[f"minos_{key}"] = np.nan
        out["hesse_ap"] = out["hesse_at"] = np.nan
        return out
    m.hesse()
    m.minos("ap", "at")
    for key in ("ap", "at") + NL:
        out[key] = m.values[key]
        out[f"hesse_{key}"] = m.errors[key]
    out["rho"] = m.covariance.correlation()["ap", "at"]
    for key in ("ap", "at"):
        out[f"minos_{key}"] = 0.5 * (m.merrors[key].upper - m.merrors[key].lower)
        out[f"minos_{key}_lo"], out[f"minos_{key}_hi"] = m.merrors[key].lower, m.merrors[key].upper

    def chi2_iso(p):
        q = dict(p)
        aiso, fap = q.pop("aiso"), q.pop("fap")
        q["ap"], q["at"] = aiso * fap ** (2 / 3), aiso * fap ** (-1 / 3)
        return chi2(q)
    best = {key: m.values[key] for key in NL}
    best["aiso"] = m.values["ap"] ** (1 / 3) * m.values["at"] ** (2 / 3)
    best["fap"] = m.values["ap"] / m.values["at"]
    lim2 = {**{key: lim[key] for key in NL}, "aiso": (.8, 1.2), "fap": (.6, 1.6)}
    m2 = minuit(chi2_iso, ("aiso", "fap") + NL, best, lim2, step)
    m2.hesse()
    m2.minos("aiso", "fap")
    for key in ("aiso", "fap"):
        out[key] = m2.values[key]
        out[f"hesse_{key}"] = m2.errors[key]
        out[f"minos_{key}"] = 0.5 * (m2.merrors[key].upper - m2.merrors[key].lower)
    return out


def run_states(fitter, cov, stacks, dest):
    pub = published_scale(fitter, cov)
    res = {"published_scale": pub}
    for st, data in stacks.items():
        first = fitter.fit(data)                      # the draft's fit path
        start = {key: first[key] for key in ("ap", "at") + NL}
        for scale in SCALES:
            s = pub if scale == "pub" else scale
            r = profile(fitter, data, s, start)
            for key, val in r.items():
                res[f"{st}_s{scale}_{key}"] = val
            print(f"{st} scale {scale}: 100 sigma MINOS ap {100 * r['minos_ap']:.3f} "
                  f"at {100 * r['minos_at']:.3f} aiso {100 * r['minos_aiso']:.3f} "
                  f"fap {100 * r['minos_fap']:.3f} | HESSE {100 * r['hesse_ap']:.3f}/"
                  f"{100 * r['hesse_at']:.3f} | Fisher {100 * r['fisher_ap']:.3f}/"
                  f"{100 * r['fisher_at']:.3f} | Sigma {r['sigmaNL_par']:.2f}/{r['sigmaNL_per']:.2f} "
                  f"chi2 {r['chi2']:.1f}", flush=True)
    np.savez(dest, **res)
    print("COMPLETE", dest.name, flush=True)


def dense_stack(model, los, kind="raw"):
    return {st: np.mean([np.load(mx.OUT / f"xi_Model{model}_LOS{los}_ph{p}.npz")
                         [f"w0.35_b512_n512_{st}_{kind}"] for p in mx.PHASES], axis=0).T.ravel()
            for st in ("pre", "post")}


def stage_profile(config, cov, model, los):
    from xi_survey_fit import make_fitter
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"profile_{config}_{cov}_Model{model}_LOS{los}.npz"
    if dest.exists():
        return
    fitter = make_fitter(config, cov)
    stacks = {st: v[fitter.mask] for st, v in dense_stack(model, los).items()}
    run_states(fitter, cov, stacks, dest)


def stage_sparse(config, cov, sparse):
    from xi_survey_fit import make_fitter
    import sparse_xi
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"sparse_{config}_{cov}_{sparse}.npz"
    if dest.exists():
        return
    fitter = make_fitter(config, cov, rtmin=sparse_xi.RTMIN)
    stacks = {st: np.mean([np.load(sparse_xi.OUT / f"xi_{sparse}_ph{p}.npz")[f"{st}_raw"]
                           for p in mx.PHASES], axis=0).T.ravel()[fitter.mask]
              for st in ("pre", "post")}
    run_states(fitter, cov, stacks, dest)


def stage_mc(config, cov, model, los, seed, n):
    from xi_survey_fit import make_fitter
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"mc_{config}_{cov}_Model{model}_LOS{los}_seed{seed}.npz"
    if dest.exists():
        return
    fitter = make_fitter(config, cov)
    stacks = {st: v[fitter.mask] for st, v in dense_stack(model, los).items()}
    chol = np.linalg.cholesky(np.linalg.inv(fitter.icov))
    rng = np.random.default_rng(1000 + seed)
    from target_bandpass import save_atomic
    res = {f"{st}_{key}": [] for st in stacks for key in ("ap", "at")}
    for i in range(n):
        noise = chol @ rng.standard_normal(len(chol))
        try:
            fits = {st: fitter.fit(stacks[st] + noise) for st in stacks}
        except RuntimeError as err:
            print("skipped realization:", str(err).splitlines()[0][:60], flush=True)
            continue
        for st, f in fits.items():
            res[f"{st}_ap"].append(f["ap"])
            res[f"{st}_at"].append(f["at"])
        save_atomic(dest, **{key: np.array(val) for key, val in res.items()})   # keep partial work
        print(f"MC {i + 1}/{n}", flush=True)
    print("COMPLETE", dest.name, flush=True)


def stage_curve(config, cov, model, los, scale="pub", npts=25):
    """Profile Delta chi^2 curves for alpha_par and alpha_perp (figure only)."""
    from xi_survey_fit import make_fitter
    dest = OUT / f"curve_{config}_{cov}_Model{model}_LOS{los}_s{scale}.npz"
    fitter = make_fitter(config, cov)
    stacks = {st: v[fitter.mask] for st, v in dense_stack(model, los).items()}
    s = published_scale(fitter, cov) if scale == "pub" else float(scale)
    prof = np.load(OUT / f"profile_{config}_{cov}_Model{model}_LOS{los}.npz")
    lim = dict(fitter.sample["limits"])
    step = {"ap": .01, "at": .01, "bias_LYA": .002, "beta_LYA": .05, "sigmaNL_par": .3, "sigmaNL_per": .3}
    res = {}
    for st, data in stacks.items():
        start = {key: float(prof[f"{st}_s{scale}_{key}"]) for key in ("ap", "at") + NL}
        m = minuit(objective_factory(fitter, data, s), ("ap", "at") + NL, start, lim, step)
        for key in ("ap", "at"):
            width = 2.6 * float(prof[f"pre_s{scale}_minos_{key}"])
            grid = np.clip(np.linspace(m.values[key] - width, m.values[key] + width, npts), .805, 1.195)
            x, y, ok = m.mnprofile(key, grid=grid)
            res[f"{st}_{key}_x"], res[f"{st}_{key}_dchi2"] = x - m.values[key], y - m.fval
            print(st, key, "done", flush=True)
    np.savez(dest, **res)
    print("COMPLETE", dest.name, flush=True)


def summary():
    def line(d, scale):
        parts = []
        kind = "minos"
        if any(not np.isfinite(d[f"{st}_s{scale}_minos_ap"]) for st in ("pre", "post")):
            kind = "fisher"
        for key in ("ap", "at", "aiso", "fap"):
            a, b = d[f"pre_s{scale}_{kind}_{key}"], d[f"post_s{scale}_{kind}_{key}"]
            parts.append(f"{key} {100 * a:.2f}->{100 * b:.2f} G {a / b:.2f}")
        parts.append(f"[{kind}]")
        sig = (f"Sigma {d[f'pre_s{scale}_sigmaNL_par']:.1f},{d[f'pre_s{scale}_sigmaNL_per']:.1f}"
               f" -> {d[f'post_s{scale}_sigmaNL_par']:.1f},{d[f'post_s{scale}_sigmaNL_per']:.1f}")
        return "  ".join(parts) + "  " + sig
    print("Profile-likelihood (MINOS) errors, six-box mean, raw (no control variate); "
          "100 sigma before->after and G")
    for f in sorted(OUT.glob("profile_*.npz")) + sorted(OUT.glob("sparse_*.npz")):
        d = np.load(f)
        for scale in SCALES:
            label = "published cov" if scale == "pub" else "draft weights "
            print(f"{f.stem:44s} {label}: {line(d, scale)}", flush=True)
    print("\nNoise Monte Carlo (noise drawn from the fit covariance, added to the six-box mean):"
          " scatter before->after, G; vs MINOS at the same normalization")
    groups = {}
    for f in sorted(OUT.glob("mc_*.npz")):
        groups.setdefault(f.stem.rsplit("_seed", 1)[0], []).append(np.load(f))
    for name, parts in groups.items():
        prof = OUT / (name.replace("mc_", "profile_") + ".npz")
        row = []
        for key in ("ap", "at"):
            pre = np.concatenate([p[f"pre_{key}"] for p in parts])
            post = np.concatenate([p[f"post_{key}"] for p in parts])
            g = pre.std(ddof=1) / post.std(ddof=1)
            boot = []
            rng = np.random.default_rng(0)
            for _ in range(2000):
                i = rng.integers(0, len(pre), len(pre))
                boot.append(pre[i].std(ddof=1) / post[i].std(ddof=1))
            txt = (f"{key} {100 * pre.std(ddof=1):.2f}->{100 * post.std(ddof=1):.2f} "
                   f"G {g:.2f}+/-{np.std(boot):.2f} (N={len(pre)})")
            if prof.exists():
                d = np.load(prof)
                txt += f" [MINOS {100 * d[f'pre_s1.0_minos_{key}']:.2f}->{100 * d[f'post_s1.0_minos_{key}']:.2f}]"
            row.append(txt)
        print(f"{name:44s} " + "   ".join(row), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "profile":
        stage_profile(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif stage == "sparse":
        stage_sparse(sys.argv[2], sys.argv[3], sys.argv[4])
    elif stage == "curve":
        stage_curve(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif stage == "mc":
        stage_mc(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]),
                 int(sys.argv[7]))
    else:
        summary()
