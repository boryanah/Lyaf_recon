"""
fit_bao_pkmu.py
---------------
Anisotropic BAO fit to P(k, mu) of the pre- and post-reconstruction Lya fields,
and the improvement factor on alpha_par / alpha_perp.

Why Fourier and not configuration space
---------------------------------------
The measurement is natively a power spectrum -- the delta_F fields come off a
periodic grid, so P(k, mu) band powers are close to diagonal, whereas the
xi(rp, pi) grid bins are strongly correlated and we have no reliable covariance
for them.  With one realization per phase that difference decides the fit.

Model
-----
    P(k, mu) = b^2 (1 + beta mu'^2)^2
               [ (P_w - P_nw)(k') exp(-k'^2 (mu'^2 Sig_par^2
                                             + (1-mu'^2) Sig_perp^2) / 2)
                 + P_nw(k') ] / (ap at^2)
               + sum_j a_j(mu) k^-j

with the Alcock-Paczynski dilation

    k'_par = k mu / ap ,   k'_perp = k sqrt(1-mu^2) / at

so ap scales line-of-sight separations and at transverse ones -- the same
convention as fit_bao_apat_standalone.py, and the standard Lya one.  ap = at =
1 recovers the fiducial cosmology.  The 1/(ap at^2) is the volume Jacobian.

The Kaiser factor and the damping use mu' (the true-frame angle), not the
observed mu, because they are properties of the true clustering.

Nonlinear parameters (Nelder-Mead): ap, at, beta, Sig_par, Sig_perp.
Linear parameters (profiled exactly at every chi2 call): the amplitude b^2 and
a per-wedge broadband  a_0 + a_1/k + a_2/k^2.  Letting the broadband float
freely per wedge is what makes this a BAO-only fit: the information comes from
the phase of the wiggles, not from the shape of the continuum.

Template
--------
CLASS + savgol de-wiggling via template_pk.  Do NOT fall back to EH98 -- its
sound horizon is 2.7% long and biases every alpha by that amount.  See
template_pk.py.

Usage
-----
  # one phase, pre vs post
  python fit_bao_pkmu.py \
      --pre  /pscratch/sd/b/boryanah/Lyaf_recon/dF_ref_Model_1_LOSz_nmesh512_*.npy \
      --post /pscratch/sd/b/boryanah/Lyaf_recon/dF_recon_Model_1_LOSz_nmesh512_R10.00_*lyadrv_*.npy \
      --plot /pscratch/sd/b/boryanah/Lyaf_recon/figs_nowiggle/pkmu_fit.png

  # several phases: pass them all, they are fitted individually and stacked
  python fit_bao_pkmu.py --pre dF_ref_*ph00?*.npy --post dF_recon_*ph00?*.npy
"""

import argparse
import gc
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

sys.path.insert(0, str(Path(__file__).parent))
from measure_pkmu import pkmu, LBOX
from fit_bao_apat_standalone import nelder_mead

PRIORS = dict(ap=(0.8, 1.2), at=(0.8, 1.2), beta=(0.0, 5.0),
              sig_par=(0.0, 20.0), sig_per=(0.0, 20.0))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Measurement
# ═════════════════════════════════════════════════════════════════════════════

def measure(fn, dk, kmax, nmu, cache=True, verbose=True, window="auto",
            workers=1):
    """P(k, mu) of a delta_F grid, cached next to the .npy.

    An unjittered zero-displacement reference is painted from particles on
    exact mesh nodes and therefore has the discrete node TSC response.  Other
    fields use the generic continuous response.  ``window='auto'`` infers
    this from the production filenames and gives node caches a distinct tag.
    """
    fn = Path(fn)
    if window == "auto":
        is_reference = (fn.stem.startswith("dF_ref_")
                        or fn.stem.startswith("dF_skref_"))
        window = "node" if is_reference and "_jit_" not in fn.stem else "continuous"
    if window not in ("node", "continuous"):
        raise ValueError("window must be 'auto', 'node', or 'continuous'")
    # not with_suffix(): these filenames already contain dots ("z2.500.npy")
    window_tag = "_winnode" if window == "node" else ""
    cfn = fn.parent / (f"{fn.stem}.pkmu_dk{dk}_kmax{kmax}_nmu{nmu}"
                       f"{window_tag}.npz")
    if cache and cfn.exists():
        # A filename alone is not sufficient provenance: legacy continuous
        # caches predate the explicit window metadata.  Reuse only a cache
        # that declares the requested compensation and assignment response.
        with np.load(cfn) as d:
            valid = ("window" in d.files
                     and str(d["window"]) == window
                     and "compensated" in d.files
                     and bool(d["compensated"]))
            if valid:
                result = tuple(np.array(d[key]) for key in ("k", "mu", "P", "N"))
            else:
                result = None
        if result is not None:
            if verbose:
                print(f"  cached  {fn.name}")
            return result
        if verbose:
            print(f"  ignoring legacy cache without matching provenance: "
                  f"{cfn.name}")
    g = np.load(fn)
    if g.ndim != 3 or g.shape[0] != g.shape[1] or g.shape[1] != g.shape[2]:
        raise ValueError(f"expected a cubic 3D field in {fn}, got {g.shape}; "
                         "this may be an MPI-local slab")
    if verbose:
        print(f"  measuring {fn.name}  ({g.shape[0]}^3, std={g.std():.5f})",
              flush=True)
    k, mu, P, N = pkmu(g, dk=dk, kmax=kmax, nmu=nmu, window=window,
                       workers=workers)
    del g; gc.collect()
    if cache:
        np.savez(cfn, k=k, mu=mu, P=P, N=N,
                 source=np.array(str(fn)), window=np.array(window),
                 compensated=np.array(True), fft_workers=np.array(workers))
    return k, mu, P, N


def bin_mask(k, P, N, kmin, kmax):
    """Which (k, mu) bins enter the fit. Computed ONCE and reused for every
    realization, so the data vectors stay the same length and the same bins."""
    return np.isfinite(P) & (N > 0) & (k >= kmin) & (k <= kmax) & (P > 0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Model
# ═════════════════════════════════════════════════════════════════════════════

class PkmuModel:
    def __init__(self, cosmo, k, mu, iwedge, nmu, nbb=3, extra_cols=None):
        self.cosmo = cosmo
        self.k, self.mu, self.iw = k, mu, iwedge
        self.nmu = nmu
        # per-wedge broadband basis: a_0 + a_1/k + a_2/k^2, one set per wedge,
        # to marginalize smooth shape in each wedge. This finite basis does
        # not guarantee that all smooth-spectrum dilation information is
        # removed; see paper_prd/referee_template_audit.py.
        cols = []
        for w in range(nmu):
            sel = (iwedge == w).astype(np.float64)
            for j in range(nbb):
                cols.append(sel * k ** (-float(j)))
        self.BB = np.vstack(cols).T if cols else np.zeros((len(k), 0))
        # Optional extra linear columns, profiled alongside the broadband.
        # Used for the sparse-skewer aliasing term P_1D(k_par) / n_2D, which is
        # white in k_perp and therefore NOT representable by the per-wedge
        # powers of |k| above -- that mismatch is what makes a naive FFT
        # estimator fail on sparse sightlines.
        if extra_cols is not None:
            e = np.atleast_2d(np.asarray(extra_cols, dtype=np.float64))
            if e.shape[0] != len(k):
                e = e.T
            self.BB = np.hstack([self.BB, e])

    def peak(self, ap, at, beta, sp, st):
        """The b^2 = 1 template, AP-dilated. Column 0 of the design matrix."""
        kpar = self.k * self.mu / ap
        kper = self.k * np.sqrt(np.maximum(1.0 - self.mu ** 2, 0.0)) / at
        kk = np.sqrt(kpar ** 2 + kper ** 2)
        kk = np.maximum(kk, 1e-8)
        mm = kpar / kk
        pw = self.cosmo.plin(kk, wiggle=True)
        pnw = self.cosmo.plin(kk, wiggle=False)
        damp = np.exp(-0.5 * kk ** 2
                      * (mm ** 2 * sp ** 2 + (1.0 - mm ** 2) * st ** 2))
        p = (pw - pnw) * damp + pnw
        return (1.0 + beta * mm ** 2) ** 2 * p / (ap * at ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Fit
# ═════════════════════════════════════════════════════════════════════════════

class PkmuFitter:
    def __init__(self, model, data, ivar, sig_prior=None):
        """Add an optional Gaussian damping prior (Sig_par0, Sig_perp0, width).

        A dense-field prior is an informative modeling assumption for sparse
        reconstruction: sparse sampling changes the inferred displacement,
        so it need not preserve the dense reconstructed damping. Report the
        prior and test sensitivity before interpreting a precision gain.
        """
        self.m, self.data, self.ivar = model, data, ivar
        self.sig_prior = sig_prior

    def _chi2(self, theta, want_coef=False):
        ap, at, beta, sp, st = theta
        pen = 0.0
        for val, (lo, hi) in zip(theta, PRIORS.values()):
            if val < lo:
                pen += 1e4 * (lo - val) ** 2
            if val > hi:
                pen += 1e4 * (val - hi) ** 2
        ap = np.clip(ap, *PRIORS['ap']); at = np.clip(at, *PRIORS['at'])
        beta = np.clip(beta, *PRIORS['beta'])
        sp = np.clip(sp, *PRIORS['sig_par']); st = np.clip(st, *PRIORS['sig_per'])

        if self.sig_prior is not None:
            sp0, st0, wsig = self.sig_prior
            pen += ((sp - sp0) / wsig) ** 2 + ((st - st0) / wsig) ** 2

        A = np.hstack([self.m.peak(ap, at, beta, sp, st)[:, None], self.m.BB])
        # Solve the weighted least-squares problem directly.  Forming the
        # normal equations squares the condition number; this matters here
        # because the k^-2 broadband columns span several orders of magnitude.
        sw = np.sqrt(self.ivar)
        Aw = A * sw[:, None]
        dw = self.data * sw
        try:
            coef = np.linalg.lstsq(Aw, dw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return (1e30, None) if want_coef else 1e30
        r = self.data - A @ coef
        c2 = float(np.sum(r * r * self.ivar)) + pen
        return (c2, coef) if want_coef else c2

    def _profile_chi2(self, best, fixed_index, fixed_value):
        """Minimize chi2 over the four parameters not held fixed."""
        free = [i for i in range(5) if i != fixed_index]
        bounds = [list(PRIORS.values())[i] for i in free]

        def objective(values):
            theta = np.array(best, copy=True)
            theta[fixed_index] = fixed_value
            theta[free] = values
            return self._chi2(theta)

        result = minimize(objective, np.asarray(best)[free], method="L-BFGS-B",
                          bounds=bounds,
                          options={"ftol": 1e-12, "gtol": 1e-8,
                                   "maxiter": 1000})
        return float(result.fun)

    def _profile_interval(self, best, chi2_min, index, initial_step=0.004):
        """Return the two roots of profile_chi2 = chi2_min + 1.

        The old implementation changed one alpha while holding beta and both
        damping scales fixed.  Those were conditional errors, although the
        output called them Delta-chi2=1 errors.  This routine reoptimizes all
        nonlinear nuisance parameters at every trial alpha.
        """
        lo_bound, hi_bound = list(PRIORS.values())[index]
        cache = {}

        def delta(value):
            key = float(value)
            if key not in cache:
                cache[key] = (self._profile_chi2(best, index, key)
                              - chi2_min - 1.0)
            return cache[key]

        roots = []
        center = float(best[index])
        for direction, bound in ((-1.0, lo_bound), (1.0, hi_bound)):
            outer = center
            step = initial_step
            while direction * (bound - outer) > 0:
                trial = np.clip(outer + direction * step, lo_bound, hi_bound)
                if delta(trial) >= 0.0:
                    roots.append(brentq(delta, min(center, trial),
                                         max(center, trial), xtol=1e-8))
                    break
                if trial == bound:
                    roots.append(np.nan)
                    break
                outer = trial
                step *= 1.5
            else:
                roots.append(np.nan)
        return roots[0], roots[1]

    def fit(self, x0=(1.0, 1.0, 1.5, 6.0, 3.0), verbose=False,
            compute_errors=True):
        f = self._chi2
        best, chi2 = nelder_mead(f, list(x0), step=[.02, .02, .3, 1.5, 1.0])
        best, chi2 = nelder_mead(f, best, step=[.005, .005, .1, .5, .5])
        _, coef = self._chi2(best, want_coef=True)
        npar = 5 + len(coef)
        res = dict(ap=best[0], at=best[1], beta=best[2], sig_par=best[3],
                   sig_per=best[4], bsq=coef[0], chi2=chi2,
                   ndof=len(self.data) - npar)
        # Profile-likelihood Delta chi^2 = 1 intervals.  beta, Sigma_parallel,
        # and Sigma_perp are reoptimized rather than held fixed.
        if compute_errors:
            for j, name in ((0, 'ap'), (1, 'at')):
                lower, upper = self._profile_interval(best, chi2, j)
                err_lo = best[j] - lower if np.isfinite(lower) else np.nan
                err_hi = upper - best[j] if np.isfinite(upper) else np.nan
                res['sigma_' + name + '_lo'] = float(err_lo)
                res['sigma_' + name + '_hi'] = float(err_hi)
                res['sigma_' + name] = float(0.5 * (err_lo + err_hi))
        else:
            res['sigma_ap'] = res['sigma_at'] = np.nan
        if verbose:
            print("    " + "  ".join(f"{k}={res[k]:.4f}" for k in
                                     ('ap', 'at', 'beta', 'sig_par', 'sig_per'))
                  + f"  chi2/dof={res['chi2']:.1f}/{res['ndof']}")
        return res


# ═════════════════════════════════════════════════════════════════════════════
# 4. Driver
# ═════════════════════════════════════════════════════════════════════════════

def analyze(files, label, cosmo, a, var_from_scatter=None):
    """Fit each file, then the stack. Returns a summary dict."""
    if not files:
        raise ValueError(f"no input files for {label}")
    print(f"\n=== {label}: {len(files)} realization(s) ===", flush=True)
    meas = [measure(f, a.dk, a.kmax_meas, a.nmu) for f in files]
    k0, mu0, P0, N0 = meas[0]
    for i, (ki, mui, Pi, Ni) in enumerate(meas[1:], start=1):
        if Pi.shape != P0.shape or not np.allclose(ki, k0, equal_nan=True) \
                or not np.allclose(mui, mu0, equal_nan=True) \
                or not np.array_equal(Ni, N0):
            raise ValueError(f"incompatible P(k,mu) grid in input {i}: "
                             f"{files[i]}")
    good = bin_mask(k0, P0, N0, a.kmin, a.kmax)
    iwedge = np.broadcast_to(np.arange(P0.shape[1])[None, :], P0.shape)
    kk, mm, NN, iw = k0[good], mu0[good], N0[good], iwedge[good]
    Ps = np.array([m[2][good] for m in meas])
    print(f"  {len(kk)} bins in the fit "
          f"({a.kmin} < k < {a.kmax}, {a.nmu} wedges)")

    model = PkmuModel(cosmo, kk, mm, iw, a.nmu, nbb=a.nbb)

    # Gaussian band-power variance: each k,mu bin averages N/2 independent
    # complex modes of a real field, so var(P) = P^2 / (N/2) = 2 P^2 / N.
    # The default uses this Gaussian variance.  Realization scatter is used
    # only when explicitly requested with --var scatter; estimating a separate
    # inverse variance for every bin from six boxes is too noisy for the
    # production fit.
    Pmean = Ps.mean(axis=0)
    var_g = 2.0 * Pmean ** 2 / np.maximum(NN, 1.0)
    if var_from_scatter is not None:
        var1 = var_from_scatter
    elif getattr(a, "var", "gauss") == "scatter" and len(Ps) >= 3:
        var1 = np.maximum(np.var(Ps, axis=0, ddof=1), 1e-3 * var_g)
        print(f"  variance from realization scatter "
              f"(median ratio to Gaussian {np.median(var1/var_g):.2f})")
        print(f"  WARNING: a per-bin variance estimated from {len(Ps)} samples "
              f"is itself noisy.\n           Weighting by 1/s^2 inflates chi2 by "
              f"(n-1)/(n-3) = {(len(Ps)-1)/(len(Ps)-3):.2f} and shrinks the "
              f"Delta chi2 = 1 errors\n           by its square root. Use "
              f"--var gauss for unbiased errors.")
    else:
        var1 = var_g
        print("  variance: Gaussian band-power counting, var = 2 P^2 / N_modes")

    per = []
    for i, P in enumerate(Ps):
        # Per-realization fits are used for their central values and scatter.
        # Profile intervals are needed only for the stacked likelihood.
        r = PkmuFitter(model, P, 1.0 / var1).fit(compute_errors=False)
        per.append(r)
        print(f"  real {i}: ap = {r['ap']:.4f}  at = {r['at']:.4f}  "
              f"beta = {r['beta']:.2f}  Sig = {r['sig_par']:.2f}/"
              f"{r['sig_per']:.2f}  chi2/dof = {r['chi2']:.1f}/{r['ndof']}",
              flush=True)

    stack = PkmuFitter(model, Pmean, len(Ps) / var1).fit()
    print(f"  stack : ap = {stack['ap']:.4f} +- {stack['sigma_ap']:.4f}  |  "
          f"at = {stack['at']:.4f} +- {stack['sigma_at']:.4f}  |  "
          f"beta = {stack['beta']:.2f}  Sig_par = {stack['sig_par']:.2f}  "
          f"Sig_perp = {stack['sig_per']:.2f}")
    out = dict(label=label, per=per, stack=stack, model=model,
               Pmean=Pmean, var1=var1, k=kk, mu=mm, iw=iw, nreal=len(Ps))
    if len(Ps) >= 2:
        out['ap_scatter'] = float(np.std([r['ap'] for r in per], ddof=1))
        out['at_scatter'] = float(np.std([r['at'] for r in per], ddof=1))
        print(f"  scatter over realizations: sigma(ap) = {out['ap_scatter']:.4f}"
              f"  sigma(at) = {out['at_scatter']:.4f}")
    return out


def report(pre, post):
    print("\n" + "=" * 70)
    print("   RECONSTRUCTION IMPROVEMENT   (sigma_pre / sigma_post)")
    print("=" * 70)
    for key, name in (('sigma_ap', 'alpha_par '), ('sigma_at', 'alpha_perp')):
        x, y = pre['stack'][key], post['stack'][key]
        print(f"  {name}  (dchi2=1 on the stack):   {x:.5f} -> {y:.5f}"
              f"    x{x / y:.2f}")
    if 'ap_scatter' in pre and 'ap_scatter' in post:
        for key, name in (('ap_scatter', 'alpha_par '),
                          ('at_scatter', 'alpha_perp')):
            x, y = pre[key], post[key]
            print(f"  {name}  (realization scatter):  {x:.5f} -> {y:.5f}"
                  f"    x{x / y:.2f}")
    print("\n  damping  Sig_par  {:.2f} -> {:.2f} Mpc/h      "
          "Sig_perp  {:.2f} -> {:.2f} Mpc/h".format(
              pre['stack']['sig_par'], post['stack']['sig_par'],
              pre['stack']['sig_per'], post['stack']['sig_per']))
    print("  beta     {:.3f} -> {:.3f}   (a large shift means the "
          "relocation-only\n           construction is distorting the Kaiser "
          "factor -- see BUGS_AND_ISSUES.md)".format(
              pre['stack']['beta'], post['stack']['beta']))


def make_plot(pre, post, path):
    """Wiggles-only view: (P - smooth part of the best fit) / smooth, per wedge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [(pre, "pre-recon"), (post, "post-recon")]
    nmu = pre['model'].nmu
    fig, axes = plt.subplots(1, nmu, figsize=(3.1 * nmu, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for res, lab in sets:
        m, st = res['model'], res['stack']
        th = (st['ap'], st['at'], st['beta'], st['sig_par'], st['sig_per'])
        A = np.hstack([m.peak(*th)[:, None], m.BB])
        sw = np.sqrt(1.0 / res['var1'])
        coef = np.linalg.lstsq(A * sw[:, None], res['Pmean'] * sw,
                              rcond=None)[0]
        # the same model with the wiggles switched off = the smooth reference
        th0 = (st['ap'], st['at'], st['beta'], 1e3, 1e3)   # damp wiggles away
        A0 = np.hstack([m.peak(*th0)[:, None], m.BB])
        smooth = A0 @ coef
        for w in range(nmu):
            s = res['iw'] == w
            o = np.argsort(res['k'][s])
            axes[w].plot(res['k'][s][o],
                         (res['Pmean'][s][o] - smooth[s][o]) / smooth[s][o],
                         lw=1.4, label=lab)
            axes[w].plot(res['k'][s][o],
                         ((A @ coef)[s][o] - smooth[s][o]) / smooth[s][o],
                         lw=1.0, ls="--", color="k", alpha=.5)
    for w in range(nmu):
        axes[w].axhline(0, color="k", lw=.6)
        axes[w].set_title(rf"$\mu \in [{w/nmu:.1f}, {(w+1)/nmu:.1f}]$",
                          fontsize=10)
        axes[w].set_xlabel("k [h/Mpc]")
    axes[0].set_ylabel(r"$P/P_{\rm smooth}-1$")
    axes[0].legend(fontsize=8)
    fig.suptitle("BAO wiggles by wedge; dashed = best-fit model", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pre', nargs='+', required=True, help='pre-recon .npy grids')
    p.add_argument('--post', nargs='+', default=None, help='post-recon .npy grids')
    p.add_argument('--kmin', type=float, default=0.02)
    p.add_argument('--kmax', type=float, default=0.40)
    p.add_argument('--dk', type=float, default=0.005)
    p.add_argument('--kmax_meas', type=float, default=0.5,
                   help='kmax for the P(k,mu) measurement/cache')
    p.add_argument('--nmu', type=int, default=5)
    p.add_argument('--nbb', type=int, default=3,
                   help='broadband terms per wedge: a_0 + a_1/k + ... + a_j/k^j')
    p.add_argument('--z', type=float, default=2.5)
    p.add_argument('--plot', default=None)
    p.add_argument('--var', default='gauss', choices=['gauss', 'scatter'],
                   help="band-power variance. 'gauss' (default) uses "
                        "2 P^2 / N_modes; 'scatter' uses the realization "
                        "scatter, which is BIASED for small n -- see the "
                        "warning it prints")
    p.add_argument('--save', default=None)
    a = p.parse_args()

    from template_pk import class_pk
    cosmo = class_pk(z=a.z)
    print(f"template: CLASS + savgol,  r_s,drag = {cosmo.rs_drag:.3f} Mpc/h")
    print(f"fit range: {a.kmin} < k < {a.kmax} h/Mpc, {a.nmu} mu wedges, "
          f"{a.nbb} broadband terms per wedge")

    pre_files = sorted(a.pre)
    post_files = sorted(a.post) if a.post else []
    if post_files:
        if len(pre_files) != len(post_files):
            raise ValueError("pre/post file counts differ: "
                             f"{len(pre_files)} pre and {len(post_files)} post")
        pre_phases = [re.search(r"ph\d+", Path(x).name) for x in pre_files]
        post_phases = [re.search(r"ph\d+", Path(x).name) for x in post_files]
        if all(pre_phases) and all(post_phases):
            pre_tags = [x.group(0) for x in pre_phases]
            post_tags = [x.group(0) for x in post_phases]
            if pre_tags != post_tags:
                raise ValueError(f"pre/post phase mismatch: {pre_tags} vs "
                                 f"{post_tags}")

    pre = analyze(pre_files, "pre-recon", cosmo, a)
    post = None
    if post_files:
        post = analyze(post_files, "post-recon", cosmo, a)
        report(pre, post)
        if a.plot:
            make_plot(pre, post, a.plot)

    if a.save:
        out = {}
        for tag, r in (("pre", pre), ("post", post)):
            if r is None:
                continue
            for k in ('ap', 'at', 'sigma_ap', 'sigma_at',
                      'sigma_ap_lo', 'sigma_ap_hi',
                      'sigma_at_lo', 'sigma_at_hi', 'beta',
                      'sig_par', 'sig_per', 'chi2'):
                out[f"{tag}_{k}"] = r['stack'][k]
            out[f"{tag}_aps"] = np.array([x['ap'] for x in r['per']])
            out[f"{tag}_ats"] = np.array([x['at'] for x in r['per']])
        np.savez(a.save, **out)
        print(f"\nSaved -> {a.save}")
