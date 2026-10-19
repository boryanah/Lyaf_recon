"""
sparse_skewers.py
-----------------
How much of the reconstruction gain survives when the Lya field is sampled
along discrete quasar sightlines instead of as a full 3D volume?

This is an experimental sparse-field calculation.  The paper does not use its
post-reconstruction results because the available pair-count and painted FFT
fields do not yet have the same effective window.  The code is retained to
construct that matched test.  Dense-field results alone do not establish how
much reconstruction gain survives survey sightline sampling.

Geometry
--------
At z = 2.5 the comoving distance is 4022 Mpc/h, so one degree subtends
70.2 Mpc/h and one square degree is 4929 (Mpc/h)^2.  DESI's Lya sample is
~50 quasars/deg^2, i.e.

    n_2D = 0.0101 per (Mpc/h)^2  ->  40580 skewers across the 2000 Mpc/h box
                                ->  f = 0.155 of the 512^2 available sightlines
                                ->  mean transverse separation 9.9 Mpc/h

which is essentially the R = 10 Mpc/h smoothing scale.  The calculation below
tests where the gain begins to fall, but its absolute alpha values require the
matched-estimator validation described above.

Estimator
---------
Both fields use the MASK-WEIGHTED estimator  delta_F * W / <W>,  built by
`paint(..., norm="global")`.  Its power is the true P(k) plus the usual Lya
aliasing pedestal.  The sampling operator is common to pre and post, but the
shape of P_1D is measured separately in the two states because reconstruction
changes the line-of-sight power.  The alternative -- dividing pointwise by the
local occupancy -- would interpolate across the gaps and invent information
the survey never had.

The reconstruction is driven from the SPARSE field too, via
`build_recon_lya(mask=...)`, which assigns the same sightline positions as
randoms so pyrecon divides out the geometry rather than reading empty cells as
voids.  That is the whole point: a real survey has to estimate Psi from what it
can see.

Usage
-----
  python sparse_skewers.py --frac 1.0 0.5 0.155 0.08 0.04 --nmesh 512
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from recon_paint_field import (build_recon_lya, paint, OUT_DIR, LBOX,
                               FOREST_RESPONSE, forest_response_tag,
                               save_npy_atomic)
from fit_bao_pkmu import measure, bin_mask, PkmuModel, PkmuFitter

SIM, Z = "AbacusSummit_base_c000_ph000", 2.5
CHI_Z25 = 4022.0                      # Mpc/h, comoving distance to z = 2.5
DEG = CHI_Z25 * np.pi / 180.0         # Mpc/h per degree


def measure_p1d(field, mask=None, window="continuous"):
    """P_1D(k_par) of the skewers, in (Mpc/h).

    This is the shape of the aliasing term that contaminates an FFT-based P_3D
    on sparse sightlines:  P_measured = P_3D + P_1D(k_par) / n_2D.  It is white
    in k_perp, so it cannot be absorbed by broadband terms built from powers of
    |k| -- it needs its own template, which is what this provides.
    """
    from scipy.fft import rfft
    n = field.shape[0]
    fz = rfft(field, axis=2) / n
    p1 = (np.abs(fz) ** 2).mean(axis=(0, 1)) * LBOX
    kz = np.fft.rfftfreq(n, d=1.0 / n) * (2 * np.pi / LBOX)
    modes = np.fft.rfftfreq(n, d=1.0 / n)
    if window == "node":
        response = 0.75 + 0.25 * np.cos(2.0 * np.pi * modes / n)
    elif window == "continuous":
        response = np.sinc(modes / n) ** 3
    else:
        raise ValueError("window must be 'node' or 'continuous'")
    p1 /= response ** 2
    return kz, p1


def frac_to_qso_per_deg2(frac, nmesh):
    """Translate a sightline fraction into quasars per square degree."""
    n2d = frac * nmesh ** 2 / LBOX ** 2
    return n2d * DEG ** 2


def make_mask(nmesh, frac, seed):
    """Random transverse sightline mask, shape (nmesh, nmesh)."""
    if frac >= 1.0:
        return None
    rng = np.random.default_rng(seed)
    m = np.zeros(nmesh * nmesh, dtype=bool)
    n = int(round(frac * nmesh * nmesh))
    m[rng.choice(nmesh * nmesh, size=n, replace=False)] = True
    return m.reshape(nmesh, nmesh)


def field_path(a, frac, mode, sim):
    tag = "dense" if frac >= 1.0 else f"f{frac:.3f}"
    if mode == "ref":
        if frac >= 1.0:
            return OUT_DIR / (f"dF_ref_Model_{a.model}_LOS{a.los}"
                              f"_nmesh{a.nmesh}_{sim}_z{Z:.3f}.npy")
        return OUT_DIR / (f"dF_skref_{tag}_Model_{a.model}_LOS{a.los}"
                          f"_nmesh{a.nmesh}_seed{a.seed}_{sim}_z{Z:.3f}.npy")
    if frac >= 1.0:
        response = forest_response_tag(a.bias_F, a.beta_F)
        return OUT_DIR / (f"dF_recon_Model_{a.model}_LOS{a.los}"
                          f"_nmesh{a.nmesh}_R{a.sr:.2f}_nr{a.nmesh_recon}_"
                          f"{a.rectype}_{response}_lyadrv_{sim}_z{Z:.3f}.npy")
    response = forest_response_tag(a.bias_F, a.beta_F)
    return OUT_DIR / (f"dF_skrec_{tag}_Model_{a.model}_LOS{a.los}"
                      f"_nmesh{a.nmesh}_R{a.sr:.2f}_nr{a.nmesh_recon}_"
                      f"{a.rectype}_{response}_seed{a.seed}_"
                      f"{sim}_z{Z:.3f}.npy")


def ensure(a, frac, mask, sim):
    """Build the sparse pre/post pair for one sightline fraction and phase."""
    fp = field_path(a, frac, "ref", sim)
    fq = field_path(a, frac, "recon", sim)
    norm = "local" if frac >= 1.0 else "global"

    if not fp.exists():
        g = paint(sim, a.model, a.los, a.nmesh, "ref", mask=mask, norm=norm,
                  verbose=False, workers=a.ncpu)
        save_npy_atomic(fp, g)
        print(f"    ref  std={g.std():.5f}", flush=True)
        del g; gc.collect()
    if not fq.exists():
        from cosmoprimo.fiducial import AbacusSummit
        rsd_f = AbacusSummit().growth_rate(Z)
        recon = build_recon_lya(sim, a.model, a.los, Z, a.nmesh_recon, a.sr,
                                a.rectype, a.bias_F, a.beta_F, a.ncpu,
                                mask=mask, verbose=False)
        g = paint(sim, a.model, a.los, a.nmesh, "recon", recon=recon,
                  rsd_f=rsd_f, mask=mask, norm=norm, verbose=False,
                  workers=a.ncpu)
        save_npy_atomic(fq, g)
        print(f"    post std={g.std():.5f}", flush=True)
        del g, recon; gc.collect()
    return fp, fq


def fit(fn, cosmo, a, shared=None, p1d=None, sig_prior=None):
    k, mu, P, N = measure(fn, a.dk, a.kmax_meas, a.nmu, verbose=False,
                          workers=a.ncpu)
    if shared is None:
        good = bin_mask(k, P, N, a.kmin, a.kmax)
        iw = np.broadcast_to(np.arange(P.shape[1])[None, :], P.shape)[good]
        shared = dict(good=good, N=N[good], k=k[good], mu=mu[good], iw=iw)
    g = shared["good"]
    extra = None
    if p1d is not None:
        kz, p1 = p1d
        kpar = shared["k"] * shared["mu"]
        extra = np.interp(kpar, kz, p1)[:, None]
    model = PkmuModel(cosmo, shared["k"], shared["mu"], shared["iw"], a.nmu,
                      nbb=a.nbb, extra_cols=extra)
    Pv = P[g]
    var = 2.0 * Pv ** 2 / np.maximum(shared["N"], 1.0)
    return PkmuFitter(model, Pv, 1.0 / var, sig_prior=sig_prior).fit(), shared


def stack_fit(a, frac, cosmo, sims, sig_prior_pre, sig_prior_post, use_alias):
    """Build (if needed) and fit the stacked pre/post pair at one density."""
    mask = make_mask(a.nmesh, frac, a.seed)
    fps, fqs = [], []
    for sim in sims:
        fp, fq = ensure(a, frac, mask, sim)
        fps.append(fp); fqs.append(fq)

    # The selection is shared, but reconstruction changes P_1D itself.  Use a
    # phase-averaged template for each state rather than imposing the pre shape
    # on the post spectrum.  A free amplitude absorbs the known 1/n_2D factor.
    p1d = {"pre": None, "post": None}
    if use_alias and frac < 1.0:
        # Average each shape over every phase.  Using the first phase alone
        # would imprint its one-dimensional sample variance on the stack.
        for state, paths, window in (("pre", fps, "node"),
                                     ("post", fqs, "continuous")):
            measured = [measure_p1d(np.load(path), window=window)
                        for path in paths]
            p1d[state] = (measured[0][0],
                          np.mean([item[1] for item in measured], axis=0))

    def one(files, sig_prior, alias_shape):
        meas = [measure(f, a.dk, a.kmax_meas, a.nmu, verbose=False,
                        workers=a.ncpu) for f in files]
        k0, mu0, P0, N0 = meas[0]
        good = bin_mask(k0, P0, N0, a.kmin, a.kmax)
        iw = np.broadcast_to(np.arange(P0.shape[1])[None, :], P0.shape)[good]
        Ps = np.array([m[2][good] for m in meas])
        Pm = Ps.mean(axis=0)
        var = 2.0 * Pm ** 2 / np.maximum(N0[good], 1.0)
        extra = None
        if alias_shape is not None:
            extra = np.interp(k0[good] * mu0[good], alias_shape[0],
                              alias_shape[1])[:, None]
        model = PkmuModel(cosmo, k0[good], mu0[good], iw, a.nmu, nbb=a.nbb,
                          extra_cols=extra)
        return PkmuFitter(model, Pm, len(Ps) / var,
                          sig_prior=sig_prior).fit(), len(Ps)

    rp, n = one(fps, sig_prior_pre, p1d["pre"])
    rq, _ = one(fqs, sig_prior_post, p1d["post"])
    return rp, rq, n, p1d


def main(a):
    from template_pk import class_pk
    cosmo = class_pk(z=Z)
    sims = [f"AbacusSummit_base_c000_ph{p}" for p in a.phases]
    if a.nmesh_recon != a.nmesh and any(frac < 1.0 for frac in a.frac):
        raise ValueError("sparse masks are defined on the painting mesh; "
                         "nmesh_recon must equal nmesh for sparse runs")
    print(f"template r_s = {cosmo.rs_drag:.3f} Mpc/h")
    print(f"grid {a.nmesh}^3, cell {LBOX/a.nmesh:.2f} Mpc/h, R = {a.sr} Mpc/h")
    print(f"phases: {', '.join(a.phases)}")
    print(f"aliasing template: {'ON' if a.alias else 'off'}   "
          f"damping prior: {'OFF' if a.sig_width <= 0 else f'width {a.sig_width} Mpc/h'}\n")

    # Sigma is a property of the field and of the reconstruction, not of the
    # sampling, so the sparse fits are held near the DENSE best fit rather than
    # allowed to rail against zero.  Pre and post get their own centres.
    pri_pre = pri_post = None
    if a.sig_width > 0:
        pri_pre = (a.sig_pre[0], a.sig_pre[1], a.sig_width)
        pri_post = (a.sig_post[0], a.sig_post[1], a.sig_width)
        print(f"  prior centres: pre Sig = {a.sig_pre}, post Sig = {a.sig_post}\n")

    rows = []
    for frac in a.frac:
        nsl = int(round(frac * a.nmesh ** 2))
        sep = np.sqrt(LBOX ** 2 / max(nsl, 1))
        q = frac_to_qso_per_deg2(frac, a.nmesh)
        print(f"  f = {frac:.3f}   {nsl:7d} skewers   sep = {sep:5.1f} Mpc/h   "
              f"~{q:6.1f} QSO/deg2", flush=True)
        # Pre and post carry their own damping-prior centres.  Fit the pair in
        # one pass so the P1D template and cached measurements are identical.
        rp2, rq2, n, p1d = stack_fit(a, frac, cosmo, sims, pri_pre, pri_post,
                                     a.alias)
        gp = rp2['sigma_ap'] / rq2['sigma_ap']
        gt = rp2['sigma_at'] / rq2['sigma_at']
        rows.append(dict(frac=frac, nsl=nsl, sep=sep, q=q, pre=rp2, post=rq2,
                         gain_ap=gp, gain_at=gt, nreal=n, p1d=p1d))
        print(f"      pre : ap={rp2['ap']:.4f}+-{rp2['sigma_ap']:.4f} "
              f"at={rp2['at']:.4f}+-{rp2['sigma_at']:.4f} "
              f"Sig={rp2['sig_par']:.2f}/{rp2['sig_per']:.2f} "
              f"beta={rp2['beta']:.2f} c2={rp2['chi2']/rp2['ndof']:.2f}")
        print(f"      post: ap={rq2['ap']:.4f}+-{rq2['sigma_ap']:.4f} "
              f"at={rq2['at']:.4f}+-{rq2['sigma_at']:.4f} "
              f"Sig={rq2['sig_par']:.2f}/{rq2['sig_per']:.2f} "
              f"beta={rq2['beta']:.2f} c2={rq2['chi2']/rq2['ndof']:.2f}")
        print(f"      GAIN: ap x{gp:.2f}   at x{gt:.2f}\n", flush=True)

    q = np.array([r['q'] for r in rows])
    gp = np.array([r['gain_ap'] for r in rows])
    gt = np.array([r['gain_at'] for r in rows])
    print("  QSO/deg2   sep   gain_ap gain_at   sig_ap_pre  sig_ap_post")
    for i, r in enumerate(rows):
        print(f"  {q[i]:8.1f}  {r['sep']:4.1f}   x{gp[i]:.2f}   x{gt[i]:.2f}"
              f"     {r['pre']['sigma_ap']:.5f}    {r['post']['sigma_ap']:.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].semilogx(q, gp, "o-", label=r"$\alpha_\parallel$")
    ax[0].semilogx(q, gt, "s-", label=r"$\alpha_\perp$")
    ax[0].axhline(1, color="k", lw=.7)
    ax[0].set_ylabel(r"$\sigma_{\rm pre}/\sigma_{\rm post}$")
    ax[0].set_title(f"gain vs sightline density ({len(a.phases)} phases)")
    for key, lab, mk in (("sig_par", r"$\Sigma_\parallel$", "o"),
                         ("sig_per", r"$\Sigma_\perp$", "s")):
        ax[1].semilogx(q, [r['post'][key] for r in rows], mk + "-", label=lab + " post")
        ax[1].semilogx(q, [r['pre'][key] for r in rows], mk + "--", alpha=.5,
                       label=lab + " pre")
    ax[1].set_ylabel(r"$\Sigma$ [Mpc/$h$]"); ax[1].set_title("residual damping")
    ax[2].loglog(q, [r['pre']['sigma_ap'] for r in rows], "o--", alpha=.6,
                 label=r"$\sigma(\alpha_\parallel)$ pre")
    ax[2].loglog(q, [r['post']['sigma_ap'] for r in rows], "o-",
                 label=r"$\sigma(\alpha_\parallel)$ post")
    ax[2].set_ylabel(r"$\sigma(\alpha_\parallel)$")
    ax[2].set_title("absolute error")
    for A in ax:
        A.axvline(50, color="crimson", ls="--", lw=1.2)
        A.set_xlabel("QSO per deg²"); A.legend(fontsize=8); A.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(a.plot, dpi=110)
    print(f"\n  figure -> {a.plot}")

    alias_products = {}
    for r in rows:
        tag = f"f{r['frac']:.3f}".replace(".", "p")
        for state in ("pre", "post"):
            item = r["p1d"][state]
            if item is not None:
                alias_products[f"alias_kpar_{state}_{tag}"] = item[0]
                alias_products[f"alias_p1d_{state}_{tag}"] = item[1]

    np.savez(a.save, qso_per_deg2=q, gain_ap=gp, gain_at=gt,
             frac=np.array([r['frac'] for r in rows]),
             sep=np.array([r['sep'] for r in rows]),
             phases=np.array(a.phases), model=np.array(a.model),
             los=np.array(a.los), redshift=np.array(Z),
             nmesh=np.array(a.nmesh), nmesh_recon=np.array(a.nmesh_recon),
             smoothing_radius=np.array(a.sr), rectype=np.array(a.rectype),
             mask_seed=np.array(a.seed), bias_F=np.array(a.bias_F),
             beta_F=np.array(a.beta_F), kmin=np.array(a.kmin),
             kmax=np.array(a.kmax), kmax_meas=np.array(a.kmax_meas),
             dk=np.array(a.dk), nmu=np.array(a.nmu), nbb=np.array(a.nbb),
             alias_template=np.array(a.alias),
             alias_template_state_specific=np.array(a.alias),
             damping_prior_width=np.array(a.sig_width),
             damping_prior_pre=np.array(a.sig_pre),
             damping_prior_post=np.array(a.sig_post),
             nreal=np.array([r['nreal'] for r in rows]),
             **alias_products,
             **{f"{w}_{k}": np.array([r[w][k] for r in rows])
                for w in ("pre", "post")
                for k in ("ap", "at", "sigma_ap", "sigma_at", "beta",
                          "sig_par", "sig_per", "chi2", "ndof")})
    print(f"  saved   -> {a.save}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phases", nargs="+",
                   default=["000", "001", "002", "003", "004", "005"])
    p.add_argument("--alias", action="store_true", default=True,
                   help="include the P_1D(k_par)/n_2D aliasing template")
    p.add_argument("--no-alias", dest="alias", action="store_false")
    p.add_argument("--sig_width", type=float, default=1.0,
                   help="Gaussian prior width on Sigma, Mpc/h. 0 disables it")
    p.add_argument("--sig_pre", type=float, nargs=2, default=[4.61, 2.81],
                   help="prior centre (Sig_par, Sig_perp) for the PRE fit, "
                        "from the dense 6-phase result")
    p.add_argument("--sig_post", type=float, nargs=2, default=[2.16, 1.36],
                   help="same for the POST fit")
    p.add_argument("--frac", type=float, nargs="+",
                   default=[1.0, 0.5, 0.25, 0.155, 0.08, 0.04, 0.02],
                   help="fraction of the nmesh^2 transverse cells kept as "
                        "sightlines. 0.155 is DESI-like at nmesh=512")
    p.add_argument("--nmesh", type=int, default=512)
    p.add_argument("--nmesh_recon", type=int, default=512)
    p.add_argument("--sr", type=float, default=10.0)
    p.add_argument("--rectype", default="MG")
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--los", default="z", choices=["z"])
    p.add_argument("--bias_F", type=float, default=None)
    p.add_argument("--beta_F", type=float, default=None)
    p.add_argument("--ncpu", type=int, default=1,
                   help="FFT and reconstruction threads; set this explicitly "
                        "inside an allocated batch job")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--kmin", type=float, default=0.02)
    p.add_argument("--kmax", type=float, default=0.40)
    p.add_argument("--dk", type=float, default=0.005)
    p.add_argument("--kmax_meas", type=float, default=0.5)
    p.add_argument("--nmu", type=int, default=5)
    p.add_argument("--nbb", type=int, default=3)
    p.add_argument("--plot", default="/pscratch/sd/b/boryanah/Lyaf_recon/"
                                     "figs_nowiggle/sparse_skewers.png")
    p.add_argument("--save", default="/pscratch/sd/b/boryanah/Lyaf_recon/"
                                     "sparse_skewers_Model_1_LOSz_ph000.npz")
    a = p.parse_args()
    if a.model not in FOREST_RESPONSE:
        p.error(f"no calibrated forest response for Model {a.model}")
    default_bias, default_beta = FOREST_RESPONSE[a.model]
    if a.bias_F is None:
        a.bias_F = default_bias
    if a.beta_F is None:
        a.beta_F = default_beta
    main(a)
