"""
scan_smoothing.py
-----------------
Pick the reconstruction smoothing radius R.

R = 12.5 Mpc/h is inherited from low-z galaxy surveys, where the displacements
are much larger.  At z = 2.5 the per-axis rms displacement is only ~1.5 Mpc/h,
so a smaller R plausibly recovers more -- but as R shrinks, halo shot noise and
nonlinear power leak into Psi and the reconstruction starts adding noise
instead of removing smearing.  There is an optimum; this finds it.

For each R: build the recon, paint the post-recon field (skipped if the .npy is
already on disk), measure P(k, mu), fit, and record sigma(alpha) and Sigma.
The pre-recon reference is fit once and drawn as a horizontal line.

Usage
-----
  python scan_smoothing.py --driver lya --sr 8 10 12.5 15 \
      --nmesh 512 --nmesh_recon 512 --model 1
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
from recon_paint_field import (build_recon, build_recon_lya, paint, OUT_DIR,
                               LBOX, FOREST_RESPONSE, forest_response_tag,
                               save_npy_atomic)
from fit_bao_pkmu import measure, bin_mask, PkmuModel, PkmuFitter

SIM, Z = "AbacusSummit_base_c000_ph000", 2.5


def field_path(a, sr):
    if a.driver == "lya":
        bt = forest_response_tag(a.bias_F, a.beta_F)
        driver_tag = f"_{bt}_lyadrv"
    else:
        driver_tag = f"_b{a.bias:.1f}"
    return OUT_DIR / (f"dF_recon_Model_{a.model}_LOS{a.los}_nmesh{a.nmesh}"
                      f"_R{sr:.2f}_nr{a.nmesh_recon}_{a.rectype}"
                      f"{driver_tag}_{SIM}_z{Z:.3f}.npy")


def ref_path(a):
    return OUT_DIR / (f"dF_ref_Model_{a.model}_LOS{a.los}_nmesh{a.nmesh}"
                      f"_{SIM}_z{Z:.3f}.npy")


def ensure_field(a, sr):
    fn = field_path(a, sr)
    if fn.exists():
        print(f"  R={sr:5.2f}  field already on disk", flush=True)
        return fn
    print(f"  R={sr:5.2f}  building ...", flush=True)
    rsd_f = None
    if a.driver == "halos":
        recon, _ = build_recon(SIM, Z, a.nmesh_recon, sr, a.rectype,
                               a.bias, a.ncpu, 0, verbose=False)
    else:
        from cosmoprimo.fiducial import AbacusSummit
        recon = build_recon_lya(SIM, a.model, a.los, Z, a.nmesh_recon,
                                sr, a.rectype, a.bias_F, a.beta_F, a.ncpu,
                                verbose=False)
        rsd_f = AbacusSummit().growth_rate(Z)
    field = paint(SIM, a.model, a.los, a.nmesh, "recon", recon=recon,
                  rsd_f=rsd_f, verbose=False, workers=a.ncpu)
    save_npy_atomic(fn, field)
    print(f"  R={sr:5.2f}  std={field.std():.5f}  saved", flush=True)
    del recon, field
    gc.collect()
    return fn


def fit_one(fn, cosmo, a, shared=None):
    """Measure + fit one field. `shared` carries the bin selection so every R
    is fitted on exactly the same (k, mu) bins."""
    k, mu, P, N = measure(fn, a.dk, a.kmax_meas, a.nmu, verbose=False)
    if shared is None:
        good = bin_mask(k, P, N, a.kmin, a.kmax)
        iwedge = np.broadcast_to(np.arange(P.shape[1])[None, :], P.shape)
        shared = dict(good=good, k=k[good], mu=mu[good], N=N[good],
                      iw=iwedge[good])
        shared["model"] = PkmuModel(cosmo, shared["k"], shared["mu"],
                                    shared["iw"], a.nmu, nbb=a.nbb)
    g = shared["good"]
    Pv = P[g]
    var = 2.0 * Pv ** 2 / np.maximum(shared["N"], 1.0)
    res = PkmuFitter(shared["model"], Pv, 1.0 / var).fit()
    return res, shared


def main(a):
    from template_pk import class_pk
    cosmo = class_pk(z=Z)
    print(f"template: CLASS + savgol, r_s = {cosmo.rs_drag:.3f} Mpc/h")
    print(f"displacement tracer: {a.driver}")
    print(f"fit: {a.kmin} < k < {a.kmax}, {a.nmu} wedges, nbb={a.nbb}\n")

    print("pre-recon reference:")
    pre, shared = fit_one(ref_path(a), cosmo, a)
    print(f"  ap = {pre['ap']:.4f} +- {pre['sigma_ap']:.4f} | "
          f"at = {pre['at']:.4f} +- {pre['sigma_at']:.4f} | "
          f"Sig = {pre['sig_par']:.2f}/{pre['sig_per']:.2f} | "
          f"beta = {pre['beta']:.2f}\n")

    rows = []
    for sr in a.sr:
        fn = ensure_field(a, sr)
        res, _ = fit_one(fn, cosmo, a, shared=shared)
        rows.append((sr, res))
        print(f"  R={sr:5.2f}  ap = {res['ap']:.4f} +- {res['sigma_ap']:.4f}"
              f"  (x{pre['sigma_ap']/res['sigma_ap']:.2f})"
              f" | at = {res['at']:.4f} +- {res['sigma_at']:.4f}"
              f"  (x{pre['sigma_at']/res['sigma_at']:.2f})"
              f" | Sig = {res['sig_par']:.2f}/{res['sig_per']:.2f}"
              f" | beta = {res['beta']:.2f}"
              f" | chi2/dof = {res['chi2']:.0f}/{res['ndof']}", flush=True)

    R = np.array([r[0] for r in rows])
    gp = np.array([pre['sigma_ap'] / r[1]['sigma_ap'] for r in rows])
    gt = np.array([pre['sigma_at'] / r[1]['sigma_at'] for r in rows])
    sp = np.array([r[1]['sig_par'] for r in rows])
    st = np.array([r[1]['sig_per'] for r in rows])
    be = np.array([r[1]['beta'] for r in rows])

    # Do NOT just take argmax(gain).  sigma(alpha) falls monotonically as R
    # shrinks, but so does the validity of the fit: past a point the model no
    # longer describes the post-recon P(k,mu) and the "improvement" is the fit
    # absorbing a model failure, not a real gain.  Three physical cuts:
    #
    #   Sigma away from 0    A parameter on its lower bound is not a resolved
    #                        damping measurement.  The expected ordering
    #                        Sigma_par > Sigma_perp is checked on the six-phase
    #                        production stack, not imposed on one noisy phase.
    #   Sig_perp < pre       reconstruction cannot INCREASE transverse damping.
    #   chi2/dof, beta       the model must still fit, and the Kaiser factor
    #                        must not have drifted far from its pre-recon value.
    chi2dof = np.array([r[1]['chi2'] / r[1]['ndof'] for r in rows])
    ok = ((sp > 0.2) & (st > 0.2) & (st < pre['sig_per'] * 1.02)
          & (chi2dof < a.max_chi2dof)
          & (np.abs(be - pre['beta']) < a.max_dbeta))
    print("\n  R      gain_ap  gain_at  Sig_par  Sig_perp   beta   chi2/dof  ok")
    for i in range(len(R)):
        print(f"  {R[i]:5.1f}   x{gp[i]:.2f}    x{gt[i]:.2f}   "
              f"{sp[i]:6.2f}   {st[i]:6.2f}   {be[i]:5.2f}   {chi2dof[i]:6.3f}"
              f"    {'yes' if ok[i] else 'NO'}")
    if ok.any():
        cand = np.where(ok)[0]
        ibest = int(cand[np.argmax(gp[cand])])
        print(f"\n  >>> recommended R = {R[ibest]} Mpc/h   "
              f"(x{gp[ibest]:.2f} on alpha_par, x{gt[ibest]:.2f} on alpha_perp)")
        iraw = int(np.argmax(gp))
        if not ok[iraw]:
            print(f"      best raw gain was R = {R[iraw]} "
                  f"(x{gp[iraw]:.2f}) but it fails the sanity cuts above")
        else:
            print("      the radius with the best raw gain also passes the "
                  "stated sanity cuts")
    else:
        ibest = int(np.argmax(gp))
        print("\n  >>> NO radius passes the sanity cuts -- the post-recon model "
              "is wrong everywhere; do not quote a gain")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot(R, gp, "o-", label=r"$\alpha_\parallel$")
    ax[0].plot(R, gt, "s-", label=r"$\alpha_\perp$")
    ax[0].axhline(1, color="k", lw=.7)
    ax[0].set_ylabel(r"$\sigma_{\rm pre}/\sigma_{\rm post}$")
    ax[0].set_title("improvement vs smoothing radius")

    ax[1].plot(R, sp, "o-", label=r"$\Sigma_\parallel$ post")
    ax[1].plot(R, st, "s-", label=r"$\Sigma_\perp$ post")
    ax[1].axhline(pre['sig_par'], color="C0", ls="--", lw=1,
                  label=r"$\Sigma_\parallel$ pre")
    ax[1].axhline(pre['sig_per'], color="C1", ls="--", lw=1,
                  label=r"$\Sigma_\perp$ pre")
    ax[1].set_ylabel(r"$\Sigma$  [Mpc/$h$]")
    ax[1].set_title("residual BAO damping")

    ax[2].plot(R, be, "o-", color="C3")
    ax[2].axhline(pre['beta'], color="C3", ls="--", lw=1, label=r"$\beta$ pre")
    ax[2].set_ylabel(r"$\beta$")
    ax[2].set_title(r"Kaiser $\beta$ (should stay near its pre value)")

    for A in ax:
        A.set_xlabel("R [Mpc/$h$]")
        A.legend(fontsize=9)
        A.grid(alpha=.25)
    fig.tight_layout()
    suffix = "lyadrv" if a.driver == "lya" else "halodrv"
    plot = a.plot or str(OUT_DIR / "figs_nowiggle" /
                         f"scan_R_{suffix}.png")
    save = a.save or str(OUT_DIR /
                         f"scan_R_Model_{a.model}_LOS{a.los}_ph000_{suffix}.npz")
    Path(plot).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=110)
    print(f"  figure -> {plot}")

    np.savez(save, R=R, gain_ap=gp, gain_at=gt, sig_par=sp, sig_per=st,
             beta=be,
             bias_F=np.array(a.bias_F), beta_F=np.array(a.beta_F),
             source=np.array([str(field_path(a, value)) for value in R]),
             pre_sigma_ap=pre['sigma_ap'], pre_sigma_at=pre['sigma_at'],
             pre_sig_par=pre['sig_par'], pre_sig_per=pre['sig_per'],
             pre_beta=pre['beta'],
             ap=np.array([r[1]['ap'] for r in rows]),
             at=np.array([r[1]['at'] for r in rows]))
    print(f"  saved   -> {save}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sr", type=float, nargs="+",
                   default=[5.0, 7.5, 10.0, 12.5, 15.0, 20.0])
    p.add_argument("--nmesh", type=int, default=512)
    p.add_argument("--nmesh_recon", type=int, default=512)
    p.add_argument("--bias", type=float, default=2.70)
    p.add_argument("--driver", choices=["lya", "halos"], default="lya")
    p.add_argument("--bias_F", type=float, default=None)
    p.add_argument("--beta_F", type=float, default=None)
    p.add_argument("--rectype", default="MG")
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--los", default="z", choices=["z"])
    p.add_argument("--ncpu", type=int, default=1,
                   help="FFT and reconstruction threads; set this explicitly "
                        "inside an allocated batch job")
    p.add_argument("--kmin", type=float, default=0.02)
    p.add_argument("--kmax", type=float, default=0.40)
    p.add_argument("--dk", type=float, default=0.005)
    p.add_argument("--kmax_meas", type=float, default=0.5)
    p.add_argument("--nmu", type=int, default=5)
    p.add_argument("--nbb", type=int, default=3)
    p.add_argument("--max_chi2dof", type=float, default=1.10,
                   help="reject a radius whose post-recon fit is worse than this")
    p.add_argument("--max_dbeta", type=float, default=0.20,
                   help="reject a radius whose beta drifts further than this "
                        "from the pre-recon value")
    p.add_argument("--plot", default=None)
    p.add_argument("--save", default=None)
    a = p.parse_args()
    if a.model not in FOREST_RESPONSE:
        p.error(f"no calibrated forest response for Model {a.model}")
    default_bias, default_beta = FOREST_RESPONSE[a.model]
    if a.bias_F is None:
        a.bias_F = default_bias
    if a.beta_F is None:
        a.beta_F = default_beta
    main(a)
