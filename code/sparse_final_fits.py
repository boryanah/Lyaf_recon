"""
sparse_final_fits.py
--------------------
Precision gains versus sightline density on ONE footing (paper Fig. 6 and
Table IV): six-phase stacks, Model 1, LOSz, 0.02 < k < 0.35 h/Mpc, three
broadband terms per wedge, the production damping priors, and -- for sparse
masks -- a free-amplitude P_1D(k_par) alias column with the phase-averaged
shape of the fitted state.  The fully sampled forest (frac = 1) uses the same
fit without an alias column.

For each density and displacement source it stores the stacked pre and post
fits, the gains, the gains with each state's errors rescaled by
sqrt(chi2/dof), the prior-free gains, and the k <= 0.30 gains.

Inputs: fields written by sparse_skewers.py (Regime III) and
regime2_ensemble.py (Regime II).  Output:
/pscratch/sd/b/boryanah/Lyaf_recon/sparse_final_fits.npz
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "paper_prd"))

import regime2_fit as R                                     # noqa: E402
from fit_bao_pkmu import measure, bin_mask, PkmuModel, PkmuFitter  # noqa: E402
from recon_paint_field import OUT_DIR                        # noqa: E402
from regime2_ensemble import TRACERS, tracer_tag             # noqa: E402
from template_pk import class_pk                             # noqa: E402

FRACS = (1.0, 0.5, 0.25, 0.155, 0.08)
DENSE_PRE = "dF_ref_Model_1_LOSz_nmesh512_{sim}_z2.500.npy"
DENSE_POST = ("dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_"
              "bF-0.148066_bt0.957688_lyadrv_{sim}_z2.500.npy")


def dense_fitter(paths, prior):
    spectra = [measure(p, 0.005, 0.5, 5, verbose=False) for p in paths]
    k, mu, p0, n = spectra[0]
    good = bin_mask(k, p0, n, R.KMIN, R.KMAX)
    iw = np.broadcast_to(np.arange(k.shape[1])[None, :], k.shape)[good]
    powers = np.asarray([s[2][good] for s in spectra])
    mean = powers.mean(axis=0)
    var = 2.0 * mean ** 2 / np.maximum(n[good], 1.0)
    model = PkmuModel(R.COSMO, k[good], mu[good], iw, 5, nbb=3)
    return PkmuFitter(model, mean, len(powers) / var, sig_prior=prior)


def pair(frac, pre, post, kmax, prior):
    R.KMAX = kmax
    pp = R.PRE_PRIOR if prior else None
    qq = R.POST_PRIOR if prior else None
    if frac >= 1.0:
        a = dense_fitter(pre, pp).fit()
        b = dense_fitter(post, qq).fit()
    else:
        a = R.fitter(pre[0], pre[1], pp).fit()
        b = R.fitter(post[0], post[1], qq).fit()
    return a, b


def gain(a, b, rescale=False):
    fa = np.sqrt(a["chi2"] / a["ndof"]) if rescale else 1.0
    fb = np.sqrt(b["chi2"] / b["ndof"]) if rescale else 1.0
    return np.array([a["sigma_ap"] * fa / (b["sigma_ap"] * fb),
                     a["sigma_at"] * fa / (b["sigma_at"] * fb)])


def main():
    R.COSMO = class_pk(z=2.5)
    dest = OUT_DIR / "sparse_final_fits.npz"
    # Merge into an existing product so densities can be added incrementally.
    out = dict(np.load(dest)) if dest.exists() else {}
    fracs = [float(x) for x in sys.argv[1:]] or list(FRACS)
    for frac in fracs:
        tag = f"f{frac:.3f}"
        if frac >= 1.0:
            pre = [OUT_DIR / DENSE_PRE.format(sim=s) for s in R.SIMS]
            sources = {"III_self": [OUT_DIR / DENSE_POST.format(sim=s) for s in R.SIMS]}
        else:
            pre = R.state([R.pre_path(frac, s) for s in R.SIMS], "node")
            sources = {"III_self": [R.self_path(frac, s) for s in R.SIMS]}
            if frac in (0.155, 0.5):
                for t in TRACERS:
                    sources[f"II_{tracer_tag(t)}"] = [R.field_path(frac, t, s)
                                                      for s in R.SIMS]
        for label, paths in sources.items():
            post = paths if frac >= 1.0 else R.state(paths, "continuous")
            a, b = pair(frac, pre, post, 0.35, True)
            an, bn = pair(frac, pre, post, 0.35, False)
            a3, b3 = pair(frac, pre, post, 0.30, True)
            key = f"{tag}_{label}"
            out[f"{key}_gain"] = gain(a, b)
            out[f"{key}_gain_rescaled"] = gain(a, b, rescale=True)
            out[f"{key}_gain_noprior"] = gain(an, bn)
            out[f"{key}_gain_kmax030"] = gain(a3, b3)
            for st, r in (("pre", a), ("post", b)):
                for k in ("ap", "at", "sigma_ap", "sigma_at", "sig_par",
                          "sig_per", "beta", "chi2", "ndof"):
                    out[f"{key}_{st}_{k}"] = r[k]
            print(f"{key:18s} gain {out[f'{key}_gain'].round(3)} "
                  f"rescaled {out[f'{key}_gain_rescaled'].round(3)} "
                  f"noprior {out[f'{key}_gain_noprior'].round(3)} "
                  f"kmax0.30 {out[f'{key}_gain_kmax030'].round(3)} "
                  f"chi2/dof {a['chi2']/a['ndof']:.2f}->{b['chi2']/b['ndof']:.2f}",
                  flush=True)
    done = sorted({float(k.split("_")[0][1:]) for k in out
                   if k.startswith("f") and k.endswith("_gain")}, reverse=True)
    out["fracs"] = np.array(done)
    out["qso_per_deg2"] = np.array([f * 512 ** 2 / 2000.0 ** 2 * (4022.0 * np.pi / 180) ** 2
                                    for f in done])
    np.savez(dest, **out)
    print("saved", dest)


if __name__ == "__main__":
    main()
