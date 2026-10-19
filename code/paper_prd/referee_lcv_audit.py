#!/usr/bin/env python3
"""Separate LCV subtraction, fit weighting, and coefficient reuse.

Reads only archived band powers. No fields or initial conditions are loaded.
Each output retains individual phases and orientations; error estimates must
average orientations within phase before using six-phase scatter.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from fit_bao_pkmu import PkmuFitter, PkmuModel, bin_mask
from lcv_reconstruction_check import coefficient
from template_pk import class_pk

PARS = ("ap", "at", "beta", "sig_par", "sig_per", "chi2")


def source_path(model, los):
    suffix = "" if los == "z" else ("_LOSy" if model == 1 else f"_Model{model}_LOSy")
    return HERE / "data" / f"lcv_reconstruction_results{suffix}.npz"


def run(model, los, destination):
    start = time.monotonic()
    with np.load(source_path(model, los)) as src:
        d = {key: np.array(src[key]) for key in src.files}
    k, mu, count = (d[x] for x in ("k", "mu", "N"))
    control = d[f"m{model}_control_bandpower"]
    theory = d[f"m{model}_theory_bandpower"]
    raw = {s: d[f"m{model}_{s}_raw_bandpower"] for s in ("pre", "post")}
    cross = {s: d[f"m{model}_{s}_cross_bandpower"] for s in raw}
    assert control.shape == theory.shape == raw["pre"].shape == (6, 100, 5)
    good = bin_mask(k, raw["pre"].mean(0), count, 0.02, 0.40)
    wedge = np.broadcast_to(np.arange(5)[None, :], k.shape)[good]
    fit_model = PkmuModel(class_pk(z=2.5), k[good], mu[good], wedge, 5, nbb=3)
    raw_var = {s: 2 * raw[s].mean(0)**2 / np.maximum(count, 1) for s in raw}
    cv, loo, cv_var = {}, {}, {}
    out = dict(model=np.array(model), los=np.array(los), phases=np.arange(6),
               source=np.array(str(source_path(model, los))),
               description=np.array(__doc__))
    for s in raw:
        q = coefficient(cross[s], control, k, "ensemble_wedges", count)
        np.testing.assert_allclose(q, d[f"m{model}_{s}_ensemble_wedges_coefficient"],
                                   rtol=1e-12, atol=1e-12)
        cv[s] = raw[s] - q * (control-theory)
        np.testing.assert_allclose(cv[s], d[f"m{model}_{s}_ensemble_wedges_bandpower"])
        qloo = np.array([coefficient(cross[s][np.arange(6) != p],
                                    control[np.arange(6) != p], k,
                                    "ensemble_wedges", count)[0]
                         for p in range(6)])
        loo[s] = raw[s] - qloo * (control-theory)
        cbar, xbar, tbar = control.mean(0), cross[s].mean(0), raw[s].mean(0)
        qbar = q.mean(0)
        cv_var[s] = np.maximum(2 * (tbar**2 + qbar**2*cbar**2-2*qbar*xbar**2)
                               / np.maximum(count, 1), 1e-4*raw_var[s])
        out[f"{s}_loo_coefficient"] = qloo
        out[f"{s}_cv_variance"] = cv_var[s]
        out[f"{s}_raw_variance"] = raw_var[s]
    out["k"] = k
    out["mu"] = mu
    out["N"] = count

    cases = {
        "raw_common_raw": (raw, {s: raw_var["pre"] for s in raw}),
        "cv_common_raw": (cv, {s: raw_var["pre"] for s in raw}),
        "raw_common_cv": (raw, {s: cv_var["pre"] for s in raw}),
        "cv_common_cv": (cv, {s: cv_var["pre"] for s in raw}),
        "loo_common_cv": (loo, {s: cv_var["pre"] for s in raw}),
        "loo_common_raw": (loo, {s: raw_var["pre"] for s in raw}),
        "cv_state_weights": (cv, cv_var),
    }
    for case, (spectra, variances) in cases.items():
        for s in raw:
            fits = []
            for p in range(6):
                f = PkmuFitter(fit_model, spectra[s][p][good], 1/variances[s][good])
                fit = f.fit(compute_errors=False)
                fits.append(fit)
            for par in PARS:
                out[f"{case}_{s}_{par}"] = np.array([f[par] for f in fits])
        delta = 100*(out[f"{case}_post_at"]-out[f"{case}_pre_at"])
        print(f"m{model} LOS{los} {case}: delta_at {delta.mean():+.5f} "
              f"+/- {delta.std(ddof=1)/np.sqrt(6):.5f}%", flush=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, **out)

    # Hold the surrogate and coefficient fixed while shifting the target's
    # fitted acoustic template. A CV must not pin a real target shift to zero.
    injected = np.array([-0.003, 0.003])
    out["injected_delta_at"] = injected
    for case in ("cv_common_raw", "cv_common_cv"):
        spectra, variances = cases[case]
        for s in raw:
            weight = 6/variances[s][good]
            mean = spectra[s].mean(0)[good]
            fitter = PkmuFitter(fit_model, mean, weight)
            base = fitter.fit(compute_errors=False)
            theta = np.array([base[p] for p in PARS[:5]])
            recovered = []
            for shift in injected:
                shifted = theta.copy()
                shifted[1] += shift
                signal = base["bsq"]*(fit_model.peak(*shifted)-fit_model.peak(*theta))
                fit = PkmuFitter(fit_model, mean+signal, weight).fit(
                    x0=shifted, compute_errors=False)
                recovered.append(fit["at"]-base["at"])
            out[f"{case}_{s}_recovered_delta_at"] = np.array(recovered)
            print(f"m{model} LOS{los} {case} {s} injection: "
                  f"{100*np.array(recovered)} percent for {100*injected}", flush=True)
    out["elapsed_seconds"] = time.monotonic()-start
    np.savez(destination, **out)
    print(f"saved {destination}; {out['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--los", choices=("y", "z"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.los, args.output)
