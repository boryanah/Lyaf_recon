#!/usr/bin/env python3
"""Compare the legacy and response-calibrated Model 1 reconstructions.

The paired scale response is fit with a common pre-reconstruction weight so
changes in the power amplitude cannot change the objective function.  Central
stack fits are also repeated with each state's own Gaussian weight to report
the damping scales and profile-likelihood precision.  A compact stability
grid checks k cuts and broadband order.
"""

from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRATCH = Path("/pscratch/sd/b/boryanah/Lyaf_recon")
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import PkmuFitter, PkmuModel, bin_mask  # noqa: E402
from template_pk import class_pk  # noqa: E402


def phase(path):
    match = re.search(r"ph(\d{3})", path.name)
    if match is None:
        raise ValueError(f"no phase in {path}")
    return match.group(1)


def six(root, pattern):
    paths = sorted(root.glob(pattern))
    expected = [f"{index:03d}" for index in range(6)]
    if len(paths) != 6 or [phase(path) for path in paths] != expected:
        raise ValueError(f"expected {expected} for {pattern}, got {paths}")
    return paths


def load(paths):
    rows = []
    for path in paths:
        with np.load(path) as data:
            rows.append(tuple(np.array(data[key]) for key in ("k", "mu", "P", "N")))
    k, mu, _, count = rows[0]
    for path, row in zip(paths[1:], rows[1:]):
        if (row[2].shape != rows[0][2].shape
                or not np.allclose(row[0], k, equal_nan=True)
                or not np.allclose(row[1], mu, equal_nan=True)
                or not np.array_equal(row[3], count)):
            raise ValueError(f"incompatible grid in {path}")
    return rows


def fit_pair(pre_rows, post_rows, cosmo, kmin, kmax, nbb, weight_mode,
             stack_errors=False):
    k, mu, p0, count = pre_rows[0]
    good = bin_mask(k, p0, count, kmin, kmax)
    wedge = np.broadcast_to(np.arange(p0.shape[1])[None, :], p0.shape)[good]
    model = PkmuModel(cosmo, k[good], mu[good], wedge, p0.shape[1], nbb=nbb)
    pre = np.array([row[2][good] for row in pre_rows])
    post = np.array([row[2][good] for row in post_rows])
    nmode = np.maximum(count[good], 1.0)
    var_pre = 2.0 * pre.mean(0)**2 / nmode
    var_post = 2.0 * post.mean(0)**2 / nmode
    if weight_mode == "pre":
        variances = (var_pre, var_pre)
    elif weight_mode == "own":
        variances = (var_pre, var_post)
    else:
        raise ValueError(weight_mode)

    result = {}
    for state, spectra, variance in zip(("pre", "post"), (pre, post), variances):
        per = [PkmuFitter(model, spectrum, 1.0 / variance)
               .fit(compute_errors=False) for spectrum in spectra]
        stack = PkmuFitter(model, spectra.mean(0), 6.0 / variance).fit(
            compute_errors=stack_errors
        )
        result[state] = {key: np.array([row[key] for row in per])
                         for key in ("ap", "at", "beta", "sig_par", "sig_per",
                                     "chi2")}
        result[state]["stack"] = stack
    return result


def summarize_response(name, result):
    dap = 100.0 * (result["post"]["ap"] - result["pre"]["ap"])
    dat = 100.0 * (result["post"]["at"] - result["pre"]["at"])
    print(f"{name:18s}: dapar={dap.mean():+.3f}+/-"
          f"{dap.std(ddof=1)/np.sqrt(6):.3f}%  "
          f"daperp={dat.mean():+.3f}+/-"
          f"{dat.std(ddof=1)/np.sqrt(6):.3f}%")
    return dap, dat


def main():
    pre_paths = six(
        SCRATCH,
        "dF_ref_Model_1_LOSz_nmesh512_AbacusSummit_base_c000_ph00?_z2.500."
        "pkmu_dk0.005_kmax0.5_nmu5_winnode.npz",
    )
    cases = {
        "legacy_local": six(
            SCRATCH,
            "dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_bF-0.192_"
            "lyadrv_AbacusSummit_base_c000_ph00?_z2.500."
            "pkmu_dk0.005_kmax0.5_nmu5.npz",
        ),
        "legacy_global": six(
            SCRATCH,
            "dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_bF-0.192_"
            "lyadrv_global_AbacusSummit_base_c000_ph00?_z2.500."
            "pkmu_dk0.005_kmax0.5_nmu5.npz",
        ),
        "calibrated_local": six(
            SCRATCH,
            "dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_"
            "bF-0.148066_bt0.957688_"
            "lyadrv_AbacusSummit_base_c000_ph00?_z2.500."
            "pkmu_dk0.005_kmax0.5_nmu5.npz",
        ),
        "calibrated_global": six(
            SCRATCH,
            "dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_bF-0.148_bt0.96_"
            "lyadrv_global_AbacusSummit_base_c000_ph00?_z2.500."
            "pkmu_dk0.005_kmax0.5_nmu5.npz",
        ),
    }
    pre_rows = load(pre_paths)
    cosmo = class_pk(z=2.5)
    output = {"cases": np.array(list(cases)),
              "pre_source": np.array([str(path) for path in pre_paths])}

    for name, paths in cases.items():
        post_rows = load(paths)
        paired = fit_pair(pre_rows, post_rows, cosmo, 0.02, 0.40, 3, "pre")
        dap, dat = summarize_response(name, paired)
        own = fit_pair(pre_rows, post_rows, cosmo, 0.02, 0.40, 3, "own",
                       stack_errors=True)
        output[f"{name}_delta_ap_percent"] = dap
        output[f"{name}_delta_at_percent"] = dat
        output[f"{name}_source"] = np.array([str(path) for path in paths])
        for state in ("pre", "post"):
            for key in ("ap", "at", "beta", "sig_par", "sig_per", "chi2"):
                output[f"{name}_{state}_{key}_per_phase"] = own[state][key]
            for key, value in own[state]["stack"].items():
                if key != "bb":
                    output[f"{name}_{state}_{key}"] = value
        p, q = own["pre"]["stack"], own["post"]["stack"]
        print(f"  own-weight stack: alpha {p['ap']:.6f}/{p['at']:.6f} -> "
              f"{q['ap']:.6f}/{q['at']:.6f}; Sigma "
              f"{p['sig_par']:.2f}/{p['sig_per']:.2f} -> "
              f"{q['sig_par']:.2f}/{q['sig_per']:.2f}; profile gain "
              f"{p['sigma_ap']/q['sigma_ap']:.3f}/"
              f"{p['sigma_at']/q['sigma_at']:.3f}")

    configs = []
    for value in (0.25, 0.30, 0.35, 0.40):
        configs.append(("kmax", value, 0.02, value, 3))
    for value in (0.015, 0.02, 0.03, 0.04):
        configs.append(("kmin", value, value, 0.40, 3))
    for value in (2, 3, 4, 5):
        configs.append(("nbb", value, 0.02, 0.40, value))
    stability_cases, axes, values, deltas_ap, deltas_at = [], [], [], [], []
    for case in ("calibrated_local", "calibrated_global"):
        post_rows = load(cases[case])
        for axis, value, kmin, kmax, nbb in configs:
            result = fit_pair(pre_rows, post_rows, cosmo, kmin, kmax, nbb, "pre")
            dap = 100.0 * (result["post"]["ap"] - result["pre"]["ap"])
            dat = 100.0 * (result["post"]["at"] - result["pre"]["at"])
            stability_cases.append(case); axes.append(axis); values.append(value)
            deltas_ap.append(dap); deltas_at.append(dat)
            print(f"  stability {case} {axis}={value}: "
                  f"dapar={dap.mean():+.3f}+/-"
                  f"{dap.std(ddof=1)/np.sqrt(6):.3f}% "
                  f"daperp={dat.mean():+.3f}+/-"
                  f"{dat.std(ddof=1)/np.sqrt(6):.3f}%")
    output.update(stability_case=np.array(stability_cases),
                  stability_axis=np.array(axes), stability_value=np.array(values),
                  stability_delta_ap_percent=np.array(deltas_ap),
                  stability_delta_at_percent=np.array(deltas_at))

    destination = SCRATCH / "calibrated_reconstruction_checks.npz"
    np.savez(destination, **output)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
