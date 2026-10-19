#!/usr/bin/env python
"""Rebuild the paper's dense-field tables from the archived P(k, mu) files.

The script never reads a 512^3 field.  It uses the small, self-contained
band-power files in data/pkmu and applies the same fit as fit_bao_pkmu.py.
Profile-likelihood intervals reoptimize beta and both damping scales.
"""

import argparse
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import bin_mask, PkmuFitter, PkmuModel  # noqa: E402
from template_pk import class_pk                           # noqa: E402

DATA = HERE / "data" / "pkmu"
SCRATCH = Path("/pscratch/sd/b/boryanah/Lyaf_recon")
KMIN, KMAX, NMU, NBB = 0.02, 0.40, 5, 3


def phase_of(path):
    match = re.search(r"ph(\d{3})", path.name)
    if match is None:
        raise ValueError(f"phase tag missing from {path}")
    return match.group(1)


def files_for(model, state, los="z"):
    if state == "pre":
        root = SCRATCH
        pattern = (f"dF_ref_Model_{model}_LOS{los}_nmesh512_"
                   "AbacusSummit_base_c000_ph00?_z2.500."
                   "pkmu_dk0.005_kmax0.5_nmu5_winnode.npz")
    else:
        root = SCRATCH
        response_tag = {
            1: "bF-0.148066_bt0.957688",
            2: "bF-0.130939_bt0.989632",
            3: "bF-0.132453_bt2.074689",
            4: "bF-0.128523_bt2.395469",
        }[model]
        pattern = (f"dF_recon_Model_{model}_LOS{los}_nmesh512_R10.00_nr512_"
                   f"MG_{response_tag}_lyadrv_"
                   "AbacusSummit_base_c000_ph00?_z2.500."
                   "pkmu_dk0.005_kmax0.5_nmu5.npz")
    paths = sorted(root.glob(pattern))
    if len(paths) != 6:
        raise ValueError(f"expected six {state} files for Model {model}, "
                         f"found {len(paths)} with {pattern}")
    if [phase_of(p) for p in paths] != [f"{i:03d}" for i in range(6)]:
        raise ValueError(f"unexpected phase sequence for Model {model}: {paths}")
    return paths


def load(paths, expected_window):
    rows = []
    for path in paths:
        with np.load(path) as d:
            if "window" not in d.files or str(d["window"]) != expected_window:
                raise ValueError(f"{path} does not declare the expected "
                                 f"{expected_window} assignment window")
            if "compensated" not in d.files or not bool(d["compensated"]):
                raise ValueError(f"{path} is not a compensated spectrum")
            rows.append(tuple(np.array(d[key]) for key in ("k", "mu", "P", "N")))
    k0, mu0, p0, n0 = rows[0]
    for path, (k, mu, p, n) in zip(paths[1:], rows[1:]):
        if p.shape != p0.shape or not np.allclose(k, k0, equal_nan=True) \
                or not np.allclose(mu, mu0, equal_nan=True) \
                or not np.array_equal(n, n0):
            raise ValueError(f"incompatible binning in {path}")
    return rows


def fit_stack(paths, cosmo, expected_window):
    rows = load(paths, expected_window)
    k, mu, p0, n = rows[0]
    good = bin_mask(k, p0, n, KMIN, KMAX)
    iw = np.broadcast_to(np.arange(p0.shape[1])[None, :], p0.shape)[good]
    spectra = np.array([row[2][good] for row in rows])
    mean = spectra.mean(axis=0)
    var_one = 2.0 * mean ** 2 / np.maximum(n[good], 1.0)
    model = PkmuModel(cosmo, k[good], mu[good], iw, NMU, nbb=NBB)
    per = [PkmuFitter(model, p, 1.0 / var_one).fit(compute_errors=False)
           for p in spectra]
    stack = PkmuFitter(model, mean, len(spectra) / var_one).fit()
    return stack, per


def fit_paired_response(pre_paths, post_paths, cosmo):
    """Fit phase-matched before and after spectra with one common weight.

    Absolute stacked constraints use each state's own Gaussian mode-counting
    variance.  A reconstruction response is a difference, however, so using
    the pre-reconstruction variance for both sides prevents a smooth power
    change from changing the objective function itself.
    """
    pre_rows = load(pre_paths, "node")
    post_rows = load(post_paths, "continuous")
    k, mu, p0, count = pre_rows[0]
    for path, row in zip(post_paths, post_rows):
        if (row[2].shape != p0.shape
                or not np.allclose(row[0], k, equal_nan=True)
                or not np.allclose(row[1], mu, equal_nan=True)
                or not np.array_equal(row[3], count)):
            raise ValueError(f"pre/post binning mismatch in {path}")
    good = bin_mask(k, p0, count, KMIN, KMAX)
    wedge = np.broadcast_to(np.arange(p0.shape[1])[None, :], p0.shape)[good]
    model = PkmuModel(cosmo, k[good], mu[good], wedge, NMU, nbb=NBB)
    pre = np.array([row[2][good] for row in pre_rows])
    post = np.array([row[2][good] for row in post_rows])
    variance = 2.0 * pre.mean(axis=0) ** 2 / np.maximum(count[good], 1.0)
    result = {}
    for name, spectra in (("pre", pre), ("post", post)):
        result[name] = [
            PkmuFitter(model, spectrum, 1.0 / variance)
            .fit(compute_errors=False)
            for spectrum in spectra
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--los", choices=("y", "z"), default="z")
    parser.add_argument("--destination", default=None)
    args = parser.parse_args()

    cosmo = class_pk(z=2.5)
    output = {"rs_drag_mpc_h": cosmo.rs_drag,
              "kmin_h_mpc": KMIN, "kmax_h_mpc": KMAX,
              "nmu": NMU, "nbb": NBB, "los": np.array(args.los)}
    print(f"CLASS ruler: {cosmo.rs_drag:.4f} Mpc/h")
    for model in range(1, 5):
        state_paths = {}
        for state in ("pre", "post"):
            paths = files_for(model, state, args.los)
            state_paths[state] = paths
            expected_window = "node" if state == "pre" else "continuous"
            stack, per = fit_stack(paths, cosmo, expected_window)
            prefix = f"m{model}_{state}"
            for key, value in stack.items():
                if key != "bb":
                    output[f"{prefix}_{key}"] = value
            for key in ("ap", "at", "beta", "sig_par", "sig_per", "chi2"):
                output[f"{prefix}_{key}_per_phase"] = np.array(
                    [row[key] for row in per])
            # Figure 2 is normally run from figs_code/, so retain absolute
            # provenance rather than bare names that depend on the cwd.
            output[f"{prefix}_source"] = np.array([str(p) for p in paths])
            print(f"Model {model} {state:4s}: "
                  f"ap={stack['ap']:.6f} -{stack['sigma_ap_lo']:.6f}"
                  f"/+{stack['sigma_ap_hi']:.6f}, "
                  f"at={stack['at']:.6f} -{stack['sigma_at_lo']:.6f}"
                  f"/+{stack['sigma_at_hi']:.6f}, "
                  f"Sigma={stack['sig_par']:.2f}/{stack['sig_per']:.2f}")
        paired = fit_paired_response(state_paths["pre"], state_paths["post"],
                                     cosmo)
        for par in ("ap", "at"):
            delta = 100.0 * np.array([
                after[par] - before[par]
                for before, after in zip(paired["pre"], paired["post"])
            ])
            output[f"m{model}_paired_delta_{par}_percent"] = delta
            output[f"m{model}_paired_delta_{par}_mean_percent"] = delta.mean()
            output[f"m{model}_paired_delta_{par}_sem_percent"] = (
                delta.std(ddof=1) / np.sqrt(delta.size)
            )
        print(f"Model {model} paired response: "
              f"dapar={output[f'm{model}_paired_delta_ap_mean_percent']:+.3f}"
              f"+/-{output[f'm{model}_paired_delta_ap_sem_percent']:.3f}%  "
              f"daperp={output[f'm{model}_paired_delta_at_mean_percent']:+.3f}"
              f"+/-{output[f'm{model}_paired_delta_at_sem_percent']:.3f}%")
    destination = (Path(args.destination) if args.destination else
                   HERE / "data" / ("dense_results.npz" if args.los == "z"
                                     else f"dense_results_LOS{args.los}.npz"))
    np.savez(destination, **output)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
