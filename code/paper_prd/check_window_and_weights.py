#!/usr/bin/env python3
"""Isolate assignment-window, field-definition, and weighting shifts.

The zero-displacement reference is painted from particles located exactly on
mesh nodes.  Its TSC response is the discrete node stencil, whereas displaced
particles have generic subcell positions and use the usual continuous TSC
window.  This script compares the legacy common sinc-cubed correction with an
exact node correction, a matched-jitter pair, and the standard RecSym-like
global-normalization field.  Each case is fit with state-specific and common
pre/post weights so a data-dependent weighting shift is visible separately.
"""

from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARCHIVE = HERE / "data" / "pkmu"
SCRATCH = Path("/pscratch/sd/b/boryanah/Lyaf_recon")
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import PkmuFitter, PkmuModel, bin_mask  # noqa: E402
from template_pk import class_pk  # noqa: E402


def phase(path):
    match = re.search(r"ph(\d{3})", path.name)
    if match is None:
        raise ValueError(f"phase missing from {path}")
    return match.group(1)


def six(root, pattern):
    paths = sorted(root.glob(pattern))
    expected = [f"{i:03d}" for i in range(6)]
    if len(paths) != 6 or [phase(path) for path in paths] != expected:
        raise ValueError(f"expected phases {expected} for {pattern}, got {paths}")
    return paths


def load(paths):
    rows = []
    for path in paths:
        with np.load(path) as data:
            rows.append(tuple(np.array(data[key]) for key in ("k", "mu", "P", "N")))
    k0, mu0, p0, n0 = rows[0]
    for path, (k, mu, p, n) in zip(paths[1:], rows[1:]):
        if (p.shape != p0.shape
                or not np.allclose(k, k0, equal_nan=True)
                or not np.allclose(mu, mu0, equal_nan=True)
                or not np.array_equal(n, n0)):
            raise ValueError(f"incompatible P(k,mu) grid in {path}")
    return rows


def fit_case(pre_paths, post_paths, cosmo, weight_mode):
    pre_rows, post_rows = load(pre_paths), load(post_paths)
    k, mu, p0, count = pre_rows[0]
    for path, row in zip(post_paths, post_rows):
        if (row[2].shape != p0.shape
                or not np.allclose(row[0], k, equal_nan=True)
                or not np.allclose(row[1], mu, equal_nan=True)
                or not np.array_equal(row[3], count)):
            raise ValueError(f"pre/post binning mismatch in {path}")
    good = bin_mask(k, p0, count, 0.02, 0.40)
    iw = np.broadcast_to(np.arange(p0.shape[1])[None, :], p0.shape)[good]
    model = PkmuModel(cosmo, k[good], mu[good], iw, p0.shape[1], nbb=3)
    pre = np.array([row[2][good] for row in pre_rows])
    post = np.array([row[2][good] for row in post_rows])
    nmode = np.maximum(count[good], 1.0)
    var_pre = 2.0 * pre.mean(0) ** 2 / nmode
    var_post = 2.0 * post.mean(0) ** 2 / nmode
    if weight_mode == "own":
        variances = (var_pre, var_post)
    elif weight_mode == "pre":
        variances = (var_pre, var_pre)
    elif weight_mode == "shared":
        common = 0.5 * (var_pre + var_post)
        variances = (common, common)
    else:
        raise ValueError(weight_mode)

    result = {}
    for state, spectra, variance in zip(("pre", "post"), (pre, post), variances):
        fits = [PkmuFitter(model, row, 1.0 / variance).fit(compute_errors=False)
                for row in spectra]
        result[state] = {
            "ap": np.array([fit["ap"] for fit in fits]),
            "at": np.array([fit["at"] for fit in fits]),
        }
        stack = PkmuFitter(model, spectra.mean(0), 6.0 / variance).fit(
            compute_errors=False
        )
        result[state]["stack"] = np.array([stack["ap"], stack["at"]])
    return result


def main():
    pre_legacy = six(
        ARCHIVE,
        "dF_ref_Model_1_LOSz_nmesh512_AbacusSummit_base_c000_ph00?_z2.500."
        "pkmu_dk0.005_kmax0.5_nmu5.npz",
    )
    post_local = six(
        ARCHIVE,
        "dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_bF-0.192_lyadrv_"
        "AbacusSummit_base_c000_ph00?_z2.500.pkmu_dk0.005_kmax0.5_nmu5.npz",
    )
    pre_node = six(
        SCRATCH,
        "dF_ref_Model_1_LOSz_nmesh512_AbacusSummit_base_c000_ph00?_z2.500."
        "pkmu_dk0.005_kmax0.5_nmu5_winnode.npz",
    )
    cases = {
        "legacy_window_local": (pre_legacy, post_local),
        "node_corrected_local": (pre_node, post_local),
        "matched_jitter_local": (
            six(SCRATCH, "dF_ref_Model_1_LOSz_nmesh512_jit_*ph00?*.pkmu_*.npz"),
            six(SCRATCH, "dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_"
                         "bF-0.192_lyadrv_jit_*ph00?*.pkmu_*.npz"),
        ),
    }
    global_pattern = ("dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_"
                      "bF-0.192_lyadrv_global_*ph00?*.pkmu_*.npz")
    if len(list(SCRATCH.glob(global_pattern))) == 6:
        cases["node_corrected_global"] = (
            pre_node, six(SCRATCH, global_pattern)
        )

    cosmo = class_pk(z=2.5)
    output = {"case": np.array(list(cases)),
              "weight_mode": np.array(("own", "pre", "shared"))}
    for name, paths in cases.items():
        for weight_mode in output["weight_mode"]:
            result = fit_case(*paths, cosmo, str(weight_mode))
            prefix = f"{name}_{weight_mode}"
            for par in ("ap", "at"):
                pre = result["pre"][par]
                post = result["post"][par]
                delta = post - pre
                output[f"{prefix}_pre_{par}"] = pre
                output[f"{prefix}_post_{par}"] = post
                output[f"{prefix}_delta_{par}"] = delta
            output[f"{prefix}_pre_stack"] = result["pre"]["stack"]
            output[f"{prefix}_post_stack"] = result["post"]["stack"]
            dap = 100.0 * output[f"{prefix}_delta_ap"]
            dat = 100.0 * output[f"{prefix}_delta_at"]
            print(f"{name:24s} {weight_mode:6s}: "
                  f"dapar={dap.mean():+.3f}+/-{dap.std(ddof=1)/np.sqrt(6):.3f}%  "
                  f"daperp={dat.mean():+.3f}+/-{dat.std(ddof=1)/np.sqrt(6):.3f}%  "
                  f"stack pre={result['pre']['stack'][0]:.6f}/"
                  f"{result['pre']['stack'][1]:.6f} post="
                  f"{result['post']['stack'][0]:.6f}/"
                  f"{result['post']['stack'][1]:.6f}", flush=True)

    # Home has a deliberately small quota on this machine.  Keep generated
    # diagnostic products with the other reconstruction intermediates.
    destination = SCRATCH / "window_weight_checks.npz"
    np.savez(destination, **output)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
