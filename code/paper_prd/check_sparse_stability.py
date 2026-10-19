"""Fit-only stability grid for the calibrated sparse self-reconstruction test.

The expensive 512^3 fields and their P(k, mu) measurements must already exist
in project scratch.  This script reads only the compact spectra and the exact
pre/post P_1D templates saved by ``sparse_skewers.py``.  It therefore tests fit
range, broadband order, and alias-template choices without reconstructing or
Fourier transforming another large field.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import measure, bin_mask, PkmuModel, PkmuFitter
from template_pk import class_pk


SCRATCH = Path("/pscratch/sd/b/boryanah/Lyaf_recon")
RESULT = SCRATCH / "sparse_skewers_Model_1_calibrated_f0.155_seed1234.npz"
OUTPUT = Path(__file__).parent / "data" / "sparse_stability.npz"
PRE_PRIOR = (4.61, 2.81, 1.0)
POST_PRIOR = (2.16, 1.36, 1.0)


def fields():
    pre = sorted(SCRATCH.glob(
        "dF_skref_f0.155_Model_1_LOSz_nmesh512_seed1234_"
        "AbacusSummit_base_c000_ph00?_z2.500.npy"))
    post = sorted(SCRATCH.glob(
        "dF_skrec_f0.155_Model_1_LOSz_nmesh512_R10.00_nr512_MG_"
        "bF-0.148066_bt0.957688_seed1234_"
        "AbacusSummit_base_c000_ph00?_z2.500.npy"))
    expected = [f"ph{i:03d}" for i in range(6)]
    for name, paths in (("pre", pre), ("post", post)):
        found = [next((part for part in path.stem.split("_")
                       if part.startswith("ph")), "") for path in paths]
        if found != expected:
            raise RuntimeError(f"{name} fields have phases {found}, expected {expected}")
    return pre, post


def spectra(paths):
    return [measure(path, 0.005, 0.5, 5, verbose=False) for path in paths]


def fit_stack(measured, cosmo, kmax, nbb, alias, prior):
    k, mu, _, counts = measured[0]
    good = bin_mask(k, measured[0][2], counts, 0.02, kmax)
    iw = np.broadcast_to(np.arange(k.shape[1])[None, :], k.shape)[good]
    powers = np.asarray([item[2][good] for item in measured])
    mean_power = powers.mean(axis=0)
    variance = 2.0 * mean_power ** 2 / np.maximum(counts[good], 1.0)
    extra = None
    if alias is not None:
        kpar, p1d = alias
        extra = np.interp(k[good] * mu[good], kpar, p1d)[:, None]
    model = PkmuModel(cosmo, k[good], mu[good], iw, 5, nbb=nbb,
                      extra_cols=extra)
    return PkmuFitter(model, mean_power, len(powers) / variance,
                      sig_prior=prior).fit()


def main():
    pre_paths, post_paths = fields()
    pre_spectra, post_spectra = spectra(pre_paths), spectra(post_paths)
    with np.load(RESULT) as data:
        alias_pre = (np.array(data["alias_kpar_pre_f0p155"]),
                     np.array(data["alias_p1d_pre_f0p155"]))
        alias_post = (np.array(data["alias_kpar_post_f0p155"]),
                      np.array(data["alias_p1d_post_f0p155"]))
    cosmo = class_pk(z=2.5)

    rows = []
    for nbb in (2, 3, 4):
        for kmax in (0.30, 0.35, 0.40):
            pre = fit_stack(pre_spectra, cosmo, kmax, nbb, alias_pre,
                            PRE_PRIOR)
            post = fit_stack(post_spectra, cosmo, kmax, nbb, alias_post,
                             POST_PRIOR)
            rows.append(("state_specific", nbb, kmax, pre, post))

    # At the central fit choice, isolate what was changed in the sparse alias
    # treatment.  ``reused_pre`` reproduces the old conceptual mistake; the
    # no-alias result shows how much broadband leakage the explicit template
    # is absorbing.
    for label, post_alias in (("reused_pre", alias_pre), ("no_alias", None)):
        pre_alias = alias_pre if label != "no_alias" else None
        pre = fit_stack(pre_spectra, cosmo, 0.35, 3, pre_alias, PRE_PRIOR)
        post = fit_stack(post_spectra, cosmo, 0.35, 3, post_alias, POST_PRIOR)
        rows.append((label, 3, 0.35, pre, post))

    keys = ("ap", "at", "sigma_ap", "sigma_at", "sig_par", "sig_per",
            "beta", "chi2", "ndof")
    output = {
        "alias_choice": np.array([row[0] for row in rows]),
        "nbb": np.array([row[1] for row in rows]),
        "kmax": np.array([row[2] for row in rows]),
        "phases": np.array([f"{i:03d}" for i in range(6)]),
        "source_result": np.array(str(RESULT)),
    }
    for state, index in (("pre", 3), ("post", 4)):
        for key in keys:
            output[f"{state}_{key}"] = np.array([row[index][key]
                                                  for row in rows])
    output["gain_ap"] = output["pre_sigma_ap"] / output["post_sigma_ap"]
    output["gain_at"] = output["pre_sigma_at"] / output["post_sigma_at"]
    output["delta_ap_percent"] = 100.0 * (output["post_ap"] - output["pre_ap"])
    output["delta_at_percent"] = 100.0 * (output["post_at"] - output["pre_at"])
    np.savez(OUTPUT, **output)

    print("alias             nbb kmax  pre(ap,at)       post(ap,at)      "
          "delta[%]       gain")
    for i, row in enumerate(rows):
        print(f"{row[0]:17s} {row[1]:3d} {row[2]:.2f}  "
              f"{output['pre_ap'][i]:.5f},{output['pre_at'][i]:.5f}  "
              f"{output['post_ap'][i]:.5f},{output['post_at'][i]:.5f}  "
              f"{output['delta_ap_percent'][i]:+.3f},"
              f"{output['delta_at_percent'][i]:+.3f}  "
              f"{output['gain_ap'][i]:.2f},{output['gain_at'][i]:.2f}")
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()
