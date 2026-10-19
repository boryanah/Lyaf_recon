#!/usr/bin/env python3
"""Measure the large-scale forest response directly from matched IC phases.

For every forest model and phase this measures

    <delta_F delta_L*> / <|delta_L|^2> = b_F (1 + beta_F mu^2)

at low wave number.  The initial grid is loaded only once per phase and the
small band-power cache is written to scratch, so interrupted runs can resume.
The reference TSC field is deconvolved with the exact node-centred window.
"""

from pathlib import Path
import gc
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from lcv_shift_check import (FIELD_ROOT, NMU, bin_values, geometry,  # noqa: E402
                             linear_fourier, target_fourier)

KMIN, KMAX = 0.005, 0.05
MODELS = (1, 2, 3, 4)


def fit_response(k, mu2, count, p_linear, cross):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = cross / p_linear
    good = ((k >= KMIN) & (k < KMAX) & (count > 0)
            & np.isfinite(ratio))
    # The response is linear in mu^2.  Use the mode-weighted <mu^2> in each
    # wedge, not <mu>^2.  Their difference is small, but it exceeds the
    # statistical error on b_F for these very large boxes.
    design = np.column_stack((np.ones(good.sum()), mu2[good]))
    root_weight = np.sqrt(count[good])
    b_f, b_beta = np.linalg.lstsq(
        design * root_weight[:, None], ratio[good] * root_weight,
        rcond=None,
    )[0]
    return b_f, b_beta / b_f


def displacement_fidelity(k, mu, count, p_linear, target, cross,
                          b_f, beta_f):
    """Fidelity of the R=10 smoothed displacement power and cross power."""
    response = b_f * (1.0 + beta_f * mu ** 2)
    smooth2 = np.exp(-(10.0 * k) ** 2)
    output = {}
    for name, angular in (("parallel", mu ** 2),
                          ("transverse", 1.0 - mu ** 2),
                          ("all", np.ones_like(mu))):
        with np.errstate(divide="ignore", invalid="ignore"):
            base = count * smooth2 * angular / np.maximum(k, 1.0e-9) ** 2
            truth = base * p_linear
            estimate = base * target / response ** 2
            covariance = base * cross / response
        good = ((k > 0.0) & (k < 0.5) & (count > 0)
                & np.isfinite(truth) & np.isfinite(estimate)
                & np.isfinite(covariance))
        tt = truth[good].sum()
        ee = estimate[good].sum()
        et = covariance[good].sum()
        output[f"r_{name}"] = et / np.sqrt(tt * ee)
        output[f"transfer_{name}"] = et / tt
        output[f"rms_ratio_{name}"] = np.sqrt(ee / tt)
    return output


def cache_phase(phase, geo, growth):
    destination = FIELD_ROOT / f"forest_response_direct_ph{phase}.npz"
    if destination.exists():
        print(f"  phase {phase}: using {destination.name}", flush=True)
        return destination

    delta = linear_fourier(phase, growth, workers=1)
    p_linear = bin_values(
        (np.abs(delta) ** 2 * 2000.0 ** 3).astype(np.float64), geo)
    output = {"k": geo["kmean"], "mu": geo["mumean"],
              "N": geo["count"].reshape(-1, NMU), "linear": p_linear}
    sim = f"AbacusSummit_base_c000_ph{phase}"
    for model in MODELS:
        path = FIELD_ROOT / (
            f"dF_ref_Model_{model}_LOSz_nmesh512_{sim}_z2.500.npy"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        target = target_fourier(path, geo, workers=1)
        output[f"m{model}_target"] = bin_values(
            (np.abs(target) ** 2 * 2000.0 ** 3).astype(np.float64), geo)
        output[f"m{model}_cross"] = bin_values(
            (np.real(target * np.conj(delta)) * 2000.0 ** 3)
            .astype(np.float64), geo)
        del target
        gc.collect()
    del delta
    np.savez(destination, **output)
    print(f"  phase {phase}: saved {destination.name}", flush=True)
    return destination


def main():
    from cosmoprimo.fiducial import AbacusSummit

    cosmology = AbacusSummit()
    growth = cosmology.growth_factor(2.5) / cosmology.growth_factor(99.0)
    geo = geometry()
    paths = [cache_phase(f"{phase:03d}", geo, growth)
             for phase in range(6)]
    rows = [np.load(path) for path in paths]
    k, count = (rows[0][key] for key in ("k", "N"))
    mu, mu2 = geo["mumean"], geo["mu2mean"]
    output = {"kmin": KMIN, "kmax": KMAX,
              "source": np.array([str(path) for path in paths])}
    print(f"response fit over {KMIN:.3f} <= k < {KMAX:.3f} h/Mpc")
    for model in MODELS:
        per_phase = np.array([
            fit_response(k, mu2, count, row["linear"], row[f"m{model}_cross"])
            for row in rows
        ])
        mean = per_phase.mean(axis=0)
        sem = per_phase.std(axis=0, ddof=1) / np.sqrt(per_phase.shape[0])
        output[f"m{model}_b_f_per_phase"] = per_phase[:, 0]
        output[f"m{model}_beta_f_per_phase"] = per_phase[:, 1]
        output[f"m{model}_b_f_mean"] = mean[0]
        output[f"m{model}_b_f_sem"] = sem[0]
        output[f"m{model}_beta_f_mean"] = mean[1]
        output[f"m{model}_beta_f_sem"] = sem[1]
        fidelity = displacement_fidelity(
            k, mu, count,
            np.mean([row["linear"] for row in rows], axis=0),
            np.mean([row[f"m{model}_target"] for row in rows], axis=0),
            np.mean([row[f"m{model}_cross"] for row in rows], axis=0),
            mean[0], mean[1],
        )
        for key, value in fidelity.items():
            output[f"m{model}_{key}"] = value
        print(f"  Model {model}: b_F={mean[0]:+.6f} +/- {sem[0]:.6f}, "
              f"beta_F={mean[1]:.6f} +/- {sem[1]:.6f}; "
              f"R=10 all r={fidelity['r_all']:.5f}, "
              f"T={fidelity['transfer_all']:.5f}, "
              f"rms={fidelity['rms_ratio_all']:.5f}")
    destination = FIELD_ROOT / "forest_response_all_models.npz"
    np.savez(destination, **output)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
