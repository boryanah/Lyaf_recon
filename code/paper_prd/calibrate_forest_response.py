#!/usr/bin/env python3
"""Legacy Model 1 response cross-check.

Production calibration is performed by ``calibrate_all_forest_response.py``,
which reads the exact node-corrected reference fields for all four models.
This older diagnostic is retained only to reproduce the comparison with the
archived pre-window-fix control-variate band powers.

The phase-matched IC band powers were produced by ``lcv_shift_check.py``.
At low k their cross spectrum measures

    delta_F / delta_L = b_F (1 + beta_F mu^2)

without relying on an assumed halo bias.  This script fits that response in
each phase, compares it with the independent broadband amplitude in the BAO
fit, and reports the fidelity of the R=10 Mpc/h smoothed displacement proxy.
"""

from pathlib import Path
import argparse

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SAVE = Path("/pscratch/sd/b/boryanah/Lyaf_recon") / (
    "forest_response_calibration_legacy.npz"
)
B0, BETA0 = -0.192, 0.95


def response_fit(k, mu, count, linear, cross, kmin, kmax):
    q0 = B0 * (1.0 + BETA0 * mu**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        response = q0 * cross / linear
    good = ((k >= kmin) & (k < kmax) & (count > 0)
            & np.isfinite(response))
    design = np.column_stack((np.ones(good.sum()), mu[good]**2))
    root_weight = np.sqrt(count[good])
    coefficient = np.linalg.lstsq(
        design * root_weight[:, None], response[good] * root_weight,
        rcond=None,
    )[0]
    b_f, b_beta = coefficient
    return b_f, b_beta / b_f


def displacement_fidelity(k, mu, count, linear, target, cross, b_f, beta_f):
    q0 = B0 * (1.0 + BETA0 * mu**2)
    q = b_f * (1.0 + beta_f * mu**2)
    smooth2 = np.exp(-(10.0 * k)**2)
    result = {}
    for name, angular in (("parallel", mu**2),
                          ("transverse", 1.0 - mu**2),
                          ("all", np.ones_like(mu))):
        with np.errstate(divide="ignore", invalid="ignore"):
            base = count * smooth2 * angular / np.maximum(k, 1.0e-9)**2
            true = base * linear / q0**2
            estimate = base * target / q**2
            covariance = base * cross / (q0 * q)
        good = ((k > 0.0) & (k < 0.5) & (count > 0)
                & np.isfinite(true) & np.isfinite(estimate)
                & np.isfinite(covariance))
        tt, ee, et = true[good].sum(), estimate[good].sum(), covariance[good].sum()
        result[f"r_{name}"] = et / np.sqrt(tt * ee)
        result[f"transfer_{name}"] = et / tt
        result[f"rms_ratio_{name}"] = np.sqrt(ee / tt)
    return result


def main(args):
    paths = [HERE / "data" / f"lcv_bandpowers_ph{i:03d}.npz"
             for i in range(6)]
    rows = [np.load(path) for path in paths]
    k, mu, count = (np.array(rows[0][key]) for key in ("k", "mu", "N"))
    per_phase = np.array([
        response_fit(k, mu, count, row["linear"], row["cross_pre"],
                     args.kmin, args.kmax)
        for row in rows
    ])
    mean = per_phase.mean(axis=0)
    sem = per_phase.std(axis=0, ddof=1) / np.sqrt(len(per_phase))

    linear = np.mean([row["linear"] for row in rows], axis=0)
    target = np.mean([row["pre"] for row in rows], axis=0)
    cross = np.mean([row["cross_pre"] for row in rows], axis=0)
    fidelity = displacement_fidelity(k, mu, count, linear, target, cross,
                                     mean[0], mean[1])

    with np.load(HERE / "data" / "dense_results.npz") as dense:
        bao_b = -np.sqrt(float(dense["m1_pre_bsq"]))
        bao_beta = float(dense["m1_pre_beta"])

    print(f"IC cross response ({args.kmin:.3f} <= k < {args.kmax:.3f}):")
    print(f"  b_F    = {mean[0]:+.6f} +/- {sem[0]:.6f}")
    print(f"  beta_F = {mean[1]:.6f} +/- {sem[1]:.6f}")
    print("independent BAO broadband fit:")
    print(f"  b_F    = {bao_b:+.6f}")
    print(f"  beta_F = {bao_beta:.6f}")
    print(f"old/new displacement amplitude = {mean[0] / B0:.4f}")
    print("R=10 smoothed displacement proxy versus phase-matched IC:")
    for direction in ("parallel", "transverse", "all"):
        print(f"  {direction:10s}: r={fidelity[f'r_{direction}']:.5f}, "
              f"cross transfer={fidelity[f'transfer_{direction}']:.5f}, "
              f"rms ratio={fidelity[f'rms_ratio_{direction}']:.5f}")

    output = {
        "kmin": args.kmin, "kmax": args.kmax,
        "b_f_per_phase": per_phase[:, 0],
        "beta_f_per_phase": per_phase[:, 1],
        "b_f_mean": mean[0], "b_f_sem": sem[0],
        "beta_f_mean": mean[1], "beta_f_sem": sem[1],
        "bao_b_f": bao_b, "bao_beta_f": bao_beta,
        "old_b_f": B0, "old_beta_f": BETA0,
        **fidelity,
    }
    if args.save:
        np.savez(args.save, **output)
        print(f"saved {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmin", type=float, default=0.005)
    parser.add_argument("--kmax", type=float, default=0.05)
    parser.add_argument("--save", type=Path, default=DEFAULT_SAVE)
    main(parser.parse_args())
