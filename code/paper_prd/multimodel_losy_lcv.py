#!/usr/bin/env python3
"""Production-resolution LOSy LCV measurement for one forest model.

This generalizes the Model 1 counterpart of lcv_reconstruction_check.py. It keeps
LOS-y and LOS-z as separate views of the same six matter phases. The script
rotates both the shipped forest modes and the phase-matched initial density so
that the physical line of sight is always the last, half-complex FFT axis.
It never treats the two orientations as independent volumes.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import bin_mask, PkmuFitter, PkmuModel  # noqa: E402
from lcv_reconstruction_check import coefficient          # noqa: E402
from lcv_shift_check import (FIELD_ROOT, NMU, bin_values, geometry,  # noqa: E402
                             linear_fourier, target_fourier)
from recon_paint_field import FOREST_RESPONSE, forest_response_tag  # noqa: E402
from template_pk import class_pk                          # noqa: E402

LBOX = 2000.0
PHASES = tuple(f"{index:03d}" for index in range(6))
KMIN = 0.02
COEFFICIENT_MODES = ("ensemble_wedges", "ensemble_monopole",
                     "phase_wedges", "phase_monopole")
FIT_GRID = ((0.30, 3), (0.35, 3), (0.40, 2), (0.40, 3), (0.40, 4))


def field_paths(phase: str, los: str, model: int) -> dict[str, Path]:
    sim = f"AbacusSummit_base_c000_ph{phase}"
    b_f, beta_f = FOREST_RESPONSE[model]
    response = forest_response_tag(b_f, beta_f)
    return {
        "pre": FIELD_ROOT /
        f"dF_ref_Model_{model}_LOS{los}_nmesh512_{sim}_z2.500.npy",
        "post": FIELD_ROOT /
        (f"dF_recon_Model_{model}_LOS{los}_nmesh512_R10.00_nr512_MG_"
         f"{response}_lyadrv_{sim}_z2.500.npy"),
    }


def cache_path(phase: str, los: str, model: int) -> Path:
    return (FIELD_ROOT /
            f"lcv_reconstruction_calibrated_Model{model}_LOS{los}_ph{phase}.npz")


def cache_phase(phase: str, los: str, geo: dict,
                density_theory: np.ndarray, growth: float, model: int,
                workers: int, rebuild: bool = False) -> Path:
    destination = cache_path(phase, los, model)
    if destination.exists() and not rebuild:
        print(f"phase {phase} LOS{los}: using {destination.name}", flush=True)
        return destination

    delta = linear_fourier(phase, growth, workers=workers, los=los)
    b_f, beta_f = FOREST_RESPONSE[model]
    response = np.asarray(
        b_f * (1.0 + beta_f * geo["mu"] ** 2), dtype=np.float32)
    linear = delta * response
    del delta
    control = bin_values(
        (np.abs(linear) ** 2 * LBOX ** 3).astype(np.float64), geo)
    theory = bin_values(
        (density_theory * response ** 2).astype(np.float64), geo)
    output = {
        "k": geo["kmean"], "mu": geo["mumean"],
        "N": geo["count"].reshape(-1, NMU),
        "control": control, "theory": theory,
        "phase": np.array(phase), "los": np.array(los),
        "model": np.array(model),
        "growth_z99_to_z2p5": np.array(growth),
    }
    for state, source in field_paths(phase, los, model).items():
        if not source.exists():
            raise FileNotFoundError(source)
        target = target_fourier(source, geo, workers=workers)
        output[state] = bin_values(
            (np.abs(target) ** 2 * LBOX ** 3).astype(np.float64), geo)
        output[f"cross_{state}"] = bin_values(
            (np.real(target * np.conj(linear)) * LBOX ** 3)
            .astype(np.float64), geo)
        del target
        gc.collect()
    del linear, response
    np.savez(destination, **output)
    print(f"phase {phase} LOS{los}: saved {destination}", flush=True)
    return destination


def fit_spectra(spectra: np.ndarray, variance: np.ndarray, k: np.ndarray,
                mu: np.ndarray, count: np.ndarray, cosmo, kmax: float,
                nbb: int) -> tuple[list[dict], dict]:
    good = bin_mask(k, spectra.mean(axis=0), count, KMIN, kmax)
    iwedge = np.broadcast_to(np.arange(NMU)[None, :], k.shape)[good]
    model = PkmuModel(cosmo, k[good], mu[good], iwedge, NMU, nbb=nbb)
    per_phase = [
        PkmuFitter(model, row[good], 1.0 / variance[good])
        .fit(compute_errors=False)
        for row in spectra
    ]
    stack = PkmuFitter(
        model, spectra.mean(axis=0)[good], len(spectra) / variance[good]
    ).fit(compute_errors=False)
    return per_phase, stack


def add_fit(output: dict, prefix: str, spectra: np.ndarray,
            variance: np.ndarray, k: np.ndarray, mu: np.ndarray,
            count: np.ndarray, cosmo, kmax: float, nbb: int) -> None:
    per_phase, stack = fit_spectra(
        spectra, variance, k, mu, count, cosmo, kmax, nbb)
    for parameter in ("ap", "at", "beta", "sig_par", "sig_per", "chi2"):
        output[f"{prefix}_{parameter}_phase"] = np.array(
            [fit[parameter] for fit in per_phase])
        output[f"{prefix}_{parameter}_stack"] = stack[parameter]


def summarize(paths: list[Path], los: str, model: int, cosmo) -> Path:
    rows = [np.load(path) for path in paths]
    prefix = f"m{model}"
    k, mu, count = rows[0]["k"], rows[0]["mu"], rows[0]["N"]
    control = np.array([row["control"] for row in rows])
    theory = np.array([row["theory"] for row in rows])
    output = {
        "k": k, "mu": mu, "N": count,
        "models": np.array([model]), "phases": np.array(PHASES),
        "los": np.array(los),
        "primary_coefficient": np.array("ensemble_wedges"),
        f"{prefix}_control_bandpower": control,
        f"{prefix}_theory_bandpower": theory,
    }

    for state in ("pre", "post"):
        target = np.array([row[state] for row in rows])
        cross = np.array([row[f"cross_{state}"] for row in rows])
        output[f"{prefix}_{state}_raw_bandpower"] = target
        output[f"{prefix}_{state}_cross_bandpower"] = cross
        target_mean = target.mean(axis=0)
        control_mean = control.mean(axis=0)
        cross_mean = cross.mean(axis=0)
        raw_variance = 2.0 * target_mean ** 2 / np.maximum(count, 1.0)

        for kmax, nbb in FIT_GRID:
            grid = f"kmax{kmax:.2f}_nbb{nbb}"
            add_fit(output, f"{prefix}_{state}_raw_{grid}", target, raw_variance,
                    k, mu, count, cosmo, kmax, nbb)

        for mode in COEFFICIENT_MODES:
            q = coefficient(cross, control, k, mode, count)
            controlled = target - q * (control - theory)
            qmean = q.mean(axis=0)
            controlled_variance = (
                2.0 / np.maximum(count, 1.0)
                * (target_mean ** 2 + qmean ** 2 * control_mean ** 2
                   - 2.0 * qmean * cross_mean ** 2)
            )
            controlled_variance = np.maximum(
                controlled_variance, 1.e-4 * raw_variance)
            output[f"{prefix}_{state}_{mode}_coefficient"] = q
            output[f"{prefix}_{state}_{mode}_bandpower"] = controlled
            for kmax, nbb in FIT_GRID:
                grid = f"kmax{kmax:.2f}_nbb{nbb}"
                add_fit(output, f"{prefix}_{state}_{mode}_{grid}", controlled,
                        controlled_variance, k, mu, count, cosmo, kmax, nbb)

    print(f"\nLOS{los}", flush=True)
    for kmax, nbb in FIT_GRID:
        grid = f"kmax{kmax:.2f}_nbb{nbb}"
        for kind in ("raw",) + COEFFICIENT_MODES:
            pre = f"{prefix}_pre_{kind}_{grid}"
            post = f"{prefix}_post_{kind}_{grid}"
            for parameter in ("ap", "at"):
                delta = (output[f"{post}_{parameter}_phase"]
                         - output[f"{pre}_{parameter}_phase"])
                output[f"{prefix}_{kind}_{grid}_delta_{parameter}"] = delta
            delta_ap = output[f"{prefix}_{kind}_{grid}_delta_ap"]
            delta_at = output[f"{prefix}_{kind}_{grid}_delta_at"]
            print(
                f"  {grid} {kind:18s}: "
                f"dap={100*delta_ap.mean():+.4f} +/- "
                f"{100*delta_ap.std(ddof=1)/np.sqrt(6):.4f}%  "
                f"dat={100*delta_at.mean():+.4f} +/- "
                f"{100*delta_at.std(ddof=1)/np.sqrt(6):.4f}%",
                flush=True,
            )

    destination = (HERE / "data" /
                   f"lcv_reconstruction_results_Model{model}_LOS{los}.npz")
    np.savez(destination, **output)
    print(f"saved {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--los", choices=("y", "z"), required=True)
    parser.add_argument("--model", type=int, choices=tuple(FOREST_RESPONSE),
                        required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--central-only", action="store_true",
                        help="fit only kmax=0.40, nbb=3 with the primary LCV")
    args = parser.parse_args()

    if args.central_only:
        global FIT_GRID, COEFFICIENT_MODES
        FIT_GRID = ((0.40, 3),)
        COEFFICIENT_MODES = ("ensemble_wedges",)

    from cosmoprimo.fiducial import AbacusSummit

    fiducial = AbacusSummit()
    growth = (fiducial.growth_factor(2.5)
              / fiducial.growth_factor(99.0))
    cosmo = class_pk(z=2.5)
    geo = geometry()
    density_theory = cosmo.plin(geo["k"], wiggle=True)
    density_theory[0, 0, 0] = 0.0
    paths = []
    for phase in PHASES:
        destination = cache_path(phase, args.los, args.model)
        if args.fit_only:
            if not destination.exists():
                raise FileNotFoundError(destination)
        else:
            destination = cache_phase(
                phase, args.los, geo, density_theory, growth, args.model,
                args.workers, rebuild=args.rebuild)
        paths.append(destination)
    del density_theory, geo
    gc.collect()
    summarize(paths, args.los, args.model, cosmo)


if __name__ == "__main__":
    main()
