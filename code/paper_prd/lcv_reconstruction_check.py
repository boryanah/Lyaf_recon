#!/usr/bin/env python3
"""Linear-control-variate test before and after reconstruction.

This is the production replacement for the exploratory ``lcv_shift_check.py``.
It uses calibrated forest responses and the active pre/post fields for all
four forest models.  For every phase it measures

    P_cv = P_F - c (P_L,phase - P_L,theory),

where ``delta_L^F = b_F (1 + beta_F mu^2) delta_L`` and the Gaussian optimal
coefficient is ``c=(P_FL/P_LL)^2``.  Following the established
AbacusLymanAlpha analysis, c is fixed to one on linear scales, smoothed in k,
and tapered at high k.  Ensemble and phase-by-phase coefficient estimates are
both reported.  Large Fourier fields are never written; small band-power
caches go to pscratch and the compact fit summary goes to ``paper_prd/data``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
# Historical builder: its cached `theory` uses a separate target-redshift
# spectrum rather than the expectation of the backscaled IC field. Preserve
# it for reproduction; adopted results use correct_lcv_mean.py followed by
# run_icmean_pk_refits.sh. See ESTIMATOR_VALIDATION_20260908.md.
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import PkmuFitter, PkmuModel, bin_mask  # noqa: E402
from lcv_shift_check import (FIELD_ROOT, NMU, bin_values, geometry,  # noqa: E402
                             linear_fourier, target_fourier)
from recon_paint_field import FOREST_RESPONSE, forest_response_tag  # noqa: E402
from template_pk import class_pk  # noqa: E402

LBOX = 2000.0
PHASES = tuple(f"{i:03d}" for i in range(6))
MODELS = (1, 2, 3, 4)
KMIN = 0.02


def field_paths(model: int, phase: str) -> dict[str, Path]:
    """Return the active, calibrated reference and reconstructed fields."""
    sim = f"AbacusSummit_base_c000_ph{phase}"
    b_f, beta_f = FOREST_RESPONSE[model]
    tag = forest_response_tag(b_f, beta_f)
    return {
        "pre": FIELD_ROOT /
        f"dF_ref_Model_{model}_LOSz_nmesh512_{sim}_z2.500.npy",
        "post": FIELD_ROOT /
        (f"dF_recon_Model_{model}_LOSz_nmesh512_R10.00_nr512_MG_"
         f"{tag}_lyadrv_{sim}_z2.500.npy"),
    }


def cache_path(phase: str) -> Path:
    return FIELD_ROOT / f"lcv_reconstruction_calibrated_ph{phase}.npz"


def cache_phase(phase: str, geo: dict, density_theory: np.ndarray,
                growth: float, workers: int, rebuild: bool = False) -> Path:
    """Measure target, linear-control, and cross band powers for one phase."""
    destination = cache_path(phase)
    if destination.exists() and not rebuild:
        print(f"phase {phase}: using {destination.name}", flush=True)
        return destination

    delta = linear_fourier(phase, growth, workers=workers)
    output = {
        "k": geo["kmean"],
        "mu": geo["mumean"],
        "N": geo["count"].reshape(-1, NMU),
        "growth_z99_to_z2p5": growth,
    }

    for model in MODELS:
        b_f, beta_f = FOREST_RESPONSE[model]
        response = np.asarray(
            b_f * (1.0 + beta_f * geo["mu"] ** 2), dtype=np.float32)
        linear = delta * response
        control = bin_values(
            (np.abs(linear) ** 2 * LBOX ** 3).astype(np.float64), geo)
        control_theory = bin_values(
            (density_theory * response ** 2).astype(np.float64), geo)
        output[f"m{model}_control"] = control
        output[f"m{model}_theory"] = control_theory

        for state, path in field_paths(model, phase).items():
            if not path.exists():
                raise FileNotFoundError(path)
            target = target_fourier(path, geo, workers=workers)
            output[f"m{model}_{state}"] = bin_values(
                (np.abs(target) ** 2 * LBOX ** 3).astype(np.float64), geo)
            output[f"m{model}_cross_{state}"] = bin_values(
                (np.real(target * np.conj(linear)) * LBOX ** 3)
                .astype(np.float64), geo)
            del target
            gc.collect()
        del linear, response
        gc.collect()
        print(f"phase {phase}: measured Model {model}", flush=True)

    del delta
    gc.collect()
    np.savez(destination, **output)
    print(f"phase {phase}: saved {destination}", flush=True)
    return destination


def smooth_coefficient(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Apply the smoothing and high-k taper of the established LCV method."""
    q = np.asarray(q, dtype=np.float64)
    out = q * 0.5 * (1.0 - np.tanh((k - 0.618) / 0.167))
    out[k < 0.05] = 1.0
    for wedge in range(out.shape[-1]):
        valid = np.isfinite(out[:, wedge])
        if valid.sum() < 7:
            raise ValueError("too few finite coefficient bins")
        filled = np.interp(np.arange(len(valid)), np.flatnonzero(valid),
                           out[valid, wedge])
        window = min(21, len(filled) if len(filled) % 2 else len(filled) - 1)
        out[:, wedge] = savgol_filter(filled, window, 3)
    return np.clip(out, 0.0, 2.0)


def coefficient(cross: np.ndarray, control: np.ndarray, k: np.ndarray,
                mode: str, count: np.ndarray) -> np.ndarray:
    """Return smoothed coefficients with shape (phase,k,mu)."""
    if mode.startswith("ensemble"):
        x = cross.mean(axis=0, keepdims=True)
        c = control.mean(axis=0, keepdims=True)
    elif mode.startswith("phase"):
        x, c = cross, control
    else:
        raise ValueError(mode)

    if mode.endswith("monopole"):
        weights = np.broadcast_to(count, x.shape)
        norm = np.maximum(np.sum(weights, axis=-1, keepdims=True), 1.0)
        x0 = np.sum(weights * x, axis=-1, keepdims=True) / norm
        c0 = np.sum(weights * c, axis=-1, keepdims=True) / norm
        raw = np.broadcast_to((x0 / np.maximum(c0, 1e-30)) ** 2,
                              x.shape)
    elif mode.endswith("wedges"):
        raw = (x / np.maximum(c, 1e-30)) ** 2
    else:
        raise ValueError(mode)

    smoothed = np.array([smooth_coefficient(row, k) for row in raw])
    return np.broadcast_to(smoothed, cross.shape).copy()


def fit_spectra(spectra: np.ndarray, variance: np.ndarray, k: np.ndarray,
                mu: np.ndarray, count: np.ndarray, cosmo, kmax: float,
                nbb: int) -> tuple[list[dict], dict]:
    good = bin_mask(k, spectra.mean(axis=0), count, KMIN, kmax)
    iw = np.broadcast_to(np.arange(NMU)[None, :], k.shape)[good]
    model = PkmuModel(cosmo, k[good], mu[good], iw, NMU, nbb=nbb)
    per = [PkmuFitter(model, row[good], 1.0 / variance[good])
           .fit(compute_errors=False) for row in spectra]
    # The user-requested diagnostic is the shift, not the precision gain after
    # applying a control variate.  Profile intervals are also very expensive
    # for the full robustness grid.  Phase scatter supplies the uncertainty on
    # the paired shift below, so central values are sufficient here.
    stack = PkmuFitter(model, spectra.mean(axis=0)[good],
                       len(spectra) / variance[good]).fit(compute_errors=False)
    return per, stack


def add_fit(output: dict, prefix: str, spectra: np.ndarray,
            variance: np.ndarray, k: np.ndarray, mu: np.ndarray,
            count: np.ndarray, cosmo, kmax: float, nbb: int):
    per, stack = fit_spectra(spectra, variance, k, mu, count, cosmo,
                             kmax=kmax, nbb=nbb)
    for par in ("ap", "at", "beta", "sig_par", "sig_per", "chi2"):
        output[f"{prefix}_{par}_phase"] = np.array([row[par] for row in per])
        output[f"{prefix}_{par}_stack"] = stack[par]
    return per, stack


def summarize(rows: list[np.lib.npyio.NpzFile], cosmo) -> Path:
    k, mu, count = rows[0]["k"], rows[0]["mu"], rows[0]["N"]
    output = {
        "k": k, "mu": mu, "N": count,
        "models": np.array(MODELS), "phases": np.array(PHASES),
        "primary_coefficient": np.array("ensemble_wedges"),
    }
    coefficient_modes = ("ensemble_wedges", "ensemble_monopole",
                         "phase_wedges", "phase_monopole")
    fit_grid = ((0.30, 3), (0.35, 3), (0.40, 2), (0.40, 3), (0.40, 4))

    for model in MODELS:
        control = np.array([row[f"m{model}_control"] for row in rows])
        theory = np.array([row[f"m{model}_theory"] for row in rows])
        fits = {}
        for state in ("pre", "post"):
            target = np.array([row[f"m{model}_{state}"] for row in rows])
            cross = np.array([row[f"m{model}_cross_{state}"] for row in rows])
            output[f"m{model}_{state}_raw_bandpower"] = target
            output[f"m{model}_{state}_cross_bandpower"] = cross
            output[f"m{model}_control_bandpower"] = control
            output[f"m{model}_theory_bandpower"] = theory

            tbar, cbar, xbar = target.mean(0), control.mean(0), cross.mean(0)
            raw_variance = 2.0 * tbar ** 2 / np.maximum(count, 1.0)
            for kmax, nbb in fit_grid:
                grid = f"kmax{kmax:.2f}_nbb{nbb}"
                prefix = f"m{model}_{state}_raw_{grid}"
                fits[(state, "raw", grid)] = add_fit(
                    output, prefix, target, raw_variance, k, mu, count,
                    cosmo, kmax, nbb)

            for cmode in coefficient_modes:
                q = coefficient(cross, control, k, cmode, count)
                cv = target - q * (control - theory)
                qbar = q.mean(axis=0)
                cv_variance = (2.0 / np.maximum(count, 1.0)
                               * (tbar ** 2 + qbar ** 2 * cbar ** 2
                                  - 2.0 * qbar * xbar ** 2))
                cv_variance = np.maximum(cv_variance, 1e-4 * raw_variance)
                output[f"m{model}_{state}_{cmode}_coefficient"] = q
                output[f"m{model}_{state}_{cmode}_bandpower"] = cv
                for kmax, nbb in fit_grid:
                    grid = f"kmax{kmax:.2f}_nbb{nbb}"
                    prefix = f"m{model}_{state}_{cmode}_{grid}"
                    fits[(state, cmode, grid)] = add_fit(
                        output, prefix, cv, cv_variance, k, mu, count,
                        cosmo, kmax, nbb)

        print(f"\nModel {model}", flush=True)
        for kmax, nbb in fit_grid:
            grid = f"kmax{kmax:.2f}_nbb{nbb}"
            for kind in ("raw",) + coefficient_modes:
                pre_key = f"m{model}_pre_{kind}_{grid}"
                post_key = f"m{model}_post_{kind}_{grid}"
                delta_at = (output[f"{post_key}_at_phase"]
                            - output[f"{pre_key}_at_phase"])
                delta_ap = (output[f"{post_key}_ap_phase"]
                            - output[f"{pre_key}_ap_phase"])
                output[f"m{model}_{kind}_{grid}_delta_at"] = delta_at
                output[f"m{model}_{kind}_{grid}_delta_ap"] = delta_ap
                at_pre = output[f"{pre_key}_at_stack"]
                at_post = output[f"{post_key}_at_stack"]
                print(f"  {grid} {kind:18s}: "
                      f"at {at_pre:.6f} -> {at_post:.6f}; "
                      f"paired d_at={100*delta_at.mean():+.4f} +/- "
                      f"{100*delta_at.std(ddof=1)/np.sqrt(len(delta_at)):.4f}%  "
                      f"d_ap={100*delta_ap.mean():+.4f} +/- "
                      f"{100*delta_ap.std(ddof=1)/np.sqrt(len(delta_ap)):.4f}%",
                      flush=True)

    destination = HERE / "data" / "lcv_reconstruction_results.npz"
    destination.parent.mkdir(exist_ok=True)
    np.savez(destination, **output)
    print(f"saved compact results -> {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    from cosmoprimo.fiducial import AbacusSummit

    cosmo_fid = AbacusSummit()
    growth = (cosmo_fid.growth_factor(2.5)
              / cosmo_fid.growth_factor(99.0))
    cosmo = class_pk(z=2.5)
    geo = geometry()
    density_theory = cosmo.plin(geo["k"], wiggle=True)
    density_theory[0, 0, 0] = 0.0
    paths = []
    for phase in PHASES:
        path = cache_path(phase)
        if args.fit_only:
            if not path.exists():
                raise FileNotFoundError(path)
        else:
            path = cache_phase(phase, geo, density_theory, growth,
                               args.workers, rebuild=args.rebuild)
        paths.append(path)
    del density_theory, geo
    gc.collect()
    rows = [np.load(path) for path in paths]
    summarize(rows, cosmo)


if __name__ == "__main__":
    main()
