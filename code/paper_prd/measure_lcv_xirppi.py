#!/usr/bin/env python3
"""Build matched pre/post configuration-space LCV correlations.

This script transforms the same calibrated Model 1 fields used by
``lcv_reconstruction_check.py``.  It applies the established, ensemble-level
monopole LCV coefficient to individual Fourier modes and then bins the
periodic correlation into 4 Mpc/h bins of transverse and line-of-sight
separation.  The pre and post fields use their correct assignment responses:
the exact node-centred TSC stencil before reconstruction and the continuous
TSC window after reconstruction.

The output is compact.  No three-dimensional Fourier or correlation fields
are written to disk.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
from scipy.fft import irfftn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from lcv_reconstruction_check import field_paths  # noqa: E402
from lcv_shift_check import geometry, linear_fourier, target_fourier  # noqa: E402
from measure_xirppi_fields import NGRID, bin_rppi  # noqa: E402
from recon_paint_field import FOREST_RESPONSE  # noqa: E402
from template_pk import class_pk  # noqa: E402

LBOX = 2000.0
MODEL = 1
SUMMARY = HERE / "data" / "lcv_reconstruction_results.npz"


def expand_coefficient(q: np.ndarray, geo: dict) -> np.ndarray:
    """Expand a binned q(k,mu) coefficient onto the rFFT mesh."""
    out = np.zeros(geo["k"].size, dtype=np.float32)
    out[geo["ok"]] = np.asarray(q, dtype=np.float32).ravel()[geo["index"]]
    return out.reshape(geo["k"].shape)


def measure_one(phase: str, state: str, q: np.ndarray, geo: dict,
                density_theory: np.ndarray, growth: float,
                workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and LCV xi(r_perp,r_parallel) for one matched field."""
    b_f, beta_f = FOREST_RESPONSE[MODEL]
    response = np.asarray(b_f * (1.0 + beta_f * geo["mu"] ** 2),
                          dtype=np.float32)
    linear = linear_fourier(phase, growth, workers=workers) * response
    target = target_fourier(field_paths(MODEL, phase)[state], geo,
                            workers=workers)

    raw_power = np.asarray(np.abs(target) ** 2, dtype=np.float32)
    control_delta = np.asarray(
        np.abs(linear) ** 2 - density_theory * response ** 2 / LBOX ** 3,
        dtype=np.float32,
    )
    qmode = expand_coefficient(q, geo)
    cv_power = raw_power - qmode * control_delta
    raw_power[0, 0, 0] = 0.0
    cv_power[0, 0, 0] = 0.0
    del target, linear, response, control_delta, qmode
    gc.collect()

    raw_xi = irfftn(raw_power, s=(NGRID,) * 3, workers=workers,
                    overwrite_x=True)
    raw_xi *= np.float32(NGRID ** 3)
    del raw_power
    cv_xi = irfftn(cv_power, s=(NGRID,) * 3, workers=workers,
                   overwrite_x=True)
    cv_xi *= np.float32(NGRID ** 3)
    del cv_power
    gc.collect()

    edges, raw = bin_rppi(raw_xi)
    del raw_xi
    _, cv = bin_rppi(cv_xi)
    del cv_xi
    gc.collect()
    return edges, np.stack((raw, cv))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--phases", nargs="+",
                        default=[f"{i:03d}" for i in range(6)])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if not SUMMARY.exists():
        raise FileNotFoundError(SUMMARY)
    summary = np.load(SUMMARY)
    geo = geometry()

    from cosmoprimo.fiducial import AbacusSummit

    cosmo_fid = AbacusSummit()
    growth = (cosmo_fid.growth_factor(2.5)
              / cosmo_fid.growth_factor(99.0))
    cosmo = class_pk(z=2.5)
    density_theory = np.asarray(cosmo.plin(geo["k"], wiggle=True),
                                dtype=np.float32)
    density_theory[0, 0, 0] = 0.0

    # This is the closest 512^3 counterpart of the historical configuration-
    # space prescription, whose coefficient was derived from the monopole and
    # held fixed across realizations.  The stored array is repeated over phase;
    # use row zero explicitly to document that it is an ensemble coefficient.
    for phase in args.phases:
        ip = int(phase)
        for state in ("pre", "post"):
            destination = (HERE / "data" /
                           f"xi_lcv_{state}_Model1_ph{phase}.npz")
            if destination.exists() and not args.rebuild:
                print(f"exists {destination}", flush=True)
                continue
            key = f"m1_{state}_ensemble_monopole_coefficient"
            q = summary[key][0]
            print(f"phase {phase} {state}: transform", flush=True)
            edges, xi = measure_one(phase, state, q, geo, density_theory,
                                    growth, args.workers)
            np.savez(destination, xirppi_raw=xi[0], xirppi_lcv=xi[1],
                     rp_bins=edges, pi_bins=edges,
                     coefficient=np.array("ensemble_monopole"),
                     phase_index=np.array(ip),
                     tsc_window=np.array("node" if state == "pre"
                                         else "continuous"))
            print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
