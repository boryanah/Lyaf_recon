#!/usr/bin/env python3
"""Combine matched sparse pair-count fits without double-counting phases.

The two sightline masks are applied to the same six simulation phases.  We
therefore average the mask responses within each phase and estimate the
standard error from the six phase averages, rather than treating the twelve
mask-phase measurements as independent realizations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CONFIGS = ("central", "kmax025", "kmax035", "nbb2", "nbb4",
           "thin1", "weak_sigma", "state_weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=2, type=Path,
                        help="compact fit files for the two mask seeds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    masks = [np.load(path) for path in args.inputs]
    phases = masks[0]["phases"]
    if not np.array_equal(phases, masks[1]["phases"]):
        raise ValueError("mask files do not contain the same ordered phases")
    for key in ("frac", "z_stride"):
        if not np.array_equal(masks[0][key], masks[1][key]):
            raise ValueError(f"mask files disagree on {key}")

    saved = {
        "phases": phases,
        "frac": masks[0]["frac"],
        "z_stride": masks[0]["z_stride"],
        "input_files": np.array([str(path.resolve()) for path in args.inputs]),
        "seeds": np.array([int(mask["seed"]) for mask in masks]),
        "combination": np.array("mask average within phase; SEM over phases"),
    }
    nphase = len(phases)
    for config in CONFIGS:
        for par in ("ap", "at"):
            values = np.stack([mask[f"{config}_delta_{par}"]
                               for mask in masks])
            mean_by_phase = values.mean(axis=0)
            difference_by_phase = values[1] - values[0]
            saved[f"{config}_mask_delta_{par}"] = values
            saved[f"{config}_combined_delta_{par}"] = mean_by_phase
            saved[f"{config}_mask_difference_{par}"] = difference_by_phase
            saved[f"{config}_mean_delta_{par}"] = mean_by_phase.mean()
            saved[f"{config}_sem_delta_{par}"] = (
                mean_by_phase.std(ddof=1) / np.sqrt(nphase))
            saved[f"{config}_mean_mask_difference_{par}"] = (
                difference_by_phase.mean())
            saved[f"{config}_sem_mask_difference_{par}"] = (
                difference_by_phase.std(ddof=1) / np.sqrt(nphase))
            for state in ("pre", "post"):
                state_values = np.stack(
                    [mask[f"{config}_{state}_{par}"] for mask in masks])
                combined_state = state_values.mean(axis=0)
                saved[f"{config}_mask_{state}_{par}"] = state_values
                saved[f"{config}_combined_{state}_{par}"] = combined_state
                saved[f"{config}_mean_{state}_{par}"] = combined_state.mean()
                saved[f"{config}_sem_{state}_{par}"] = (
                    combined_state.std(ddof=1) / np.sqrt(nphase))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **saved)
    for par in ("ap", "at"):
        print(
            f"central d{par}="
            f"{100 * saved[f'central_mean_delta_{par}']:+.3f}+-"
            f"{100 * saved[f'central_sem_delta_{par}']:.3f}%  "
            f"mask2-mask1="
            f"{100 * saved[f'central_mean_mask_difference_{par}']:+.3f}+-"
            f"{100 * saved[f'central_sem_mask_difference_{par}']:.3f}%")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
