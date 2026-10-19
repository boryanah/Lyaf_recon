#!/usr/bin/env python3
"""Fit the matched sparse pair-count spectra before and after reconstruction.

The central paired fit uses weights derived from each pre-reconstruction
spectrum in both states.  ``state_weights`` is the explicit alternative.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fit_bao_multipoles import (MultipoleFitter, MultipoleModel, gauss_var)  # noqa: E402
from template_pk import class_pk  # noqa: E402

ELLS = (0, 2)


def spectrum_path(root, state, phase, frac, z_stride):
    tag = f"matched_M1_f{frac}_{state}_ph{phase}_zs{z_stride}"
    return (Path(root) / state / "out" /
            f"{tag}_power_spectrum_n66_l4_R0200.txt")


def fit_one(power, weight_power, model, kbins, kmin, kmax, nbb, thin, sig_prior):
    kc = kbins.mean(axis=1)
    kk = np.tile(kc, len(ELLS))
    mask = (kk > kmin) & (kk < kmax)
    if thin > 1:
        keep = np.zeros_like(mask)
        for j in range(len(ELLS)):
            idx = np.arange(j * len(kc), (j + 1) * len(kc))
            keep[idx[::thin]] = True
        mask &= keep
    pl = power[:, :len(ELLS)]
    weight_pl = weight_power[:, :len(ELLS)]
    var = gauss_var(kbins, weight_pl, ELLS)
    fit = MultipoleFitter(model, pl.T.ravel()[mask],
                          (1.0 / var.T.ravel())[mask], mask,
                          sig_prior=sig_prior)
    return fit.fit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",
                        default="/pscratch/sd/b/boryanah/Lyaf_p3d/matched")
    parser.add_argument("--binfile", default=str(
        ROOT.parent / "power_3d_code" / "for_Boryana" / "P3D" /
        "k_files" / "k_binning_lin_0.01_0.50_dk_0.0075.txt"))
    parser.add_argument("--frac", type=float, default=0.155)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--z-stride", type=int, default=2)
    parser.add_argument("--phases", nargs="+",
                        default=[f"{i:03d}" for i in range(6)])
    parser.add_argument("--out", default=str(
        ROOT / "paper_prd" / "data" / "sparse_pair_matched_results.npz"))
    args = parser.parse_args()

    kbins = np.loadtxt(args.binfile)
    powers = {}
    for state in ("pre", "post"):
        powers[state] = []
        for phase in args.phases:
            path = spectrum_path(args.root, state, phase, args.frac,
                                 args.z_stride)
            if not path.exists():
                raise FileNotFoundError(path)
            powers[state].append(np.loadtxt(path))

    cosmo = class_pk(z=2.5)
    configs = [
        ("central", 0.031, 0.30, 3, 2, 2.0),
        ("kmax025", 0.031, 0.25, 3, 2, 2.0),
        ("kmax035", 0.031, 0.35, 3, 2, 2.0),
        ("nbb2", 0.031, 0.30, 2, 2, 2.0),
        ("nbb4", 0.031, 0.30, 4, 2, 2.0),
        ("thin1", 0.031, 0.30, 3, 1, 2.0),
        ("weak_sigma", 0.031, 0.30, 3, 2, 4.0),
        ("state_weights", 0.031, 0.30, 3, 2, 2.0),
    ]
    saved = {"phases": np.array(args.phases),
             "frac": np.array(args.frac), "seed": np.array(args.seed),
             "z_stride": np.array(args.z_stride),
             "source_root": np.array(str(Path(args.root).resolve())),
             "central_weights": np.array("pre-derived for both states")}
    for label, kmin, kmax, nbb, thin, sig_width in configs:
        model = MultipoleModel(cosmo, kbins, ELLS, R0=200.0, nbb=nbb)
        results = {}
        for state, centre in (("pre", (4.61, 2.81)),
                              ("post", (2.16, 1.36))):
            prior = (*centre, sig_width)
            use_state_weights = label == "state_weights"
            rows = [fit_one(power, power if use_state_weights
                            else powers["pre"][i], model, kbins, kmin,
                            kmax, nbb, thin, prior)
                    for i, power in enumerate(powers[state])]
            results[state] = rows
            for key in rows[0]:
                saved[f"{label}_{state}_{key}"] = np.array(
                    [row[key] for row in rows])
        print(label)
        for par in ("ap", "at"):
            pre = saved[f"{label}_pre_{par}"]
            post = saved[f"{label}_post_{par}"]
            delta = post - pre
            saved[f"{label}_delta_{par}"] = delta
            print(f"  d{par}={100*delta.mean():+.3f}+-"
                  f"{100*delta.std(ddof=1)/np.sqrt(len(delta)):.3f}%  "
                  f"pre={pre.mean():.5f} post={post.mean():.5f}")

    np.savez(args.out, **saved)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
