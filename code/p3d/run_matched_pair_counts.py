#!/usr/bin/env python3
"""Run resumable matched sparse pre/post pair counts in pscratch.

The driver retains every second line-of-sight pixel.  This halves the catalog
size while leaving the central fit range (k <= 0.30 h/Mpc) below the resulting
0.402 h/Mpc line-of-sight Nyquist frequency.  The same pixels, flux values,
mask, pair exclusion, and radial window enter the pre and post measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P3D = ROOT.parent / "power_3d_code" / "for_Boryana" / "P3D"
DEFAULT_SCRATCH = Path("/pscratch/sd/b/boryanah/Lyaf_p3d/matched")


def run_one(a, state, phase):
    tag = f"matched_M1_f{a.frac}_{state}_ph{phase}_zs{a.z_stride}"
    state_root = a.root / state
    catalog = state_root / f"{tag}.txt"
    outdir = state_root / "out"
    result = outdir / f"{tag}_power_spectrum_n66_l4_R0{a.R0:g}.txt"
    outdir.mkdir(parents=True, exist_ok=True)

    if result.exists():
        print(f"exists {result}", flush=True)
    else:
        if not catalog.exists():
            subprocess.run([
                sys.executable, str(HERE / "skewers_to_p3d.py"),
                "--phase", phase, "--state", state,
                "--frac", str(a.frac), "--seed", str(a.seed),
                "--z-stride", str(a.z_stride), "--ncpu", str(a.ncpu),
                "--out", str(catalog),
            ], check=True, cwd=ROOT)
        subprocess.run([
            str(P3D / "power_periodic"),
            "-in", str(catalog),
            "-binfile", str(P3D / "k_files" /
                            "k_binning_lin_0.01_0.50_dk_0.0075.txt"),
            "-output", str(outdir), "-out_string", tag,
            "-max_l", "4", "-R0", str(a.R0),
            "-nthread", str(a.nthread), "-nbin_RR", "1000",
            "-max_l_RR", "8", "-fast", "1", "-perbox",
        ], check=True, cwd=ROOT)
        if not result.exists():
            raise RuntimeError(f"pair counter did not create {result}")

    if catalog.exists() and not a.keep_catalogs:
        catalog.unlink()
        print(f"removed temporary catalog {catalog}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--phases", nargs="+",
                        default=[f"{i:03d}" for i in range(6)])
    parser.add_argument("--states", nargs="+", choices=("pre", "post"),
                        default=["pre", "post"])
    parser.add_argument("--frac", type=float, default=0.155)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--z-stride", type=int, default=2)
    parser.add_argument("--R0", type=float, default=200.0)
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--nthread", type=int, default=32)
    parser.add_argument("--keep-catalogs", action="store_true")
    a = parser.parse_args()
    for phase in a.phases:
        phase = f"{int(phase):03d}"
        for state in a.states:
            run_one(a, state, phase)


if __name__ == "__main__":
    main()
