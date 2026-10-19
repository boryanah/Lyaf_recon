#!/usr/bin/env python3
"""Verify the exact Fourier response of the zero-displacement TSC paint.

The reference particles sit exactly on the 512^3 mesh nodes.  Consequently
their TSC window is the transform of the discrete [1/8, 3/4, 1/8] stencil,
not the sinc-cubed ensemble window for particles at generic subcell positions.
This check compares the painted reference field with the original band-limited
grid mode by mode.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfftn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from make_dF_grid import LBOX, load_dF_grid  # noqa: E402


def main():
    nmesh = 512
    sim = "AbacusSummit_base_c000_ph000"
    ref_path = Path("/pscratch/sd/b/boryanah/Lyaf_recon") / (
        "dF_ref_Model_1_LOSz_nmesh512_"
        f"{sim}_z2.500.npy"
    )

    original = load_dF_grid(sim, 1, "z", nmesh, verbose=True)
    painted = np.load(ref_path)
    fk_original = rfftn(original, workers=-1)
    fk_painted = rfftn(painted, workers=-1)
    del original, painted

    mx = np.fft.fftfreq(nmesh, d=1.0 / nmesh)
    mz = np.fft.rfftfreq(nmesh, d=1.0 / nmesh)
    wx = 0.75 + 0.25 * np.cos(2.0 * np.pi * mx / nmesh)
    wz = 0.75 + 0.25 * np.cos(2.0 * np.pi * mz / nmesh)
    window = (wx[:, None, None] * wx[None, :, None]
              * wz[None, None, :])

    kf = 2.0 * np.pi / LBOX
    kx = mx * kf
    kz = mz * kf
    kk = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2
                 + kz[None, None, :] ** 2)
    prediction = fk_original * window
    good = ((kk > 0.0) & (kk < 0.5)
            & (np.abs(fk_original) > 1.0e-7 * np.max(np.abs(fk_original))))

    ratio = fk_painted[good] / prediction[good]
    residual = fk_painted[good] - prediction[good]
    rms_relative = np.sqrt(np.mean(np.abs(residual) ** 2)
                           / np.mean(np.abs(prediction[good]) ** 2))
    phase_rms = np.sqrt(np.mean(np.angle(ratio) ** 2))
    print(f"modes: {good.sum():,}")
    print(f"median real ratio: {np.median(ratio.real):.9f}")
    print(f"median imaginary ratio: {np.median(ratio.imag):+.3e}")
    print(f"rms complex residual: {rms_relative:.3e}")
    print(f"rms phase error: {phase_rms:.3e} rad")


if __name__ == "__main__":
    main()
