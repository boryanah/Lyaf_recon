"""
make_dF_grid.py
---------------
Build a real-space delta_F grid directly from the *shipped* Fourier-space field
`complex_dF_Model_{m}_LOS{d}.asdf`, bypassing the tau chunks entirely.

Why
---
The public AbacusLymanAlpha v1 data already ships rfftn(delta_F) of the full
6912^3 grid, low-pass filtered to kperp < 2, klos < 4 h/Mpc, and stored as
real+imag float32.  LOSz has shape (1273, 1273, 1274), while LOSy has shape
(1273, 1274, 1273) because its half-complex axis is the physical y direction.
Everything downstream of that (the pre-recon xi in data_fft/) is built from
these products.  To *displace* the field for
post-reconstruction we need it back in real space -- but only band-limited to
whatever grid we intend to paint onto.  So: take the low-k block, inverse-FFT
onto an nmesh^3 grid, done.  No tau files, no downsample.py, no 1.3 TB of I/O.

Conventions (matched to get_corr_fft.py)
----------------------------------------
  * stored array is the UNNORMALIZED rfftn (get_corr_fft.py divides by Ndim^3)
  * LOSz is stored as (kx, ky, kz_half)
  * LOSy is stored as (kx, ky_half, kz)
  * for LOSy we crop in that native layout and transpose (0, 2, 1), exactly as
    in the original get_corr_fft.py and control-variate scripts; the returned
    array is therefore always (kx, ktransverse, klos_half)
  * DC mode is zeroed (get_corr_fft.py does `raw_p3d[0,0,0] = 0.`)

Normalization
-------------
numpy's irfftn divides by the size of the *output* grid, so recovering
delta_F(x) = (1/Ndim^3) sum_k F(k) e^{ikx} on an nmesh^3 grid requires an extra
factor (nmesh/Ndim)^3.

Taper
-----
The shipped field has a sharp low-pass cut; re-cutting it at the coarse Nyquist
would ring in real space.  A tanh taper (same functional form get_corr_fft.py
uses) is applied by default over the top 10% of the retained band.  Pre- and
post-recon fields must use the SAME taper so the window cancels in the
comparison.

Usage
-----
  python make_dF_grid.py --sim_name AbacusSummit_base_c000_ph000 \
                         --model 1 --los z --nmesh 512
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import asdf
from scipy.fft import irfftn

NDIM_FULL = 6912          # full grid the shipped rfftn was taken from
LBOX = 2000.0             # Mpc/h

LYA_DIR = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/")


def _open_asdf(path):
    """Open lazily with both the current and DESI-environment ASDF APIs."""
    try:
        return asdf.open(path, lazy_load=True, memmap=True)
    except TypeError as error:
        if "memmap" not in str(error):
            raise
        return asdf.open(path, lazy_load=True)


def _taper(k, k_cut, frac=0.1):
    """tanh roll-off over the top `frac` of the band (get_corr_fft.py form)."""
    kp = (1.0 - frac) * k_cut
    kw = frac * k_cut / 2.0
    return 0.5 * (1.0 - np.tanh((np.abs(k) - kp) / kw))


def load_dF_grid(sim_name, model, los, nmesh, redshift=2.5, taper=True,
                 lya_dir=LYA_DIR, verbose=True, workers=1):
    """
    Return real-space delta_F on an (nmesh, nmesh, nmesh) float32 grid, with
    the line of sight on axis 2.

    Only modes with |kx|, |ky|, kz <= (nmesh//2 - 1) are kept; the Nyquist
    plane is left at zero to sidestep Hermitian-symmetry bookkeeping.
    """
    if los not in ("y", "z"):
        raise ValueError("los must be 'y' or 'z'")
    if nmesh < 4 or nmesh % 2:
        raise ValueError("nmesh must be an even integer of at least 4")
    K = nmesh // 2 - 1                       # highest retained mode index
    kf = 2.0 * np.pi / LBOX
    k_nyq_coarse = K * kf
    if verbose:
        print(f"  nmesh={nmesh}  cell={LBOX/nmesh:.3f} Mpc/h  "
              f"keeping |k| <= {k_nyq_coarse:.4f} h/Mpc ({K} modes)")

    fn = (lya_dir / sim_name / f"z{redshift:.3f}" / f"Model_{model:d}"
          / f"complex_dF_Model_{model:d}_LOS{los}.asdf")
    if verbose:
        print(f"  reading {fn.name}")

    # Full FFT axes are fftfreq-ordered, so the |k|<=K block is two
    # contiguous runs:
    # positive: indices 0 .. K; negative: indices -K .. -1.
    lo = slice(0, K + 1)
    hi = slice(-K, None)                     # last K entries

    with _open_asdf(fn) as af:
        re = af["data"]["ComplexDeltaFluxReal"]
        im = af["data"]["ComplexDeltaFluxImag"]
        expected_half_axis = 2 if los == "z" else 1
        full_axes = (0, 1) if los == "z" else (0, 2)
        shape = tuple(re.shape)
        layout_ok = (
            re.ndim == 3
            and tuple(im.shape) == shape
            and shape[expected_half_axis] >= K + 1
            and all(shape[axis] >= 2 * K + 1 for axis in full_axes)
        )
        if not layout_ok:
            raise ValueError(
                f"unexpected LOS{los} Fourier layout real={shape}, "
                f"imag={tuple(im.shape)}; expected half-complex axis "
                f"{expected_half_axis} with at least {K + 1} entries and "
                f"full axes {full_axes} with at least {2 * K + 1} entries"
            )
        # Validate source support before allocating what can be a multi-GB
        # target array. This makes an accidentally oversized nmesh fail safely.
        out = np.zeros(
            (nmesh, nmesh, nmesh // 2 + 1), dtype=np.complex64)
        for s0, d0 in ((lo, lo), (hi, slice(nmesh - K, nmesh))):
            for st, dt in ((lo, lo), (hi, slice(nmesh - K, nmesh))):
                if los == "z":
                    blk = (np.asarray(re[s0, st, : K + 1], dtype=np.float32)
                           + 1j * np.asarray(im[s0, st, : K + 1],
                                             dtype=np.float32))
                    out[d0, dt, : K + 1] = blk
                else:
                    # Native LOSy order is (kx, ky_half, kz). Move the
                    # half-complex LOS axis last after cropping both full
                    # axes. This is the legacy transpose (0, 2, 1), applied
                    # before any rFFT interpretation.
                    blk = (np.asarray(re[s0, : K + 1, st], dtype=np.float32)
                           + 1j * np.asarray(im[s0, : K + 1, st],
                                             dtype=np.float32))
                    out[d0, dt, : K + 1] = np.transpose(blk, (0, 2, 1))
                del blk
        gc.collect()

    out[0, 0, 0] = 0.0                       # zero the DC mode

    if taper:
        kx = np.fft.fftfreq(nmesh, d=1.0 / nmesh)[:, None, None] * kf
        ky = np.fft.fftfreq(nmesh, d=1.0 / nmesh)[None, :, None] * kf
        kz = np.fft.rfftfreq(nmesh, d=1.0 / nmesh)[None, None, :] * kf
        out *= (_taper(kx, k_nyq_coarse) * _taper(ky, k_nyq_coarse)
                * _taper(kz, k_nyq_coarse)).astype(np.float32)
        del kx, ky, kz; gc.collect()

    # irfftn divides by nmesh^3; we want the 1/NDIM_FULL^3 of the original FFT
    field = irfftn(out, s=(nmesh, nmesh, nmesh), workers=int(workers))
    del out; gc.collect()
    field = (field * (float(nmesh) / NDIM_FULL) ** 3).astype(np.float32)

    if verbose:
        print(f"  delta_F grid: mean={field.mean():.3e}  std={field.std():.5f}  "
              f"min={field.min():.4f}  max={field.max():.4f}")
    return field


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim_name", default="AbacusSummit_base_c000_ph000")
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--los", default="z", choices=["y", "z"],
                   help="line of sight in the shipped half-complex field")
    p.add_argument("--nmesh", type=int, default=512)
    p.add_argument("--redshift", type=float, default=2.5)
    p.add_argument("--ncpu", type=int, default=1)
    p.add_argument("--no-taper", action="store_true")
    p.add_argument("--out", default=None, help="path to save the .npy grid")
    a = p.parse_args()

    g = load_dF_grid(a.sim_name, a.model, a.los, a.nmesh,
                     redshift=a.redshift, taper=not a.no_taper,
                     workers=a.ncpu)
    if a.out:
        np.save(a.out, g)
        print(f"  saved -> {a.out}")
