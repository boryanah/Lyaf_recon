#!/usr/bin/env python3
"""Like-for-like high-resolution pre/post LCV correlations for LOS-y or LOS-z.

The target grid has nmesh^3 cells (1152^3 in production), while the smoothed
reconstruction displacement may use the validated 512^3 production mesh.
Both target states are evaluated on the same Fourier support. The initial
density and the shipped LOS-y forest use the same (0, 2, 1) rotation, placing
the physical line of sight on the final rFFT axis.

The control coefficient follows the archived high-resolution analysis: it is
estimated separately for every phase, state, and orientation from the
monopole of the target-linear cross spectrum. Only compact 4 Mpc/h binned
correlations are written.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
from scipy.fft import irfftn, rfftn
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
# Historical control mean. Its raw fields remain the measured input, but
# adopted LCV results use matched_lcv_bins.py -> correct_lcv_mean.py ->
# run_icmean_xi_refits.sh. See ESTIMATOR_VALIDATION_20260908.md.
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from make_dF_grid import LBOX, _open_asdf, load_dF_grid  # noqa: E402
from recon_paint_field import (FOREST_RESPONSE, OUT_DIR,  # noqa: E402
                               forest_response_tag)
from template_pk import class_pk  # noqa: E402

IC_ROOT = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/ic")
NIC = 1152
MODEL = 1
DBIN = 4.0
RMAX = 200.0


def linear_modes(phase: str, los: str, nmesh: int, growth: float,
                 workers: int) -> np.ndarray:
    """Read, rotate, transform, and crop the phase-matched initial density."""
    sim = f"AbacusSummit_base_c000_ph{phase}"
    source = IC_ROOT / sim / "ic_dens_N1152.asdf"
    with _open_asdf(source) as af:
        density = np.array(
            af["data"]["density"], dtype=np.float32, copy=True)
    if los == "y":
        density = np.ascontiguousarray(density.transpose(0, 2, 1))
    fourier = rfftn(density, workers=workers) / np.float32(NIC ** 3)
    del density
    gc.collect()
    if nmesh != NIC:
        if nmesh > NIC or nmesh % 2:
            raise ValueError("nmesh must be even and no larger than 1152")
        index = np.r_[
            np.arange(nmesh // 2),
            np.arange(NIC - nmesh // 2, NIC),
        ]
        fourier = np.ascontiguousarray(
            fourier[index][:, index, :nmesh // 2 + 1])
    fourier[nmesh // 2, :, :] = 0
    fourier[:, nmesh // 2, :] = 0
    fourier[:, :, nmesh // 2] = 0
    fourier[0, 0, 0] = 0
    fourier *= np.float32(growth)
    return fourier


def post_path(phase: str, los: str, nmesh: int,
              nmesh_recon: int, field_root: Path) -> Path:
    sim = f"AbacusSummit_base_c000_ph{phase}"
    b_f, beta_f = FOREST_RESPONSE[MODEL]
    response = forest_response_tag(b_f, beta_f)
    return field_root / (
        f"dF_recon_Model_{MODEL}_LOS{los}_nmesh{nmesh}_R10.00_"
        f"nr{nmesh_recon}_MG_{response}_lyadrv_{sim}_z2.500.npy"
    )


def target_modes(phase: str, state: str, los: str, nmesh: int,
                 nmesh_recon: int, field_root: Path,
                 workers: int) -> np.ndarray:
    """Return normalized, assignment-corrected target Fourier modes."""
    if state == "pre":
        field = load_dF_grid(
            f"AbacusSummit_base_c000_ph{phase}", MODEL, los, nmesh,
            workers=workers)
    else:
        source = post_path(
            phase, los, nmesh, nmesh_recon, field_root)
        if not source.exists():
            raise FileNotFoundError(source)
        field = np.load(source, mmap_mode="r")
    fourier = rfftn(field, workers=workers) / np.float32(nmesh ** 3)
    del field
    gc.collect()
    if state == "post":
        modes = np.fft.fftfreq(nmesh, d=1.0 / nmesh)
        modes_half = np.fft.rfftfreq(nmesh, d=1.0 / nmesh)
        transverse = np.sinc(modes / nmesh).astype(np.float32) ** 3
        radial = np.sinc(modes_half / nmesh).astype(np.float32) ** 3
        fourier /= (
            transverse[:, None, None]
            * transverse[None, :, None]
            * radial[None, None, :]
        )
    fourier[0, 0, 0] = 0
    return fourier


class Geometry:
    """Small two-dimensional geometry reused while streaming rFFT planes."""

    def __init__(self, nmesh: int):
        self.nmesh = nmesh
        self.kf = 2.0 * np.pi / LBOX
        self.kfull = (
            np.fft.fftfreq(nmesh, d=1.0 / nmesh).astype(np.float32)
            * np.float32(self.kf)
        )
        self.khalf = (
            np.fft.rfftfreq(nmesh, d=1.0 / nmesh).astype(np.float32)
            * np.float32(self.kf)
        )
        self.kperp2 = (
            self.kfull[:, None] ** 2 + self.kfull[None, :] ** 2
        ).astype(np.float32)
        self.kny = np.pi * nmesh / LBOX
        self.edges = np.arange(
            0.0, self.kny + self.kf * 1.01, self.kf)
        self.centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.weight = np.full(len(self.khalf), 2.0)
        self.weight[0] = 1.0
        if nmesh % 2 == 0:
            self.weight[-1] = 1.0

    def plane(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        k2 = self.kperp2 + self.khalf[index] ** 2
        kval = np.sqrt(k2, dtype=np.float32)
        mu2 = np.zeros_like(kval)
        positive = k2 > 0
        mu2[positive] = self.khalf[index] ** 2 / k2[positive]
        return kval, mu2


def phase_coefficient(target: np.ndarray, delta: np.ndarray,
                      geometry: Geometry, b_f: float,
                      beta_f: float) -> np.ndarray:
    """Measure and smooth the archived phase-specific monopole coefficient."""
    nbin = len(geometry.centres)
    control = np.zeros(nbin)
    cross = np.zeros(nbin)
    count = np.zeros(nbin)
    for iz in range(len(geometry.khalf)):
        kval, mu2 = geometry.plane(iz)
        good = kval < geometry.kny
        index = np.floor(kval[good] / geometry.kf).astype(np.int32)
        response = np.float32(b_f) * (
            1.0 + np.float32(beta_f) * mu2[good])
        linear = delta[:, :, iz][good] * response
        weight = geometry.weight[iz]
        count += np.bincount(
            index, weights=np.full(index.size, weight), minlength=nbin)[:nbin]
        control += np.bincount(
            index, weights=weight * np.abs(linear) ** 2,
            minlength=nbin)[:nbin]
        cross += np.bincount(
            index,
            weights=weight * np.real(
                target[:, :, iz][good] * np.conjugate(linear)),
            minlength=nbin,
        )[:nbin]
    control /= np.maximum(count, 1.0)
    cross /= np.maximum(count, 1.0)
    raw = np.square(cross / np.maximum(control, 1.e-30))
    taper = 0.5 * (
        1.0 - np.tanh((geometry.centres - 0.618) / 0.167))
    coefficient = raw * taper
    coefficient[geometry.centres < 0.05] = 1.0
    window = min(21, len(coefficient) if len(coefficient) % 2
                 else len(coefficient) - 1)
    coefficient = savgol_filter(coefficient, window, 3)
    return np.clip(coefficient, 0.0, 2.0)


def correlation_from_power(power: np.ndarray, nmesh: int,
                           workers: int) -> np.ndarray:
    power[0, 0, 0] = 0
    xi = irfftn(
        power, s=(nmesh,) * 3, workers=workers, overwrite_x=True)
    xi *= np.float32(nmesh ** 3)
    return xi


def bin_rppi(xi: np.ndarray, nmesh: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(0.0, RMAX + DBIN, DBIN)
    nbin = len(edges) - 1
    index = np.arange(nmesh)
    coordinate = (
        np.minimum(index, nmesh - index).astype(np.float64)
        * LBOX / nmesh
    )
    rperp = np.sqrt(
        coordinate[:, None] ** 2 + coordinate[None, :] ** 2)
    transverse_bin = np.floor(rperp / DBIN).astype(np.int16)
    good = transverse_bin < nbin
    flat_bin = transverse_bin[good]
    transverse_count = np.bincount(
        flat_bin, minlength=nbin).astype(np.int64)
    total = np.zeros((nbin, nbin), dtype=np.float64)
    count = np.zeros((nbin, nbin), dtype=np.int64)
    for iz, separation in enumerate(coordinate):
        radial_bin = int(np.floor(separation / DBIN))
        if radial_bin >= nbin:
            continue
        total[:, radial_bin] += np.bincount(
            flat_bin, weights=xi[:, :, iz][good],
            minlength=nbin)[:nbin]
        count[:, radial_bin] += transverse_count
    if np.any(count == 0):
        raise RuntimeError("empty configuration-space bin")
    return edges, total / count


def controlled_power(target: np.ndarray, delta: np.ndarray,
                     coefficient: np.ndarray, geometry: Geometry,
                     cosmo, b_f: float, beta_f: float) -> np.ndarray:
    output = np.empty(target.shape, dtype=np.float32)
    for iz in range(len(geometry.khalf)):
        kval, mu2 = geometry.plane(iz)
        response = np.float32(b_f) * (
            1.0 + np.float32(beta_f) * mu2)
        linear = delta[:, :, iz] * response
        q = np.interp(
            kval, geometry.centres, coefficient,
            left=coefficient[0], right=0.0).astype(np.float32)
        theory = np.zeros_like(kval)
        active = (q > 1.e-5) & (kval > 0.0)
        theory[active] = (
            cosmo.plin(kval[active], wiggle=True)
            * response[active] ** 2 / LBOX ** 3
        )
        output[:, :, iz] = (
            np.abs(target[:, :, iz]) ** 2
            - q * (np.abs(linear) ** 2 - theory)
        )
    return output


def measure_state(phase: str, state: str, los: str, nmesh: int,
                  nmesh_recon: int, field_root: Path,
                  delta: np.ndarray, geometry: Geometry, cosmo,
                  workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_f, beta_f = FOREST_RESPONSE[MODEL]
    target = target_modes(
        phase, state, los, nmesh, nmesh_recon, field_root, workers)
    coefficient = phase_coefficient(
        target, delta, geometry, b_f, beta_f)
    raw_power = np.asarray(np.abs(target) ** 2, dtype=np.float32)
    raw_xi = correlation_from_power(raw_power, nmesh, workers)
    edges, raw = bin_rppi(raw_xi, nmesh)
    del raw_xi
    gc.collect()
    cv_power = controlled_power(
        target, delta, coefficient, geometry, cosmo, b_f, beta_f)
    del target
    gc.collect()
    cv_xi = correlation_from_power(cv_power, nmesh, workers)
    _, controlled = bin_rppi(cv_xi, nmesh)
    del cv_xi
    gc.collect()
    return edges, np.stack((raw, controlled)), coefficient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--los", choices=("y", "z"), required=True)
    parser.add_argument("--nmesh", type=int, default=1152)
    parser.add_argument("--nmesh-recon", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--field-root",
        default="/pscratch/sd/b/boryanah/Lyaf_recon/two_orientation_1152")
    parser.add_argument(
        "--output-root",
        default="/pscratch/sd/b/boryanah/Lyaf_recon/two_orientation_1152/xi")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    phase = f"{int(args.phase):03d}"
    destination = (
        Path(args.output_root)
        / f"xi_highres_Model1_LOS{args.los}_ph{phase}.npz")
    if destination.exists() and not args.rebuild:
        print(f"exists {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    from cosmoprimo.fiducial import AbacusSummit

    fiducial = AbacusSummit()
    growth = (
        fiducial.growth_factor(2.5) / fiducial.growth_factor(99.0))
    cosmo = class_pk(z=2.5)
    geometry = Geometry(args.nmesh)
    print(f"phase {phase} LOS{args.los}: loading oriented IC", flush=True)
    delta = linear_modes(
        phase, args.los, args.nmesh, growth, args.workers)
    saved = {
        "phase": np.array(phase), "los": np.array(args.los),
        "nmesh": np.array(args.nmesh),
        "nmesh_recon": np.array(args.nmesh_recon),
        "coefficient_estimator": np.array("phase_monopole"),
        "growth_z99_to_z2p5": np.array(growth),
    }
    for state in ("pre", "post"):
        print(f"phase {phase} LOS{args.los}: {state}", flush=True)
        edges, xi, coefficient = measure_state(
            phase, state, args.los, args.nmesh, args.nmesh_recon,
            Path(args.field_root), delta, geometry, cosmo, args.workers)
        saved[f"{state}_raw"] = xi[0]
        saved[f"{state}_lcv"] = xi[1]
        saved[f"{state}_coefficient"] = coefficient
    saved["rp_bins"] = edges
    saved["pi_bins"] = edges
    np.savez(destination, **saved)
    print(f"saved {destination}", flush=True)


if __name__ == "__main__":
    main()
