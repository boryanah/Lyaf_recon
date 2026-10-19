#!/usr/bin/env python3
"""Fit the matched configuration-space raw and LCV correlations with Vega.

The model, covariance weighting, coordinates, and 12-term broadband reproduce
the historical configuration-space analysis in the parent ``Lyaf_shift``
directory.  Only the input correlations change: this script reads the
window-corrected, phase-matched pre/post products made by
``measure_lcv_xirppi.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from vega import VegaInterface
from vega.coordinates import RtRpCoordinates
from vega.minimizer import Minimizer

HERE = Path(__file__).resolve().parent
LYAF_SHIFT = HERE.parents[3]
DATA = HERE / "data"
OUTPUT = DATA / "xi_lcv_vega_results.npz"


def load_fields() -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for state in ("pre", "post"):
        files = sorted(DATA.glob(f"xi_lcv_{state}_Model1_ph???.npz"))
        if len(files) != 6:
            raise RuntimeError(f"expected six {state} files, found {len(files)}")
        rows = [np.load(path) for path in files]
        out[(state, "raw")] = np.array([row["xirppi_raw"] for row in rows])
        out[(state, "lcv")] = np.array([row["xirppi_lcv"] for row in rows])
    return out


def setup_vega(small_scale_nl="dnl_arinyo", coordinate_nmesh=0):
    # The historical data product named in lyaxlya_bb.ini has been removed
    # from pscratch.  Vega reads that file only to initialize its data object;
    # this script replaces the measured vector immediately below.  Point a
    # temporary copy of the unchanged model configuration at the surviving
    # eBOSS correlation instead of requiring the obsolete placeholder.
    os.chdir(LYAF_SHIFT)
    eboss = Path("/global/cfs/projectdirs/desi/science/lya/eboss_dr16/"
                 "public_dr16_correlations/cf_z_0_10-exp.fits")
    with TemporaryDirectory(prefix="lyaf_vega_") as temporary:
        temporary = Path(temporary)
        child = temporary / "lyaxlya_bb.ini"
        child_text = (HERE / "configs" / "lyaxlya_bb.ini").read_text()
        child_lines = [f"filename = {eboss}" if line.startswith("filename =")
                       else line for line in child_text.splitlines()]
        if small_scale_nl == "none":
            child_lines = [line for line in child_lines
                           if not line.startswith("small scale nl =")]
        elif small_scale_nl != "dnl_arinyo":
            raise ValueError("small_scale_nl must be dnl_arinyo or none")
        child.write_text("\n".join(child_lines) + "\n")

        main = temporary / "main_bb.ini"
        main_text = (HERE / "configs" / "main_bb.ini").read_text()
        main_text = main_text.replace("ini files = configs/lyaxlya_bb.ini",
                                      f"ini files = {child}")
        main.write_text(main_text)
        vega = VegaInterface(str(main))
    edges = np.linspace(0.0, 200.0, 51)
    centres = 0.5 * (edges[1:] + edges[:-1])
    extra = {}
    if coordinate_nmesh:
        radial, transverse = lattice_bin_means(coordinate_nmesh)
        extra = dict(rp_grid=np.repeat(radial, 50),
                     rt_grid=np.tile(transverse, 50))
    coords = RtRpCoordinates(rp_min=0.0, rp_max=200.0, rt_max=200.0,
                             rp_nbins=50, rt_nbins=50, z_eff=2.5, **extra)
    vega.corr_items["lyaxlya"].init_coordinates(coords)
    rp = np.tile(centres, (50, 1)).T
    rt = rp.T
    radius = np.sqrt(rp ** 2 + rt ** 2)
    return vega, rp, rt, radius


def lattice_bin_means(nmesh):
    """Mean lattice separations in the actual cylindrical correlation bins.

    This provides a leading bin-centre sensitivity test. It is not a full
    average of the nonlinear model over every lattice pair in each bin.
    """
    index = np.arange(nmesh)
    x = np.minimum(index, nmesh-index) * (2000.0/nmesh)
    def means(values):
        take = values < 200.0
        ibin = np.floor(values[take]/4.0).astype(int)
        return (np.bincount(ibin, weights=values[take], minlength=50)
                / np.bincount(ibin, minlength=50))
    return means(x), means(np.hypot(x[:, None], x[None, :]).ravel())


def covariance_and_mask(rp, rt, radius, rmin, rmax):
    mask = ((rp.ravel() > 0.0) & (rp.ravel() < 200.0)
            & (rt.ravel() > 0.0) & (rt.ravel() < 200.0)
            & (radius.ravel() > rmin) & (radius.ravel() < rmax))
    path = Path("/global/cfs/projectdirs/desi/science/lya/eboss_dr16/"
                "public_dr16_correlations/cf_z_0_10-exp.fits")
    with fits.open(path) as hdul:
        cov = np.array(hdul["COR"].data["CO"], dtype=np.float64) / 20.0
        drp = np.tile(2.0 + 4.0 * np.arange(50), (50, 1)).T.ravel()
        drt = np.tile(2.0 + 4.0 * np.arange(50), (50, 1)).ravel()
    dr = np.sqrt(drp ** 2 + drt ** 2)
    dmask = ((drp > 0.0) & (drp < 200.0) & (drt > 0.0) & (drt < 200.0)
             & (dr > rmin) & (dr < rmax))
    if not np.array_equal(mask, dmask):
        raise RuntimeError("simulation and covariance masks do not match")
    cov = cov[np.ix_(dmask, dmask)]
    return mask, np.linalg.inv(cov)


def broadband_columns(rp, rt, mask):
    """The archived additive r^p L_ell(mu) basis, with scaled columns."""
    radius = np.sqrt(rp ** 2 + rt ** 2).ravel()[mask]
    mu = rp.ravel()[mask] / radius
    legendre = {
        0: np.ones_like(mu),
        2: 0.5 * (3.0 * mu ** 2 - 1.0),
        4: (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0,
        6: (231.0 * mu ** 6 - 315.0 * mu ** 4
            + 105.0 * mu ** 2 - 5.0) / 16.0,
    }
    return np.column_stack([radius ** power * legendre[ell]
                            for power in (-2, -1, 0)
                            for ell in (0, 2, 4, 6)])


def run(args):
    # Resolve before setup_vega changes the working directory.
    args.output = str(Path(args.output).resolve())
    fields = load_fields()
    vega, rp, rt, radius = setup_vega(
        small_scale_nl=getattr(args, "small_scale_nl", "dnl_arinyo"),
        coordinate_nmesh=getattr(args, "coordinate_nmesh", 0))
    initialized = False
    saved = {"phases": np.array([f"{i:03d}" for i in range(6)]),
             "central_rmin": np.array(args.central_rmin),
             "rmax": np.array(args.rmax),
             "broadband": np.array("additive r^[-2,-1,0] L_[0,2,4,6]"),
             "small_scale_nl": np.array(getattr(args, "small_scale_nl", "dnl_arinyo")),
             "coordinate_nmesh": np.array(getattr(args, "coordinate_nmesh", 0))}
    for config in ("main_bb.ini", "lyaxlya_bb.ini"):
        saved[f"config_sha256_{config}"] = np.array(hashlib.sha256(
            (HERE / "configs" / config).read_bytes()).hexdigest())

    physical = ("ap", "at", "bias_LYA", "beta_LYA", "sigmaNL_par",
                "sigmaNL_per")
    sample = {group: {name: value for name, value in values.items()
                      if name in physical}
              for group, values in vega.sample_params.items()}

    def one(xi, mask, icov, broadband):
        nonlocal initialized
        data = np.asarray(xi, dtype=np.float64).T.ravel()[mask]
        # Profile the 12 additive broadband coefficients analytically.  This
        # is the same likelihood as the archived 18-parameter Minuit fit, but
        # avoids asking Minuit to explore a linear subspace at every call.
        icov_b = icov @ broadband
        scale = np.sqrt(np.diag(broadband.T @ icov_b))
        basis = broadband / scale[None, :]
        icov_b = icov @ basis
        normal = basis.T @ icov_b

        def chisq(params):
            nonlocal initialized
            full = vega.params.copy()
            full.update(params)
            model = vega.compute_model(full, run_init=not initialized)["lyaxlya"]
            initialized = True
            diff = model[mask] - data
            rhs = basis.T @ (icov @ diff)
            coefficients = np.linalg.solve(normal, rhs)
            residual = diff - basis @ coefficients
            return float(residual @ icov @ residual)

        minimizer = Minimizer(chisq, sample)
        minimizer.minimize()
        values = minimizer.values
        errors = minimizer.errors
        if not minimizer.fmin.is_valid:
            raise RuntimeError(f"Vega fit did not converge: {minimizer.fmin}")
        return {name: float(values[name]) for name in physical} | {
                    "sigma_ap": float(errors["ap"]),
                    "sigma_at": float(errors["at"]),
                    "chi2": float(chisq(values)),
                    "valid_minimum": bool(minimizer.fmin.is_valid),
                    "parameters_at_limit": bool(minimizer.fmin.has_parameters_at_limit),
                    "edm": float(minimizer.fmin.edm),
                }

    for rmin in args.rmin:
        mask, icov = covariance_and_mask(rp, rt, radius, rmin, args.rmax)
        broadband = broadband_columns(rp, rt, mask)
        for (state, kind), xis in fields.items():
            tag = f"{state}_{kind}_rmin{rmin:g}"
            result = one(xis.mean(axis=0), mask, icov, broadband)
            for key, value in result.items():
                saved[f"{tag}_{key}_stack"] = value
            print(f"{tag:22s} ap={result['ap']:.6f}+-{result['sigma_ap']:.6f} "
                  f"at={result['at']:.6f}+-{result['sigma_at']:.6f}",
                  flush=True)
            if np.isclose(rmin, args.central_rmin):
                phase_results = [one(xi, mask, icov, broadband) for xi in xis]
                for key in phase_results[0]:
                    saved[f"{tag}_{key}_phase"] = np.array(
                        [item[key] for item in phase_results])

    c = f"rmin{args.central_rmin:g}"
    for kind in ("raw", "lcv"):
        for par in ("ap", "at"):
            pre = saved[f"pre_{kind}_{c}_{par}_phase"]
            post = saved[f"post_{kind}_{c}_{par}_phase"]
            delta = post - pre
            saved[f"{kind}_delta_{par}"] = delta
            print(f"paired {kind:3s} d{par}: {100*delta.mean():+.4f}+-"
                  f"{100*delta.std(ddof=1)/np.sqrt(len(delta)):.4f}%",
                  flush=True)
    np.savez(args.output, **saved)
    print(f"saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmin", nargs="+", type=float,
                        default=[20.0, 30.0, 40.0, 50.0])
    parser.add_argument("--central-rmin", type=float, default=30.0)
    parser.add_argument("--rmax", type=float, default=180.0)
    parser.add_argument("--output", default=str(OUTPUT),
                        help="destination; defaults to the paper data product")
    run(parser.parse_args())
