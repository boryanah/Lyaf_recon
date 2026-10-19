#!/usr/bin/env python
"""Legacy linear-control-variate failure diagnostic.

The six Abacus initial-condition grids share phases with the Lyman-alpha
fields.  This script downsamples each initial density field in Fourier space,
constructs the linear forest field, and measures its auto power and cross
power with the pre/post reconstructed fields.  It then applies the Gaussian
optimal auto-power control variate

    P_cv = P_target - c (P_linear,realization - P_linear,theory),
    c = (P_target,linear / P_linear,realization)^2.

This script intentionally reads the old ``b_F=-0.192`` reconstructed fields
so the archived failure can be reproduced.  It is not a production estimator
and its output is excluded from the paper results; see ``BUGS_AND_ISSUES.md``.
Use ``calibrate_all_forest_response.py`` and
``check_calibrated_reconstruction.py`` for the active response checks.

Only one phase is held in memory at a time.  The peak allocation is set by a
1152^3 float32 grid and its complex64 rFFT and is expected to remain below
20 GiB.  Cached small band powers make repeated fits inexpensive.
"""

from pathlib import Path
import gc
import sys

import numpy as np
from scipy.fft import rfftn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from fit_bao_pkmu import PkmuFitter, PkmuModel, bin_mask  # noqa: E402
from make_dF_grid import _open_asdf                         # noqa: E402
from template_pk import class_pk                            # noqa: E402

LBOX = 2000.0
NIC, N = 1152, 512
DK, KMAX_MEAS, NMU = 0.005, 0.50, 5
KMIN, KMAX, NBB = 0.02, 0.40, 3
LEGACY_B_F, LEGACY_BETA_F = -0.192, 0.95
IC_ROOT = Path("/global/cfs/cdirs/desi/cosmosim/Abacus/ic")
FIELD_ROOT = Path("/pscratch/sd/b/boryanah/Lyaf_recon")


def geometry():
    kf = 2 * np.pi / LBOX
    kx = np.fft.fftfreq(N, d=1 / N) * kf
    kz = np.fft.rfftfreq(N, d=1 / N) * kf
    k2 = (kx[:, None, None]**2 + kx[None, :, None]**2
          + kz[None, None, :]**2)
    kval = np.sqrt(k2, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.abs(kz[None, None, :] / kval)
    mu[0, 0, 0] = 0
    k_edges = np.arange(0, KMAX_MEAS + DK, DK)
    mu_edges = np.linspace(0, 1, NMU + 1)
    ik = np.digitize(kval.ravel(), k_edges) - 1
    im = np.clip(np.digitize(mu.ravel(), mu_edges) - 1, 0, NMU - 1)
    ok = (ik >= 0) & (ik < len(k_edges) - 1)
    index = (ik[ok] * NMU + im[ok]).astype(np.int32)
    weight_z = np.full(len(kz), 2.0, dtype=np.float32)
    weight_z[0] = weight_z[-1] = 1.0
    weights = np.broadcast_to(weight_z[None, None, :], kval.shape).ravel()[ok]
    nbin = (len(k_edges) - 1) * NMU
    count = np.bincount(index, weights=weights, minlength=nbin)
    kmean = np.bincount(index, weights=weights * kval.ravel()[ok],
                        minlength=nbin) / np.maximum(count, 1)
    mumean = np.bincount(index, weights=weights * mu.ravel()[ok],
                         minlength=nbin) / np.maximum(count, 1)
    mu2mean = np.bincount(index, weights=weights * mu.ravel()[ok] ** 2,
                          minlength=nbin) / np.maximum(count, 1)

    kny = np.pi * N / LBOX
    wx = np.sinc((np.pi * kx / (2 * kny)) / np.pi)**3
    wz = np.sinc((np.pi * kz / (2 * kny)) / np.pi)**3
    window_continuous = (wx[:, None, None] * wx[None, :, None]
                         * wz[None, None, :]).astype(np.float32)
    mx = np.fft.fftfreq(N, d=1.0 / N)
    mz = np.fft.rfftfreq(N, d=1.0 / N)
    wx = 0.75 + 0.25 * np.cos(2.0 * np.pi * mx / N)
    wz = 0.75 + 0.25 * np.cos(2.0 * np.pi * mz / N)
    window_node = (wx[:, None, None] * wx[None, :, None]
                   * wz[None, None, :]).astype(np.float32)
    return {"k": kval, "mu": mu.astype(np.float32), "ok": ok,
            "index": index, "weights": weights, "count": count,
            "kmean": kmean.reshape(-1, NMU),
            "mumean": mumean.reshape(-1, NMU),
            "mu2mean": mu2mean.reshape(-1, NMU),
            "window_continuous": window_continuous,
            "window_node": window_node,
            "nbin": nbin}


def bin_values(values, geo):
    total = np.bincount(geo["index"],
                        weights=geo["weights"] * values.ravel()[geo["ok"]],
                        minlength=geo["nbin"])
    return (total / np.maximum(geo["count"], 1)).reshape(-1, NMU)


def linear_fourier(phase, growth, workers=-1, los="z"):
    """Return the phase-matched IC modes with the requested LOS last."""
    if los not in ("y", "z"):
        raise ValueError("los must be 'y' or 'z'")
    path = IC_ROOT / f"AbacusSummit_base_c000_ph{phase}" / "ic_dens_N1152.asdf"
    with _open_asdf(path) as af:
        density = np.array(af["data"]["density"], dtype=np.float32, copy=True)
    if los == "y":
        # Match the forest convention: physical (x,y,z) becomes (x,z,y), so
        # scipy's last, half-complex axis is always the physical line of sight.
        density = np.ascontiguousarray(np.transpose(density, (0, 2, 1)))
    print(f"  phase {phase} LOS{los}: loaded IC, rms={density.std():.5f}",
          flush=True)
    fourier = rfftn(density, workers=workers) / np.float32(NIC**3)
    del density
    gc.collect()
    ind = np.r_[np.arange(N // 2), np.arange(NIC - N // 2, NIC)]
    cropped = np.ascontiguousarray(fourier[ind][:, ind, :N // 2 + 1])
    del fourier
    gc.collect()
    cropped[N // 2, :, :] = 0
    cropped[:, N // 2, :] = 0
    cropped[:, :, N // 2] = 0
    cropped[0, 0, 0] = 0
    cropped *= np.float32(growth)
    return cropped


def target_fourier(path, geo, workers=-1):
    field = np.load(path, mmap_mode="r")
    if field.shape != (N, N, N):
        raise ValueError(f"unexpected field shape {field.shape} in {path}")
    fourier = rfftn(field, workers=workers) / np.float32(N**3)
    del field
    is_reference = path.name.startswith("dF_ref_") and "_jit_" not in path.stem
    window = geo["window_node"] if is_reference else geo["window_continuous"]
    fourier /= window
    return fourier


def phase_cache(phase, geo, theory, growth):
    destination = FIELD_ROOT / f"lcv_bandpowers_windowfix_ph{phase}.npz"
    if destination.exists():
        print(f"  phase {phase}: using {destination.name}", flush=True)
        return
    delta = linear_fourier(phase, growth)
    linear = delta * (LEGACY_B_F * (1 + LEGACY_BETA_F * geo["mu"]**2))
    del delta
    p_lin = bin_values((np.abs(linear)**2 * LBOX**3).astype(np.float64), geo)
    output = {"k": geo["kmean"], "mu": geo["mumean"],
              "N": geo["count"].reshape(-1, NMU), "linear": p_lin,
              "theory": theory}
    sim = f"AbacusSummit_base_c000_ph{phase}"
    names = {
        "pre": f"dF_ref_Model_1_LOSz_nmesh512_{sim}_z2.500.npy",
        "post": ("dF_recon_Model_1_LOSz_nmesh512_R10.00_nr512_MG_"
                 f"bF-0.192_lyadrv_{sim}_z2.500.npy"),
    }
    for state, name in names.items():
        path = FIELD_ROOT / name
        if not path.exists():
            raise FileNotFoundError(path)
        target = target_fourier(path, geo)
        output[state] = bin_values(
            (np.abs(target)**2 * LBOX**3).astype(np.float64), geo)
        output[f"cross_{state}"] = bin_values(
            (np.real(target * np.conj(linear)) * LBOX**3).astype(np.float64),
            geo)
        del target
        gc.collect()
    del linear
    np.savez(destination, **output)
    print(f"  phase {phase}: saved {destination.name}", flush=True)


def fit_spectra(spectra, k, mu, count, cosmo, variance):
    good = bin_mask(k, spectra[0], count, KMIN, KMAX)
    iw = np.broadcast_to(np.arange(NMU)[None, :], k.shape)[good]
    model = PkmuModel(cosmo, k[good], mu[good], iw, NMU, nbb=NBB)
    per = [PkmuFitter(model, row[good], 1 / variance[good])
           .fit(compute_errors=False) for row in spectra]
    mean = spectra.mean(axis=0)
    stack = PkmuFitter(model, mean[good], len(spectra) / variance[good]).fit()
    return per, stack


def summarize(cosmo):
    rows = [np.load(FIELD_ROOT / f"lcv_bandpowers_windowfix_ph{i:03d}.npz")
            for i in range(6)]
    k, mu, count = rows[0]["k"], rows[0]["mu"], rows[0]["N"]
    linear = np.array([row["linear"] for row in rows])
    theory = rows[0]["theory"]
    result = {"k": k, "mu": mu, "N": count, "theory": theory}
    fitted = {}
    for state in ("pre", "post"):
        target = np.array([row[state] for row in rows])
        cross = np.array([row[f"cross_{state}"] for row in rows])
        tbar, cbar, xbar = target.mean(0), linear.mean(0), cross.mean(0)
        coefficient = np.square(xbar / np.maximum(cbar, 1e-30))
        cv = target - coefficient * (linear - theory)
        var_raw = 2 * tbar**2 / np.maximum(count, 1)
        var_cv = (2 / np.maximum(count, 1)
                  * (tbar**2 + coefficient**2 * cbar**2
                     - 2 * coefficient * xbar**2))
        var_cv = np.maximum(var_cv, 1e-4 * var_raw)
        raw_per, raw_stack = fit_spectra(target, k, mu, count, cosmo, var_raw)
        cv_per, cv_stack = fit_spectra(cv, k, mu, count, cosmo, var_cv)
        fitted[(state, "raw")] = (raw_per, raw_stack)
        fitted[(state, "cv")] = (cv_per, cv_stack)
        result[f"{state}_raw"] = target
        result[f"{state}_cv"] = cv
        result[f"{state}_cross"] = cross
        result[f"{state}_coefficient"] = coefficient
        result[f"{state}_var_cv"] = var_cv
        print(f"  {state:4s} raw: ap={raw_stack['ap']:.6f} +/- "
              f"{raw_stack['sigma_ap']:.6f}, at={raw_stack['at']:.6f} +/- "
              f"{raw_stack['sigma_at']:.6f}")
        print(f"  {state:4s} LCV: ap={cv_stack['ap']:.6f} +/- "
              f"{cv_stack['sigma_ap']:.6f}, at={cv_stack['at']:.6f} +/- "
              f"{cv_stack['sigma_at']:.6f}")

    for kind in ("raw", "cv"):
        pre = fitted[("pre", kind)][0]
        post = fitted[("post", kind)][0]
        for par in ("ap", "at"):
            delta = np.array([b[par] - a[par] for a, b in zip(pre, post)])
            print(f"  {kind:3s} paired d{par}: {100*delta.mean():+.4f} +/- "
                  f"{100*delta.std(ddof=1)/np.sqrt(6):.4f} percent")
            result[f"{kind}_delta_{par}"] = delta
    destination = FIELD_ROOT / "lcv_shift_results_windowfix.npz"
    np.savez(destination, **result)
    print(f"saved {destination}")


def main():
    from cosmoprimo.fiducial import AbacusSummit

    cosmo_fid = AbacusSummit()
    growth = (cosmo_fid.growth_factor(2.5)
              / cosmo_fid.growth_factor(99.0))
    cosmo = class_pk(z=2.5)
    geo = geometry()
    theory_mode = (cosmo.plin(geo["k"], wiggle=True)
                   * (LEGACY_B_F
                      * (1 + LEGACY_BETA_F * geo["mu"]**2))**2)
    theory = bin_values(theory_mode.astype(np.float64), geo)
    del theory_mode
    print(f"growth from z=99 to z=2.5: {growth:.6f}")
    for phase in (f"{i:03d}" for i in range(6)):
        phase_cache(phase, geo, theory, growth)
    summarize(cosmo)


if __name__ == "__main__":
    main()
