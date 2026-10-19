"""
scenario2_external_recon.py
---------------------------
External reconstruction evaluated at skewers (Regime II of the paper).

A discrete tracer sample (halos, optionally thinned to LAE-like density, or a
real LAE mock via --halo-file) provides the density field for the Zel'dovich
solve. The resulting displacement field is evaluated at the skewer cell
positions via pyrecon read_shifts, the RecSym shift is built with the TRUE
growth rate (Psi + f*(Psi.z)zhat, f=0.9702), and the sparse forest field is
advected + TSC-repainted through the same path as the zero-displacement
reference (mask-weighted estimator, global-mean normalisation).

Only ph000 has a halo catalogue. Single-phase paired pre/post gains only.
Random thinning of halos preserves the bias (same population), so
bias=2.695 stays correct for the thinned runs.

Usage:
  python scenario2_external_recon.py --frac 0.155 --thin-n 1e-3
  # full suite: sbatch run_scenario2.slurm
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from cosmoprimo.fiducial import AbacusSummit

sys.path.insert(0, str(Path(__file__).parent))
from recon_paint_field import LBOX, HALO_DIR, load_dF_grid
from sparse_skewers import make_mask, frac_to_qso_per_deg2, measure_p1d
from fit_bao_pkmu import bin_mask, PkmuModel, PkmuFitter
from measure_pkmu import pkmu
from template_pk import class_pk

OUT_DIR = Path("/pscratch/sd/b/boryanah/Lyaf_recon")
REDSHIFT = 2.5
# dense-field damping values, used as Gaussian priors (width 1 Mpc/h) exactly
# as in the production sparse fits (sparse_skewers.py)
SIG_PRIOR_PRE = (4.61, 2.81, 1.0)
SIG_PRIOR_POST = (2.16, 1.36, 1.0)


def build_recon_external(full_sim, sr, bias, thin_n, seed, nmesh, ncpu=32,
                         halo_file=None, verbose=True):
    """pyrecon MultiGrid solve from the (optionally thinned) halo catalog.
    Mirrors recon_paint_field.build_recon (periodic branch, no randoms)."""
    from pyrecon import MultiGridReconstruction

    cosmo = AbacusSummit()
    ff = cosmo.growth_rate(REDSHIFT)

    path = Path(halo_file) if halo_file else \
        HALO_DIR / f"halos_{full_sim}_z{REDSHIFT:.3f}.npz"
    f = np.load(path)
    pos = np.asarray(f["x_L2com"], dtype=np.float64) + LBOX / 2.0
    vel = np.asarray(f["v_L2com"], dtype=np.float64)
    vz2mpc = float(f["VelZSpace_to_kms"]) / LBOX   # km/s per Mpc/h
    del f; gc.collect()

    if thin_n is not None:
        n_keep = max(1, int(round(thin_n * LBOX ** 3)))
        if n_keep > len(pos):
            raise ValueError(f"thin_n={thin_n} exceeds catalog density")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pos), size=n_keep, replace=False)
        pos, vel = pos[idx], vel[idx]
    n_tracer = len(pos) / LBOX ** 3

    pos[:, 2] += vel[:, 2] / vz2mpc          # RSD along z (per-Mpc/h factor!)
    pos %= LBOX
    del vel; gc.collect()
    if verbose:
        print(f"  tracer: {len(pos):,}  (n = {n_tracer:.3e} (Mpc/h)^-3, "
              f"b = {bias})")

    recon = MultiGridReconstruction(
        f=ff, bias=bias, nmesh=nmesh, los="z", positions=pos, boxsize=LBOX,
        boxcenter=(LBOX / 2,) * 3, nthreads=int(ncpu), fft_engine="fftw",
        fft_plan="estimate", dtype="f4", wrap=True)
    recon.assign_data(pos)
    del pos; gc.collect()
    recon.set_density_contrast(smoothing_radius=sr)
    recon.run()
    if verbose:
        print("  external recon done", flush=True)
    return recon, ff, n_tracer


def paint_sparse(grid, mask, nmesh, recon=None, rsd_f=None, n_slab=32,
                 verbose=True):
    """Mask-weighted advect-and-repaint. recon=None -> zero-displacement
    reference through the identical path (window symmetry)."""
    from pmesh.pm import ParticleMesh

    cell = LBOX / nmesh
    pm = ParticleMesh(BoxSize=LBOX, Nmesh=[nmesh] * 3, dtype="f4")
    m_w = pm.create(type="real"); m_w[...] = 0.0
    m_n = pm.create(type="real"); m_n[...] = 0.0
    ax = np.arange(nmesh, dtype=np.float32) * cell

    for i0 in range(0, nmesh, n_slab):
        i1 = min(i0 + n_slab, nmesh)
        gx, gy, gz = np.meshgrid(ax[i0:i1], ax, ax, indexing="ij")
        pos = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)
        del gx, gy, gz
        w = np.ascontiguousarray(grid[i0:i1].ravel())
        if mask is not None:
            keep = np.repeat(mask[i0:i1].ravel(), nmesh)
            pos, w = pos[keep], w[keep]
        if recon is not None:
            # shifts at the SKEWER CELL positions; RecSym with the true f
            d = recon.read_shifts(pos, field="disp")
            d[:, 2] *= (1.0 + rsd_f)
            pos = (pos - d) % LBOX
            del d
        pm.paint(pos, mass=w, resampler="tsc", out=m_w, hold=True)
        pm.paint(pos, mass=None, resampler="tsc", out=m_n, hold=True)
        del pos, w; gc.collect()

    dens_w = np.asarray(m_w)
    dens_n = np.asarray(m_n)
    if mask is None:                      # dense: local normalisation
        dF = dens_w / np.maximum(dens_n, 1e-6)
        dF[dens_n <= 0] = 0.0
    else:                                 # sparse: GLOBAL mean occupancy
        dF = dens_w / dens_n.mean()
    dF -= dF.mean()
    del m_w, m_n, dens_w, dens_n; gc.collect()
    return np.ascontiguousarray(dF, dtype=np.float32)


def measure_and_fit(field, cosmo_t, kmin, kmax, nmu=5, nbb=3, sig_prior=None,
                    label="", window="continuous", workers=1, verbose=True):
    k, mu, P, N = pkmu(field, dk=0.005, kmax=0.5, nmu=nmu,
                       window=window, workers=workers)
    good = bin_mask(k, P, N, kmin, kmax)
    iw = np.broadcast_to(np.arange(P.shape[1])[None, :], P.shape)[good]
    # Sparse FFT power contains P_1D(k_parallel)/n_2D.  Reconstruction changes
    # P_1D, so derive the alias shape from the state being fitted rather than
    # reusing a pre-reconstruction template.
    kz, p1d = measure_p1d(field, window=window)
    alias_shape = np.interp(k[good] * mu[good], kz, p1d)[:, None]
    model = PkmuModel(cosmo_t, k[good], mu[good], iw, nmu, nbb=nbb,
                      extra_cols=alias_shape)
    Pv = P[good]
    var = 2.0 * Pv ** 2 / np.maximum(N[good], 1.0)
    res = PkmuFitter(model, Pv, 1.0 / var, sig_prior=sig_prior).fit()
    if verbose:
        print(f"  {label}: ap={res['ap']:.4f}+-{res['sigma_ap']:.4f}  "
              f"at={res['at']:.4f}+-{res['sigma_at']:.4f}  "
              f"chi2/dof={res['chi2']:.0f}/{res['ndof']}", flush=True)
    measurement = {"k": k, "mu": mu, "P": P, "N": N}
    return res, measurement


def main(a):
    full_sim = f"AbacusSummit_base_c000_{a.phase}"
    if a.halo_file is None and a.phase != "ph000":
        raise ValueError("halo catalog exists for ph000 only; use --halo-file")
    print(f"\nScenario 2: external recon at skewers\n"
          f"  {full_sim}  Model {a.model}  frac={a.frac} "
          f"({frac_to_qso_per_deg2(a.frac, a.nmesh):.0f}/deg^2)  "
          f"thin_n={a.thin_n}  sr={a.sr}  fit k in [{a.kmin},{a.kmax}]\n",
          flush=True)

    recon, ff, n_tracer = build_recon_external(
        full_sim, a.sr, a.bias, a.thin_n, a.seed, a.nmesh, ncpu=a.ncpu,
        halo_file=a.halo_file)

    grid = load_dF_grid(full_sim, a.model, a.los, a.nmesh)
    mask = make_mask(a.nmesh, a.frac, a.seed)

    print("  painting pre (zero displacement, same path)", flush=True)
    dF_pre = paint_sparse(grid, mask, a.nmesh, recon=None)
    print("  painting post (external shifts at skewers)", flush=True)
    dF_post = paint_sparse(grid, mask, a.nmesh, recon=recon, rsd_f=ff)
    del grid, recon; gc.collect()

    cosmo_t = class_pk(z=REDSHIFT)
    rp, mp = measure_and_fit(dF_pre, cosmo_t, a.kmin, a.kmax,
                             sig_prior=SIG_PRIOR_PRE, label="pre",
                             window="node", workers=a.ncpu)
    rq, mq = measure_and_fit(dF_post, cosmo_t, a.kmin, a.kmax,
                             sig_prior=SIG_PRIOR_POST, label="post",
                             window="continuous", workers=a.ncpu)

    gain_ap = rp["sigma_ap"] / rq["sigma_ap"]
    gain_at = rp["sigma_at"] / rq["sigma_at"]
    print(f"  GAIN: ap x{gain_ap:.3f}   at x{gain_at:.3f}", flush=True)

    tag = (f"{a.phase}_m{a.model}_f{a.frac:.3f}"
           + (f"_thin{a.thin_n:.0e}" if a.thin_n is not None else "_full")
           + f"_sr{a.sr:.0f}_kmax{a.kmax:.2f}_n{a.nmesh}"
           + f"_seed{a.seed}_b{a.bias:.3f}")
    out = OUT_DIR / f"scenario2_{tag}.npz"
    save = dict(phase=a.phase, model=a.model, frac=a.frac,
                qso_per_deg2=frac_to_qso_per_deg2(a.frac, a.nmesh),
                thin_n=(np.nan if a.thin_n is None else a.thin_n),
                n_tracer=n_tracer, bias=a.bias, sr=a.sr, seed=a.seed,
                kmin=a.kmin, kmax=a.kmax,
                alias_template=np.array(True),
                alias_template_state_specific=np.array(True),
                damping_prior_pre=np.array(SIG_PRIOR_PRE),
                damping_prior_post=np.array(SIG_PRIOR_POST),
                gain_ap=gain_ap, gain_at=gain_at)
    for wname, r in (("pre", rp), ("post", rq)):
        for kk, vv in r.items():
            save[f"{wname}_{kk}"] = vv
    for wname, measurement in (("pre", mp), ("post", mq)):
        for kk, vv in measurement.items():
            save[f"{wname}_pkmu_{kk}"] = vv
    np.savez(out, **save)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="ph000")
    p.add_argument("--model", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--los", default="z", choices=["z"])
    p.add_argument("--nmesh", type=int, default=512)
    p.add_argument("--ncpu", type=int, default=1,
                   help="FFT and reconstruction threads")
    p.add_argument("--frac", type=float, default=0.155,
                   help="0.155=DESI 50/deg^2, 0.5=DESI-II 161/deg^2")
    p.add_argument("--thin-n", type=float, default=None,
                   help="tracer number density (Mpc/h)^-3; None=full halos")
    p.add_argument("--bias", type=float, default=2.695)
    p.add_argument("--sr", type=float, default=10.0)
    p.add_argument("--kmin", type=float, default=0.02)
    p.add_argument("--kmax", type=float, default=0.35,
                   help="0.35 removes the aliasing-driven alpha bias")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--halo-file", default=None,
                   help="external (LAE) mock npz with x_L2com/v_L2com/"
                        "VelZSpace_to_kms; overrides the ph000 halos")
    main(p.parse_args())
