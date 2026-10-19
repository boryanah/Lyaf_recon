"""
recon_paint_field.py
--------------------
Produce a pair of continuous Lya delta_F fields:

    --mode ref    zero displacement  -> pre-recon  reference
    --mode recon  Zeldovich shifts   -> post-recon field

Both go through the same advect + TSC-repaint code path.  An unjittered
reference is nevertheless a special case because its particles lie exactly
on grid nodes, whereas displaced particles have generic subcell positions.
Use ``--jitter`` to match their assignment statistics, or apply the exact
node-stencil correction to the reference spectrum.

Two reconstructed-field definitions are available.  ``--norm local``
transports delta_F as a scalar by dividing its shifted weighted paint by the
shifted sampling density.  ``--norm global`` keeps the Jacobian contribution
and is equivalent, on this dense grid, to the standard RecSym displaced-data
minus shifted-random construction.  The distinction is a scientific choice,
so it is explicit in both the command line and output filename.

Method
------
1. Either the forest or a halo catalog drives pyrecon.  The production paper
   result uses the forest.  pyrecon's `f` is the growth rate, not the growth
   factor.
2. delta_F comes from make_dF_grid.load_dF_grid: an inverse FFT of the shipped
   Fourier-space field, band-limited to the painting grid's Nyquist.
3. Each grid cell is advected as a tracer carrying weight delta_F, then TSC
   repainted.  The Lya field is recovered as

       delta_F_rec = (TSC of delta_F-weighted points) / (TSC of the same
                      points unweighted)

   The unweighted denominator plays the role that shifted randoms play for a
   point sample: it divides out the displacement-induced modulation of the
   cell number density, leaving the advected field itself.

Convention note
---------------
The field is advected with the RecSym redshift-space shift.  The usual RecSym
Kaiser-preservation argument is derived for a density tracer with displaced
data and shifted random catalogs.  Equation above is a relocation of a
continuous flux value instead, so that derivation does not apply unchanged.
The same BAO template is used on both sides, but beta and the broadband are fit
freely and their changes are diagnostics of the construction.

Usage
-----
  python recon_paint_field.py --mode ref   --nmesh 512
  python recon_paint_field.py --mode recon --driver lya --nmesh 512 \
      --nmesh_recon 512 --sr 10 --rectype MG
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np

from make_dF_grid import load_dF_grid, LBOX

HALO_DIR = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/Lyaf/")
OUT_DIR = Path("/pscratch/sd/b/boryanah/Lyaf_recon/")

# Large-scale forest responses measured directly from each phase-matched
# initial density field (paper_prd/calibrate_all_forest_response.py).
FOREST_RESPONSE = {
    1: (-0.148066, 0.957688),
    2: (-0.130939, 0.989632),
    3: (-0.132453, 2.074689),
    4: (-0.128523, 2.395469),
}


def forest_response_tag(b_f, beta_f):
    """Unambiguous cache tag for the two forest-response inputs."""
    return f"bF{b_f:.6f}_bt{beta_f:.6f}"


def save_npy_atomic(path, array):
    """Write a large NumPy field without exposing a partial final file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_recon(sim_name, redshift, nmesh_recon, sr, rectype, bias, ncpu,
                rands_fac=0, seed=300, verbose=True):
    """Run pyrecon on the RSD halo catalog and return the fitted instance."""
    from pyrecon import (MultiGridReconstruction, IterativeFFTReconstruction,
                         IterativeFFTParticleReconstruction)
    from cosmoprimo.fiducial import AbacusSummit

    recfunc = {"MG": MultiGridReconstruction,
               "IFT": IterativeFFTReconstruction,
               "IFTP": IterativeFFTParticleReconstruction}[rectype]

    cosmo = AbacusSummit()
    ff = cosmo.growth_rate(redshift)          # growth RATE, not growth factor
    if verbose:
        print(f"  f(z={redshift}) = {ff:.4f}   bias = {bias}   R = {sr} Mpc/h")

    f = np.load(HALO_DIR / f"halos_{sim_name}_z{redshift:.3f}.npz")
    pos = np.asarray(f["x_L2com"], dtype=np.float64) + LBOX / 2.0
    vel = np.asarray(f["v_L2com"], dtype=np.float64)
    vz2mpc = float(f["VelZSpace_to_kms"]) / LBOX   # km/s per Mpc/h
    del f; gc.collect()

    # RSD along z.  VelZSpace_to_kms spans the FULL box (215321 km/s), so the
    # per-Mpc/h conversion needs the /Lbox -- this is the factor-2000 bug.
    pos[:, 2] += vel[:, 2] / vz2mpc
    pos %= LBOX
    del vel; gc.collect()
    if verbose:
        print(f"  halos: {len(pos):,}")

    recon = recfunc(f=ff, bias=bias, nmesh=nmesh_recon, los="z",
                    positions=pos, boxsize=LBOX,
                    boxcenter=(LBOX / 2, LBOX / 2, LBOX / 2),
                    nthreads=int(ncpu), fft_engine="fftw", fft_plan="estimate",
                    dtype="f4", wrap=True)
    recon.assign_data(pos)
    del pos; gc.collect()

    if rands_fac > 0:
        rng = np.random.default_rng(seed)
        n_r = int(rands_fac * recon.mesh_data.value.sum())
        if verbose:
            print(f"  randoms: {n_r:,}")
        recon.assign_randoms(rng.random((n_r, 3)) * LBOX)
    recon.set_density_contrast(smoothing_radius=sr)
    recon.run()
    if verbose:
        print("  recon done", flush=True)
    return recon, ff


def build_recon_lya(sim_name, model, los, redshift, nmesh_recon, sr, rectype,
                    b_F, beta_F, ncpu, n_slab=32, mask=None, verbose=True):
    """Drive the reconstruction from the Lya field itself instead of the halos.

    This is the *realistic* case.  build_recon() uses a dense halo catalog that
    knows the density field almost perfectly, so the gain it produces is an
    upper bound; a real survey has to estimate Psi from the forest.

    Two subtleties, both handled here:

    1. b_F is NEGATIVE (about -0.15 for Model 1): more gas means more
       absorption means less
       flux.  pyrecon's periodic branch computes  delta = (n/nbar - 1) / bias,
       so rather than hand it a negative bias we do the division ourselves in
       the weights -- paint  w = 1 + delta_F / b_F  and pass bias = 1.  The
       mean weight is 1 by construction; individual cells go negative, which is
       fine (the periodic branch has no thresholding and the Zel'dovich solve
       is linear in delta).

    2. beta_F is NOT f / b_F.  Lya has a velocity-gradient bias b_eta on top of
       the density bias, so delta_F = (b_F + b_eta f mu^2) delta and
       beta_F = b_eta f / b_F ~ 0.96, where f / b_F would be about -7.
       pyrecon
       assumes beta = f / bias internally, so we pass f = beta_F with bias = 1.
       That makes the RSD-corrected Zel'dovich solve right, but it also means
       pyrecon's own 'disp+rsd' would apply beta_F where the gas actually moves
       with the true growth rate.  So read 'disp' and build the RSD part
       ourselves -- see paint(rsd_f=...).
    """
    if b_F == 0:
        raise ValueError("b_F must be nonzero")
    if mask is not None and np.shape(mask) != (nmesh_recon, nmesh_recon):
        raise ValueError("sightline mask shape must match the reconstruction "
                         f"mesh: got {np.shape(mask)}, expected "
                         f"{(nmesh_recon, nmesh_recon)}")

    from pyrecon import (MultiGridReconstruction, IterativeFFTReconstruction,
                         IterativeFFTParticleReconstruction)
    recfunc = {"MG": MultiGridReconstruction,
               "IFT": IterativeFFTReconstruction,
               "IFTP": IterativeFFTParticleReconstruction}[rectype]

    if verbose:
        print(f"  Lya-driven recon: b_F = {b_F}  beta_F = {beta_F}  "
              f"R = {sr} Mpc/h  nmesh_recon = {nmesh_recon}")

    # los="z" always: load_dF_grid puts the line of sight on axis 2 whatever
    # the shipped file's orientation, so the recon must agree with the grid,
    # not with the file's label.
    recon = recfunc(f=beta_F, bias=1.0, nmesh=nmesh_recon, los="z",
                    boxsize=LBOX, boxcenter=(LBOX / 2,) * 3,
                    nthreads=int(ncpu), fft_engine="fftw",
                    fft_plan="estimate", dtype="f4", wrap=True)

    grid = load_dF_grid(sim_name, model, los, nmesh_recon, redshift=redshift,
                        verbose=verbose, workers=ncpu)
    cell = LBOX / nmesh_recon
    ax = np.arange(nmesh_recon, dtype=np.float32) * cell
    for i0 in range(0, nmesh_recon, n_slab):
        i1 = min(i0 + n_slab, nmesh_recon)
        X, Y, Z = np.meshgrid(ax[i0:i1], ax, ax, indexing="ij")
        pos = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        del X, Y, Z
        w = 1.0 + np.ascontiguousarray(grid[i0:i1].ravel()) / b_F
        if mask is not None:
            # sparse sightlines: only cells on a selected skewer carry data.
            # The same positions are also assigned as randoms, so pyrecon's
            # randoms branch divides out the sightline geometry instead of
            # reading the empty cells as enormous voids -- which is what the
            # periodic branch would do, and would wreck Psi.
            keep = np.repeat(mask[i0:i1].ravel(), nmesh_recon)
            pos, w = pos[keep], w[keep]
        recon.assign_data(pos, weights=w)
        if mask is not None:
            recon.assign_randoms(pos)
        del pos, w; gc.collect()
        if verbose:
            print(f"    assigned {i1}/{nmesh_recon}", end="\r", flush=True)
    if verbose:
        print()
    del grid; gc.collect()

    recon.set_density_contrast(smoothing_radius=sr)
    recon.run()
    if verbose:
        print("  recon done", flush=True)
    return recon


def paint(sim_name, model, los, nmesh, mode, recon=None, n_slab=32,
          rsd_f=None, mask=None, norm="local", jitter=False, jitter_seed=99,
          verbose=True, workers=1):
    """Advect the delta_F grid and TSC-repaint. Returns delta_F_rec (nmesh^3).

    Painting uses pmesh rather than abacusnbody.analysis.tsc: the cosmodesi
    environment that carries pyrecon has a broken numba (coverage 6.5.0 vs the
    version numba needs), so tsc_parallel cannot be imported there.  pmesh is
    present in the same env, so recon and painting stay in one process.
    """
    if mode == "recon" and recon is None:
        raise ValueError("mode='recon' requires a reconstruction object")
    if mask is not None and np.shape(mask) != (nmesh, nmesh):
        raise ValueError("sightline mask shape must match the painting mesh: "
                         f"got {np.shape(mask)}, expected {(nmesh, nmesh)}")
    if norm not in ("local", "global", "both"):
        raise ValueError("norm must be 'local', 'global', or 'both'")

    from pmesh.pm import ParticleMesh

    grid = load_dF_grid(sim_name, model, los, nmesh, verbose=verbose,
                        workers=workers)
    cell = LBOX / nmesh

    pm = ParticleMesh(BoxSize=LBOX, Nmesh=[nmesh] * 3, dtype="f4")
    m_w = pm.create(type="real"); m_w[...] = 0.0
    m_n = pm.create(type="real"); m_n[...] = 0.0

    # BUGFIX: grid index i sits at position i*cell in the SAME [0, L) frame
    # pyrecon uses (halo positions are x_L2com + L/2).  check_alignment.py
    # verifies this directly: the delta_F grid and the halo density grid
    # cross-correlate at r = -0.97 with zero lag, index against index.
    #
    # The previous `- LBOX/2` here assumed the delta_F grid started at the
    # Abacus box corner -L/2 and wrapped it back, which evaluated Psi half a
    # box away from every cell it was applied to.  A smoothed displacement
    # field read at the wrong place is still smooth, so this did not blow up
    # -- it just decorrelated Psi from the field and quietly removed the whole
    # reconstruction signal (halo BAO sharpened, Lya BAO did not).
    ax = np.arange(nmesh, dtype=np.float32) * cell
    for i0 in range(0, nmesh, n_slab):
        i1 = min(i0 + n_slab, nmesh)
        X, Y, Z = np.meshgrid(ax[i0:i1], ax, ax, indexing="ij")
        pos = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        del X, Y, Z
        # Shared sub-cell jitter.  Without it the REFERENCE points sit exactly
        # on grid nodes in all three axes (TSC window 0.75/0.125/0.125 per
        # axis) while the RECON points are displaced by ~1.5/1.5/3.0 Mpc/h rms
        # -- so the pre/post window mismatch is itself ANISOTROPIC, much larger
        # along the line of sight than across it, and can leak into alpha_par
        # vs alpha_perp rather than staying in the broadband.  Drawing the same
        # offsets (same seed, same slab) for both modes makes the two windows
        # identical by construction.  Drawn before any masking so the sequence
        # does not depend on the sightline selection.
        if jitter:
            rng = np.random.default_rng(jitter_seed + i0)
            pos += ((rng.random(pos.shape, dtype=np.float32) - 0.5)
                    * np.float32(cell))
        pos %= LBOX
        w = np.ascontiguousarray(grid[i0:i1].ravel())
        if mask is not None:
            keep = np.repeat(mask[i0:i1].ravel(), nmesh)
            pos, w = pos[keep], w[keep]

        if mode == "recon":
            if rsd_f is None:
                pos -= recon.read_shifts(pos, field="disp+rsd")
            else:
                # Build the RecSym shift ourselves: disp+rsd = disp + f (disp.z)z,
                # i.e. only the LOS component is multiplied by (1+f).  Needed
                # when the recon was constructed with an f that is not the true
                # growth rate -- the Lya driver passes f = beta_F to get the
                # density solve right, but the gas still moves with f_true.
                d = recon.read_shifts(pos, field="disp")
                d[:, 2] *= (1.0 + rsd_f)
                pos -= d
                del d
            pos %= LBOX

        pm.paint(pos, mass=w, resampler="tsc", out=m_w, hold=True)
        pm.paint(pos, mass=None, resampler="tsc", out=m_n, hold=True)
        del pos, w; gc.collect()
        if verbose:
            print(f"    painted {i1}/{nmesh}", end="\r", flush=True)
    if verbose:
        print()
    del grid; gc.collect()

    dens_w = np.asarray(m_w)
    dens_n = np.asarray(m_n)
    empty = dens_n <= 0
    if verbose:
        print(f"  empty cells: {int(empty.sum()):,} / {nmesh**3:,} "
              f"({100.*empty.sum()/nmesh**3:.4f}%)")
    outputs = {}
    if norm in ("local", "both"):
        # divide pointwise: recovers the field value wherever there is support.
        # Right for the dense case, where every cell carries data.
        out = dens_w / np.maximum(dens_n, 1e-6)
        out[empty] = 0.0
        out -= out.mean()
        outputs["local"] = np.ascontiguousarray(out, dtype=np.float32)
    if norm in ("global", "both"):
        # divide by the GLOBAL mean occupancy: this is the mask-weighted
        # estimator  delta_F * W / <W>,  which is what a sparse survey actually
        # measures.  Its power is P(k) plus the usual Lya aliasing pedestal.
        # The selection is common to pre and post, while the P_1D shape of that
        # pedestal must be measured separately for the two physical fields.
        # Pointwise division would instead interpolate across the gaps and
        # quietly invent information.
        out = dens_w / dens_n.mean()
        out -= out.mean()
        outputs["global"] = np.ascontiguousarray(out, dtype=np.float32)
    return outputs if norm == "both" else outputs[norm]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim_name", default="AbacusSummit_base_c000_ph000")
    p.add_argument("--redshift", type=float, default=2.5)
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--los", default="z", choices=["y", "z"],
                   help="physical line of sight; the loader rotates it to axis 2")
    p.add_argument("--nmesh", type=int, default=512, help="Lya painting grid")
    p.add_argument("--nmesh_recon", type=int, default=512)
    p.add_argument("--sr", type=float, default=10.0)
    p.add_argument("--rectype", default="MG", choices=["MG", "IFT", "IFTP"])
    p.add_argument("--bias", type=float, default=2.70, help="halo bias")
    p.add_argument("--driver", default="lya", choices=["halos", "lya"],
                   help="what estimates the displacement field. 'halos' is the "
                        "optimistic case (a dense catalog that knows the "
                        "density almost perfectly); 'lya' is the realistic one")
    p.add_argument("--bias_F", type=float, default=None,
                   help="Lya flux bias; default is the measured response for "
                        "the selected model")
    p.add_argument("--beta_F", type=float, default=None,
                   help="Lya RSD parameter; default is measured for the "
                        "selected model. NOT f/b_F -- see build_recon_lya")
    p.add_argument("--rands_fac", type=int, default=0,
                   help="0 = use the periodic mean density (no randoms)")
    p.add_argument("--ncpu", type=int, default=1,
                   help="FFT and reconstruction threads; batch drivers pass "
                        "their allocation explicitly")
    p.add_argument("--n_slab", type=int, default=32,
                   help="number of x planes assigned or painted per batch")
    p.add_argument("--mode", default="recon", choices=["ref", "recon"])
    p.add_argument("--norm", default="local", choices=["local", "global", "both"],
                   help="local transports the flux scalar; global implements "
                        "the dense displaced-data minus shifted-random field; "
                        "both saves both definitions from one paint")
    p.add_argument("--jitter", action="store_true",
                   help="shared sub-cell jitter, identical in both modes, so "
                        "the TSC window is common to ref and recon")
    p.add_argument("--no_rsd_shift", action="store_true",
                   help="diagnostic: apply the Zeldovich shift with NO (1+f) "
                        "boost on the LOS component, making the residual "
                        "displacement nearly isotropic")
    p.add_argument("--out_dir", default=str(OUT_DIR))
    a = p.parse_args()
    if a.model not in FOREST_RESPONSE:
        p.error(f"no calibrated forest response for Model {a.model}; pass "
                "--bias_F and --beta_F explicitly")
    default_bias, default_beta = FOREST_RESPONSE[a.model]
    if a.bias_F is None:
        a.bias_F = default_bias
    if a.beta_F is None:
        a.beta_F = default_beta

    # This program assigns a different simulation in each invocation and
    # saves a plain NumPy array.  It is not an MPI-distributed field writer.
    # If several invocations are launched as ranks of one srun, pmesh silently
    # distributes each nominal mesh over COMM_WORLD; np.asarray then returns
    # only the local slab and collectives can mix the simulations.  Refuse that
    # configuration before reading any large inputs.
    try:
        from mpi4py import MPI
        world_size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    if world_size != 1:
        raise RuntimeError(
            "recon_paint_field.py requires one MPI rank per invocation; "
            f"found COMM_WORLD size {world_size}. Launch phases as separate "
            "one-task jobs, not as tasks of one multi-rank srun."
        )

    os.makedirs(a.out_dir, exist_ok=True)
    recon, rsd_f = None, None
    if a.mode == "recon":
        if a.driver == "halos":
            recon, _ = build_recon(a.sim_name, a.redshift, a.nmesh_recon, a.sr,
                                   a.rectype, a.bias, a.ncpu, a.rands_fac)
        else:
            from cosmoprimo.fiducial import AbacusSummit
            rsd_f = 0.0 if a.no_rsd_shift else AbacusSummit().growth_rate(a.redshift)
            recon = build_recon_lya(a.sim_name, a.model, a.los, a.redshift,
                                    a.nmesh_recon, a.sr, a.rectype,
                                    a.bias_F, a.beta_F, a.ncpu,
                                    n_slab=a.n_slab)
    field = paint(a.sim_name, a.model, a.los, a.nmesh, a.mode, recon=recon,
                  n_slab=a.n_slab, rsd_f=rsd_f, norm=a.norm,
                  jitter=a.jitter, workers=a.ncpu)
    fields = field if isinstance(field, dict) else {a.norm: field}
    expected_shape = (a.nmesh,) * 3
    for norm_name, norm_field in fields.items():
        if norm_field.shape != expected_shape:
            raise RuntimeError(f"paint returned local or malformed shape "
                               f"{norm_field.shape}; expected {expected_shape}")

    bt = (f"b{a.bias:.1f}" if a.driver == "halos"
          else forest_response_tag(a.bias_F, a.beta_F))
    for norm_name, norm_field in fields.items():
        drv = "" if a.driver == "halos" else "_lyadrv"
        drv += "_global" if norm_name == "global" else ""
        drv += (("_jit" if a.jitter else "")
                + ("_f0" if a.no_rsd_shift else ""))
        tag = (f"{a.mode}_Model_{a.model}_LOS{a.los}_nmesh{a.nmesh}"
               + (f"_R{a.sr:.2f}_nr{a.nmesh_recon}_{a.rectype}_{bt}{drv}"
                  if a.mode == "recon" else ("_jit" if a.jitter else "")))
        fn = Path(a.out_dir) / f"dF_{tag}_{a.sim_name}_z{a.redshift:.3f}.npy"
        save_npy_atomic(fn, norm_field)
        print(f"  {norm_name} std={norm_field.std():.5f}  saved -> {fn}")
