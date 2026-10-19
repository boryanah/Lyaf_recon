"""
box_to_p3d.py
-------------
Convert a periodic delta_F box + a transverse sightline mask into the 7-column
text format the P3D pair-count estimator reads:

    x y z w wg tid hpx

Contract (from power_mod/power_counts.h::count_pairs, not from the README):
  * the estimator uses  d_i = wg_i - w_i  as the density and w_i as the weight,
    so a unit-weight forest pixel is  w = 1,  wg = 1 + delta_F  ->  d = delta_F;
  * pairs with equal `tid` are SKIPPED.  One tid per skewer is what removes the
    P_1D/n_2D aliasing pedestal that the FFT estimator cannot avoid, so tid must
    be constant along a sightline and unique across sightlines;
  * `hpx` is jackknife bookkeeping only; 0 everywhere is fine.

Geometry.  Our LOS is axis 2 of the grid and the box is periodic, so the
matching mu convention is the compile-time PERIODIC branch of `cleanup_l`
(mu = |dz|/r).  Positions are written in the box frame [0, LBOX), which is what
that branch expects.  Do NOT feed these files to a non-periodic binary without
first moving the box to a large radial distance -- the default branch would
otherwise use a midpoint LOS pointing at the origin, which for a box straddling
the origin is not the z axis and not even approximately constant.

--random generates a Gaussian-noise field with the identical geometry, for
throughput calibration: the pair count and hence the run time do not depend on
the pixel values, only on the sampling.  That avoids the expensive asdf read
when all we want is a timing number.

Usage
-----
  # timing input: 2% of sightlines, random values
  python box_to_p3d.py --frac 0.02 --random --out /pscratch/.../p3d_f0.02.txt

  # real field
  python box_to_p3d.py --model 1 --frac 0.619 --out /pscratch/.../p3d_f0.619.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NB: LBOX is duplicated rather than imported from make_dF_grid, because that
# module imports asdf at the top and asdf is not in the plain python module we
# use here (only in the desi env).  --random / --field then need no asdf at
# all; load_dF_grid is imported lazily below, in the one branch that needs it.
LBOX = 2000.0                          # Mpc/h, matches make_dF_grid.LBOX

CHI_Z25 = 4022.0                       # Mpc/h, comoving distance to z = 2.5
DEG = CHI_Z25 * np.pi / 180.0          # 70.19 Mpc/h per degree


def qso_per_deg2(frac, nmesh):
    """Sightline fraction -> quasars per square degree (nmesh=512: frac*322.9)."""
    return frac * nmesh ** 2 / LBOX ** 2 * DEG ** 2


def frac_for_qso(q, nmesh):
    """Inverse of qso_per_deg2."""
    return q * LBOX ** 2 / (nmesh ** 2 * DEG ** 2)


def make_mask(nmesh, frac, seed=42):
    """Random transverse sightline mask, shape (nmesh, nmesh).

    Same construction as sparse_skewers.make_mask so the P3D and FFT runs can
    be given literally the same sightlines.
    """
    if frac >= 1.0:
        return np.ones((nmesh, nmesh), dtype=bool)
    rng = np.random.default_rng(seed)
    m = np.zeros(nmesh * nmesh, dtype=bool)
    m[rng.choice(nmesh * nmesh, int(round(frac * nmesh * nmesh)),
                 replace=False)] = True
    return m.reshape(nmesh, nmesh)


def n_pairs_estimate(nsk, nmesh, R0):
    """Rough pair count within R0 (ignores the tid exclusion and edges)."""
    n3d = nsk * nmesh / LBOX ** 3
    return 0.5 * (nsk * nmesh) * n3d * (4. / 3. * np.pi * R0 ** 3)


def ghost_corners(eps=1e-6):
    """Eight zero-weight particles pinning the position range to exactly LBOX.

    In PERIODIC mode `driver.h` infers the period from the RANGE of the input
    positions, ignoring -boxsize.  Our cell centres span 0 .. (nmesh-1)*cell =
    1996.09, so without this the code would use 1996.09 as the period and every
    wrapped separation would be wrong by one cell (3.9 Mpc/h) -- a systematic
    on the ~20% of pairs that cross a face, and on the normalization volume.

    These particles have w = 0 and wg = 0, hence d = wg - w = 0, so they
    contribute exactly zero to power_counts (weighted by d_i d_j) and zero to
    RR_counts (weighted by w_i w_j).  They only inflate `used_pairs`.  Placed
    at LBOX - eps rather than LBOX so they cannot land in cell index `nside`.
    """
    c = [0.0, LBOX - eps]
    return np.array([[x, y, z, 0.0, 0.0, -1 - i, 0]
                     for i, (x, y, z) in enumerate(
                         (a, b, d) for a in c for b in c for d in c)])


def write_txt(out, grid, mask, nmesh, chunk=32, periodic=True, distant=0.0):
    """Stream the selected skewers to disk.  Never materializes N x 7 in RAM."""
    cell = LBOX / nmesh
    zc = np.arange(nmesh, dtype=np.float64) * cell
    ix, iy = np.nonzero(mask)
    nsk = len(ix)
    print(f"  {nsk:,} skewers x {nmesh} pixels = {nsk*nmesh:,} particles")

    with open(out, "w") as fh:
        # ``chunk`` is a number of sightlines.  Multiplying it by nmesh here
        # used to materialize millions of rows at once and defeated the
        # streaming design.
        for s0 in range(0, nsk, chunk):
            s1 = min(s0 + chunk, nsk)
            n = s1 - s0
            x = np.repeat(ix[s0:s1] * cell, nmesh)
            y = np.repeat(iy[s0:s1] * cell, nmesh)
            z = np.tile(zc, n)
            # tid: unique per sightline, int64, constant along the skewer
            tid = np.repeat(ix[s0:s1].astype(np.int64) * nmesh
                            + iy[s0:s1].astype(np.int64), nmesh)
            d = grid[ix[s0:s1], iy[s0:s1], :].ravel().astype(np.float64)
            w = np.ones_like(d)
            np.savetxt(fh, np.column_stack([x, y, z + distant, w, 1.0 + d, tid,
                                            np.zeros_like(tid)]),
                       fmt="%.6f %.6f %.6f %.6f %.8f %d %d")
            print(f"    wrote {s1}/{nsk} skewers", end="\r", flush=True)
        if periodic:
            g = ghost_corners()
            g[:, 2] += distant
            np.savetxt(fh, g, fmt="%.6f %.6f %.6f %.6f %.8f %d %d")
            print("    + 8 zero-weight ghost corners (pins period to LBOX)")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim_name", default="AbacusSummit_base_c000_ph000")
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--los", default="z", choices=["y", "z"])
    p.add_argument("--nmesh", type=int, default=512)
    p.add_argument("--redshift", type=float, default=2.5)
    p.add_argument("--frac", type=float, default=None,
                   help="fraction of the nmesh^2 sightlines to keep")
    p.add_argument("--qso", type=float, default=None,
                   help="quasars per deg^2 (alternative to --frac)")
    p.add_argument("--seed", type=int, default=42, help="mask seed")
    p.add_argument("--random", action="store_true",
                   help="Gaussian-noise field instead of the real delta_F; "
                        "same geometry, so timing is identical")
    p.add_argument("--field", default=None,
                   help="path to a saved .npy delta_F grid (e.g. a post-recon "
                        "field from recon_paint_field.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--no-ghosts", action="store_true",
                   help="omit the zero-weight corner particles (only for the "
                        "non-periodic / distant-observer binary)")
    p.add_argument("--distant", type=float, default=0.0,
                   help="add this to z, in Mpc/h.  For the NON-periodic binary: "
                        "at D >> LBOX the midpoint LOS becomes the z axis to "
                        "~LBOX/2D, reproducing the flat-sky mu convention. "
                        "Positions are double precision so D ~ 1e7 is safe.")
    a = p.parse_args()

    if (a.frac is None) == (a.qso is None):
        raise SystemExit("give exactly one of --frac / --qso")
    frac = a.frac if a.frac is not None else frac_for_qso(a.qso, a.nmesh)
    print(f"  frac = {frac:.4f}  ->  {qso_per_deg2(frac, a.nmesh):.1f} QSO/deg^2"
          f"   (mean transverse separation "
          f"{LBOX/(a.nmesh*np.sqrt(frac)):.1f} Mpc/h)")

    mask = make_mask(a.nmesh, frac, a.seed)
    nsk = int(mask.sum())
    for R0 in (150., 200.):
        print(f"  estimated pairs within R0={R0:.0f}: "
              f"{n_pairs_estimate(nsk, a.nmesh, R0):.2e}")

    if a.random:
        print("  --random: Gaussian noise field (timing only, NOT physics)")
        grid = np.random.default_rng(a.seed + 1).normal(
            0., 0.1, (a.nmesh, a.nmesh, a.nmesh)).astype(np.float32)
    elif a.field:
        grid = np.load(a.field)
        print(f"  loaded {a.field}  std={grid.std():.5f}")
    else:
        from make_dF_grid import load_dF_grid
        grid = load_dF_grid(a.sim_name, a.model, a.los, a.nmesh,
                            redshift=a.redshift)

    write_txt(a.out, grid, mask, a.nmesh, periodic=not a.no_ghosts,
              distant=a.distant)
    print(f"  saved -> {a.out}")
