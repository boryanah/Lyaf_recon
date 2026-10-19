"""
regime2_ensemble.py
-------------------
Regime II (external-tracer displacement) for one phase, built so that it is
directly comparable with the Regime III sparse self-reconstruction of
sparse_skewers.py:

  * the SAME sightline mask (seed 1234) and the SAME zero-displacement
    reference fields dF_skref_f{frac}_..._seed1234_... (written by
    sparse_skewers.py; this script never rewrites them);
  * the SAME painter (recon_paint_field.paint, global normalisation) and
    the SAME RecSym shift built from pure 'disp' with the true growth rate;
  * only the displacement SOURCE differs: redshift-space CompaSO halos
    (N >= 150 particles, b = 2.695), optionally randomly thinned to
    LAE-like number densities (random thinning preserves the bias).

For each tracer density it writes the post-reconstruction field
    dF_exrec_f{frac}_Model_1_LOSz_nmesh512_R10.00_nr512_MG_halo{tag}_b2.695_seed1234_{sim}_z2.500.npy
its compensated P(k, mu) cache (continuous window) and a compact P_1D file.
regime2_fit.py stacks the six phases exactly as sparse_skewers.stack_fit does.

Usage:
  python regime2_ensemble.py --phase 003 --ncpu 32
"""
import argparse
import gc
from pathlib import Path

import numpy as np

from recon_paint_field import OUT_DIR, paint, save_npy_atomic
from sparse_skewers import make_mask, measure_p1d
from fit_bao_pkmu import measure
from scenario2_external_recon import build_recon_external

ARCHIVED_PH000 = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/"
                      "Lyaf/halos_AbacusSummit_base_c000_ph000_z2.500.npz")
BUILT = OUT_DIR / "halos"
BIAS = 2.695          # measured large-scale bias of this halo sample
MASK_SEED = 1234      # identical to the Regime III production mask
THIN_SEED = 20250926  # which halos survive random thinning
FRACS = (0.155, 0.5)  # ~50 and ~161 sightlines per deg^2
TRACERS = (None, 1e-3, 1e-4)
SR, NMESH, MODEL, LOS, Z = 10.0, 512, 1, "z", 2.5


def tracer_tag(thin_n):
    return "full" if thin_n is None else f"n{thin_n:.0e}"


def field_path(frac, thin_n, sim):
    return OUT_DIR / (f"dF_exrec_f{frac:.3f}_Model_{MODEL}_LOS{LOS}_nmesh{NMESH}"
                      f"_R{SR:.2f}_nr{NMESH}_MG_halo{tracer_tag(thin_n)}"
                      f"_b{BIAS:.3f}_seed{MASK_SEED}_{sim}_z{Z:.3f}.npy")


def p1d_path(field):
    return field.with_name(field.stem + ".p1d.npz")


def main(a):
    sim = f"AbacusSummit_base_c000_ph{a.phase}"
    halo_file = (ARCHIVED_PH000 if a.phase == "000"
                 else BUILT / f"halos_{sim}_z{Z:.3f}.npz")
    if not halo_file.exists():
        raise FileNotFoundError(f"missing halo catalogue {halo_file}")
    masks = {frac: make_mask(NMESH, frac, MASK_SEED) for frac in FRACS}
    for thin_n in TRACERS:
        todo = [frac for frac in FRACS
                if not field_path(frac, thin_n, sim).exists()]
        if todo:
            recon, ff, n_tracer = build_recon_external(
                sim, SR, BIAS, thin_n, THIN_SEED, NMESH, ncpu=a.ncpu,
                halo_file=str(halo_file))
            print(f"  {sim} tracer {tracer_tag(thin_n)}: n={n_tracer:.3e}",
                  flush=True)
            for frac in todo:
                g = paint(sim, MODEL, LOS, NMESH, "recon", recon=recon,
                          rsd_f=ff, mask=masks[frac], norm="global",
                          verbose=False, workers=a.ncpu)
                path = field_path(frac, thin_n, sim)
                save_npy_atomic(path, g)
                print(f"    f={frac:.3f} post std={g.std():.5f} -> {path.name}",
                      flush=True)
                del g; gc.collect()
            del recon; gc.collect()
        for frac in FRACS:
            path = field_path(frac, thin_n, sim)
            measure(path, 0.005, 0.5, 5, verbose=False, workers=a.ncpu)
            pp = p1d_path(path)
            if not pp.exists():
                kz, p1 = measure_p1d(np.load(path), window="continuous")
                np.savez(pp, kz=kz, p1d=p1, window=np.array("continuous"))
    print("DONE", sim, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True)
    p.add_argument("--ncpu", type=int, default=32)
    main(p.parse_args())
