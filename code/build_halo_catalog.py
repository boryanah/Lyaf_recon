"""
build_halo_catalog.py
---------------------
Write a halo catalogue for one AbacusSummit base phase at z=2.5 in the same
format as the archived phase-000 file used by the external-tracer
(Regime II) reconstruction:

    /global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/Lyaf/
        halos_AbacusSummit_base_c000_ph000_z2.500.npz
    keys: x_L2com, v_L2com (float32, box-centred), M, VelZSpace_to_kms

The archived phase-000 file contains 45,737,872 CompaSO halos with
M >= 3.164e11 Msun/h, i.e. N >= 150 particles.  ``--check`` rebuilds
phase 000 with this cut, for cleaned and uncleaned catalogues, and reports
which reproduces the archived count exactly.  Use that setting for the
other phases so all six share one tracer definition.

Usage:
  python build_halo_catalog.py --phase 000 --check
  python build_halo_catalog.py --phase 001 --cleaned {0,1}
"""
import argparse
from pathlib import Path

import numpy as np

SIM_ROOT = Path("/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit")
ARCHIVED = Path("/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/Lyaf/"
                "halos_AbacusSummit_base_c000_ph000_z2.500.npz")
OUT = Path("/pscratch/sd/b/boryanah/Lyaf_recon/halos")
NMIN = 150


def load(phase, cleaned):
    from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
    path = (SIM_ROOT / f"AbacusSummit_base_c000_ph{phase}" / "halos"
            / "z2.500")
    cat = CompaSOHaloCatalog(path, fields=["x_L2com", "v_L2com", "N"],
                             cleaned=bool(cleaned))
    h = cat.halos
    keep = np.asarray(h["N"]) >= NMIN
    header = cat.header
    mpart = float(header["ParticleMassHMsun"])
    vz = float(header["VelZSpace_to_kms"])
    x = np.asarray(h["x_L2com"])[keep].astype(np.float32)
    v = np.asarray(h["v_L2com"])[keep].astype(np.float32)
    m = np.asarray(h["N"])[keep].astype(np.float64) * mpart
    return x, v, m, vz, mpart


def main(a):
    if a.check:
        with np.load(ARCHIVED) as d:
            n_arch = len(d["M"])
            m_arch = float(d["M"].min())
        print(f"archived ph000: {n_arch:,} halos, M_min {m_arch:.4e}")
        for cleaned in (1, 0):
            x, v, m, vz, mpart = load("000", cleaned)
            print(f"cleaned={cleaned}: {len(m):,} halos with N>={NMIN}, "
                  f"M_min {m.min():.4e}, m_part {mpart:.4e}, "
                  f"match={'YES' if len(m) == n_arch else 'no'}", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"halos_AbacusSummit_base_c000_ph{a.phase}_z2.500.npz"
    if dest.exists():
        print("exists", dest)
        return
    x, v, m, vz, mpart = load(a.phase, a.cleaned)
    tmp = dest.with_name("." + dest.name + ".tmp.npz")
    np.savez(tmp, x_L2com=x, v_L2com=v, M=m, VelZSpace_to_kms=vz,
             cleaned=np.array(bool(a.cleaned)), nmin=np.array(NMIN))
    tmp.rename(dest)
    print(f"ph{a.phase}: {len(m):,} halos -> {dest}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="000")
    p.add_argument("--cleaned", type=int, default=1)
    p.add_argument("--check", action="store_true")
    main(p.parse_args())
