#!/usr/bin/env python
"""Re-run the two headline gain measurements from the data bundled here.

1. Dense (Regime I) Model 1, LOSz: the six-phase stacked P(k,mu) fits before and
   after reconstruction with profile-likelihood errors, exactly as
   code/paper_prd/recompute_dense_results.py does them (functions imported from
   that script; only its SCRATCH root is pointed at data/).  Expected (paper Table
   II, data/paper_prd_data/dense_results.npz): sigma(alpha_par) 0.00286 -> 0.00210
   (gain 1.363), sigma(alpha_perp) 0.00150 -> 0.00125 (gain 1.203), damping
   (4.61, 2.81) -> (2.16, 1.36) Mpc/h.

2. Sparse self-reconstruction (Regime III) at 50 sightlines/deg^2 (mask fraction
   0.155, seed 1234), Model 1, LOSz: the six-phase stacked fits with the
   state-specific P_1D alias template and the damping priors recorded in
   data/pscratch/Lyaf_recon/sparse_skewers_Model_1_calibrated_f0.155_seed1234.npz,
   using the shared stack fitter of code/paper_prd/check_sparse_stability.py.
   Expected: gains 1.4126 (alpha_par) and 1.1409 (alpha_perp) stored in that file
   and 1.413 / 1.141 in sparse_final_fits.npz (f0.155_III_self_gain).

Usage (repository root, cosmodesi environment):
  python tools/reproduce_gains.py            # both tests, Model 1
  python tools/reproduce_gains.py dense 2    # dense test for another model
  python tools/reproduce_gains.py sparse
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
SCR = ROOT / 'data' / 'pscratch' / 'Lyaf_recon'
PD = ROOT / 'data' / 'paper_prd_data'
sys.path.insert(0, str(CODE / 'paper_prd'))
sys.path.insert(0, str(CODE))

from fit_bao_pkmu import measure  # noqa: E402
from template_pk import class_pk  # noqa: E402
import recompute_dense_results as rd  # noqa: E402
import check_sparse_stability as cs  # noqa: E402

SIMS = [f'AbacusSummit_base_c000_ph{p:03d}' for p in range(6)]


def scalar(x):
    """Provenance fields of the npz products are stored as 0-d or 1-element arrays."""
    return float(np.atleast_1d(x)[0])


def dense(model, cosmo):
    rd.SCRATCH = SCR                      # read the bundled caches instead of pscratch
    ref = np.load(PD / 'dense_results.npz')
    out = {}
    for state, window in (('pre', 'node'), ('post', 'continuous')):
        t0 = time.time()
        stack, per = rd.fit_stack(rd.files_for(model, state, 'z'), cosmo, window)
        out[state] = stack
        print(f'  Model {model} {state:4s}: ap={stack["ap"]:.6f} sigma_ap={stack["sigma_ap"]:.5f} '
              f'at={stack["at"]:.6f} sigma_at={stack["sigma_at"]:.5f} '
              f'Sigma=({stack["sig_par"]:.2f}, {stack["sig_per"]:.2f})  [{time.time()-t0:.0f} s]')
        for key in ('sigma_ap', 'sigma_at', 'sig_par', 'sig_per', 'ap', 'at'):
            stored = float(ref[f'm{model}_{state}_{key}'])
            print(f'      {key:8s} re-run {float(stack[key]):.6f}  stored {stored:.6f}  '
                  f'diff {float(stack[key]) - stored:+.2e}')
    g_ap = out['pre']['sigma_ap'] / out['post']['sigma_ap']
    g_at = out['pre']['sigma_at'] / out['post']['sigma_at']
    s_ap = float(ref[f'm{model}_pre_sigma_ap'] / ref[f'm{model}_post_sigma_ap'])
    s_at = float(ref[f'm{model}_pre_sigma_at'] / ref[f'm{model}_post_sigma_at'])
    print(f'  GAIN Model {model}: alpha_par {g_ap:.4f} (stored {s_ap:.4f})   '
          f'alpha_perp {g_at:.4f} (stored {s_at:.4f})')
    return g_ap, g_at


def sparse(cosmo):
    rec = np.load(SCR / 'sparse_skewers_Model_1_calibrated_f0.155_seed1234.npz', allow_pickle=True)
    kmax = scalar(rec['kmax']); nbb = int(scalar(rec['nbb'])); width = scalar(rec['damping_prior_width'])
    prior_pre = tuple(float(x) for x in np.atleast_1d(rec['damping_prior_pre'])[:2]) + (width,)
    prior_post = tuple(float(x) for x in np.atleast_1d(rec['damping_prior_post'])[:2]) + (width,)
    alias_pre = (np.array(rec['alias_kpar_pre_f0p155']), np.array(rec['alias_p1d_pre_f0p155']))
    alias_post = (np.array(rec['alias_kpar_post_f0p155']), np.array(rec['alias_p1d_post_f0p155']))
    print(f'  recorded provenance: kmax={kmax}, nbb={nbb}, priors {prior_pre} / {prior_post}, '
          f'mask seed {int(scalar(rec["mask_seed"]))}, {scalar(rec["qso_per_deg2"]):.1f} sightlines/deg2')
    pre = [SCR / f'dF_skref_f0.155_Model_1_LOSz_nmesh512_seed1234_{s}_z2.500.npy' for s in SIMS]
    post = [SCR / ('dF_skrec_f0.155_Model_1_LOSz_nmesh512_R10.00_nr512_MG_bF-0.148066_bt0.957688_'
                   f'seed1234_{s}_z2.500.npy') for s in SIMS]
    # measure() returns the cached P(k,mu) next to the (absent) field file
    t0 = time.time()
    pre_s = [measure(p, 0.005, 0.5, 5, verbose=False) for p in pre]
    post_s = [measure(p, 0.005, 0.5, 5, verbose=False) for p in post]
    a = cs.fit_stack(pre_s, cosmo, kmax, nbb, alias_pre, prior_pre)
    b = cs.fit_stack(post_s, cosmo, kmax, nbb, alias_post, prior_post)
    print(f'  pre : ap={a["ap"]:.5f}+-{a["sigma_ap"]:.5f} at={a["at"]:.5f}+-{a["sigma_at"]:.5f} '
          f'Sigma=({a["sig_par"]:.2f}, {a["sig_per"]:.2f})')
    print(f'  post: ap={b["ap"]:.5f}+-{b["sigma_ap"]:.5f} at={b["at"]:.5f}+-{b["sigma_at"]:.5f} '
          f'Sigma=({b["sig_par"]:.2f}, {b["sig_per"]:.2f})  [{time.time()-t0:.0f} s]')
    for key in ('pre_ap', 'pre_at', 'post_ap', 'post_at', 'pre_sigma_ap', 'post_sigma_ap', 'pre_sigma_at', 'post_sigma_at'):
        state, par = key.split('_', 1)
        val = (a if state == 'pre' else b)[par]
        print(f'      {key:14s} re-run {float(val):.6f}  stored {scalar(rec[key]):.6f}')
    g_ap, g_at = a['sigma_ap'] / b['sigma_ap'], a['sigma_at'] / b['sigma_at']
    sff = np.load(SCR / 'sparse_final_fits.npz')['f0.155_III_self_gain']
    print(f'  GAIN sparse 50/deg2: alpha_par {g_ap:.4f} (stored {scalar(rec["gain_ap"]):.4f}; '
          f'sparse_final_fits {sff[0]:.4f})   alpha_perp {g_at:.4f} (stored {scalar(rec["gain_at"]):.4f}; '
          f'sparse_final_fits {sff[1]:.4f})')
    return g_ap, g_at


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    model = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cosmo = class_pk(z=2.5)               # cached CLASS template in code/templates
    print(f'CLASS drag scale {cosmo.rs_drag:.4f} Mpc/h')
    if which in ('dense', 'both'):
        print('== dense Regime I stack fits')
        dense(model, cosmo)
    if which in ('sparse', 'both'):
        print('== sparse Regime III stack fits (50 sightlines/deg2)')
        sparse(cosmo)


if __name__ == '__main__':
    main()
