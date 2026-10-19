# Inputs that are not bundled

Measured on 2026-10-18 (`du -sh`, `ls -l`). `$SCR = /pscratch/sd/b/boryanah/Lyaf_recon`
(NERSC scratch, purged periodically). The full repository `lyaf_recon_paper`
carries a longer manifest (`MANIFEST_HEAVY_INPUTS.md`).

| Input | Location | Size | Produced by | Needed by |
|---|---|---|---|---|
| AbacusLymanAlpha v1 flux fields `complex_dF_Model_{1..4}_LOS{y,z}.asdf` | `/global/cfs/cdirs/desi/public/cosmosim/AbacusLymanAlpha/v1/AbacusSummit_base_c000_ph00{0..5}/z2.500/Model_m/` | 13.9 GB each, 48 files (0.67 TB; the directories also hold 8.3 TB/phase of optical-depth skewers that are not read) | public release (Hadzhiyska et al. 2023) | `code/make_dF_grid.py` |
| AbacusSummit IC density `ic_dens_N1152.asdf` (`ic_dens_N576.asdf`) | `/global/cfs/cdirs/desi/cosmosim/Abacus/ic/AbacusSummit_base_c000_ph00N/` | 3.2 GB (405 MB) per phase | public AbacusSummit ICs | LCV surrogate and IC template (`lcv_reconstruction_check.py`, `matched_lcv_bins.py`, `ic_reference_spectrum.py`, `calibrate_all_forest_response.py`) |
| CompaSO halo catalogues z = 2.5 | `/global/cfs/cdirs/desi/public/cosmosim/AbacusSummit/`; archived phase-000 file `/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/new/Lyaf/halos_AbacusSummit_base_c000_ph000_z2.500.npz` (1.46 GB); phases 001-005 `$SCR/halos/` (6.9 GB) | | `code/build_halo_catalog.py` | Regime II (`regime2_ensemble.py`, `scenario2_external_recon.py`) |
| Reference and reconstructed `512^3` fields `dF_ref_*`, `dF_recon_*`, sparse `dF_skref_*`, `dF_skrec_*`, Regime II `dF_exrec_*` (`.npy`, 536,871,040 bytes each) | `$SCR/` | 957 files, 202 GB in total at the top level (fields, band powers, LCV caches) | `code/recon_paint_field.py`, `sparse_skewers.py`, `regime2_ensemble.py` via `code/jobs/*.slurm` | everything in section 10 of `RESULTS.md`; their `P(k,mu)` / `P_1D` caches **are** bundled |
| `1152^3` Model 1 targets (12 fields, about 6.1 GB each) and their correlations | `$SCR/two_orientation_1152/` | 69 GB | `recon_paint_field.py --nmesh 1152 --nmesh_recon 512`, `highres_two_orientation_xirppi.py` | configuration-space checks; the 12 `xi_highres_*.npz` (1.1 MB) are bundled |
| LCV per-phase band-power caches `lcv_reconstruction_calibrated_*_ph00N.npz` | `$SCR/` | 30 files, 1.6 MB | `lcv_reconstruction_check.py`, `multimodel_losy_lcv.py` | rebuilding `lcv_reconstruction_results*.npz` (bundled) |
| Exact Fourier-to-bin kernels `xi_exact_bin_kernels_n{512,1152}.npz` | `$SCR/estimator_validation_20260908/` | 41 MB each | `matched_lcv_bins.py` | `correct_lcv_mean.py`, `target_bandpass.py` |
| Matched-control correlations before the IC-mean correction `matched_xi_LOS*_ph*.npz` | same | 12 x 260 KB | `matched_lcv_bins.py` | `correct_lcv_mean.py` (its outputs `icmean_xi_*` are bundled) |
| Pair-count text catalogues `dF_M1_f0.155.txt`, `dF_M1_f0.619.txt` and the P3D C++ tree | `/pscratch/sd/b/boryanah/Lyaf_p3d/` (6.2 GB); C++ in `power_3d_code/for_Boryana/P3D` next to the original tree | | `code/p3d/box_to_p3d.py`; build recipe `code/p3d/BUILD_NERSC.md` | `run_matched_pair_counts.py` (its outputs are bundled) |
| eBOSS DR16 correlation/covariance `cf_z_0_10-exp.fits` | `/global/cfs/projectdirs/desi/science/lya/eboss_dr16/public_dr16_correlations/` | | public eBOSS release | all Vega fits (weight matrix, divided by 20) |
| Lya x QSO study (`qso_cross_20260929/`, `qso_desi_20260930/` with `model_xi/` 1.1 GB) | `$SCR/` | 19 GB + 62 GB | collaborator-exchange scripts (full repository) | not part of the core tables; the four `fits_xi_Model*.npz` are bundled |
