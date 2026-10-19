# Lyman-alpha forest BAO reconstruction on AbacusSummit boxes: core code and light data

Supplementary code for *Improving cosmological distance measurements at
high-redshift using Lyman-alpha forest BAO reconstruction* (B. Hadzhiyska and
R. de Belsunce) and its long validation draft. This is the lean companion of
the full reproducibility repository `lyaf_recon_paper` (same authors' analysis
tree, assembled 2026-10-18): it keeps the analysis code, the job scripts and
the light products needed to re-run the fits and the precision-gain
measurements, and leaves out manuscripts, figures, notes and the heavy fields.
Every number quoted here was read from a product in `data/` or from the
summaries copied into it; `RESULTS.md` lists them table by table with the
script and input that produce each row.

```
lyaf_recon_core/
  README.md        this file: scenarios, method, measurement, fits, gain, environment
  RESULTS.md       the headline tables as plain numbers + what cannot be re-run
  HEAVY_INPUTS.md  the fields and catalogues that are not bundled (paths, sizes, producers)
  code/            analysis code (flat modules), jobs/ (Slurm), paper_prd/ (table-producing fit
                   scripts + run_*.sh drivers + Vega configs), p3d/ (pair-count wrappers),
                   from_roger/ (all-model xi fit, xi profile-likelihood gain, FoM), templates/ (CLASS cache)
  data/            507 light products (31.7 MB) with MANIFEST.json (repo path -> original path -> sha256)
  tools/           reproduce_gains.py (re-runs the two headline gains), relocate_paths.sh, check_doc_paths.py
  environment/     the two Python environments (pip freezes) and the exact source commands
```

## (a) The scenarios (regimes) and which code realises them

The study uses the six public AbacusLymanAlpha boxes (AbacusSummit base
cosmology, phases 000-005, z = 2.5, L = 2000 Mpc/h, 6912^3 particles), each
with four forest prescriptions (Models 1-4) and two lines of sight (LOSz,
LOSy). All regimes work on a 512^3 mesh of the flux fluctuation `delta_F`
(cell 3.906 Mpc/h) built from the public Fourier-space fields; Model 1 also at
1152^3 for the configuration-space checks.

| Regime | What is moved / what supplies the displacement | Code | Products in `data/` |
|---|---|---|---|
| **I. Dense (continuous) self-reconstruction** | every mesh cell of `delta_F` is a sample; the smoothed forest field itself gives the displacement | `code/recon_paint_field.py --mode ref` / `--mode recon --driver lya`, run by `code/jobs/run_recon.slurm` (Model 1) and `run_models.slurm` (Models 2-4); band powers by `code/measure_pkmu.py` | `pscratch/Lyaf_recon/dF_{ref,recon}_Model_*_LOS*_...pkmu_dk0.005_kmax0.5_nmu5[_winnode].npz` (96 files), `paper_prd_data/dense_results{,_LOSy}.npz` |
| **II. External displacement (survey-like, DESI-II-style tracer)** | the forest pixels on a sparse sightline mask are moved by a displacement solved from redshift-space CompaSO halos (N >= 150 particles, n = 5.7e-3 h^3/Mpc^3, b = 2.7), randomly thinned to 1e-3 and 1e-4 h^3/Mpc^3 | `code/build_halo_catalog.py`, `code/scenario2_external_recon.py` (solver), `code/regime2_ensemble.py` (six phases), `code/regime2_fit.py`, run by `code/jobs/run_regime2.slurm` / `run_final_runs.slurm` | `dF_exrec_f{0.155,0.500}_..._halo{full,n1e-03,n1e-04}_...{pkmu,p1d}.npz`, `regime2_regime3_sparse_fits.npz`, `sparse_final_fits.npz` |
| **III. Sparse self-reconstruction (DESI-like sightlines)** | a fixed random subset of transverse mesh columns (fraction 0.155 = 50 sightlines/deg^2, mean separation 9.9 Mpc/h, 40,632 sightlines; also 0.08/0.25/0.5/1.0 = 26/81/161/323 per deg^2) supplies both the samples and the displacement | `code/sparse_skewers.py` (mask, solve with the observed columns as data and randoms, P_1D alias template, stacked fits), run by `code/jobs/run_skewers.slurm` / `run_density_scan.slurm`; density-scan refit `code/sparse_final_fits.py`; fit grid `code/paper_prd/check_sparse_stability.py` | `dF_{skref,skrec}_f*_...{pkmu,p1d}.npz`, `sparse_skewers_Model_1_calibrated_f0.155_seed1234.npz`, `sparse_skewers_Model_1_LOSz_6phases_calibrated.npz`, `sparse_final_fits.npz`, `paper_prd_data/sparse_stability.npz` |
| Exact same-pixel pair counts (alias-free check of III) | same pixels, fluxes, weights and sightline ids before/after; only coordinates change; two masks (seeds 1234, 5678) | `code/p3d/run_matched_pair_counts.py` (+ external C++ `power_periodic`, `code/p3d/BUILD_NERSC.md`), `fit_matched_reconstruction.py`, `combine_matched_masks.py` | `pscratch/Lyaf_p3d/matched*/`, `paper_prd_data/sparse_pair_matched*.npz` |
| Lya x QSO at survey densities | discussed only in the collaborator report; not in the core tables | not included (`code/from_roger/` holds only the xi fitters used for the tables) | - |

## (b) The reconstruction, step by step as the code does it

1. **Forest response calibration** (`code/paper_prd/calibrate_all_forest_response.py`,
   `calibrate_forest_response.py`): the cross spectrum of `delta_F` with the
   phase-matched initial density `delta_L` (AbacusSummit IC file), fitted as
   `R(k,mu) = b_F + b_F beta_F <mu^2>` over `0.005 <= k < 0.05 h/Mpc` with the
   mode-weighted `<mu^2>` of each wedge, jointly over the six phases. Result
   (`data/pscratch/Lyaf_recon/forest_response_all_models.npz`): `b_F =
   -0.14807, -0.13094, -0.13245, -0.12852` and `beta_F = 0.9577, 0.9896, 2.0747,
   2.3955` for Models 1-4 (statistical errors 1e-4 and 3e-3 to 7e-3). These
   values are the `FOREST_RESPONSE` table in `code/recon_paint_field.py` and are
   also encoded in the output file names (`bF-0.148066_bt0.957688`, ...).
2. **The `delta_F` grid** (`code/make_dF_grid.py`): the public
   `complex_dF_Model_m_LOS{y,z}.asdf` (real FFT of the 6912^3 field, low-k
   block) is cropped to the modes of the 512^3 target mesh, its Nyquist planes
   zeroed, the upper 10% of the retained band smoothly tapered, and inverse
   transformed with the factor `(512/6912)^3`. LOSy is stored as
   `(kx, ky_half, kz)` and is transposed `(0, 2, 1)` so that the last axis is
   the line of sight in both cases (the IC density receives the same rotation).
3. **Smoothing and displacement solve** (`code/recon_paint_field.py`,
   `build_recon_lya`): the unit-bias field `1 + delta_F / b_F` (negative cells
   are fine for a periodic linear solve) is passed to pyrecon's multigrid solver
   (`MultiGridReconstruction`, `--rectype MG`, 512^3 reconstruction mesh,
   periodic box) with the Gaussian smoothing `S(k) = exp(-k^2 R^2 / 2)`,
   `R = 10 Mpc/h` (`--sr 10`), and with the forest anisotropy `beta_F` in place
   of pyrecon's `f` so that the solver inverts
   `div Psi + beta_F div[(Psi.z) z] = -delta_F / b_F`. The real-space `Psi` is
   read back from the solver.
4. **Redshift-space displacement (RecSym convention)**: the samples are moved by
   `Psi_s = Psi + f (Psi.z) z` with the cosmological growth rate `f = 0.9702` at
   z = 2.5 (not `beta_F`), from `x` to `y = x - Psi_s(x)`.
5. **Repainting the flux** (`paint` in `recon_paint_field.py`): each mesh cell
   carries its scalar `delta_F`; the moved samples are TSC-painted and divided
   by the TSC paint of the same moved samples without weights
   (`--norm local`, the production choice): `delta_F_rec(y) = sum_i delta_F(x_i)
   W(y - y_i) / sum_i W(y - y_i)`. The unweighted denominator plays the role of
   shifted randoms. **No smoothing is applied to the displaced field**; the
   Gaussian kernel enters only the displacement solve. `--norm global` keeps the
   Jacobian instead (used as a check: Model 1 response `(+0.24 +/- 0.23,
   -0.02 +/- 0.10)%`).
6. **Reference field**: `--mode ref` runs the same paint with zero displacement
   (samples exactly on mesh nodes), so pre and post fields share the pipeline.
7. **Sparse variants**: Regime III passes the selected columns to the solver as
   both data and randoms (the solve divides by the selection, so unobserved
   columns are not read as voids) and keeps the global mean in the painted
   field's denominator; Regime II solves the standard biased-tracer equation on
   the halo catalogue and reads `Psi` at the forest pixels. Same `R = 10
   Mpc/h`, same mesh, same painting.

## (c) The measurement

* **`P(k, mu)`** (`code/measure_pkmu.py`, cached by `fit_bao_pkmu.measure`):
  FFT of the grid, five equal `mu = |k_par|/k` wedges, `dk = 0.005 h/Mpc`, power
  kept to `0.5 h/Mpc`, divided by the squared TSC assignment window:
  node-centred `W_node(m) = 3/4 + cos(2 pi m/N)/4` per axis for the unshifted
  reference (`--window node`, cache suffix `_winnode`), continuous
  `sinc^3(m/N)` for the displaced field (`--window continuous`). Every cache
  records window, compensation state and source. The Gaussian weight of the
  six-box mean is `Var P = 2 P^2 / (N_box N_mode)`.
* **Sparse estimator term**: for complete random sightlines
  `P_obs = P_3D + P_1D(k_par) / n_2D`; the sparse FFT fits add the measured
  `P_1D(k_par)` of each state (phase-averaged, assignment-corrected) as a
  free-amplitude column (`sparse_skewers.measure_p1d`, `.p1d_*.npz` caches).
  The pair-count estimator excludes same-sightline pairs instead.
* **Configuration space** (`code/paper_prd/measure_lcv_xirppi.py`,
  `highres_two_orientation_xirppi.py`, `measure_common_target.py`,
  `measure_matched_model.py`): `xi(r_par, r_perp)` from the Fourier field in
  4 Mpc/h cylindrical bins to 200 Mpc/h (exact lattice-bin kernels; optional
  common low-pass `W(k)` from 0.35 to 0.5 h/Mpc), at 512^3 and 1152^3.
* **Linear control variate (LCV)** (`lcv_reconstruction_check.py`,
  `multimodel_losy_lcv.py`, `matched_lcv_bins.py`, `ic_reference_spectrum.py`,
  `correct_lcv_mean.py`): the linear forest `delta_F^L = b_F (1 + beta_F mu^2)
  delta_L` from the phase's IC density; `P_CV = P_F - c (P_L - <P_L>)` with
  `c = (P_FL/P_LL)^2` from the six phases (1 below 0.05 h/Mpc, tapered where the
  linear field decorrelates; held-out coefficients for the xi fits); the
  expectation `<P_L>` is the IC generating spectrum times
  `(ZD_Pk_sigma_ratio x applied growth)^2`, not a separate Boltzmann spectrum
  (2.4% different). Applied separately before and after reconstruction; the
  expectation of the spectrum is unchanged, so LCV cannot hide a shift.

## (d) The fits

* **Reference Fourier model** (`code/fit_bao_pkmu.py`, `PkmuModel`/`PkmuFitter`;
  template `code/template_pk.py` = CLASS linear spectrum for the AbacusSummit
  cosmology with a Savitzky-Golay no-wiggle companion, cached in
  `code/templates/pk_AbacusSummit_savgol_z2.500.npz`, `r_d = 99.084 Mpc/h`):
  `P = b^2 (1 + beta mu'^2)^2 [P_nw + (P_lin - P_nw) e^{-D/2}](k')/(apar aperp^2)
  + sum_j a_wj k^-j (+ A_alias P_1D)`, `D = k'^2 [mu'^2 Sigma_par^2 + (1 - mu'^2)
  Sigma_perp^2]`, AP mapping `k'_par = k mu/apar`, `k'_perp = k (1-mu^2)^0.5/aperp`;
  nonlinear parameters `apar, aperp, beta, Sigma_par, Sigma_perp` (bounds
  0.8-1.2, 0-5, 0-20 Mpc/h), amplitude and three broadband terms per wedge
  solved by weighted least squares at every step; fit range `0.02 <= k <= 0.40
  h/Mpc` (0.35 for the sparse density scan); optional Gaussian priors on the
  damping (sparse fits: pre `(4.61, 2.81)`, post `(2.16, 1.36)`, width 1 Mpc/h).
  Errors: one-dimensional profile likelihood, `Delta chi^2 = 1`, with the other
  nonlinear parameters re-optimised and the linear ones re-solved at each point.
* **Bin-averaged refits** (`estimator_bins.py`, `fit_binned_pk.py`,
  `refit_binned_pk.py`): the model and the broadband columns are averaged over
  the actual periodic modes of each `(k, mu)` bin (six-node rule), for raw and
  LCV spectra under state-specific, raw-pre and LCV-pre weights, with full or
  peak-only dilation and 3 or 4 broadband terms.
* **Baseline spline model of the submitted paper** (`spline_bao_fit.py`,
  `ic_bao_template.py`, `refit_spline_bao.py`, `refit_spline_extra.py`): the
  spectrum that generated the ICs as template, peak-only dilation, an
  independent cubic-spline broadband per wedge with knot spacing `0.06 h/Mpc`
  (0.05 and 0.08 as variants), bin averaging, LCV, fixed raw pre-reconstruction
  weights for both states (`cv / raw` rows of the summaries) or the
  stress-test LCV-pre weights (`cv / cv`).
* **Configuration space** (`fit_lcv_xirppi_vega.py` -> `fit_binned_xi.py`,
  `refit_binned_xi.py`, `fit_common_target.py`, `refit_common_target.py`,
  `refit_ic_xi.py`; `matched_model_fit.py` + `run_matched_model_fits.py`;
  `from_roger/model_xi.py`, `xi_profile.py`): Vega fits of `xi(r_par, r_perp)`
  over `30 < r < 180 Mpc/h` weighted by the public eBOSS DR16 covariance
  divided by 20 (a fixed weight, not a calibrated covariance), either with the
  twelve polynomial broadband terms `r^{-2,-1,0} x L_{0,2,4,6}` (reference,
  Arinyo correction on or off) or with the Hankel transform of the spline basis
  (matched likelihood); bin-averaged over the lattice separations. These need
  the Vega environment and the eBOSS file (`environment/README.md`).

## (e) The precision gain and the paired response

* **Gain** `G = sigma_pre / sigma_post`, where `sigma` is the profile-likelihood
  error of a single fit to the mean spectrum of the six boxes (the combined
  noiseless volume of 48 (Gpc/h)^3), each state with its own Gaussian variance.
  It measures the sharpening of the acoustic feature, not the box-to-box
  scatter. For the sparse regimes the same stack fit carries the alias column
  and the damping priors; variants without the prior, with `k <= 0.30`, and
  with errors rescaled by `sqrt(chi^2/dof)` are stored next to the central gain.
* **Paired response** `Delta alpha = alpha_post - alpha_pre` per phase, both
  states fitted with the same (raw pre-reconstruction) weights; LOSy and LOSz
  are averaged within each phase first, and the quoted error is the standard
  error over the six phases (never twelve).
* **Commands that produce the headline numbers** (from `code/`, with
  `PYTHONPATH=$PWD:$PWD/paper_prd`; the drivers hard-code the original NERSC
  paths, run `tools/relocate_paths.sh` once to point them at `data/`):

```bash
# Regime I, Table II: damping scales, profile errors, gains, paired responses (reads the 96 band powers)
python paper_prd/recompute_dense_results.py --los z        # -> paper_prd/data/dense_results.npz
python paper_prd/recompute_dense_results.py --los y
# Regime III at 50/deg^2 (the production run; needs the fields) and the k<=0.35 density scan / Regime II table
python sparse_skewers.py --model 1 --nmesh 512 --nmesh_recon 512 --phases 000 001 002 003 004 005 \
       --frac 1.0 0.5 0.25 0.155 0.08 --alias --sig_width 1.0 --sig_pre 4.61 2.81 --sig_post 2.16 1.36
python regime2_fit.py && python sparse_final_fits.py        # -> sparse_final_fits.npz (tab:sparse, fig06)
python paper_prd/check_sparse_stability.py                  # fit grid at 50/deg^2 (from the cached spectra)
# LCV, bin-averaged Fourier fits (Tables III-V) and the spline baseline (Fig. 4 / tab:variants / radial shift)
bash paper_prd/run_icmean_pk_refits.sh && python paper_prd/summarize_estimator_validation.py
for m in 1 2 3 4; do for l in y z; do python paper_prd/refit_spline_bao.py --model $m --los $l; \
    python paper_prd/refit_spline_extra.py --model $m --los $l; done; done
python paper_prd/summarize_template_closure.py
# configuration space (Vega python): Table V/VII refits and the matched likelihoods (Table VI, Model 1 xi numbers)
bash paper_prd/run_icmean_xi_refits.sh z; bash paper_prd/run_icmean_xi_refits.sh y
bash paper_prd/run_matched_measurements.sh; $VEGA_PY paper_prd/run_matched_model_fits.py --space xi --validate; \
    $VEGA_PY paper_prd/run_matched_model_fits.py --space xi
# exact matched pair counts (Table VIII) from the bundled pair-count multipoles
python p3d/fit_matched_reconstruction.py --root ../data/pscratch/Lyaf_p3d/matched --seed 1234 --out m1234.npz
python p3d/fit_matched_reconstruction.py --root ../data/pscratch/Lyaf_p3d/matched_seed5678 --seed 5678 --out m5678.npz
python p3d/combine_matched_masks.py m1234.npz m5678.npz --out combined.npz
# the two re-runs performed when this bundle was built (needs only data/ and code/templates)
python tools/reproduce_gains.py            # dense Model 1 LOSz gains 1.363 / 1.203; sparse 50/deg^2 gains 1.4126 / 1.1409
```

What each re-run checks is recorded in `RESULTS.md` ("Verification").

## (f) Environment and heavy inputs

* Fourier pipeline, fits, re-runs: `source
  /global/common/software/desi/users/adematti/cosmodesi_environment.sh main`
  (Python 3.12.9, numpy 2.4.3, scipy 1.17.1, pyrecon 1.0.0, pmesh 0.1.58,
  cosmoprimo 1.0.0, asdf 5.2.0, abacusutils 2.1.3; full list
  `environment/cosmodesi_main_pip_freeze.txt`). The bundled re-runs need only
  numpy and scipy plus the cached CLASS template.
* Configuration-space fits: `/global/common/software/desi/users/acuceu/vega_master/bin/python`
  (Python 3.11.13, vega 1.8.1.dev5, picca 9.21.4, iminuit 2.33.0;
  `environment/vega_master_pip_freeze.txt`) and the eBOSS DR16 file
  `/global/cfs/projectdirs/desi/science/lya/eboss_dr16/public_dr16_correlations/cf_z_0_10-exp.fits`.
* Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for the fits.
* Heavy inputs (public flux fields 13.9 GB each, IC densities 3.2 GB each, the
  512^3/1152^3 fields, halo catalogues, pair-count catalogues) are listed with
  sizes and producing scripts in `HEAVY_INPUTS.md`; the batch scripts in
  `code/jobs/` regenerate the fields (about 18.5 GiB and under two hours per
  phase on one Perlmutter CPU node for the 512^3 production).
* Layout note: the modules in `code/` import each other by bare name and the
  `paper_prd/` scripts add `..` to `sys.path` themselves, so run from `code/`
  with `PYTHONPATH=$PWD:$PWD/paper_prd`. `tools/relocate_paths.sh` rewrites the
  two absolute NERSC prefixes inside `code/` to this bundle and creates the
  link `paper_prd/data` (inside `code/`) to `data/paper_prd_data`; `tools/reproduce_gains.py`
  works without it.
