# Headline results, the script and input behind each, and what cannot be re-run

All values below were read on 2026-10-18 from the products in `data/` (or from
the text summaries copied into `data/`, which were written by the summarize
scripts named here). Offsets and responses are in per cent; errors are
standard errors over the six phases after averaging LOSy and LOSz within each
phase, unless a row says "profile". Paths are relative to `data/`;
`EV = pscratch/Lyaf_recon/estimator_validation_20260908`,
`TC = pscratch/Lyaf_recon/template_closure_20260909`,
`MM = pscratch/Lyaf_recon/matched_model_20260909`,
`TT = pscratch/Lyaf_recon/target_transfer_20260908`.

## 1. Forest response and displacement fidelity (Table I / `tab:response`)

Product `pscratch/Lyaf_recon/forest_response_all_models.npz`
(`code/paper_prd/calibrate_all_forest_response.py`, keys `m*_b_f_mean`,
`m*_beta_f_mean`, `m*_r_all`, `m*_transfer_all`).

| Model | b_F | beta_F | r_Psi | T_Psi |
|---|---|---|---|---|
| 1 | -0.14807 +/- 1.1e-4 | 0.9577 +/- 3.3e-3 | 0.9871 | 0.9991 |
| 2 | -0.13094 +/- 9.3e-5 | 0.9896 +/- 3.2e-3 | 0.9866 | 0.9993 |
| 3 | -0.13245 +/- 9.0e-5 | 2.0747 +/- 5.6e-3 | 0.9900 | 1.0002 |
| 4 | -0.12852 +/- 1.0e-4 | 2.3955 +/- 6.5e-3 | 0.98935 | 1.0006 |

## 2. Regime I precision gain, reference model, LOSz (Table II)

Product `paper_prd_data/dense_results.npz` (`code/paper_prd/recompute_dense_results.py --los z`,
reading the 96 band powers in `pscratch/Lyaf_recon/`). Profile errors of the
six-box stack; damping in Mpc/h; `G = sigma_pre / sigma_post`.

| Model | (Sigma_par, Sigma_perp) pre | post | sigma(apar) pre -> post | sigma(aperp) pre -> post | G(apar) | G(aperp) |
|---|---|---|---|---|---|---|
| 1 | 4.61, 2.81 | 2.16, 1.36 | 0.00286 -> 0.00210 | 0.00150 -> 0.00125 | 1.363 | 1.203 |
| 2 | 4.62, 2.82 | 2.15, 1.38 | 0.00286 -> 0.00210 | 0.00151 -> 0.00126 | 1.364 | 1.201 |
| 3 | 4.81, 2.91 | 3.00, 2.11 | 0.00279 -> 0.00208 | 0.00146 -> 0.00122 | 1.340 | 1.197 |
| 4 | 4.86, 2.88 | 3.01, 2.03 | 0.00276 -> 0.00207 | 0.00146 -> 0.00121 | 1.336 | 1.198 |

Paired responses with common raw-pre weights (`EV/binned_pk_Model*_LOS*_o6_icmean.npz`,
variant `full3_raw_raw`; `code/paper_prd/summarize_estimator_validation.py`,
printed in `EV/adopted_summary.txt`): `100 Delta aperp = -0.074 +/- 0.087,
-0.075 +/- 0.086, -0.149 +/- 0.074, -0.152 +/- 0.075` (Models 1-4). The
alpha_iso / F_AP gains and the (apar, aperp) correlation of the same
polynomial model, Model 1 LOSz: `100 sigma(par, perp) 0.285/0.151 -> 0.208/0.125`,
`rho = -0.42`, `G(iso) 1.27`, `G(AP) 1.32` (`pscratch/Lyaf_recon/qso_desi_20260930/log_fom.txt`,
`code/from_roger/fom.py`).

## 3. Transverse scales with and without LCV (Table III) and fixed-weight tests (Table IV)

`EV/binned_pk_Model*_LOS{y,z}_o6_icmean.npz` (`code/paper_prd/refit_binned_pk.py --control-mean ic`
on the band powers of `paper_prd_data/lcv_reconstruction_results*.npz` and the
corrected LCV spectra `EV/icmean_pk_*`), summarised in `EV/adopted_summary.txt`.

| Model | raw pre aperp | raw post | LCV pre | LCV post | raw 100 Delta | LCV 100 Delta |
|---|---|---|---|---|---|---|
| 1 | 1.000248 | 0.999607 | 1.000908 | 1.000289 | -0.064 +/- 0.087 | -0.062 +/- 0.043 |
| 2 | 1.000227 | 0.999573 | 1.000862 | 1.000249 | -0.065 +/- 0.086 | -0.061 +/- 0.042 |
| 3 | 1.000413 | 0.999008 | 1.000478 | 0.999821 | -0.141 +/- 0.075 | -0.066 +/- 0.030 |
| 4 | 1.000475 | 0.999049 | 1.000628 | 0.999908 | -0.143 +/- 0.076 | -0.072 +/- 0.030 |

Controlled transverse response of Models 3 / 4 under fixed weights (same files,
variants `full3_cv_raw`, `full3_cv_cv`, `full4_cv_*`, `peak3_cv_*`):

| Template | raw-pre weights M3 / M4 | LCV-pre weights M3 / M4 |
|---|---|---|
| full dilation, 3 terms | -0.101 +/- 0.057 / -0.104 +/- 0.056 | -0.271 +/- 0.051 / -0.269 +/- 0.050 |
| full dilation, 4 terms | -0.043 +/- 0.063 / -0.046 +/- 0.064 | -0.128 +/- 0.042 / -0.129 +/- 0.041 |
| peak-only, 3 terms | -0.030 +/- 0.053 / -0.028 +/- 0.053 | -0.060 +/- 0.026 / -0.056 +/- 0.028 |

## 4. The submitted paper's baseline (spline) fits: bias and radial shift

`TC/spline_Model{1..4}_LOS{y,z}.npz` (`code/paper_prd/refit_spline_bao.py`), rows
`spline06_ic / cv / raw` of `TC/final_summary.md` (`summarize_template_closure.py`):
IC template, peak-only dilation, 0.06 h/Mpc splines, LCV, raw-pre weights.

| Model | 100(apar-1) pre | post | 100 Delta apar | 100(aperp-1) pre | post | 100 Delta aperp |
|---|---|---|---|---|---|---|
| 1 | +0.269 +/- 0.112 | +0.100 +/- 0.066 | -0.169 +/- 0.077 | +0.027 +/- 0.069 | +0.006 +/- 0.015 | -0.021 +/- 0.058 |
| 2 | +0.274 +/- 0.119 | +0.099 +/- 0.071 | -0.175 +/- 0.084 | +0.018 +/- 0.075 | -0.002 +/- 0.019 | -0.020 +/- 0.059 |
| 3 | +0.350 +/- 0.119 | +0.100 +/- 0.091 | -0.249 +/- 0.067 | +0.037 +/- 0.072 | 0.000 +/- 0.019 | -0.037 +/- 0.055 |
| 4 | +0.332 +/- 0.113 | +0.102 +/- 0.084 | -0.230 +/- 0.064 | +0.042 +/- 0.069 | +0.005 +/- 0.018 | -0.037 +/- 0.055 |

This is the "+0.3% -> +0.10 +/- 0.07%" radial statement and the
"transverse unbiased at 0.02%" statement. Model dependence (`tab:variants`,
Models 1 / 3; post `100(aperp-1)` and `100 Delta apar`), same files plus
`TC/spline_extra_*` for the Hinton de-wiggling: baseline `+0.006(15) / -0.17(8)`,
`0.000(19) / -0.25(7)`; knots 0.05 `+0.011(15) / -0.17(8)`, `-0.004(19) / -0.24(7)`;
knots 0.08 `+0.004(17) / -0.17(7)`, `-0.004(23) / -0.24(5)`; CLASS P_lin
`+0.024(16) / -0.17(8)`, `+0.018(20) / -0.25(7)`; polynomial P_nw `+0.004(16) / -0.16(8)`,
`-0.004(19) / -0.24(7)`; polynomial broadband `-0.007(17) / -0.22(8)`,
`-0.002(23) / -0.27(7)`; stress (LCV-pre) weights `+0.002(18) / -0.15(9)`,
`-0.028(24) / -0.19(9)`; largest offset of all variants `-0.063 +/- 0.019`
(Model 3, knots 0.08, stress weights). Known-scale validation
(`TC/spline_validation.npz`): smooth distortions move a 3-term polynomial fit by
up to 0.49% / 0.11%, the 0.06 spline by 0.0038% / 0.0304%, retaining >= 93% of
the dilation sensitivity.

## 5. Configuration space

* Bin-averaged, matched-control Vega fits of Model 1 (Table V;
  `EV/binned_xi_n{512,1152}_LOS*_o6{,_icmean}.npz`, `refit_binned_xi.py`):
  512^3 raw `(-0.048 +/- 0.092, -0.087 +/- 0.078)`, 512^3 LCV
  `(-0.198 +/- 0.080, -0.035 +/- 0.071)`, 1152^3 raw `(-0.031 +/- 0.096, -0.120 +/- 0.077)`,
  1152^3 LCV `(-0.186 +/- 0.080, -0.065 +/- 0.076)` for `(100 Delta apar, 100 Delta aperp)`;
  1152^3 LCV absolute offsets `+0.282 +/- 0.045 -> +0.096 +/- 0.083` (radial),
  `-0.059 +/- 0.081 -> -0.124 +/- 0.038` (transverse); no-Arinyo
  `(-0.176 +/- 0.072, -0.066 +/- 0.069)`, CLASS/no-Arinyo post transverse `-0.108 +/- 0.032`.
* Matched Fourier / configuration likelihoods, Model 1, 512^3, LCV (Table VI;
  `MM/fits_pk.npz`, `MM/fits_xi.npz`, `run_matched_model_fits.py`):

| Space | 100(aperp-1) before | after | Delta | 100(apar-1) before | after | Delta |
|---|---|---|---|---|---|---|
| Fourier | +0.021 +/- 0.066 | +0.001 +/- 0.014 | -0.020 +/- 0.055 | +0.269 +/- 0.112 | +0.106 +/- 0.064 | -0.164 +/- 0.080 |
| Configuration | -0.074 +/- 0.119 | -0.058 +/- 0.061 | +0.016 +/- 0.094 | +0.386 +/- 0.067 | +0.087 +/- 0.087 | -0.299 +/- 0.088 |

  (the submitted paper quotes the configuration row as `+0.39 +/- 0.07 -> +0.09 +/- 0.09`,
  `-0.07 +/- 0.12 -> -0.06 +/- 0.06`; the all-model repeat
  `pscratch/Lyaf_recon/qso_desi_20260930/model_xi/fits_xi_Model{1..4}.npz`,
  `code/from_roger/model_xi.py`, agrees for Model 1 to <= 0.002%).
* Common output band, 1152^3 target, 512^3 lattice, LCV (Table VII;
  `TT/fits_common_LOS*_b512_w0.35{,_nonl,_nonl_class}.npz`,
  `TC/ic_xi_LOS*_common_*.npz`; `TT/final_summary.txt`, `TC/final_summary.md`):
  archived spectrum + Arinyo `-0.198 +/- 0.079 -> -0.187 +/- 0.046` (paired `+0.011 +/- 0.064`);
  IC spectrum + Arinyo `-0.131 +/- 0.071 -> -0.097 +/- 0.044`; archived, no Arinyo
  `-0.061 +/- 0.077 -> -0.066 +/- 0.044`; CLASS, no Arinyo `-0.096 +/- 0.070 -> -0.067 +/- 0.043`;
  IC, no Arinyo `-0.089 +/- 0.069 -> -0.061 +/- 0.043`. Target-grid difference
  with the common band `+0.0010 +/- 0.0024` (LCV) versus `-0.0299 +/- 0.0100` native.

## 6. Sparse regimes

* Regime III at 50 sightlines/deg^2, painted FFT, Model 1 LOSz, six phases
  (`pscratch/Lyaf_recon/sparse_skewers_Model_1_calibrated_f0.155_seed1234.npz`,
  `code/sparse_skewers.py`; fit grid `paper_prd_data/sparse_stability.npz`):
  `(apar, aperp) = (1.0034, 0.9909) -> (1.0023, 0.9952)`, profile-error gains
  `1.4126` and `1.1409`; the transverse response ranges from +0.29 to +0.92%
  over the fit grid (methodology diagnostic, not a scale measurement).
* Density scan and Regime II (`pscratch/Lyaf_recon/sparse_final_fits.npz`,
  `code/sparse_final_fits.py`, k <= 0.35; `tab:sparse`), `G(apar) / G(aperp)`:

| Displacement source | 50 /deg^2 | 161 /deg^2 |
|---|---|---|
| sightlines (Regime III) | 1.413 / 1.141 | 1.493 / 1.221 |
| halos, n = 5.7e-3 | 1.412 / 1.125 | 1.437 / 1.185 |
| halos, n = 1e-3 | 1.385 / 1.114 | 1.411 / 1.177 |
| halos, n = 1e-4 | 1.176 / 1.032 | 1.210 / 1.080 |

  Self-reconstruction versus density: 323 /deg^2 `1.379 / 1.195`, 161 `1.493 / 1.221`,
  81 `1.470 / 1.163`, 50 `1.413 / 1.141`, 26 `1.253 / 1.097` (mean separation
  3.9, 5.5, 7.8, 9.9, 13.8 Mpc/h). chi^2/dof of the stacks: 50 /deg^2
  `363.6/308 -> 396.9/308`, 161 /deg^2 post `446.3/308`; most conservative
  161 /deg^2 variant (`sqrt(chi^2/dof)`-rescaled) `1.306 / 1.068`. Per-phase
  responses at 161 /deg^2 (`regime2_regime3_sparse_fits.npz`, `log_regime2_fit.txt`):
  `100 Delta aperp = +0.088 +/- 0.076, +0.060 +/- 0.072, +0.045 +/- 0.092,
  +0.089 +/- 0.126`.
* Exact matched pair counts at 50 /deg^2 (Table VIII;
  `paper_prd_data/sparse_pair_matched{,_seed5678,_combined}_results.npz`,
  `code/p3d/fit_matched_reconstruction.py`, `combine_matched_masks.py`; raw
  multipoles in `pscratch/Lyaf_p3d/matched*/`):

| Mask | pre apar | post apar | 100 Delta apar | pre aperp | post aperp | 100 Delta aperp |
|---|---|---|---|---|---|---|
| 1234 | 1.00175 | 0.99933 | -0.242 +/- 0.407 | 0.99399 | 0.99927 | +0.528 +/- 0.236 |
| 5678 | 0.99461 | 0.99424 | -0.038 +/- 0.292 | 1.00008 | 1.00004 | -0.003 +/- 0.138 |
| phase-wise average | 0.99818 | 0.99678 | -0.140 +/- 0.133 | 0.99703 | 0.99966 | +0.262 +/- 0.174 |

## 7. Smoothing radius (Model 1, phase 000, LOSz)

`pscratch/Lyaf_recon/scan_R_Model_1_LOSz_ph000_calibrated_ext.npz`
(`code/scan_smoothing.py --driver lya --sr 4 6 8 10 12.5 15 20`; `log_scan_ext.txt`):
`G(apar) / G(aperp) = 1.40/1.23 (R=4), 1.45/1.22 (6), 1.50/1.20 (8), 1.49/1.19 (10),
1.46/1.17 (12.5), 1.42/1.16 (15), 1.36/1.14 (20)`; for `R <= 8` the fitted radial
damping of the reconstructed field is at its zero bound, so `R = 10 Mpc/h` is
the smallest radius with both damping scales resolved.

## 8. Numerical validation numbers quoted in the papers

Node-window closure rms `1.5e-7` over 8.48e6 modes, phase rms `5.2e-7` rad
(`code/paper_prd/check_node_window_closure.py`); known-dilation recovery
`0.00011%` Fourier (`EV/fourier_dilation_validation.npz`), `0.000006%` xi
(`EV/xi_dilation_validation.npz`), `< 0.002%` matched likelihood
(`MM/validation_{pk,xi}.npz`); painter versus analytic TSC stencil `7.9e-6`,
translation offsets up to 0.043% at 3.9 Mpc/h and < 0.0017% at 1.7 Mpc/h,
sinusoidal flow `1.19% -> 0.11%` (`TT/*_check.npz`, `paint_translation_validation.npz`,
`common_target_validation.npz`); common-band known scales recovered within
0.0002% (24 inputs). The two-term broadband artefact `+0.699 +/- 0.257%` and
the 3/4/5-term values `+0.170, +0.038, +0.041%` are in
`pscratch/Lyaf_recon/calibrated_reconstruction_checks.npz`
(`check_calibrated_reconstruction.py`).

## 9. Verification performed when this bundle was built (2026-10-18)

`python tools/reproduce_gains.py` (cosmodesi environment, two BLAS threads,
about 20 s in total) re-ran the two headline gain measurements from the data
in this bundle, using the production functions imported from
`code/paper_prd/recompute_dense_results.py` and
`code/paper_prd/check_sparse_stability.py` with the pscratch root pointed at
`data/`:

* **Dense Regime I, Model 1, LOSz** (six-box stack, reference model, profile
  errors): pre `apar = 0.999690`, `aperp = 0.999869`, `sigma = 0.002857 / 0.001499`,
  `Sigma = (4.607, 2.808)`; post `apar = 1.000795`, `aperp = 0.999134`,
  `sigma = 0.002097 / 0.001246`, `Sigma = (2.164, 1.355)`. Every value agrees
  with `paper_prd_data/dense_results.npz` to better than 1e-6 (largest
  difference 9e-7 in a damping scale, 1e-12 in the errors). **Gains 1.3626 and
  1.2030** (stored 1.3626 / 1.2030; paper Table II 1.363 / 1.203).
* **Sparse Regime III, 50 sightlines/deg^2, Model 1, LOSz** (six-box stack with
  the state-specific `P_1D` alias column, `k <= 0.35`, 3 broadband terms,
  damping priors `(4.61, 2.81) / (2.16, 1.36)` of width 1 Mpc/h, all read from
  the provenance fields of `sparse_skewers_Model_1_calibrated_f0.155_seed1234.npz`):
  pre `(1.003376, 0.990858)`, `sigma = 0.004777 / 0.003486`, `Sigma = (4.73, 3.11)`;
  post `(1.002348, 0.995164)`, `sigma = 0.003382 / 0.003056`, `Sigma = (2.00, 1.82)`;
  all eight stored values reproduced to six decimals. **Gains 1.4126 and
  1.1409** (stored 1.4126 / 1.1409; `sparse_final_fits.npz` `f0.155_III_self_gain`
  1.4126 / 1.1409; paper 1.41 / 1.14).

Also checked: `python -m py_compile` on all 62 `.py` files in `code/` and
`tools/` (pass); `tools/check_doc_paths.py` on the four Markdown files (all
repository paths exist); `tools/relocate_paths.sh` on a scratch copy (no
absolute NERSC prefix left in `code/`, link created); zip round trip (see the
hand-off report). The figure PDFs, manuscripts and notes were deliberately not
copied; the full repository `lyaf_recon_paper` has them.

## 10. What cannot be re-run from this bundle, and why

1. **Fields.** Nothing that reads a `512^3`/`1152^3` field or the public
   `complex_dF` ASDF files: `recon_paint_field.py`, `make_dF_grid.py`,
   `measure_pkmu.py` on fields, `sparse_skewers.py` (field stage),
   `regime2_ensemble.py`, `scenario2_external_recon.py`, `scan_smoothing.py`,
   `build_halo_catalog.py`, the LCV builders (`lcv_reconstruction_check.py`,
   `multimodel_losy_lcv.py`, `matched_lcv_bins.py`, `correct_lcv_mean.py`,
   `ic_reference_spectrum.py` need the IC files and 41 MB bin-kernel caches),
   `measure_*_xirppi.py`, `measure_common_target.py`, `measure_matched_model.py`,
   `calibrate_*_forest_response.py`, `check_calibrated_reconstruction.py`.
   Their outputs are bundled; the inputs are listed in `HEAVY_INPUTS.md`.
2. **Vega fits** (`fit_lcv_xirppi_vega.py` and everything importing it,
   `run_matched_model_fits.py --space xi`, `model_xi.py`, `xi_profile.py`)
   need the Vega environment and the eBOSS DR16 correlation file; the bundled
   `xi(r_par, r_perp)` measurements and fit products allow the fits to be
   repeated on NERSC but not offline.
3. **The submitted paper's precision table** (`tab:precision`: spline-model
   six-box stack errors, damping `4.7, 3.2 -> 2.2, 2.1`, `G = 1.38/1.20/1.26/1.32`
   for Model 1 and the Models 2-4 rows, the LOSy gain ranges, and the
   alpha_iso / F_AP columns of `tab:sparse`) came from scratchpad scripts of the
   2026-09-26 review that were not archived; no spline-model stack fit exists on
   disk. The nearest archived numbers are Table II above (polynomial model,
   `1.363/1.203`) and `log_fom.txt` (`1.27/1.32`).
4. **Pair counts**: the C++ `power_periodic` estimator is not included (build
   recipe in `code/p3d/BUILD_NERSC.md`); its outputs are, so Table VIII refits.
5. The Lya x QSO study, the figure scripts and the manuscripts are in the full
   repository, not here.
