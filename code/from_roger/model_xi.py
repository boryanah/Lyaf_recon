#!/usr/bin/env python3
"""Configuration-space BAO fits for any forest model (Model 3 requested).

Reproduces the paper's configuration-space measurement and fit for Model 1
(paper_prd/measure_common_target.py -> run_matched_model_fits.py --space xi)
for another forest model:

  coef     leave-one-phase-out monopole LCV coefficients, from the existing
           512^3 band powers of that model (lcv_reconstruction_calibrated_*):
           raw = (cross / control)^2 of the k-bin monopole, tanh taper at
           0.618 h/Mpc, unity below 0.05, Savitzky-Golay smoothing, clipped to
           [0, 2] -- the recipe of highres_two_orientation_xirppi.
           phase_coefficient, on the 512 band-power bins instead of the 1152
           fundamental-mode bins.  The coefficient affects only the variance
           of the control variate, not its mean.
  measure  xi(r_par, r_perp) in 4 Mpc/h bins from the 512^3 reference and
           reconstructed fields of that model, raw and with the control
           variate (exact IC mean, the model's b_F and beta_F), with the same
           0.35 low-pass filter and exact bin kernels as the paper
           (key w0.35_b512_n512_<state>_<raw|lcv>).
  fit      the paper's xi fitter (matched_model_fit.xi_fitter: IC template,
           spline broadband, eBOSS DR16 / 20 weight) per phase and LOS.

Model 1 is rerun with the same 512 coefficients as a check of the
substitution against the paper's numbers.
"""
import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

CLEAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN))
sys.path.insert(0, str(CLEAN / "paper_prd"))

SCR = Path("/pscratch/sd/b/boryanah/Lyaf_recon")
OUT = SCR / "qso_desi_20260930" / "model_xi"
RESP = {1: "bF-0.148066_bt0.957688", 2: "bF-0.130939_bt0.989632",
        3: "bF-0.132453_bt2.074689", 4: "bF-0.128523_bt2.395469"}
PHASES = [f"{i:03d}" for i in range(6)]


def lcv_rows(model, los):
    rows = []
    for p in PHASES:
        if los == "z":
            d = np.load(SCR / f"lcv_reconstruction_calibrated_ph{p}.npz")
            rows.append({key: d[f"m{model}_{key}"] for key in ("control", "cross_pre", "cross_post")}
                        | {"k": d["k"], "N": d["N"]})
        else:
            tag = "" if model == 1 else f"Model{model}_"
            d = np.load(SCR / f"lcv_reconstruction_calibrated_{tag}LOSy_ph{p}.npz")
            rows.append({key: d[key] for key in ("control", "cross_pre", "cross_post", "k", "N")})
    return rows


def coefficients(model):
    from scipy.signal import savgol_filter
    OUT.mkdir(parents=True, exist_ok=True)
    for los in "zy":
        rows = lcv_rows(model, los)
        k = np.nanmean(np.where(rows[0]["N"] > 0, rows[0]["k"], np.nan), axis=1)
        n = rows[0]["N"]
        raw = {st: [] for st in ("pre", "post")}
        for r in rows:
            for st in raw:
                x = np.sum(n * r[f"cross_{st}"], axis=1) / np.maximum(n.sum(1), 1)
                c = np.sum(n * r["control"], axis=1) / np.maximum(n.sum(1), 1)
                raw[st].append(np.square(x / np.maximum(c, 1e-30)))
        for i, p in enumerate(PHASES):
            out = {"coefficient_k": k, "growth": 27.59212449891135,
                   "coefficient": np.array("leave-one-phase-out 512 band-power monopole")}
            for st in raw:
                mean = np.mean([raw[st][j] for j in range(len(PHASES)) if j != i], axis=0)
                coef = mean * 0.5 * (1.0 - np.tanh((k - 0.618) / 0.167))
                coef[k < 0.05] = 1.0
                coef = np.clip(savgol_filter(coef, 13, 3), 0.0, 2.0)
                out[f"{st}_coefficient"] = coef
            np.savez(OUT / f"coef_Model{model}_LOS{los}_ph{p}.npz", **out)
    print("coefficients written for Model", model)


def measure(model, phase, los, workers=4):
    from target_bandpass import (IMAX, MODES, compact_modes, pass_power, plane,
                                 add_plane, load_kernels, save_atomic)
    from highres_two_orientation_xirppi import linear_modes
    from recon_paint_field import FOREST_RESPONSE
    from ic_reference_spectrum import spectrum
    dest = OUT / f"xi_Model{model}_LOS{los}_ph{phase}.npz"
    if dest.exists():
        print("exists", dest, flush=True)
        return
    begin = time.monotonic()
    prior = np.load(OUT / f"coef_Model{model}_LOS{los}_ph{phase}.npz")
    spec = importlib.util.spec_from_file_location(
        "lyaf_abacus_asdf", "/global/u1/b/boryanah/.local/perlmutter/python-3.11/lib/"
        "python3.11/site-packages/abacusnbody/data/asdf.py")
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    import asdf
    ext.set_nthreads(workers)
    asdf.get_config().add_extension(ext.AbacusExtension())
    delta = linear_modes(phase, los, 512, float(prior["growth"]), workers)
    linear = np.ascontiguousarray(delta[np.ix_(MODES % 512, MODES % 512, np.arange(IMAX + 1))])
    del delta
    gc.collect()
    sim = f"AbacusSummit_base_c000_ph{phase}"
    pre = compact_modes(SCR / f"dF_ref_Model_{model}_LOS{los}_nmesh512_{sim}_z2.500.npy",
                        "node", workers)
    post = compact_modes(SCR / (f"dF_recon_Model_{model}_LOS{los}_nmesh512_R10.00_nr512_MG_"
                                f"{RESP[model]}_lyadrv_{sim}_z2.500.npy"), "continuous", workers)
    kernels = load_kernels(512)
    reference = spectrum()
    bf, beta = FOREST_RESPONSE[model]
    out = {f"w0.35_b512_n512_{st}_{kind}": np.zeros((50, 50))
           for st in ("pre", "post") for kind in ("raw", "lcv")}
    for iz in range(IMAX + 1):
        k, mu2 = plane(iz)
        good = (k > 0) & (k < 0.5)
        response = bf * (1 + beta * mu2)
        mean = np.zeros_like(k)
        mean[good] = reference.plin(k[good]) * response[good] ** 2 / 2000.0 ** 3
        residual = np.abs(linear[:, :, iz] * response) ** 2 - mean
        window = np.where(good, pass_power(k, 0.35), 0.0)
        for st, modes in (("pre", pre), ("post", post)):
            q = np.interp(k, prior["coefficient_k"], prior[f"{st}_coefficient"],
                          left=prior[f"{st}_coefficient"][0], right=0.0)
            target = np.abs(modes[:, :, iz]) ** 2
            for kind in ("raw", "lcv"):
                power = window * (target if kind == "raw" else target - q * residual)
                add_plane(out[f"w0.35_b512_n512_{st}_{kind}"], power, kernels, iz)
    out["elapsed_seconds"] = time.monotonic() - begin
    save_atomic(dest, **out)
    print("COMPLETE", dest, round(out["elapsed_seconds"]), "s", flush=True)


def fit(model):
    from matched_model_fit import xi_fitter
    fitter = xi_fitter()
    res = {}
    for los in "zy":
        for st in ("pre", "post"):
            for kind in ("raw", "lcv"):
                fits = [fitter.fit(np.load(OUT / f"xi_Model{model}_LOS{los}_ph{p}.npz")
                                   [f"w0.35_b512_n512_{st}_{kind}"]) for p in PHASES]
                for key in ("ap", "at", "chi2"):
                    res[f"LOS{los}_{st}_{kind}_{key}"] = np.array([f[key] for f in fits])
    np.savez(OUT / f"fits_xi_Model{model}.npz", **res)
    sem = lambda v: v.std(ddof=1) / np.sqrt(len(v))            # noqa: E731
    print(f"Model {model}: 100(alpha-1), LOSz and LOSy averaged within phase")
    for kind in ("raw", "lcv"):
        for par in ("ap", "at"):
            vals = {st: 100 * (0.5 * (res[f"LOSz_{st}_{kind}_{par}"]
                                      + res[f"LOSy_{st}_{kind}_{par}"]) - 1)
                    for st in ("pre", "post")}
            d = vals["post"] - vals["pre"]
            print(f"  {kind:3s} {par}: pre {vals['pre'].mean():+.3f}+/-{sem(vals['pre']):.3f}"
                  f"  post {vals['post'].mean():+.3f}+/-{sem(vals['post']):.3f}"
                  f"  paired {d.mean():+.3f}+/-{sem(d):.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("coef", "measure", "fit"))
    p.add_argument("--model", type=int, default=3)
    p.add_argument("--phase", default="000")
    p.add_argument("--los", choices=("y", "z"), default="z")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.stage == "coef":
        coefficients(a.model)
    elif a.stage == "measure":
        measure(a.model, f"{int(a.phase):03d}", a.los, a.workers)
    else:
        fit(a.model)
