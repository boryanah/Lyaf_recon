# Environments (NERSC Perlmutter, verified 2026-10-18)

## Fourier pipeline, fits, figures, the bundled re-runs

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
# -> /global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/conda/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/lyaf_audit_mpl NUMBA_CACHE_DIR=/tmp/lyaf_audit_numba
```

Python 3.12.9; numpy 2.4.3, scipy 1.17.1, matplotlib 3.10.8, pyrecon 1.0.0,
pmesh 0.1.58, cosmoprimo 1.0.0, classy 3.3.4.0, iminuit 2.32.0, asdf 5.2.0,
abacusutils 2.1.3.dev17, fitsio 1.3.0, mpi4py 4.1.1. Full list:
`cosmodesi_main_pip_freeze.txt` (433 packages). The `run_*.sh` drivers in
`code/paper_prd/` call this interpreter `audit_python` / `closure_py` /
`matched_python`. The older batch scripts additionally source
`/global/common/software/desi/desi_environment.sh 23.1` and `module load
python/3.13-26.1.0`; the first was reported broken on 2026-10-17 and is not
needed by the current code.

`tools/reproduce_gains.py` needs only numpy and scipy from this stack (the CLASS
template is cached in `code/templates/`).

## Configuration-space (Vega) fits

```bash
VEGA_PY=/global/common/software/desi/users/acuceu/vega_master/bin/python
```

Python 3.11.13; vega 1.8.1.dev5+gae92fa5ca, picca 9.21.4, iminuit 2.33.0,
numpy 2.4.6, scipy 1.17.1, astropy 8.0.1, camb 2.0.4, fitsio 1.4.2,
mcfit 0.0.22, numba 0.68.0. Full list: `vega_master_pip_freeze.txt` (195
packages). The drivers call it `vega_python` / `closure_vega`. The fits read
the eBOSS DR16 file
`/global/cfs/projectdirs/desi/science/lya/eboss_dr16/public_dr16_correlations/cf_z_0_10-exp.fits`
and the Vega fiducial `Planck18/Planck18_z_2.406.fits` (inside Vega); the
frozen Vega configuration is `code/paper_prd/configs/`.

## Pair counter

External patched P3D C++ code (`power_periodic`), GCC 14, GSL, cfitsio; build
and runtime notes (`ulimit -s unlimited`, `OMP_STACKSIZE=512M`) in
`code/p3d/BUILD_NERSC.md`. Not included.
