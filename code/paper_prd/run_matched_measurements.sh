#!/usr/bin/env bash
set -eu
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/lyaf_audit_mpl NUMBA_CACHE_DIR=/tmp/lyaf_audit_numba
matched_python=/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/conda/bin/python
for phase in 0 1 2 3 4 5; do
  for los in z y; do
    "$matched_python" -u paper_prd/measure_matched_model.py --phase "$phase" --los "$los" --workers 2
  done
done
"$matched_python" -u paper_prd/run_matched_model_fits.py --space pk --validate
"$matched_python" -u paper_prd/run_matched_model_fits.py --space pk
