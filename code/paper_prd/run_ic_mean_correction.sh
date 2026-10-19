#!/usr/bin/env bash
set -eu
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/lyaf_audit_mpl
audit_python=/global/common/software/desi/users/adematti/perlmutter/cosmodesiconda/20260321-1.0.0/conda/bin/python
"$audit_python" -u paper_prd/correct_lcv_mean.py
for model in 3 4 1 2; do
  for los in z y; do
    "$audit_python" -u paper_prd/refit_binned_pk.py --model "$model" --los "$los" --control-mean ic
  done
done
