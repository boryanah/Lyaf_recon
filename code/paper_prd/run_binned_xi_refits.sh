#!/usr/bin/env bash
set -eu
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/lyaf_audit_mpl NUMBA_CACHE_DIR=/tmp/lyaf_audit_numba
vega_python=/global/common/software/desi/users/acuceu/vega_master/bin/python
for nmesh in 512 1152; do
  for los in z y; do
    "$vega_python" -u paper_prd/refit_binned_xi.py --nmesh "$nmesh" --los "$los"
  done
done
