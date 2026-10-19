#!/usr/bin/env bash
set -eu
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/lyaf_audit_mpl NUMBA_CACHE_DIR=/tmp/lyaf_audit_numba
vega_python=/global/common/software/desi/users/acuceu/vega_master/bin/python
los=$1
while ! test -f /pscratch/sd/b/boryanah/Lyaf_recon/target_transfer_20260908/common_target_validation.npz; do sleep 5; done
"$vega_python" -u paper_prd/refit_common_target.py --los "$los" --measure 512 --start .35
"$vega_python" -u paper_prd/refit_common_target.py --los "$los" --measure 1152 --start .35
"$vega_python" -u paper_prd/refit_common_target.py --los "$los" --measure 512 --start .40
