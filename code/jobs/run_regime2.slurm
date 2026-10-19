#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --constraint=cpu
#SBATCH --array=0-5
#SBATCH --job-name=lya_regime2
#SBATCH --output=/pscratch/sd/b/boryanah/Lyaf_recon/slurm-lya_regime2-%A_%a.out

# Regime II (external-tracer displacement) ensemble: one array task per phase.
# Phases 001-005 first build a halo catalogue matching the archived phase-000
# file exactly (cleaned CompaSO, N >= 150; verified by
# `build_halo_catalog.py --check`: 45,737,872 halos, identical M_min).
# regime2_fit.py then stacks all six phases together with Regime III.

set -eo pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
PH=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
export OMP_NUM_THREADS=32
if [ "$PH" != "000" ]; then
    python -u build_halo_catalog.py --phase "$PH" --cleaned 1
fi
python -u regime2_ensemble.py --phase "$PH" --ncpu 32
echo done "$PH"
