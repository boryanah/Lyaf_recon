#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=28G
#SBATCH --constraint=cpu
#SBATCH --array=0-5
#SBATCH --job-name=lya_pairmatch
#SBATCH --output=/pscratch/sd/b/boryanah/Lyaf_p3d/matched/slurm-matched-%A_%a.out

set -eo pipefail

source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

ulimit -s unlimited
export OMP_STACKSIZE=512M

cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
phase=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")
python -u p3d/run_matched_pair_counts.py --phases "${phase}" \
  --states pre post --z-stride 2 --ncpu 8 --nthread 32
