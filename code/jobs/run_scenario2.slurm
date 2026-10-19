#!/bin/bash
#SBATCH --job-name=scenario2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --mem=30G
#SBATCH --time=01:45:00
#SBATCH --output=scenario2_%A_%a.log

# Scenario 2: External reconstruction evaluated at skewers
# Runs one (frac, thin_n) combination per array task on ph000.
# Output: /pscratch/sd/b/boryanah/Lyaf_recon/scenario2_*.npz

# exact env of the proven production scripts (run_recon.slurm)
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean

echo "Scenario 2: External reconstruction (halos at skewers)"
echo "Starting: $(date)"

# frac: 0.155 (DESI 50/deg^2), 0.5 (DESI-II 161/deg^2)
# thin_n: None (full), 1e-3 (LAE-like), 1e-4 (sparse LAE-like)
FRACS=(0.155 0.155 0.155 0.5 0.5 0.5)
THIN=("" "1e-3" "1e-4" "" "1e-3" "1e-4")
frac=${FRACS[$SLURM_ARRAY_TASK_ID]}
thin_n=${THIN[$SLURM_ARRAY_TASK_ID]}

if [ -z "$thin_n" ]; then
  echo "  Running: frac=$frac (full catalog)"
  python scenario2_external_recon.py --phase ph000 --model 1 \
      --frac "$frac" --sr 10 --kmax 0.35 --seed 42 \
      --ncpu "$SLURM_CPUS_PER_TASK"
else
  echo "  Running: frac=$frac thin_n=$thin_n"
  python scenario2_external_recon.py --phase ph000 --model 1 \
      --frac "$frac" --thin-n "$thin_n" --sr 10 --kmax 0.35 --seed 42 \
      --ncpu "$SLURM_CPUS_PER_TASK"
fi

echo "Finished: $(date)"
