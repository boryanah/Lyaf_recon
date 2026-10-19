#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --constraint=cpu
#SBATCH --job-name=lya_scanR_ext
#SBATCH --output=/pscratch/sd/b/boryanah/Lyaf_recon/slurm-lya_scanR_ext-%j.out

# Extend the calibrated phase-000 smoothing scan beyond the adopted R = 10
# (the archived calibrated scan stops at 10).  Existing fields are reused.
set -eo pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
python -u scan_smoothing.py --driver lya --model 1 --los z \
    --nmesh 512 --nmesh_recon 512 --beta_F 0.957688 --bias_F -0.148066 \
    --sr 4 6 8 10 12.5 15 20 --ncpu 32 \
    --plot /pscratch/sd/b/boryanah/Lyaf_recon/figs_nowiggle/scan_R_lyadrv_ext.png \
    --save /pscratch/sd/b/boryanah/Lyaf_recon/scan_R_Model_1_LOSz_ph000_calibrated_ext.npz
