#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=28G
#SBATCH --constraint=cpu
#SBATCH --job-name=lya_rscan
#SBATCH --output=slurm-lya-rscan-%j.out

# One forest-driven smoothing-radius scan on ph000.  The R=10 field is reused
# from production; only missing radii are constructed.
set -eo pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean

NCPU=${LYA_NCPU:-32}
python -u scan_smoothing.py --driver lya --model 1 --los z \
    --nmesh 512 --nmesh_recon 512 --beta_F 0.957688 --bias_F -0.148066 \
    --sr 4 6 8 10 12.5 15 --ncpu "$NCPU" \
    --plot /pscratch/sd/b/boryanah/Lyaf_recon/figs_nowiggle/scan_R_lyadrv.png \
    --save /pscratch/sd/b/boryanah/Lyaf_recon/scan_R_Model_1_LOSz_ph000_lyadrv.npz
