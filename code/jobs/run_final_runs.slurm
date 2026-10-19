#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=lya_final_runs
#SBATCH --output=/pscratch/sd/b/boryanah/Lyaf_recon/slurm-lya_final_runs-%j.out

# One node, three independent workloads in parallel, then the joint fit:
#   (1) calibrated sightline-density scan (Regime III), 6 phases  [32 threads]
#   (2) smoothing-radius scan extended to R = 12.5, 15, 20 (ph000) [24 threads]
#   (3) Regime II external-halo displacement, 6 phases, two waves of three
#       phases [24 threads each]; phases 001-005 first build their halo
#       catalogue (cleaned CompaSO, N >= 150, verified against phase 000)
#   (4) regime2_fit.py: six-phase stacked fits of pre / Regime III / Regime II
set -o pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
D=/pscratch/sd/b/boryanah/Lyaf_recon
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

(OMP_NUM_THREADS=32 python -u sparse_skewers.py --ncpu 32 \
    --phases 000 001 002 003 004 005 --frac 1.0 0.5 0.25 0.155 0.08 \
    --alias --sig_width 1.0 --sig_pre 4.61 2.81 --sig_post 2.16 1.36 \
    --plot $D/figs_nowiggle/sparse_skewers_6ph_calibrated.png \
    --save $D/sparse_skewers_Model_1_LOSz_6phases_calibrated.npz \
    > $D/log_density_scan.txt 2>&1; echo "density scan exit $?") &

(OMP_NUM_THREADS=24 python -u scan_smoothing.py --driver lya --model 1 --los z \
    --nmesh 512 --nmesh_recon 512 --beta_F 0.957688 --bias_F -0.148066 \
    --sr 4 6 8 10 12.5 15 20 --ncpu 24 \
    --plot $D/figs_nowiggle/scan_R_lyadrv_ext.png \
    --save $D/scan_R_Model_1_LOSz_ph000_calibrated_ext.npz \
    > $D/log_scan_ext.txt 2>&1; echo "scan exit $?") &

run_phase() {
    PH=$1
    if [ "$PH" != "000" ]; then
        OMP_NUM_THREADS=24 python -u build_halo_catalog.py --phase "$PH" --cleaned 1 || return 1
    fi
    OMP_NUM_THREADS=24 python -u regime2_ensemble.py --phase "$PH" --ncpu 24
}
(for PH in 000 001 002; do run_phase $PH > $D/log_regime2_$PH.txt 2>&1 & done; wait
 for PH in 003 004 005; do run_phase $PH > $D/log_regime2_$PH.txt 2>&1 & done; wait
 echo "regime2 done") &

wait
OMP_NUM_THREADS=32 python -u regime2_fit.py > $D/log_regime2_fit.txt 2>&1
echo "fit exit $?"
