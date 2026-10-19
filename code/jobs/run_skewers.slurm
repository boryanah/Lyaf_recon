#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --constraint=cpu
#SBATCH --job-name=lya_skewers
#SBATCH --output=slurm-lya_skewers-%j.out

# Sparse-sightline scan, 6 phases, with the two fixes the ph000 run showed were
# needed:
#   * a Gaussian prior on Sigma, centred on the dense 6-phase result, so the
#     damping stops railing against zero when the aliasing noise leaves it
#     unconstrained;
#   * an explicit P_1D(k_par)/n_2D aliasing template in the linear basis, since
#     that contaminant is white in k_perp and cannot be absorbed by broadband
#     terms built from powers of |k|.
#
# Both are the minimal in-Fourier version of what a specialised Lya P3D
# estimator does properly.  The definitive route is xi from pair counts, which
# has no same-sightline aliasing term.  See BUGS_AND_ISSUES.md for the
# remaining mismatch between the archived pair-count and painted fields.

set -eo pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
D=/pscratch/sd/b/boryanah/Lyaf_recon
NCPU=${LYA_NCPU:-32}

python -u sparse_skewers.py --ncpu "$NCPU" \
    --phases 000 001 002 003 004 005 \
    --frac 1.0 0.5 0.25 0.155 0.08 \
    --alias --sig_width 1.0 \
    --sig_pre 4.61 2.81 --sig_post 2.16 1.36 \
    --plot $D/figs_nowiggle/sparse_skewers_6ph.png \
    --save $D/sparse_skewers_Model_1_LOSz_6phases_calibrated.npz

echo done
