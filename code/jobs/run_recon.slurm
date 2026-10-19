#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=28G
#SBATCH --constraint=cpu
#SBATCH --array=0-5
#SBATCH --job-name=lya_recon
#SBATCH --output=slurm-lya-recon-%A_%a.out
##SBATCH --account=desi

# Production pre/post-recon field pairs for the Lya BAO reconstruction study.
#
# One phase is run per array task.  The production result in the paper uses a
# 512^3 displacement mesh driven by the Lya field itself.
#
# Completed production tasks peak at about 18.5 GiB.  The request leaves
# headroom without reserving a needlessly large node share.
#
# Note LOSy is not yet supported by make_dF_grid.load_dF_grid (the shipped
# LOSy field needs a transpose applied before the rfft half-axis is
# interpreted).  Keep LOS=z until that is added.

set -eo pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean

NMESH=512            # Lya painting grid
NMESH_RECON=512      # pyrecon grid for the Lya displacement field
SR=10.0             # smoothing radius, Mpc/h -- chosen by scan_smoothing.py
BIAS_F=-0.148066      # six-phase mean from phase-matched IC cross spectra
BETA_F=0.957688
NCPU=${LYA_NCPU:-32}
MODEL=1
LOS=z

printf -v PH "%03d" "$SLURM_ARRAY_TASK_ID"
SIM=AbacusSummit_base_c000_ph${PH}
echo "=================== $SIM  Model_$MODEL  LOS$LOS ==================="

D=/pscratch/sd/b/boryanah/Lyaf_recon
REF="$D/dF_ref_Model_${MODEL}_LOS${LOS}_nmesh${NMESH}_${SIM}_z2.500.npy"
POST="$D/dF_recon_Model_${MODEL}_LOS${LOS}_nmesh${NMESH}_R10.00_nr${NMESH_RECON}_MG_bF-0.148066_bt0.957688_lyadrv_${SIM}_z2.500.npy"
if [[ ! -f "$REF" ]]; then
    python -u recon_paint_field.py --sim_name "$SIM" --model "$MODEL" --los "$LOS" \
        --mode ref --nmesh "$NMESH" --ncpu "$NCPU"
fi
if [[ ! -f "$POST" ]]; then
    python -u recon_paint_field.py --sim_name "$SIM" --model "$MODEL" --los "$LOS" \
        --mode recon --driver lya --nmesh "$NMESH" --nmesh_recon "$NMESH_RECON" \
        --sr "$SR" --rectype MG --bias_F "$BIAS_F" --beta_F "$BETA_F" \
        --ncpu "$NCPU"
fi

# ---------------------------------------------------------------------------
# Smoothing-radius scan (one phase is enough to pick R; then rerun the loop).
# Already done for ph000: use scan_smoothing.py, not this loop.  Result was
# R = 10 Mpc/h.  Radii from 4 through 8 give a larger apparent gain but put
# Sigma_parallel on its zero bound in phase 000.  R = 10 is the first tested
# radius with both damping scales resolved.  The expected Sigma_parallel >
# Sigma_perp hierarchy is recovered by the six-phase stack.  See AUDIT.md.
# ---------------------------------------------------------------------------
# for SR in 4 6 8 10 12.5 15 20; do
#     python -u recon_paint_field.py --mode recon --nmesh $NMESH \
#         --nmesh_recon $NMESH_RECON --sr $SR --bias $BIAS --ncpu $NCPU
# done

# ---------------------------------------------------------------------------
# Measure P(k, mu) for the pair produced by this task.
# ---------------------------------------------------------------------------
python -u measure_pkmu.py --nmu 5 --kmax 0.5 --window node --cache-each --ncpu "$NCPU" \
    --files "$REF"
python -u measure_pkmu.py --nmu 5 --kmax 0.5 --window continuous --cache-each --ncpu "$NCPU" \
    --files "$POST"

echo "done"
