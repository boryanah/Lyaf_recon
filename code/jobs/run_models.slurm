#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=01:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=28G
#SBATCH --constraint=cpu
#SBATCH --array=0-17
#SBATCH --job-name=lya_models
#SBATCH --output=slurm-lya-models-%A_%a.out

# Models 2, 3, 4: 6 phases each, Lya-driven Psi, R = 10 Mpc/h.
# Model 1 is already archived; see AUDIT.md.
#
# The reconstruction solve must use the measured beta_F for each model.  Using
# Model 1's value for Models 3 and 4 biases the paired alpha_perp shift and
# inflates the apparent alpha_parallel gain.

set -eo pipefail
source /global/common/software/desi/desi_environment.sh 23.1
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
cd /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean
MODEL=$((2 + SLURM_ARRAY_TASK_ID / 6))
PHASE_ID=$((SLURM_ARRAY_TASK_ID % 6))
printf -v PH "%03d" "$PHASE_ID"
BIAS_F=(0 0 -0.130939 -0.132453 -0.128523)
BETA_F=(0 0 0.989632 2.074689 2.395469)
BIAS_TAG=(0 0 -0.130939 -0.132453 -0.128523)
BETA_TAG=(0 0 0.989632 2.074689 2.395469)
SIM=AbacusSummit_base_c000_ph${PH}
NCPU=${LYA_NCPU:-32}
D=/pscratch/sd/b/boryanah/Lyaf_recon
REF="$D/dF_ref_Model_${MODEL}_LOSz_nmesh512_${SIM}_z2.500.npy"
POST="$D/dF_recon_Model_${MODEL}_LOSz_nmesh512_R10.00_nr512_MG_bF${BIAS_TAG[$MODEL]}_bt${BETA_TAG[$MODEL]}_lyadrv_${SIM}_z2.500.npy"

echo "===== Model_$MODEL  $SIM  b_F=${BIAS_F[$MODEL]} beta_F=${BETA_F[$MODEL]} ====="
if [[ ! -f "$REF" ]]; then
    python -u recon_paint_field.py --sim_name "$SIM" --model "$MODEL" --los z \
        --mode ref --nmesh 512 --ncpu "$NCPU"
fi
if [[ ! -f "$POST" ]]; then
    python -u recon_paint_field.py --sim_name "$SIM" --model "$MODEL" --los z \
        --mode recon --driver lya --nmesh 512 --nmesh_recon 512 \
        --sr 10.0 --rectype MG --bias_F "${BIAS_F[$MODEL]}" \
        --beta_F "${BETA_F[$MODEL]}" --ncpu "$NCPU"
fi

python -u measure_pkmu.py --nmu 5 --kmax 0.5 --window node --cache-each --ncpu "$NCPU" \
    --files "$REF"
python -u measure_pkmu.py --nmu 5 --kmax 0.5 --window continuous --cache-each --ncpu "$NCPU" \
    --files "$POST"

echo done
