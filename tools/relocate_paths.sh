#!/usr/bin/env bash
# Point the scripts at this bundle instead of the original NERSC locations.
#
# The analysis scripts hard-code two absolute prefixes:
#   /pscratch/sd/b/boryanah/Lyaf_recon   (and .../Lyaf_p3d)   -> generated products
#   /global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean -> the code tree
# and the paper_prd scripts read their compact inputs from paper_prd/data.
# This helper rewrites the prefixes in code/ (in place; run it once after
# unpacking) and links code/paper_prd/data to data/paper_prd_data.  The
# verification script tools/reproduce_gains.py does not need it (it redirects
# the paths programmatically); the original drivers do.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
grep -rl -E '/pscratch/sd/b/boryanah/Lyaf_(recon|p3d)|/global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean' \
    "$ROOT/code" --include='*.py' --include='*.sh' --include='*.slurm' | while read -r f; do
  sed -i \
    -e "s#/pscratch/sd/b/boryanah/Lyaf_recon#$ROOT/data/pscratch/Lyaf_recon#g" \
    -e "s#/pscratch/sd/b/boryanah/Lyaf_p3d#$ROOT/data/pscratch/Lyaf_p3d#g" \
    -e "s#/global/u1/b/boryanah/Lyaf_shift/reconstruction/new/clean#$ROOT/code#g" "$f"
  echo "relocated $f"
done
ln -sfn ../../data/paper_prd_data "$ROOT/code/paper_prd/data"
echo "linked code/paper_prd/data -> data/paper_prd_data"
echo "Remember: PYTHONPATH=$ROOT/code:$ROOT/code/paper_prd"
