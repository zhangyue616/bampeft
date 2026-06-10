from pathlib import Path
import os
#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kapt
cd "${BAM_REPO_ROOT:-$(pwd)}"

mkdir -p logs/n5_rank_sensitivity
mkdir -p dumped/n5_rank_sensitivity_splits

{
  echo "n5 scaffold split precompute start $(date -Is)"
  for dataset in freesolv esol lipo bace bbbp sider; do
    for seed in 0 1 2; do
      out="dumped/n5_rank_sensitivity_splits/${dataset}_scaffold_seed${seed}.json"
      if [[ -s "$out" ]]; then
        echo "exists $out"
      else
        python scripts/splits/scaffold_make_split.py \
          --dataset "$dataset" \
          --seed "$seed" \
          --out "$out" \
          --split_sizes 0.8 0.1 0.1
        echo "created $out"
      fi
    done
  done
  echo "n5 scaffold split precompute done $(date -Is)"
} 2>&1 | tee logs/n5_rank_sensitivity/precompute.log
