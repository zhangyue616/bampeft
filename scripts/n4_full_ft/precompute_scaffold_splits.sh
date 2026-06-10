from pathlib import Path
import os
#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kapt
cd "${BAM_REPO_ROOT:-$(pwd)}"

mkdir -p dumped/n4_full_ft_splits
for dataset in freesolv esol lipo bace bbbp sider; do
  for seed in 0 1 2 10 100 1000; do
    out="dumped/n4_full_ft_splits/${dataset}_scaffold_seed${seed}.json"
    if [[ ! -s "$out" ]]; then
      python scripts/splits/scaffold_make_split.py \
        --dataset "$dataset" \
        --seed "$seed" \
        --out "$out" \
        --split_sizes 0.8 0.1 0.1
    fi
  done
done
