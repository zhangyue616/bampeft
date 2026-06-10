from pathlib import Path
import os
#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-production}"
if [[ "$MODE" != "smoke" && "$MODE" != "production" ]]; then
  echo "Usage: bash scripts/bootstrap_ci/run_bootstrap_ci.sh [smoke|production]" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kapt-5090
cd "${BAM_REPO_ROOT:-$(pwd)}"
mkdir -p logs/bootstrap_ci

python scripts/bootstrap_ci/bootstrap_ci_compute.py "$MODE" 2>&1 | tee "logs/bootstrap_ci/${MODE}.log"
