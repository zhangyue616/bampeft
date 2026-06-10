from pathlib import Path
import os
#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-production}"
if [[ "$MODE" != "smoke" && "$MODE" != "production" && "$MODE" != "audit-only" && "$MODE" != "report-only" ]]; then
  echo "Usage: bash scripts/n5_rank_sensitivity/run_n5_rank.sh [smoke|production|audit-only|report-only]" >&2
  exit 2
fi

cd "${BAM_REPO_ROOT:-$(pwd)}"
mkdir -p logs/n5_rank_sensitivity

if [[ "$MODE" == "smoke" || "$MODE" == "production" ]]; then
  bash scripts/n5_rank_sensitivity/precompute_scaffold_splits.sh
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kapt-5090

if [[ "$MODE" == "audit-only" || "$MODE" == "report-only" ]]; then
  python scripts/n5_rank_sensitivity/n5_finalize.py "$MODE"
else
  python scripts/n5_rank_sensitivity/n5_rank_runner.py "$MODE"
  if [[ "$MODE" == "production" ]]; then
    python scripts/n5_rank_sensitivity/n5_finalize.py production
  fi
fi
