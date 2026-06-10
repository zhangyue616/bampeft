from pathlib import Path
import os
#!/usr/bin/env bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kapt-5090
cd "${BAM_REPO_ROOT:-$(pwd)}"
python scripts/n4_full_ft/n4_full_ft_runner.py
