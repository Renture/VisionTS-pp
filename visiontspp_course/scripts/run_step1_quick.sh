#!/usr/bin/env bash
# Step 1 fast acceptance on physical GPU 1:
# ETTm1 (96,192,336,720) + Weather (96,192) | colorized window_corr K=7 | MSE + LN-only | half vs align_const
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

python -u visiontspp_course/run_experiment.py \
    --stage step1_quick \
    --gpu "${GPU:-1}" \
    --seeds 2021 \
    "$@"
