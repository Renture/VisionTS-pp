#!/usr/bin/env bash
# Step 1 fast acceptance in parallel on physical GPUs 2 and 3:
# ETTm1 (96,192,336,720) + Weather (96,192) | colorized window_corr K=7 | MSE + LN-only | half vs align_const
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

GPUS_STR="${GPUS:-2 3}"
read -r -a GPU_ARR <<< "$GPUS_STR"

python -u visiontspp_course/run_experiment.py \
    --stage step1_quick \
    --gpus "${GPU_ARR[@]}" \
    --max-imgs-per-step "${MAX_IMGS:-32}" \
    --save-dir "${SAVE_DIR:-save/course_exp}" \
    --seeds 2021 \
    "$@"
