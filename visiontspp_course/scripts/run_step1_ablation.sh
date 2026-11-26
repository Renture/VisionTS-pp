#!/usr/bin/env bash
# Step 1 controlled ablations in parallel on physical GPUs 2 and 3
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

GPUS_STR="${GPUS:-2 3}"
read -r -a GPU_ARR <<< "$GPUS_STR"

python -u visiontspp_course/run_experiment.py \
    --stage step1_ablation \
    --gpus "${GPU_ARR[@]}" \
    --max-imgs-per-step "${MAX_IMGS:-32}" \
    --save-dir "${SAVE_DIR:-save/course_exp}" \
    --align-mode "${ALIGN_MODE:-half}" \
    --seeds 2021 \
    "$@"
