#!/usr/bin/env bash
# Step 2c: Fine-tuning parameter scope comparison (ln_head vs ln_head_patch vs ln_head_lora) on GPUs 2 and 3
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

GPUS_STR="${GPUS:-2 3}"
read -r -a GPU_ARR <<< "$GPUS_STR"

python -u visiontspp_course/run_experiment.py \
    --stage step2_ft_scope \
    --gpus "${GPU_ARR[@]}" \
    --max-imgs-per-step "${MAX_IMGS:-32}" \
    --save-dir "${SAVE_DIR:-save/course_exp}" \
    --align-mode "${ALIGN_MODE:-half}" \
    --quantile-mode "${QUANTILE_MODE:-q5}" \
    --clip-mode "${CLIP_MODE:-soft_clip}" \
    --epochs "${EPOCHS:-3}" \
    --seeds 2021 \
    "$@"
