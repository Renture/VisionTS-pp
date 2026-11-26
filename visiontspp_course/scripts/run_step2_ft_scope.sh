#!/usr/bin/env bash
# Step 2c: Fine-tuning parameter scope comparison (ln_head vs ln_head_patch vs ln_head_lora)
# on ETTm1 (96, 192), 3 epochs with validation early stopping
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

python -u visiontspp_course/run_experiment.py \
    --stage step2_ft_scope \
    --gpu "${GPU:-1}" \
    --align-mode "${ALIGN_MODE:-half}" \
    --quantile-mode "${QUANTILE_MODE:-q5}" \
    --clip-mode "${CLIP_MODE:-soft_clip}" \
    --epochs "${EPOCHS:-3}" \
    --seeds 2021 \
    "$@"
