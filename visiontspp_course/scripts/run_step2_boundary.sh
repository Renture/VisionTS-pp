#!/usr/bin/env bash
# Step 2b: Boundary adaptation comparison (none vs hard_clip vs filter_train vs soft_clip)
# on ETTm2 (96, 192) and Weather (96, 192)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

python -u visiontspp_course/run_experiment.py \
    --stage step2_boundary \
    --gpu "${GPU:-1}" \
    --align-mode "${ALIGN_MODE:-half}" \
    --quantile-mode "${QUANTILE_MODE:-q5}" \
    --ft-type "${FT_TYPE:-ln_head}" \
    --epochs "${EPOCHS:-3}" \
    --seeds 2021 \
    "$@"
