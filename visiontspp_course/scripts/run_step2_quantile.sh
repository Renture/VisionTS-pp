#!/usr/bin/env bash
# Step 2a: Multi-quantile heads (q5 & q9) + Pinball Loss + CRPS on ETTm1 (all 4) & Weather (96, 192)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

python -u visiontspp_course/run_experiment.py \
    --stage step2_quantile \
    --gpu "${GPU:-1}" \
    --align-mode "${ALIGN_MODE:-half}" \
    --ft-type "${FT_TYPE:-ln_head}" \
    --epochs "${EPOCHS:-3}" \
    --seeds 2021 \
    "$@"
