#!/usr/bin/env bash
# Step 1 controlled ablations on physical GPU 1:
# all vs sequential vs train_corr vs window_corr (pearson/cosine, K=7/K=4) & gray vs color
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

python -u visiontspp_course/run_experiment.py \
    --stage step1_ablation \
    --gpu "${GPU:-1}" \
    --align-mode "${ALIGN_MODE:-half}" \
    --seeds 2021 \
    "$@"
