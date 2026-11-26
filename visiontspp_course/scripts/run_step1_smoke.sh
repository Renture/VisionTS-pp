#!/usr/bin/env bash
# ~30s smoke test on physical GPU 1 (verifies Step 1 & Step 2 code paths end-to-end)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

python -u visiontspp_course/run_experiment.py \
    --stage step1_smoke \
    --gpu "${GPU:-1}" \
    --overwrite \
    "$@"
