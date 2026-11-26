#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

SRC_CKPT="${1:-$HOME/VisionTS/ckpt}"
SRC_DATA="${2:-$HOME/VisionTS/long_term_tsf/dataset}"

if [ ! -e "$REPO_DIR/ckpt" ]; then
    ln -s "$SRC_CKPT" "$REPO_DIR/ckpt"
    echo "Linked ckpt -> $SRC_CKPT"
fi

if [ ! -e "$REPO_DIR/dataset" ]; then
    ln -s "$SRC_DATA" "$REPO_DIR/dataset"
    echo "Linked dataset -> $SRC_DATA"
fi

echo "--- Checking assets ---"
ls -lh "$REPO_DIR/ckpt/mae_visualize_vit_base.pth"
ls -lh "$REPO_DIR/dataset/ETT-small/ETTm1.csv" "$REPO_DIR/dataset/ETT-small/ETTm2.csv" "$REPO_DIR/dataset/weather/weather.csv"

echo "--- Checking Python environment ---"
python -c "import torch, torchvision, timm, einops, pandas, sklearn, tqdm; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'timm:', timm.__version__)"

echo "--- Checking physical GPU 1 ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "Setup OK. Ready to run Step 1 / Step 2 scripts on GPU 1."
