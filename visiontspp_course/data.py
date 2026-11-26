"""Standardized LTSF dataset loader matching VisionTS Table 9 / Table 21 splits exactly."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch


@dataclass(frozen=True)
class DatasetSpec:
    rel_path: str
    kind: str  # 'ett_hour', 'ett_minute', 'custom'
    context: int
    period: int
    norm: float = 0.4
    align: float = 0.4
    horizons: Tuple[int, ...] = (96, 192, 336, 720)


# Exact Table 21 specifications from VisionTS
DATASETS = {
    "ETTh1": DatasetSpec("ETT-small/ETTh1.csv", "ett_hour", 1152, 24, 0.4, 0.4),
    "ETTh2": DatasetSpec("ETT-small/ETTh2.csv", "ett_hour", 1152, 24, 0.4, 0.4),
    "ETTm1": DatasetSpec("ETT-small/ETTm1.csv", "ett_minute", 2304, 96, 0.4, 0.4),
    "ETTm2": DatasetSpec("ETT-small/ETTm2.csv", "ett_minute", 1152, 96, 0.4, 0.4),
    "Weather": DatasetSpec("weather/weather.csv", "custom", 576, 144, 1.0, 0.7),
    "Electricity": DatasetSpec("electricity/electricity.csv", "custom", 1152, 24, 0.4, 0.4),
    "Traffic": DatasetSpec("traffic/traffic.csv", "custom", 1152, 24, 0.4, 0.4),
    "Illness": DatasetSpec("illness/national_illness.csv", "custom", 104, 52, 1.0, 0.4, (24, 36, 48, 60)),
}


class MultivariateWindows:
    """Enumerate multivariate sliding windows (history: [B, L, M], future: [B, H, M])."""

    def __init__(self, data: np.ndarray, context: int, horizon: int):
        self.data = np.asarray(data, dtype=np.float32)
        self.context = int(context)
        self.horizon = int(horizon)
        self.nvars = int(self.data.shape[1])
        self.count = len(self.data) - self.context - self.horizon + 1
        if self.count <= 0:
            raise ValueError(
                f"Split length {len(self.data)} too short for context={context}, horizon={horizon}"
            )
        self._hist_offsets = np.arange(self.context, dtype=np.int64)[None, :]
        self._fut_offsets = (self.context + np.arange(self.horizon, dtype=np.int64))[None, :]

    def take(self, indices: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = np.asarray(indices, dtype=np.int64)[:, None]
        hist = self.data[starts + self._hist_offsets]  # [B, L, M]
        fut = self.data[starts + self._fut_offsets]    # [B, H, M]
        return (
            torch.from_numpy(hist).to(device, non_blocking=True),
            torch.from_numpy(fut).to(device, non_blocking=True),
        )


def load_splits(
    dataset_name: str,
    dataset_root: Path,
    horizon: int,
    context_override: int = 0,
) -> Tuple[Dict[str, MultivariateWindows], DatasetSpec, np.ndarray]:
    """Load train/val/test MultivariateWindows using exact VisionTS StandardScaler protocol."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset={dataset_name!r}. Choose from {list(DATASETS)}")

    spec = DATASETS[dataset_name]
    context = int(context_override) if context_override and context_override > 0 else spec.context
    csv_path = Path(dataset_root) / spec.rel_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    if spec.kind == "ett_hour":
        border1s = [0, 12 * 30 * 24 - context, 12 * 30 * 24 + 4 * 30 * 24 - context]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        df_data = df_raw[df_raw.columns[1:]]
    elif spec.kind == "ett_minute":
        border1s = [0, 12 * 30 * 24 * 4 - context, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - context]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        df_data = df_raw[df_raw.columns[1:]]
    else:
        # Match Dataset_Custom exactly: ['date', ...other features..., 'OT']
        cols = list(df_raw.columns)
        if "date" in cols:
            cols.remove("date")
        if "OT" in cols:
            cols.remove("OT")
            cols = cols + ["OT"]
        df_data = df_raw[cols]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - context, len(df_raw) - num_test - context]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

    scaler = StandardScaler()
    train_raw = df_data.values[border1s[0] : border2s[0]]
    scaler.fit(train_raw)
    scaled = scaler.transform(df_data.values).astype(np.float32)

    splits = {}
    for idx, flag in enumerate(("train", "val", "test")):
        segment = scaled[border1s[idx] : border2s[idx]]
        splits[flag] = MultivariateWindows(segment, context=context, horizon=horizon)

    train_scaled = scaled[border1s[0] : border2s[0]]
    return splits, spec, train_scaled
