"""Multivariate grouping strategies for colorized subfigure stacking.

Supports:
- univariate: each variable in its own image (VisionTS baseline)
- all: all variables stacked into a single image (raw VisionTS++)
- sequential: balanced contiguous column-order chunks of size <= max_group_size
- train_corr: fixed balanced groups computed from training-split correlations
- window_corr: dynamic per-window balanced groups computed from look-back window similarity
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch


def balanced_group_sizes(nvars: int, max_group_size: int) -> List[int]:
    """Partition nvars into balanced group sizes with each size <= max_group_size."""
    if nvars <= 0:
        raise ValueError("nvars must be positive")
    if max_group_size <= 0:
        raise ValueError("max_group_size must be positive")
    if nvars <= max_group_size:
        return [nvars]
    num_groups = int(math.ceil(float(nvars) / float(max_group_size)))
    base = nvars // num_groups
    rem = nvars % num_groups
    return [base + 1] * rem + [base] * (num_groups - rem)


def compute_similarity_matrix(
    x: torch.Tensor, metric: str = "pearson", use_abs: bool = True
) -> torch.Tensor:
    """Compute pairwise [B, M, M] similarity matrix along the time dimension L.

    Args:
        x: tensor of shape [B, L, M].
        metric: 'pearson' or 'cosine'.
        use_abs: if True, take absolute similarity so strongly anti-correlated
            variables are grouped together.
    """
    if metric not in ("pearson", "cosine"):
        raise ValueError(f"Unsupported sim metric: {metric!r}")

    xf = x.float()
    if metric == "pearson":
        xf = xf - xf.mean(dim=1, keepdim=True)

    norm = torch.sqrt((xf * xf).sum(dim=1, keepdim=True) + 1e-8)
    xn = xf / norm
    sim = torch.bmm(xn.transpose(1, 2), xn)
    sim = torch.clamp(sim, -1.0, 1.0)
    if use_abs:
        sim = sim.abs()

    # Zero out self-similarity diagonal for seed and neighbor selection
    m = sim.shape[-1]
    eye = torch.eye(m, dtype=torch.bool, device=sim.device).unsqueeze(0)
    sim = sim.masked_fill(eye, 0.0)
    return sim


def greedy_balanced_grouping(sim: torch.Tensor, group_sizes: List[int]) -> torch.Tensor:
    """Vectorized batched greedy non-overlapping clustering into fixed balanced group sizes.

    For each group:
      1. Pick the unassigned variable with highest total similarity to other unassigned variables.
      2. Greedily add the unassigned variable with highest sum of similarities to current group members.

    Returns:
        perm: LongTensor of shape [B, M] representing a valid permutation per sample.
    """
    batch_size, nvars, _ = sim.shape
    if sum(group_sizes) != nvars:
        raise ValueError(f"sum(group_sizes)={sum(group_sizes)} != nvars={nvars}")

    device = sim.device
    batch_idx = torch.arange(batch_size, device=device)
    assigned = torch.zeros((batch_size, nvars), dtype=torch.bool, device=device)
    perm_cols = []

    for g_size in group_sizes:
        avail = ~assigned
        avail_pair = avail.unsqueeze(2) & avail.unsqueeze(1)
        seed_score = (sim * avail_pair.float()).sum(dim=2)
        seed_score = seed_score.masked_fill(assigned, -1e9)
        seed = seed_score.argmax(dim=1)

        assigned[batch_idx, seed] = True
        perm_cols.append(seed)

        if g_size > 1:
            group_sim = sim[batch_idx, seed]
            for _ in range(g_size - 1):
                cand_score = group_sim.masked_fill(assigned, -1e9)
                nxt = cand_score.argmax(dim=1)
                assigned[batch_idx, nxt] = True
                perm_cols.append(nxt)
                group_sim = group_sim + sim[batch_idx, nxt]

    return torch.stack(perm_cols, dim=1)


def compute_window_permutation(
    x: torch.Tensor,
    group_sizes: List[int],
    metric: str = "pearson",
    use_abs: bool = True,
) -> torch.Tensor:
    """Compute per-window variable permutation [B, M] given input windows [B, L, M]."""
    with torch.no_grad():
        sim = compute_similarity_matrix(x, metric=metric, use_abs=use_abs)
        return greedy_balanced_grouping(sim, group_sizes)


def build_static_groups(
    nvars: int,
    grouping: str,
    max_group_size: int = 7,
    train_data: Optional[np.ndarray] = None,
    sim_metric: str = "pearson",
    use_abs: bool = True,
) -> Tuple[Optional[List[int]], List[int]]:
    """Build static permutation (or None for dynamic window_corr) and fixed group_sizes.

    Returns:
        (static_perm, group_sizes) where static_perm is a list of length nvars
        for static modes, or None when grouping == 'window_corr'.
    """
    valid = ("univariate", "all", "sequential", "train_corr", "window_corr")
    if grouping not in valid:
        raise ValueError(f"Unknown grouping={grouping!r}. Choose from {valid}")

    if grouping == "univariate":
        return list(range(nvars)), [1] * nvars

    if grouping == "all":
        return list(range(nvars)), [nvars]

    group_sizes = balanced_group_sizes(nvars, max_group_size)

    if grouping == "sequential":
        return list(range(nvars)), group_sizes

    if grouping == "window_corr":
        return None, group_sizes

    # train_corr: compute fixed grouping from training split only
    if train_data is None:
        raise ValueError("train_data is required for grouping='train_corr'")
    x_train = torch.from_numpy(np.asarray(train_data, dtype=np.float32)).unsqueeze(0)
    perm = compute_window_permutation(x_train, group_sizes, metric=sim_metric, use_abs=use_abs)
    return perm[0].tolist(), group_sizes
