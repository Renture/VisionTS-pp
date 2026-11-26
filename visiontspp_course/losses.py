"""Losses, probabilistic CRPS evaluation, and ImageNet boundary filtering."""

from typing import List, Tuple

import torch

from .model import IMAGENET_THRES_DOWN, IMAGENET_THRES_UP


def pinball_loss(
    y_true: torch.Tensor,
    y_quantiles: torch.Tensor,
    quantiles: List[float],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute multi-quantile Pinball Loss.

    Args:
        y_true: [B, H, M]
        y_quantiles: [B, Q, H, M]
        quantiles: list of Q quantile levels in (0, 1)
        reduction: 'mean', 'sum' (mean over Q, sum over B*H*M), or 'none' ([B, Q, H, M])
    """
    q = torch.tensor(quantiles, dtype=y_quantiles.dtype, device=y_quantiles.device).view(1, -1, 1, 1)
    err = y_true.unsqueeze(1) - y_quantiles
    loss = torch.max(q * err, (q - 1.0) * err)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.mean(dim=1).sum()
    raise ValueError(f"Unsupported reduction={reduction!r}")


def crps_sum(
    y_true: torch.Tensor,
    y_quantiles: torch.Tensor,
    quantiles: List[float],
) -> torch.Tensor:
    """Sum of discrete quantile CRPS approximation (= 2 * mean_q Pinball_q) over (B, H, M) in float64."""
    q = torch.tensor(quantiles, dtype=torch.float64, device=y_quantiles.device).view(1, -1, 1, 1)
    err = y_true.to(torch.float64).unsqueeze(1) - y_quantiles.to(torch.float64)
    pb = torch.max(q * err, (q - 1.0) * err)
    return (2.0 * pb.mean(dim=1)).sum()


def filter_in_range_mask(
    x_history: torch.Tensor,
    norm_const: float,
    lo: float = IMAGENET_THRES_DOWN,
    hi: float = IMAGENET_THRES_UP,
    max_outlier_ratio: float = 0.01,
) -> Tuple[torch.Tensor, float]:
    """Identify training windows whose normalized values stay within ImageNet valid bounds.

    Returns:
        (keep_mask [B] bool tensor, dropped_fraction float).
        Always keeps at least one window in the batch as a safety fallback.
    """
    with torch.no_grad():
        means = x_history.mean(dim=1, keepdim=True)
        x_enc = x_history - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = stdev / float(norm_const)
        x_norm = x_enc / stdev

        out_of_range = (x_norm < lo) | (x_norm > hi)
        outlier_ratio = out_of_range.float().mean(dim=(1, 2))  # [B]
        keep = outlier_ratio <= float(max_outlier_ratio)
        if not bool(keep.any().item()):
            best_idx = int(outlier_ratio.argmin().item())
            keep[best_idx] = True
        dropped_frac = float((~keep).float().mean().item())
        return keep, dropped_frac
