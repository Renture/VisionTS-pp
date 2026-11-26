#!/usr/bin/env python3
"""Multi-GPU parallel staged experiment runner for VisionTS++ course improvements.

- Supports parallel execution across multiple physical GPUs (default: --gpus 2 3) via UUID binding.
- Capped micro-batch image budget (--max-imgs-per-step 32) keeps VRAM safely ~5-7 GB per GPU
  so it can safely coexist with Table 21 (~7 GB used on 24 GB RTX 3090).
- Stores all logs, weights, metrics, runs.csv, and summary.csv under ONE unified folder (--save-dir).
- 100% compatible with Python 3.8 + PyTorch 1.7.1 + torchvision 0.8.2 + timm 0.3.2.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_gpu_uuids(gpu_indices: List[str]) -> Dict[str, str]:
    """Map physical nvidia-smi GPU indices to their UUIDs."""
    if len(gpu_indices) == 1 and gpu_indices[0].lower() == "cpu":
        return {"cpu": "cpu"}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    available = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            available[parts[0]] = parts[1]
    missing = [g for g in gpu_indices if g not in available]
    if missing:
        raise ValueError(
            f"Physical GPU(s) {missing} not found in nvidia-smi (available: {sorted(available)})"
        )
    return {g: available[g] for g in gpu_indices}


def adjust_lr_type1(optimizer, epoch: int, base_lr: float):
    lr = base_lr * (0.5 ** ((epoch - 1) // 1))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def build_stage_jobs(stage: str, args) -> List[Dict[str, Any]]:
    """Expand a named stage into an ordered list of reproducible experiment configs."""
    base = dict(
        stage=stage,
        arch=args.arch,
        lr=args.lr,
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        max_imgs_per_micro_batch=args.max_imgs_per_step,
        patience=args.patience,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        filter_outlier_ratio=args.filter_outlier_ratio,
        soft_clip_linear_ratio=args.soft_clip_linear_ratio,
        max_train_steps=args.max_train_steps,
        max_eval_steps=args.max_eval_steps,
    )

    jobs = []  # type: List[Dict[str, Any]]

    if stage == "custom":
        for ds in args.datasets:
            for h in args.horizons:
                for seed in args.seeds:
                    cfg = dict(base)
                    cfg.update(
                        tag=args.tag
                        or f"{args.grouping}_K{args.max_group_size}_{args.color_mode}_{args.align_mode}_{args.quantile_mode}_{args.clip_mode}_{args.ft_type}",
                        dataset=ds,
                        horizon=h,
                        seed=seed,
                        grouping=args.grouping,
                        max_group_size=args.max_group_size,
                        sim_metric=args.sim_metric,
                        color_mode=args.color_mode,
                        align_mode=args.align_mode,
                        quantile_mode=args.quantile_mode,
                        clip_mode=args.clip_mode,
                        ft_type=args.ft_type,
                        epochs=args.epochs,
                    )
                    jobs.append(cfg)
        return jobs

    if stage == "step1_smoke":
        smoke_specs = [
            ("smoke_s1_ettm1", "ETTm1", 96, "window_corr", 7, "pearson", "random_adj", "half", "none", "none", "ln", 1),
            ("smoke_s1_weather", "Weather", 96, "window_corr", 7, "pearson", "random_adj", "align_const", "none", "none", "ln", 1),
            ("smoke_s2_q5_soft_lora", "Weather", 96, "window_corr", 7, "pearson", "random_adj", "half", "q5", "soft_clip", "ln_head_lora", 1),
        ]
        for tag, ds, h, grp, k, sim, col, aln, qm, cm, ft, ep in smoke_specs:
            cfg = dict(base)
            cfg.update(
                tag=tag,
                dataset=ds,
                horizon=h,
                seed=2021,
                grouping=grp,
                max_group_size=k,
                sim_metric=sim,
                color_mode=col,
                align_mode=aln,
                quantile_mode=qm,
                clip_mode=cm,
                ft_type=ft,
                epochs=ep,
                max_train_steps=args.max_train_steps if args.max_train_steps > 0 else 5,
                max_eval_steps=args.max_eval_steps if args.max_eval_steps > 0 else 5,
            )
            jobs.append(cfg)
        return jobs

    if stage == "step1_quick":
        # Put Weather (larger) first so both GPUs finish closer together
        targets = [
            ("Weather", [96, 192]),
            ("ETTm1", [96, 192, 336, 720]),
        ]
        for align_mode in ("half", "align_const"):
            for ds, horizons in targets:
                for h in horizons:
                    for seed in args.seeds:
                        cfg = dict(base)
                        cfg.update(
                            tag=f"s1_wcorr_K7_{align_mode}",
                            dataset=ds,
                            horizon=h,
                            seed=seed,
                            grouping="window_corr",
                            max_group_size=7,
                            sim_metric="pearson",
                            color_mode="random_adj",
                            align_mode=align_mode,
                            quantile_mode="none",
                            clip_mode="none",
                            ft_type="ln",
                            epochs=1,
                        )
                        jobs.append(cfg)
        return jobs

    if stage == "step1_ablation":
        variants = [
            ("s1_all_color", "all", 21, "pearson", "random_adj"),
            ("s1_seq_K7_color", "sequential", 7, "pearson", "random_adj"),
            ("s1_tcorr_K7_color", "train_corr", 7, "pearson", "random_adj"),
            ("s1_wcorr_K7_gray", "window_corr", 7, "pearson", "gray"),
            ("s1_wcorr_K7_cos", "window_corr", 7, "cosine", "random_adj"),
            ("s1_wcorr_K4_color", "window_corr", 4, "pearson", "random_adj"),
        ]
        targets = [("Weather", [96, 192]), ("ETTm1", [96])]
        for tag, grp, k, sim, col in variants:
            for ds, horizons in targets:
                if ds == "ETTm1" and grp == "all":
                    continue
                for h in horizons:
                    for seed in args.seeds:
                        cfg = dict(base)
                        cfg.update(
                            tag=f"{tag}_{args.align_mode}",
                            dataset=ds,
                            horizon=h,
                            seed=seed,
                            grouping=grp,
                            max_group_size=k,
                            sim_metric=sim,
                            color_mode=col,
                            align_mode=args.align_mode,
                            quantile_mode="none",
                            clip_mode="none",
                            ft_type="ln",
                            epochs=1,
                        )
                        jobs.append(cfg)
        return jobs

    if stage == "step2_quantile":
        eff_ft = args.ft_type if args.ft_type != "ln" else "ln_head"
        eff_ep = args.epochs if args.epochs > 1 else 3
        targets = [("Weather", [96, 192]), ("ETTm1", [96, 192, 336, 720])]
        for q_mode in ("q5", "q9"):
            for ds, horizons in targets:
                for h in horizons:
                    for seed in args.seeds:
                        cfg = dict(base)
                        cfg.update(
                            tag=f"s2_{q_mode}_{args.align_mode}_{eff_ft}",
                            dataset=ds,
                            horizon=h,
                            seed=seed,
                            grouping="window_corr",
                            max_group_size=7,
                            sim_metric="pearson",
                            color_mode="random_adj",
                            align_mode=args.align_mode,
                            quantile_mode=q_mode,
                            clip_mode=args.clip_mode,
                            ft_type=eff_ft,
                            epochs=eff_ep,
                        )
                        jobs.append(cfg)
        return jobs

    if stage == "step2_boundary":
        eff_q = args.quantile_mode if args.quantile_mode != "none" else "q5"
        eff_ft = args.ft_type if args.ft_type != "ln" else "ln_head"
        eff_ep = args.epochs if args.epochs > 1 else 3
        targets = [("Weather", [96, 192]), ("ETTm2", [96, 192])]
        for clip_mode in ("none", "hard_clip", "filter_train", "soft_clip"):
            for ds, horizons in targets:
                for h in horizons:
                    for seed in args.seeds:
                        cfg = dict(base)
                        cfg.update(
                            tag=f"s2_bound_{clip_mode}_{eff_q}",
                            dataset=ds,
                            horizon=h,
                            seed=seed,
                            grouping="window_corr",
                            max_group_size=7,
                            sim_metric="pearson",
                            color_mode="random_adj",
                            align_mode=args.align_mode,
                            quantile_mode=eff_q,
                            clip_mode=clip_mode,
                            ft_type=eff_ft,
                            epochs=eff_ep,
                        )
                        jobs.append(cfg)
        return jobs

    if stage == "step2_ft_scope":
        eff_q = args.quantile_mode if args.quantile_mode != "none" else "q5"
        eff_clip = args.clip_mode if args.clip_mode != "none" else "soft_clip"
        eff_ep = args.epochs if args.epochs > 1 else 3
        for ft_type in ("ln_head", "ln_head_patch", "ln_head_lora"):
            for h in (96, 192):
                for seed in args.seeds:
                    cfg = dict(base)
                    cfg.update(
                        tag=f"s2_ft_{ft_type}_{eff_q}_{eff_clip}",
                        dataset="ETTm1",
                        horizon=h,
                        seed=seed,
                        grouping="window_corr",
                        max_group_size=7,
                        sim_metric="pearson",
                        color_mode="random_adj",
                        align_mode=args.align_mode,
                        quantile_mode=eff_q,
                        clip_mode=eff_clip,
                        ft_type=ft_type,
                        epochs=eff_ep,
                    )
                    jobs.append(cfg)
        return jobs

    raise ValueError(f"Unknown stage={stage!r}")


def job_output_dir(save_root: Path, cfg: Dict[str, Any]) -> Path:
    return save_root / "runs" / cfg["tag"] / cfg["dataset"] / str(cfg["horizon"]) / f"seed_{cfg['seed']}"


def job_log_path(save_root: Path, cfg: Dict[str, Any]) -> Path:
    log_dir = save_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{cfg['tag']}__{cfg['dataset']}_H{cfg['horizon']}_seed{cfg['seed']}.log"


def is_job_completed(out_dir: Path) -> bool:
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return math.isfinite(float(data["mse"])) and math.isfinite(float(data["mae"]))
    except Exception:
        return False


def evaluate_split(model, windows, device, batch_size: int, progress, max_steps: int = 0, desc: str = "eval"):
    import numpy as np
    import torch
    from visiontspp_course.losses import crps_sum, pinball_loss

    model.eval()
    sq_sum = torch.zeros((), device=device, dtype=torch.float64)
    abs_sum = torch.zeros((), device=device, dtype=torch.float64)
    crps_total = torch.zeros((), device=device, dtype=torch.float64)
    loss_total = torch.zeros((), device=device, dtype=torch.float64)
    total_elems = 0

    starts = list(range(0, windows.count, batch_size))
    if max_steps > 0:
        starts = starts[:max_steps]

    with torch.no_grad():
        for start in progress(starts, desc=desc):
            indices = np.arange(start, min(start + batch_size, windows.count))
            hist, fut = windows.take(indices, device)
            y_point, y_q = model(hist)
            err = y_point - fut
            sq_sum += err.pow(2).sum(dtype=torch.float64)
            abs_sum += err.abs().sum(dtype=torch.float64)
            elems = int(fut.numel())
            total_elems += elems
            if model.use_quantile:
                loss_total += pinball_loss(fut, y_q, model.quantiles, reduction="sum").to(torch.float64)
                crps_total += crps_sum(fut, y_q, model.quantiles)
            else:
                loss_total += err.pow(2).sum(dtype=torch.float64)

    mse = float(sq_sum.item() / total_elems)
    mae = float(abs_sum.item() / total_elems)
    val_loss = float(loss_total.item() / total_elems)
    crps = float(crps_total.item() / total_elems) if model.use_quantile else float("nan")
    if not (math.isfinite(mse) and math.isfinite(mae) and math.isfinite(val_loss)):
        raise ValueError(f"Non-finite evaluation metrics: mse={mse}, mae={mae}, loss={val_loss}")
    return dict(mse=mse, mae=mae, crps=crps, loss=val_loss, elems=total_elems)


def run_single_job(
    cfg: Dict[str, Any],
    ckpt_path: Path,
    dataset_root: Path,
    out_dir: Path,
    gpu_label: str,
    progress,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from visiontspp_course.data import load_splits
    from visiontspp_course.grouping import build_static_groups
    from visiontspp_course.losses import filter_in_range_mask, pinball_loss
    from visiontspp_course.model import VisionTSppCourse

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu_label != "cpu" else "cpu")
    splits, spec, train_scaled = load_splits(
        cfg["dataset"], dataset_root=dataset_root, horizon=cfg["horizon"]
    )
    nvars = splits["train"].nvars

    static_perm, group_sizes = build_static_groups(
        nvars=nvars,
        grouping=cfg["grouping"],
        max_group_size=int(cfg["max_group_size"]),
        train_data=train_scaled,
        sim_metric=cfg["sim_metric"],
        use_abs=True,
    )

    model = VisionTSppCourse(
        arch=cfg["arch"],
        ckpt_path=str(ckpt_path),
        quantile_mode=cfg["quantile_mode"],
        color_mode=cfg["color_mode"],
        clip_mode=cfg["clip_mode"],
        soft_clip_linear_ratio=cfg["soft_clip_linear_ratio"],
        ft_type=cfg["ft_type"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        grouping=cfg["grouping"],
        sim_metric=cfg["sim_metric"],
    )
    model.update_config(
        context_len=spec.context,
        pred_len=cfg["horizon"],
        periodicity=spec.period,
        norm_const=spec.norm,
        align_const=spec.align,
        align_mode=cfg["align_mode"],
        interpolation="bilinear",
        group_sizes=group_sizes,
        static_perm=static_perm,
    )
    model = model.to(device)

    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    trainable_params = int(sum(p.numel() for p in trainable.values()))
    total_params = int(sum(p.numel() for p in model.parameters()))

    train_split = splits["train"]
    batch_size = int(cfg["batch_size"])
    imgs_per_win = max(len(group_sizes), 1)
    max_imgs = max(int(cfg.get("max_imgs_per_micro_batch", 32)), imgs_per_win)
    vram_safe_micro = max(1, max_imgs // imgs_per_win)
    micro_batch = min(int(cfg["micro_batch"]), batch_size, vram_safe_micro)
    eval_batch = max(1, min(max(micro_batch, 16), max(1, (max_imgs * 2) // imgs_per_win)))

    print(
        f"[{cfg['tag']}] {cfg['dataset']} H={cfg['horizon']} seed={seed} | "
        f"GPU={gpu_label} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) | "
        f"nvars={nvars}, groups={group_sizes} ({imgs_per_win} imgs/win) | "
        f"batch={batch_size}, micro={micro_batch} ({micro_batch * imgs_per_win} imgs/step) | "
        f"align={cfg['align_mode']} (patch_in={model.num_patch_input}, extra_pad={model.extra_padding}) | "
        f"q={cfg['quantile_mode']} clip={cfg['clip_mode']} ft={cfg['ft_type']} "
        f"trainable={trainable_params:,}/{total_params:,}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.Adam(trainable.values(), lr=float(cfg["lr"])) if trainable else None
    rng = np.random.default_rng(seed)

    epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    max_train_steps = int(cfg["max_train_steps"])
    max_eval_steps = int(cfg["max_eval_steps"])

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    epoch_times = []  # type: List[float]
    dropped_ratios = []  # type: List[float]

    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        order = rng.permutation(train_split.count)
        starts = list(range(0, train_split.count, batch_size))
        if max_train_steps > 0:
            starts = starts[:max_train_steps]

        ep_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ep_elems = 0
        ep_dropped = []  # type: List[float]

        desc = f"GPU{gpu_label} {cfg['tag']} {cfg['dataset']}/{cfg['horizon']} ep{epoch}/{epochs}"
        for step, start in enumerate(progress(starts, desc=desc), 1):
            batch_idx = order[start : start + batch_size]
            if optimizer is not None:
                optimizer.zero_grad()

            micro_chunks = []
            total_kept_elems = 0
            for offset in range(0, len(batch_idx), micro_batch):
                sub_idx = batch_idx[offset : offset + micro_batch]
                hist, fut = train_split.take(sub_idx, device)
                if cfg["clip_mode"] == "filter_train":
                    keep_mask, drop_frac = filter_in_range_mask(
                        hist,
                        norm_const=spec.norm,
                        max_outlier_ratio=float(cfg["filter_outlier_ratio"]),
                    )
                    ep_dropped.append(drop_frac)
                    hist = hist[keep_mask]
                    fut = fut[keep_mask]
                micro_chunks.append((hist, fut))
                total_kept_elems += int(fut.numel())

            for hist, fut in micro_chunks:
                y_point, y_q = model(hist)
                if model.use_quantile:
                    loss_sum = pinball_loss(fut, y_q, model.quantiles, reduction="sum")
                else:
                    loss_sum = F.mse_loss(y_point, fut, reduction="sum")
                if optimizer is not None:
                    (loss_sum / float(total_kept_elems)).backward()
                ep_loss_sum += loss_sum.detach().to(torch.float64)
                ep_elems += int(fut.numel())

            if optimizer is not None:
                optimizer.step()

            if step % 200 == 0:
                print(f"  Epoch {epoch}, step {step}/{len(starts)}", flush=True)

        ep_sec = time.time() - t_ep
        epoch_times.append(ep_sec)
        train_loss = float(ep_loss_sum.item() / max(ep_elems, 1))
        if ep_dropped:
            dropped_ratios.append(float(sum(ep_dropped) / len(ep_dropped)))

        val_metrics = evaluate_split(
            model,
            splits["val"],
            device=device,
            batch_size=eval_batch,
            progress=progress,
            max_steps=max_eval_steps,
            desc=f"GPU{gpu_label} {cfg['dataset']}/{cfg['horizon']} val {epoch}",
        )
        print(
            f"  Epoch {epoch}/{epochs} ({ep_sec:.1f}s): "
            f"train_loss={train_loss:.6f} | val_loss={val_metrics['loss']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_mae={val_metrics['mae']:.6f}"
            + (f" val_crps={val_metrics['crps']:.6f}" if model.use_quantile else "")
            + (f" dropped={dropped_ratios[-1]*100:.1f}%" if ep_dropped else ""),
            flush=True,
        )

        if val_metrics["loss"] <= best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        elif epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} (best epoch={best_epoch})", flush=True)
            break

        if optimizer is not None and epoch < epochs:
            adjust_lr_type1(optimizer, epoch + 1, float(cfg["lr"]))

    train_total_sec = time.time() - t_train_start

    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(best_state[name].to(device))

    test_metrics = evaluate_split(
        model,
        splits["test"],
        device=device,
        batch_size=eval_batch,
        progress=progress,
        max_steps=max_eval_steps,
        desc=f"GPU{gpu_label} {cfg['dataset']}/{cfg['horizon']} test",
    )

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "trainable_weights.pt")

    result = dict(
        stage=cfg["stage"],
        tag=cfg["tag"],
        dataset=cfg["dataset"],
        horizon=int(cfg["horizon"]),
        seed=seed,
        gpu=gpu_label,
        grouping=cfg["grouping"],
        max_group_size=int(cfg["max_group_size"]),
        group_sizes=group_sizes,
        images_per_window=imgs_per_win,
        sim_metric=cfg["sim_metric"],
        color_mode=cfg["color_mode"],
        align_mode=cfg["align_mode"],
        num_patch_input=int(model.num_patch_input),
        quantile_mode=cfg["quantile_mode"],
        clip_mode=cfg["clip_mode"],
        ft_type=cfg["ft_type"],
        epochs=epochs,
        best_epoch=best_epoch,
        mse=test_metrics["mse"],
        mae=test_metrics["mae"],
        crps=None if math.isnan(test_metrics["crps"]) else test_metrics["crps"],
        val_loss=best_val_loss,
        epoch_time_sec=float(sum(epoch_times) / max(len(epoch_times), 1)),
        train_time_sec=float(train_total_sec),
        peak_vram_mb=peak_vram_mb,
        trainable_params=trainable_params,
        filtered_train_ratio=float(sum(dropped_ratios) / len(dropped_ratios)) if dropped_ratios else 0.0,
    )
    print(
        f"  ==> TEST MSE={result['mse']:.6f}, MAE={result['mae']:.6f}"
        + (f", CRPS={result['crps']:.6f}" if result["crps"] is not None else "")
        + f" | ep_time={result['epoch_time_sec']:.1f}s | peak_vram={peak_vram_mb:.0f}MB",
        flush=True,
    )
    return result


def write_summaries(save_root: Path):
    rows = []
    for metrics_file in sorted((save_root / "runs").glob("*/*/*/seed_*/metrics.json")):
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            rows.append(data)
        except Exception:
            continue
    if not rows:
        return

    fields = [
        "stage", "tag", "dataset", "horizon", "seed", "gpu",
        "grouping", "max_group_size", "images_per_window", "sim_metric",
        "color_mode", "align_mode", "quantile_mode", "clip_mode", "ft_type",
        "epochs", "best_epoch", "mse", "mae", "crps", "val_loss",
        "epoch_time_sec", "peak_vram_mb", "trainable_params", "filtered_train_ratio",
    ]
    runs_tmp = save_root / "runs.csv.tmp"
    with runs_tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    runs_tmp.replace(save_root / "runs.csv")

    grouped = {}  # type: Dict[Tuple[str, str, int], List[Dict[str, Any]]]
    for r in rows:
        key = (r["tag"], r["dataset"], int(r["horizon"]))
        grouped.setdefault(key, []).append(r)

    summary_tmp = save_root / "summary.csv.tmp"
    with summary_tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tag", "dataset", "horizon", "runs",
            "mse_mean", "mae_mean", "crps_mean",
            "mse_std", "mae_std",
            "epoch_time_sec", "peak_vram_mb", "images_per_window", "trainable_params",
        ])
        for (tag, ds, h), grp in sorted(grouped.items()):
            mses = [r["mse"] for r in grp]
            maes = [r["mae"] for r in grp]
            crps_vals = [r["crps"] for r in grp if r.get("crps") is not None]
            ep_times = [r["epoch_time_sec"] for r in grp]
            vrams = [r["peak_vram_mb"] for r in grp]
            writer.writerow([
                tag, ds, h, len(grp),
                f"{statistics.mean(mses):.6f}",
                f"{statistics.mean(maes):.6f}",
                f"{statistics.mean(crps_vals):.6f}" if crps_vals else "",
                f"{statistics.stdev(mses):.6f}" if len(mses) > 1 else "",
                f"{statistics.stdev(maes):.6f}" if len(maes) > 1 else "",
                f"{statistics.mean(ep_times):.2f}",
                f"{statistics.mean(vrams):.1f}",
                grp[0]["images_per_window"],
                grp[0]["trainable_params"],
            ])
    summary_tmp.replace(save_root / "summary.csv")


def worker_loop(
    gpu: str,
    gpu_uuid: str,
    job_queue,
    ckpt_path: str,
    dataset_root: str,
    save_root: str,
    position: int,
    progress_lock,
    summary_lock,
):
    if gpu_uuid == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["MPLBACKEND"] = "Agg"

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from tqdm import tqdm

    tqdm.set_lock(progress_lock)
    torch.set_num_threads(2)

    if gpu_uuid != "cpu" and torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected 1 visible CUDA device for GPU {gpu} ({gpu_uuid}), got {torch.cuda.device_count()}")

    def progress(items, desc):
        return tqdm(
            items,
            desc=desc,
            position=position,
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.__stderr__,
            disable=not sys.__stderr__.isatty(),
        )

    save_dir = Path(save_root)
    ckpt = Path(ckpt_path)
    data_root = Path(dataset_root)
    tqdm.write(f"Worker started: physical GPU {gpu} ({gpu_uuid})", file=sys.__stderr__)

    for cfg in iter(job_queue.get, None):
        out_dir = job_output_dir(save_dir, cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = job_log_path(save_dir, cfg)
        tqdm.write(
            f"[GPU {gpu}] Start {cfg['tag']} | {cfg['dataset']} H={cfg['horizon']} seed={cfg['seed']}",
            file=sys.__stderr__,
        )
        with log_file.open("a", encoding="utf-8", buffering=1) as log_fh, redirect_stdout(log_fh), redirect_stderr(log_fh):
            try:
                result = run_single_job(cfg, ckpt, data_root, out_dir, gpu_label=gpu, progress=progress)
                tmp = out_dir / "metrics.json.tmp"
                tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                tmp.replace(out_dir / "metrics.json")
                with summary_lock:
                    write_summaries(save_dir)
                crps_msg = f" CRPS={result['crps']:.4f}" if result["crps"] is not None else ""
                tqdm.write(
                    f"[GPU {gpu}] Done  {cfg['tag']} | {cfg['dataset']} H={cfg['horizon']} seed={cfg['seed']} "
                    f"-> MSE={result['mse']:.4f} MAE={result['mae']:.4f}{crps_msg} "
                    f"({result['epoch_time_sec']:.1f}s/ep, {result['peak_vram_mb']:.0f}MB)",
                    file=sys.__stderr__,
                )
            except Exception:
                traceback.print_exc()
                tqdm.write(f"[GPU {gpu}] FAILED: see {log_file}", file=sys.__stderr__)
        if gpu_uuid != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("custom", "step1_smoke", "step1_quick", "step1_ablation", "step2_quantile", "step2_boundary", "step2_ft_scope"),
        default="custom",
        help="Run a predefined experimental stage or 'custom' single/grid run.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["2", "3"],
        help="Physical GPU indices for parallel execution (default: 2 3). Use '--gpus cpu' for CPU.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Backward-compatible alias for a single GPU (overrides --gpus if set).",
    )
    parser.add_argument("--ckpt-path", type=Path, default=REPO_ROOT / "ckpt" / "mae_visualize_vit_base.pth")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "save" / "course_exp",
        help="Single unified output folder for all logs, weights, metrics, runs.csv, and summary.csv.",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional experiment tag name for --stage custom.")

    # Custom grid parameters
    parser.add_argument("--datasets", nargs="+", default=["ETTm1"], help="Datasets to run.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[96], help="Forecast horizons.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2021], help="Random seeds.")

    # Architecture & Step 1 / Step 2 knobs
    parser.add_argument("--arch", choices=("mae_base", "mae_large", "mae_huge"), default="mae_base")
    parser.add_argument("--grouping", choices=("univariate", "all", "sequential", "train_corr", "window_corr"), default="window_corr")
    parser.add_argument("--max-group-size", type=int, default=7)
    parser.add_argument("--sim-metric", choices=("pearson", "cosine"), default="pearson")
    parser.add_argument("--color-mode", choices=("random_adj", "cyclic", "gray"), default="random_adj")
    parser.add_argument("--align-mode", choices=("half", "align_const"), default="half")
    parser.add_argument("--quantile-mode", choices=("none", "q5", "q9"), default="none")
    parser.add_argument("--clip-mode", choices=("none", "clip5", "hard_clip", "soft_clip", "filter_train"), default="none")
    parser.add_argument("--filter-outlier-ratio", type=float, default=0.01)
    parser.add_argument("--soft-clip-linear-ratio", type=float, default=0.7)
    parser.add_argument("--ft-type", choices=("ln", "ln_head", "ln_head_patch", "ln_head_lora", "bias", "full", "none"), default="ln")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)

    # Optimization & VRAM safety
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32, help="Multivariate windows per optimizer step.")
    parser.add_argument("--micro-batch", type=int, default=16, help="Max multivariate windows per micro-batch.")
    parser.add_argument(
        "--max-imgs-per-step",
        type=int,
        default=32,
        help="Cap MAE images per micro-batch to bound VRAM (~5-7 GB at 32 imgs) when sharing GPUs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)

    # Debugging / Control
    parser.add_argument("--max-train-steps", type=int, default=0, help="If >0, cap training steps per epoch (smoke test).")
    parser.add_argument("--max-eval-steps", type=int, default=0, help="If >0, cap val/test batches (smoke test).")
    parser.add_argument("--overwrite", action="store_true", help="Rerun even if metrics.json already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [args.gpu] if args.gpu is not None else list(dict.fromkeys(args.gpus))

    if not args.ckpt_path.is_file():
        print(f"ERROR: Checkpoint not found: {args.ckpt_path}", file=sys.stderr)
        print("Hint: run `bash visiontspp_course/scripts/setup_container.sh`", file=sys.stderr)
        return 1
    if not args.dataset_root.is_dir():
        print(f"ERROR: Dataset root not found: {args.dataset_root}", file=sys.stderr)
        print("Hint: run `bash visiontspp_course/scripts/setup_container.sh`", file=sys.stderr)
        return 1

    gpu_map = resolve_gpu_uuids(gpus)
    all_jobs = build_stage_jobs(args.stage, args)
    save_root = args.save_dir.resolve()
    (save_root / "runs").mkdir(parents=True, exist_ok=True)
    (save_root / "logs").mkdir(parents=True, exist_ok=True)

    pending = [
        cfg for cfg in all_jobs
        if args.overwrite or not is_job_completed(job_output_dir(save_root, cfg))
    ]
    skipped = len(all_jobs) - len(pending)

    manifest_entry = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        stage=args.stage,
        gpus=gpu_map,
        max_imgs_per_step=args.max_imgs_per_step,
        total_jobs=len(all_jobs),
        pending_jobs=len(pending),
    )
    with (save_root / "launch_history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry) + "\n")

    print(
        f"Stage={args.stage} | GPUs={gpus} (max_imgs_per_step={args.max_imgs_per_step}) | "
        f"Unified dir: {save_root} | {skipped} completed, {len(pending)} pending",
        flush=True,
    )
    if not pending:
        write_summaries(save_root)
        print(f"All jobs already completed. Summary: {save_root / 'summary.csv'}", flush=True)
        return 0

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    for cfg in pending:
        job_queue.put(cfg)

    active_gpus = gpus[: len(pending)]
    progress_lock = ctx.RLock()
    summary_lock = ctx.Lock()
    processes = [
        ctx.Process(
            target=worker_loop,
            args=(
                gpu,
                gpu_map[gpu],
                job_queue,
                str(args.ckpt_path),
                str(args.dataset_root),
                str(save_root),
                pos,
                progress_lock,
                summary_lock,
            ),
        )
        for pos, gpu in enumerate(active_gpus)
    ]

    for proc in processes:
        job_queue.put(None)
        proc.start()

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join()
        write_summaries(save_root)
        return 130

    write_summaries(save_root)
    done_now = sum(1 for cfg in all_jobs if is_job_completed(job_output_dir(save_root, cfg)))
    print(f"\nCompleted {done_now}/{len(all_jobs)} jobs. Unified summary: {save_root / 'summary.csv'}", flush=True)
    return int(done_now != len(all_jobs) or any(p.exitcode for p in processes))


if __name__ == "__main__":
    sys.exit(main())
