"""VisionTSppCourse: Colorized multivariate stacking + correlation grouping +
multi-quantile heads + ImageNet boundary adaptation.

100% compatible with Python 3.8, PyTorch 1.7.1, torchvision 0.8.2, timm 0.3.2.
"""

import inspect
import math
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize

from .grouping import compute_window_permutation
from .models_mae import mae_vit_base_patch16, mae_vit_huge_patch14, mae_vit_large_patch16

MAE_ARCH = {
    "mae_base": mae_vit_base_patch16,
    "mae_large": mae_vit_large_patch16,
    "mae_huge": mae_vit_huge_patch14,
}

# ImageNet valid normalized range across RGB channels:
# [-mean/std, (1 - mean)/std], intersection across 3 channels matches official VisionTS++
IMAGENET_THRES_DOWN = -1.8044444444444445  # max([-2.1179, -2.0357, -1.8044])
IMAGENET_THRES_UP = 2.2489082969432315     # min([2.2489, 2.4286, 2.6400])

QUANTILE_LEVELS = {
    "none": [],
    "q5": [0.1, 0.3, 0.5, 0.7, 0.9],
    "q9": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}


def safe_resize(size: Tuple[int, int], interpolation: int):
    params = inspect.signature(Resize).parameters
    if "antialias" in params:
        return Resize(size, interpolation, antialias=False)
    return Resize(size, interpolation)


def soft_clip_to_range(
    x: torch.Tensor,
    lo: float = IMAGENET_THRES_DOWN,
    hi: float = IMAGENET_THRES_UP,
    linear_ratio: float = 0.7,
) -> torch.Tensor:
    """C1-smooth compression into (lo, hi) that is exact identity inside the inner linear core."""
    mid = 0.5 * (lo + hi)
    half_width = 0.5 * (hi - lo)
    r_lin = half_width * float(linear_ratio)
    r_margin = half_width - r_lin

    d = x - mid
    abs_d = d.abs()
    excess = F.relu(abs_d - r_lin)
    compressed = mid + torch.sign(d) * (r_lin + r_margin * torch.tanh(excess / r_margin))
    return torch.where(abs_d <= r_lin, x, compressed)


class VisionTSppCourse(nn.Module):
    """Unified VisionTS++ course-improvement model supporting Step 1 and Step 2 ablations."""

    def __init__(
        self,
        arch: str = "mae_base",
        ckpt_path: Optional[str] = None,
        quantile_mode: str = "none",
        color_mode: str = "random_adj",
        clip_mode: str = "none",
        soft_clip_linear_ratio: float = 0.7,
        ft_type: str = "ln",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        grouping: str = "window_corr",
        sim_metric: str = "pearson",
        use_abs_sim: bool = True,
    ):
        super().__init__()
        if arch not in MAE_ARCH:
            raise ValueError(f"Unknown arch={arch!r}. Choose from {list(MAE_ARCH.keys())}")
        if quantile_mode not in QUANTILE_LEVELS:
            raise ValueError(f"Unknown quantile_mode={quantile_mode!r}. Choose from {list(QUANTILE_LEVELS)}")
        if color_mode not in ("random_adj", "cyclic", "gray"):
            raise ValueError(f"Unknown color_mode={color_mode!r}")
        if clip_mode not in ("none", "clip5", "hard_clip", "soft_clip", "filter_train"):
            raise ValueError(f"Unknown clip_mode={clip_mode!r}")

        self.arch = arch
        self.quantile_mode = quantile_mode
        self.quantiles = list(QUANTILE_LEVELS[quantile_mode])
        self.use_quantile = len(self.quantiles) > 0
        self.color_mode = color_mode
        self.clip_mode = clip_mode
        self.soft_clip_linear_ratio = float(soft_clip_linear_ratio)
        self.ft_type = ft_type
        self.grouping = grouping
        self.sim_metric = sim_metric
        self.use_abs_sim = bool(use_abs_sim)

        q_num = len(self.quantiles) if self.use_quantile else 9
        self.vision_model = MAE_ARCH[arch](quantile=self.use_quantile, quantile_head_num=q_num)

        if ckpt_path is not None:
            self._load_pretrained(ckpt_path)

        # Configure trainable parameters (including optional LoRA injection)
        self.vision_model.configure_finetune(ft_type=ft_type, lora_r=lora_r, lora_alpha=lora_alpha)

        # Populated by update_config
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size
        self.group_sizes = None  # type: Optional[List[int]]
        self.static_perm = None  # type: Optional[List[int]]

    def _load_pretrained(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state = dict(state)

        # Compatibility with VisionTS++ checkpoint key naming (decoder_pred_1 -> decoder_pred)
        if any("decoder_pred_1" in k for k in state.keys()):
            for k in list(state.keys()):
                if "decoder_pred_1" in k:
                    state[k.replace("decoder_pred_1", "decoder_pred")] = state.pop(k)
                elif not self.use_quantile and ("decoder_pred_2" in k or "decoder_pred_3" in k or "decoder_pred_quantile" in k):
                    state.pop(k)

        has_quantile_weights = any("decoder_pred_quantile_list" in k for k in state.keys())
        missing, unexpected = self.vision_model.load_state_dict(state, strict=False)

        # When starting from ImageNet MAE (single decoder_pred) with quantile heads enabled,
        # copy pretrained decoder_pred weights into every quantile head to protect visual priors.
        if self.use_quantile and not has_quantile_weights:
            self.vision_model.init_quantile_heads_from_median()
            missing = [k for k in missing if not k.startswith("decoder_pred_quantile_list.")]

        if missing:
            raise RuntimeError(f"Unexpected missing keys when loading {ckpt_path}: {missing}")
        if unexpected:
            print(f"[VisionTSppCourse] Note: ignored unexpected checkpoint keys: {unexpected[:5]}")

    def update_config(
        self,
        context_len: int,
        pred_len: int,
        periodicity: int,
        norm_const: float = 0.4,
        align_const: float = 0.4,
        align_mode: str = "half",
        interpolation: str = "bilinear",
        padding_mode: str = "replicate",
        group_sizes: Optional[List[int]] = None,
        static_perm: Optional[List[int]] = None,
    ):
        if align_mode not in ("half", "align_const"):
            raise ValueError(f"Unknown align_mode={align_mode!r}. Choose 'half' or 'align_const'")

        self.raw_context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.periodicity = int(periodicity)
        self.norm_const = float(norm_const)
        self.align_const = float(align_const)
        self.align_mode = align_mode
        self.padding_mode = padding_mode

        interp_map = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }
        if interpolation not in interp_map:
            raise ValueError(f"Unknown interpolation={interpolation!r}")
        self.interpolation = interp_map[interpolation]

        self.pad_right = 0
        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - (self.pred_len % self.periodicity)

        # Determine input patch width and effective context length (with extra left padding if needed)
        self.extra_padding = 0
        self.context_len_eff = self.raw_context_len

        if align_mode == "half":
            self.num_patch_input = self.num_patch // 2  # 7 for 224/16 -> exact W/2 = 112px
            self.num_patch_output = self.num_patch - self.num_patch_input
            min_ctx = int(math.ceil(self.pred_len * self.num_patch_input / float(self.num_patch_output)))
            if min_ctx > self.context_len_eff:
                self.extra_padding = min_ctx - self.context_len_eff
                self.context_len_eff += self.extra_padding
        else:
            pad_left_tmp = (
                0 if self.context_len_eff % self.periodicity == 0
                else self.periodicity - (self.context_len_eff % self.periodicity)
            )
            input_ratio = float(pad_left_tmp + self.context_len_eff) / float(
                pad_left_tmp + self.context_len_eff + self.pad_right + self.pred_len
            )
            self.num_patch_input = max(1, int(input_ratio * self.num_patch * self.align_const))
            self.num_patch_output = self.num_patch - self.num_patch_input

        self.adjust_input_ratio = float(self.num_patch_input) / float(self.num_patch)
        self.input_width = int(self.image_size * self.adjust_input_ratio)
        self.output_width = self.num_patch_output * self.patch_size

        self.pad_left = (
            0 if self.context_len_eff % self.periodicity == 0
            else self.periodicity - (self.context_len_eff % self.periodicity)
        )

        # Guarantee decoded flattened length strictly covers pad_left + context_len_eff + pred_len
        while True:
            num_periods_in = (self.pad_left + self.context_len_eff) // self.periodicity
            self.scale_x = float(num_periods_in) / float(self.input_width)
            self.total_periods_out = int(round(self.image_size * self.scale_x))
            if self.total_periods_out * self.periodicity >= self.pad_left + self.context_len_eff + self.pred_len:
                break
            self.extra_padding += self.periodicity
            self.context_len_eff += self.periodicity

        self.output_resize = safe_resize(
            (self.periodicity, self.total_periods_out), interpolation=self.interpolation
        )

        # Register MAE patch mask (0 = keep left context, 1 = mask right future)
        device = self.vision_model.cls_token.device
        mask = torch.ones((self.num_patch, self.num_patch), device=device)
        mask[:, : self.num_patch_input] = 0.0
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = float(torch.mean(mask).item())

        self.group_sizes = list(group_sizes) if group_sizes is not None else None
        self.static_perm = list(static_perm) if static_perm is not None else None
        if self.static_perm is not None:
            self.register_buffer(
                "_static_perm_buf", torch.tensor(self.static_perm, dtype=torch.long, device=device)
            )
        else:
            self._static_perm_buf = None

    def _sample_color_indices(self, batch_size: int, nvars: int, device: torch.device) -> torch.Tensor:
        """Return [batch_size, nvars] tensor of RGB channel indices in {0, 1, 2}."""
        if self.color_mode == "cyclic" or (not self.training):
            base = torch.arange(nvars, device=device, dtype=torch.long) % 3
            return base.unsqueeze(0).expand(batch_size, nvars)

        # random_adj during training: adjacent variables guaranteed distinct RGB channels
        colors = torch.empty((batch_size, nvars), dtype=torch.long, device=device)
        colors[:, 0] = torch.randint(0, 3, (batch_size,), device=device)
        for k in range(1, nvars):
            step = torch.randint(1, 3, (batch_size,), device=device)  # 1 or 2 -> != prev
            colors[:, k] = (colors[:, k - 1] + step) % 3
        return colors

    def _build_color_mask(
        self, color_indices: torch.Tensor, h_per_var: int, pad_down: int
    ) -> torch.Tensor:
        """Build [N, 3, H, 1] binary channel mask selecting each variable's single RGB channel."""
        n_batch, nvars = color_indices.shape
        device = color_indices.device
        # Expand [N, nvars] -> [N, nvars * h_per_var]
        row_colors = color_indices.unsqueeze(-1).expand(n_batch, nvars, h_per_var).reshape(n_batch, -1)
        mask_active = torch.zeros(
            (n_batch, 3, nvars * h_per_var), dtype=torch.float32, device=device
        )
        mask_active.scatter_(1, row_colors.unsqueeze(1), 1.0)
        if pad_down > 0:
            pad = torch.zeros((n_batch, 3, pad_down), dtype=torch.float32, device=device)
            mask_full = torch.cat([mask_active, pad], dim=2)
        else:
            mask_full = mask_active
        return mask_full.unsqueeze(-1)  # [N, 3, H, 1]

    def _apply_clipping(self, image_input: torch.Tensor) -> torch.Tensor:
        if self.clip_mode == "none" or self.clip_mode == "filter_train":
            return image_input
        if self.clip_mode == "clip5":
            return torch.clamp(image_input, -5.0, 5.0)
        if self.clip_mode == "hard_clip":
            return torch.clamp(image_input, IMAGENET_THRES_DOWN, IMAGENET_THRES_UP)
        if self.clip_mode == "soft_clip":
            return soft_clip_to_range(
                image_input,
                lo=IMAGENET_THRES_DOWN,
                hi=IMAGENET_THRES_UP,
                linear_ratio=self.soft_clip_linear_ratio,
            )
        raise ValueError(f"Unexpected clip_mode={self.clip_mode!r}")

    def _forward_equal_size_groups(
        self, x_enc_groups: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run MAE imaging + reconstruction on normalized equal-size groups.

        Args:
            x_enc_groups: normalized tensor [N, g_size, seq_len]

        Returns:
            median_pred: [N, pred_len, g_size] (normalized scale)
            all_quantiles: [N, Q, pred_len, g_size] if use_quantile else None (normalized scale)
        """
        n_batch, g_size, _ = x_enc_groups.shape
        device = x_enc_groups.device
        dtype = x_enc_groups.dtype

        # 1. Left padding & period folding
        total_pad_left = self.pad_left + self.extra_padding
        if total_pad_left > 0:
            x_pad = F.pad(x_enc_groups, (total_pad_left, 0), mode=self.padding_mode)
        else:
            x_pad = x_enc_groups

        x_2d = einops.rearrange(x_pad, "n v (p f) -> (n v) 1 f p", f=self.periodicity)

        # 2. Safe resize per variable (C=1 guarantees compatibility with torchvision 0.8.2)
        h_per_var = self.image_size // g_size
        if h_per_var < 1:
            raise ValueError(f"g_size={g_size} exceeds image_size={self.image_size}")
        input_resize = safe_resize((h_per_var, self.input_width), interpolation=self.interpolation)
        x_resize = input_resize(x_2d)  # [(N*g_size), 1, h_per_var, input_width]
        x_resize = einops.rearrange(x_resize, "(n v) 1 h w -> n 1 (v h) w", n=n_batch, v=g_size)

        pad_down = self.image_size - x_resize.shape[2]
        if pad_down > 0:
            pad_zeros = torch.zeros(
                (n_batch, 1, pad_down, self.input_width), device=device, dtype=dtype
            )
            x_resize = torch.cat([x_resize, pad_zeros], dim=2)

        masked = torch.zeros(
            (n_batch, 1, self.image_size, self.output_width), device=device, dtype=dtype
        )
        x_canvas = torch.cat([x_resize, masked], dim=-1)  # [N, 1, 224, 224]

        # 3. Colorization
        if self.color_mode == "gray":
            image_input = x_canvas.repeat(1, 3, 1, 1)
            color_mask = None
        else:
            color_indices = self._sample_color_indices(n_batch, g_size, device)
            color_mask = self._build_color_mask(color_indices, h_per_var, pad_down).to(dtype=dtype)
            image_input = x_canvas * color_mask

        # 4. Boundary adaptation
        image_input = self._apply_clipping(image_input)

        # 5. MAE forward
        noise = einops.repeat(self.mask, "1 l -> n l", n=n_batch)
        _, raw_pred, _ = self.vision_model(image_input, mask_ratio=self.mask_ratio, noise=noise)

        def decode_tokens(tokens: torch.Tensor) -> torch.Tensor:
            img_rec = self.vision_model.unpatchify(tokens)  # [N, 3, 224, 224]
            if color_mask is None:
                y_grey = img_rec.mean(dim=1, keepdim=True)
            else:
                y_grey = (img_rec * color_mask).sum(dim=1, keepdim=True)
            if pad_down > 0:
                y_grey = y_grey[:, :, :-pad_down, :]
            y_vars = einops.rearrange(y_grey, "n 1 (v h) w -> (n v) 1 h w", v=g_size)
            y_seg = self.output_resize(y_vars)  # [(N*v), 1, periodicity, total_periods_out]
            y_flat = einops.rearrange(y_seg, "(n v) 1 f p -> n (p f) v", n=n_batch, v=g_size)
            start_idx = self.pad_left + self.context_len_eff
            end_idx = start_idx + self.pred_len
            return y_flat[:, start_idx:end_idx, :]

        if not self.use_quantile:
            y_pred = decode_tokens(raw_pred)
            return y_pred, None

        mid_tokens, other_tokens_list = raw_pred
        y_mid = decode_tokens(mid_tokens)
        y_others = [decode_tokens(tok) for tok in other_tokens_list]
        half = len(y_others) // 2
        ordered = y_others[:half] + [y_mid] + y_others[half:]
        y_all_q = torch.stack(ordered, dim=1)  # [N, Q, pred_len, g_size]
        return y_mid, y_all_q

    def forward(
        self, x: torch.Tensor, fp64: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forecast future windows from look-back window x: [B, context_len, M].

        Returns:
            (y_point, y_quantiles) where:
              - y_point has shape [B, pred_len, M] (50% median when quantile mode is on)
              - y_quantiles has shape [B, Q, pred_len, M] if quantile mode is on else None
        """
        batch_size, seq_len, nvars = x.shape
        group_sizes = self.group_sizes if self.group_sizes is not None else [nvars]

        # 1. Per-variable RevIN-style normalization (exact VisionTS / VisionTS++ formula)
        means = x.mean(dim=1, keepdim=True).detach()  # [B, 1, M]
        x_enc = x - means
        stdev = torch.sqrt(
            torch.var(x_enc.to(torch.float64) if fp64 else x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).to(x.dtype)
        stdev = stdev / self.norm_const
        x_enc = x_enc / stdev  # [B, L, M]

        # 2. Variable grouping / permutation
        if self.grouping == "window_corr":
            perm = compute_window_permutation(
                x_enc, group_sizes, metric=self.sim_metric, use_abs=self.use_abs_sim
            )  # [B, M]
            inv_perm = torch.argsort(perm, dim=1)
            x_perm = torch.gather(
                x_enc, dim=2, index=perm.unsqueeze(1).expand(batch_size, seq_len, nvars)
            )
        elif self._static_perm_buf is not None:
            perm_1d = self._static_perm_buf
            inv_perm = None
            inv_perm_1d = torch.argsort(perm_1d)
            x_perm = x_enc.index_select(dim=2, index=perm_1d)
        else:
            inv_perm = None
            inv_perm_1d = None
            x_perm = x_enc

        # Rearrange to [B, M, L] and bucket chunks by group size for maximum batch parallelism
        x_perm_bml = einops.rearrange(x_perm, "b s m -> b m s")
        buckets = {}  # type: Dict[int, List[Tuple[int, torch.Tensor]]]
        offset = 0
        for g_idx, g_size in enumerate(group_sizes):
            chunk = x_perm_bml[:, offset : offset + g_size, :]
            buckets.setdefault(g_size, []).append((g_idx, chunk))
            offset += g_size

        group_point_outs = [None] * len(group_sizes)
        group_quant_outs = [None] * len(group_sizes) if self.use_quantile else None

        for g_size, items in buckets.items():
            stacked = torch.cat([chunk for _, chunk in items], dim=0)  # [K_g * B, g_size, L]
            pred_mid, pred_q = self._forward_equal_size_groups(stacked)
            mid_splits = torch.split(pred_mid, batch_size, dim=0)
            q_splits = torch.split(pred_q, batch_size, dim=0) if self.use_quantile else None
            for idx_in_bucket, (g_idx, _) in enumerate(items):
                group_point_outs[g_idx] = mid_splits[idx_in_bucket]
                if self.use_quantile:
                    group_quant_outs[g_idx] = q_splits[idx_in_bucket]

        y_perm = torch.cat(group_point_outs, dim=2)  # [B, pred_len, M]
        if self.use_quantile:
            y_q_perm = torch.cat(group_quant_outs, dim=3)  # [B, Q, pred_len, M]
        else:
            y_q_perm = None

        # 3. Restore original CSV variable order
        if self.grouping == "window_corr":
            y_norm = torch.gather(
                y_perm, dim=2, index=inv_perm.unsqueeze(1).expand(batch_size, self.pred_len, nvars)
            )
            if self.use_quantile:
                q_num = y_q_perm.shape[1]
                y_q_norm = torch.gather(
                    y_q_perm,
                    dim=3,
                    index=inv_perm.unsqueeze(1).unsqueeze(2).expand(batch_size, q_num, self.pred_len, nvars),
                )
            else:
                y_q_norm = None
        elif inv_perm_1d is not None:
            y_norm = y_perm.index_select(dim=2, index=inv_perm_1d)
            y_q_norm = y_q_perm.index_select(dim=3, index=inv_perm_1d) if self.use_quantile else None
        else:
            y_norm = y_perm
            y_q_norm = y_q_perm

        # 4. Denormalization
        y_point = y_norm * stdev + means
        if self.use_quantile:
            y_quantiles = y_q_norm * stdev.unsqueeze(1) + means.unsqueeze(1)
        else:
            y_quantiles = None

        return y_point, y_quantiles
