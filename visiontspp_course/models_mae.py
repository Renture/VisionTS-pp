"""PyTorch 1.7.1 / timm 0.3.2 compatible MAE backbone with VisionTS++ quantile heads and LoRA."""

from functools import partial
import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from .pos_embed import get_2d_sincos_pos_embed


class LoRALinear(nn.Module):
    """Lightweight pure-PyTorch LoRA wrapper around an existing nn.Linear."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.base = base_linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)

        self.lora_A = nn.Parameter(torch.empty(self.r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out + delta


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone and VisionTS++ multi-quantile heads."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss: bool = False,
        quantile: bool = False,
        quantile_head_num: int = 9,
    ):
        super().__init__()

        self.norm_pix_loss = norm_pix_loss
        self.quantile = quantile
        self.quantile_head_num = int(quantile_head_num)
        self._lora_injected = False

        # MAE encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Primary prediction head (point forecast or 50% median head in quantile mode)
        out_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(decoder_embed_dim, out_dim, bias=True)
        if self.quantile:
            if self.quantile_head_num < 3 or self.quantile_head_num % 2 == 0:
                raise ValueError("quantile_head_num must be an odd integer >= 3 (e.g. 5 or 9)")
            self.decoder_pred_quantile_list = nn.ModuleList(
                [nn.Linear(decoder_embed_dim, out_dim, bias=True) for _ in range(self.quantile_head_num - 1)]
            )

        self.initialize_weights()

    def initialize_weights(self):
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def init_quantile_heads_from_median(self):
        """Copy pretrained decoder_pred weights into all extra quantile heads to preserve visual priors."""
        if not self.quantile:
            return
        with torch.no_grad():
            for head in self.decoder_pred_quantile_list:
                head.weight.data.copy_(self.decoder_pred.weight.data)
                if head.bias is not None and self.decoder_pred.bias is not None:
                    head.bias.data.copy_(self.decoder_pred.bias.data)

    def inject_lora(self, r: int = 8, alpha: float = 16.0, targets: Iterable[str] = ("qkv", "proj")):
        """Wrap attention linear projections in encoder and decoder blocks with LoRALinear."""
        if self._lora_injected:
            return
        targets = tuple(targets)
        for block_list in (self.blocks, self.decoder_blocks):
            for blk in block_list:
                attn = blk.attn
                for attr in targets:
                    if hasattr(attn, attr):
                        layer = getattr(attn, attr)
                        if isinstance(layer, nn.Linear):
                            setattr(attn, attr, LoRALinear(layer, r=r, alpha=alpha))
        self._lora_injected = True

    def configure_finetune(self, ft_type: str = "ln", lora_r: int = 8, lora_alpha: float = 16.0):
        """Set requires_grad according to fine-tuning scope."""
        valid = {"ln", "ln_head", "ln_head_patch", "ln_head_lora", "bias", "full", "none"}
        if ft_type not in valid:
            raise ValueError(f"Unknown ft_type={ft_type!r}. Choose from {sorted(valid)}")

        if ft_type == "ln_head_lora":
            self.inject_lora(r=lora_r, alpha=lora_alpha)

        for name, param in self.named_parameters():
            if name in ("pos_embed", "decoder_pos_embed"):
                param.requires_grad = False
                continue

            if ft_type == "full":
                param.requires_grad = True
            elif ft_type == "none":
                param.requires_grad = False
            elif ft_type == "bias":
                param.requires_grad = "bias" in name
            elif ft_type == "ln":
                param.requires_grad = "norm" in name
            elif ft_type == "ln_head":
                param.requires_grad = ("norm" in name) or ("decoder_pred" in name)
            elif ft_type == "ln_head_patch":
                param.requires_grad = (
                    ("norm" in name) or ("decoder_pred" in name) or ("patch_embed" in name)
                )
            elif ft_type == "ln_head_lora":
                param.requires_grad = (
                    ("norm" in name) or ("decoder_pred" in name) or ("lora_" in name)
                )

    def unpatchify(self, x: torch.Tensor, n_channels: int = 3) -> torch.Tensor:
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape((x.shape[0], h, w, p, p, n_channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape((x.shape[0], n_channels, h * p, w * p))

    def random_masking(self, x: torch.Tensor, mask_ratio: float, noise: torch.Tensor = None):
        n, length, dim = x.shape
        len_keep = int(round(length * (1.0 - mask_ratio)))
        if noise is None:
            noise = torch.rand(n, length, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones([n, length], device=x.device)
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float, noise: torch.Tensor = None):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x_mid = self.decoder_pred(x)[:, 1:, :]
        if not self.quantile:
            return x_mid

        x_quantiles = [head(x)[:, 1:, :] for head in self.decoder_pred_quantile_list]
        return x_mid, x_quantiles

    def forward(
        self, imgs: torch.Tensor, mask_ratio: float = 0.75, noise: torch.Tensor = None
    ) -> Tuple[None, object, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, noise)
        pred = self.forward_decoder(latent, ids_restore)
        return None, pred, mask


def mae_vit_base_patch16(**kwargs) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_vit_large_patch16(**kwargs) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_vit_huge_patch14(**kwargs) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
