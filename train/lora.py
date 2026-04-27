"""
LoRA injection for DA3's DINOv2 backbone.

Wraps the qkv and proj Linear layers in every Attention block with LoRA adapters.
Everything else (DualDPT head, norms, etc.) stays fully trainable or frozen
depending on the caller's freeze strategy.
"""

import math
import torch
import torch.nn as nn
from depth_anything_3.model.dinov2.layers.attention import Attention


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank LoRA side-path."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.in_features
        out_features = base.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Return a plain Linear with LoRA weights baked in (for export)."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        )
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged.weight = nn.Parameter(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias = nn.Parameter(self.base.bias.clone())
        return merged


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """
    Walk the model and replace qkv/proj Linear layers inside every
    Attention block with LoRALinear wrappers.

    Returns the model in-place (also returned for convenience).
    """
    replaced = 0
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        module.qkv = LoRALinear(module.qkv, rank=rank, alpha=alpha)
        module.proj = LoRALinear(module.proj, rank=rank, alpha=alpha)
        replaced += 1
    print(f"[LoRA] Injected into {replaced} Attention blocks (rank={rank}, alpha={alpha})")
    return model


def freeze_backbone(model: nn.Module):
    """
    Freeze everything in the backbone except LoRA A/B weights.
    The DPT/DualDPT head is left fully trainable.
    """
    backbone = model.model.backbone
    for name, param in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Replace all LoRALinear layers with merged plain Linears (for export)."""
    for module in model.modules():
        if not isinstance(module, Attention):
            continue
        if isinstance(module.qkv, LoRALinear):
            module.qkv = module.qkv.merge_weights()
        if isinstance(module.proj, LoRALinear):
            module.proj = module.proj.merge_weights()
    return model
