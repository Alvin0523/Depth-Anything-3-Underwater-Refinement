"""
LoRA fine-tuning script for DA3 Mono Metric Large on Unity underwater data.

Usage:
    python train/train.py --data_root data/unity --epochs 30 --batch_size 4

Checkpoints saved to: checkpoints/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from depth_anything_3.api import DepthAnything3

from dataset import UnityDepthDataset
from lora import inject_lora, freeze_backbone, count_trainable
from losses import UnderwaterDepthLoss


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    mask = (gt > 0) & torch.isfinite(gt) & (pred > 0)
    if mask.sum() == 0:
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}

    p = pred[mask]
    g = gt[mask]

    absrel = ((p - g).abs() / g).mean().item()
    rmse = ((p - g).pow(2).mean()).sqrt().item()
    ratio = torch.max(p / g, g / p)
    delta1 = (ratio < 1.25).float().mean().item()
    return {"absrel": absrel, "rmse": rmse, "delta1": delta1}


# ─── Forward pass helper ───────────────────────────────────────────────────────

def run_forward(da3_api: DepthAnything3, images: torch.Tensor) -> torch.Tensor:
    """
    images: (B, 3, H, W) normalised tensors.
    Returns pred_depth: (B, H, W).
    """
    # DA3 expects (B, N, 3, H, W) where N=1 for monocular
    x = images.unsqueeze(1)
    with torch.autocast(device_type=images.device.type):
        out = da3_api.model(x)
    # out.depth shape: (B, N, H, W) or (B, H, W) depending on model version
    depth = out.depth
    if depth.dim() == 4:
        depth = depth[:, 0]  # take first view
    return depth


# ─── Train / Val loops ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser, device, epoch):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device)   # (B, 3, H, W)
        gt_depth = batch["depth"].to(device) # (B, H, W)

        pred_depth = run_forward(model, images)

        losses = criterion(pred_depth, gt_depth)
        optimiser.zero_grad()
        losses["loss"].backward()
        optimiser.step()

        total_loss += losses["loss"].item()
        if step % 10 == 0:
            print(
                f"  [epoch {epoch} step {step}/{len(loader)}]"
                f"  loss={losses['loss'].item():.4f}"
                f"  silog={losses['silog'].item():.4f}"
                f"  grad={losses['grad'].item():.4f}"
            )
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"absrel": 0.0, "rmse": 0.0, "delta1": 0.0}
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)

        pred_depth = run_forward(model, images)
        losses = criterion(pred_depth, gt_depth)
        total_loss += losses["loss"].item()

        m = compute_metrics(pred_depth, gt_depth)
        for k in all_metrics:
            all_metrics[k] += m[k]
        n += 1

    for k in all_metrics:
        all_metrics[k] /= max(n, 1)
    return total_loss / max(n, 1), all_metrics


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data/unity")
    p.add_argument("--model_name",  default="da3metric-large",
                   help="DA3 model name: da3metric-large or da3mono-large")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lora_rank",   type=int,   default=8)
    p.add_argument("--lora_alpha",  type=float, default=16.0)
    p.add_argument("--grad_weight", type=float, default=0.5,
                   help="Weight of gradient loss term")
    p.add_argument("--input_size",  type=int,   default=518,
                   help="Square input size (must be multiple of 14)")
    p.add_argument("--val_fraction",type=float, default=0.2)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  model={args.model_name}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Load pretrained model ──────────────────────────────────────────────────
    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}")
    da3 = da3.to(device)

    # ── Inject LoRA ────────────────────────────────────────────────────────────
    inject_lora(da3.model, rank=args.lora_rank, alpha=args.lora_alpha)
    freeze_backbone(da3)
    # Head stays fully trainable (already unfrozen)

    trainable, total = count_trainable(da3)
    print(f"[LoRA] Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    # ── Dataset ────────────────────────────────────────────────────────────────
    input_size = (args.input_size, args.input_size)
    train_ds = UnityDepthDataset(args.data_root, input_size=input_size, split="train",
                                  val_fraction=args.val_fraction)
    val_ds   = UnityDepthDataset(args.data_root, input_size=input_size, split="val",
                                  val_fraction=args.val_fraction)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Loss & optimiser ───────────────────────────────────────────────────────
    criterion = UnderwaterDepthLoss(grad_weight=args.grad_weight).to(device)
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, da3.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_absrel = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        da3.load_state_dict(ckpt["model"], strict=False)
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        best_absrel = ckpt.get("best_absrel", float("inf"))
        print(f"[train] Resumed from epoch {ckpt['epoch']}")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(da3, train_loader, criterion, optimiser, device, epoch)
        val_loss, metrics = validate(da3, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[epoch {epoch}/{args.epochs}]"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  AbsRel={metrics['absrel']:.4f}"
            f"  RMSE={metrics['rmse']:.4f}"
            f"  delta1={metrics['delta1']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": da3.state_dict(),
            "optimiser": optimiser.state_dict(),
            "best_absrel": best_absrel,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pth"))

        if metrics["absrel"] < best_absrel:
            best_absrel = metrics["absrel"]
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pth"))
            print(f"  --> New best AbsRel: {best_absrel:.4f}")

    print(f"[train] Done. Best AbsRel: {best_absrel:.4f}")


if __name__ == "__main__":
    main()
