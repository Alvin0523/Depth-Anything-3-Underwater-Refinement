"""
LoRA fine-tuning script for DA3 Mono Metric Large on Unity underwater data.

Usage:
    python train/train.py --data_root data/unity --epochs 30 --batch_size 4

Checkpoints saved to: checkpoints/
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from depth_anything_3.api import DepthAnything3

from dataset import UnityDepthDataset
from lora import inject_lora, freeze_backbone, count_trainable
from losses import UnderwaterDepthLoss


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    # Clamp pred so negative model outputs don't silently exclude most pixels
    pred = pred.clamp(min=1e-4)
    mask = (gt > 0) & torch.isfinite(gt)
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
    Returns pred_depth: (B, H, W) as float32.
    """
    # DA3 expects (B, N, 3, H, W) where N=1 for monocular
    x = images.unsqueeze(1)
    # bfloat16 autocast: native on A100, no scaler needed, numerically stable
    with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
        out = da3_api.model(x)
    # out.depth shape: (B, N, H, W) or (B, H, W) depending on model version
    depth = out.depth
    if depth.dim() == 4:
        depth = depth[:, 0]  # take first view
    # Cast back to float32: loss functions and Sobel buffers are float32
    return depth.float()


# ─── Train / Val loops ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimiser, device, epoch, writer, global_step):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()
    log_interval_start = time.time()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device)   # (B, 3, H, W)
        gt_depth = batch["depth"].to(device) # (B, H, W)

        pred_depth = run_forward(model, images)

        losses = criterion(pred_depth, gt_depth)
        optimiser.zero_grad()
        losses["loss"].backward()
        # Clip gradients to prevent single-batch explosions
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimiser.step()

        total_loss += losses["loss"].item()
        global_step += 1

        # ── TensorBoard per-step scalars ──────────────────────────────────────
        writer.add_scalar("train/loss",  losses["loss"].item(),  global_step)
        writer.add_scalar("train/silog", losses["silog"].item(), global_step)
        writer.add_scalar("train/grad",  losses["grad"].item(),  global_step)

        if step % 10 == 0:
            avg_step_time = (time.time() - log_interval_start) / max(step % 10 if step % 10 else 10, 1)
            print(
                f"  [epoch {epoch} step {step}/{len(loader)}  gs={global_step}]"
                f"  loss={losses['loss'].item():.4f}"
                f"  silog={losses['silog'].item():.4f}"
                f"  grad={losses['grad'].item():.4f}"
                f"  avg_step={avg_step_time:.1f}s",
                flush=True,
            )
            log_interval_start = time.time()

    print(
        f"  [epoch {epoch}] train done in {time.time()-epoch_start:.1f}s"
        f"  avg_loss={total_loss/len(loader):.4f}",
        flush=True,
    )
    return total_loss / len(loader), global_step


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metric_sums = {"absrel": 0.0, "rmse": 0.0, "delta1": 0.0}
    metric_counts = {"absrel": 0, "rmse": 0, "delta1": 0}
    val_start = time.time()
    for batch in loader:
        images = batch["image"].to(device)
        gt_depth = batch["depth"].to(device)

        pred_depth = run_forward(model, images)
        losses = criterion(pred_depth, gt_depth)
        total_loss += losses["loss"].item()

        m = compute_metrics(pred_depth, gt_depth)
        for k in metric_sums:
            # skip NaN batches (all-zero mask) so they don't poison the average
            if not (m[k] != m[k]):  # isnan check without importing math
                metric_sums[k] += m[k]
                metric_counts[k] += 1

    all_metrics = {
        k: metric_sums[k] / metric_counts[k] if metric_counts[k] > 0 else float("nan")
        for k in metric_sums
    }
    print(f"  [val] done in {time.time()-val_start:.1f}s", flush=True)
    return total_loss / max(sum(metric_counts.values()) // len(metric_counts), 1), all_metrics


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
    p.add_argument("--grad_weight",  type=float, default=0.5,
                   help="Weight of gradient loss term")
    p.add_argument("--scale_weight", type=float, default=0.1,
                   help="Scale anchor weight in SILog (prevents AbsRel divergence)")
    p.add_argument("--grad_clip",    type=float, default=1.0,
                   help="Max gradient norm for clipping (0 = disabled)")
    p.add_argument("--input_size",  type=int,   default=518,
                   help="Square input size (must be multiple of 14)")
    p.add_argument("--val_fraction",type=float, default=0.2)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--log_dir",     default=None,
                   help="TensorBoard log directory (default: <ckpt_dir>/tb_logs)")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    p.add_argument("--hf_repo",     default="Frieddeli/COMP4471",
                   help="HuggingFace repo to upload best checkpoint after training")
    p.add_argument("--no_upload",   action="store_true",
                   help="Skip HuggingFace upload after training")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Fail fast: require a GPU ───────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Check that CUDA_HOME / LD_LIBRARY_PATH point to the pixi env lib directory."
        )
    device = torch.device("cuda")
    print(
        f"[train] device={device}  gpu={torch.cuda.get_device_name(0)}"
        f"  vram={torch.cuda.get_device_properties(0).total_memory // 1024**3}GiB"
        f"  model={args.model_name}",
        flush=True,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── TensorBoard writer ─────────────────────────────────────────────────────
    log_dir = args.log_dir or os.path.join(args.ckpt_dir, "tb_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[train] TensorBoard logs -> {log_dir}", flush=True)
    print(f"        Launch with: tensorboard --logdir {log_dir}", flush=True)

    # ── Load pretrained model ──────────────────────────────────────────────────
    print(f"[train] Loading pretrained model depth-anything/{args.model_name} ...", flush=True)
    t0 = time.time()
    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}")
    da3 = da3.to(device)
    print(f"[train] Model loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Inject LoRA ────────────────────────────────────────────────────────────
    inject_lora(da3.model, rank=args.lora_rank, alpha=args.lora_alpha)
    freeze_backbone(da3)
    # Head stays fully trainable (already unfrozen)
    # LoRA Linear layers are created on CPU by default — move everything to device
    da3 = da3.to(device)

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
    criterion = UnderwaterDepthLoss(
        grad_weight=args.grad_weight,
        scale_weight=args.scale_weight,
    ).to(device)
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, da3.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 1
    best_absrel = float("inf")
    global_step = 0

    if args.resume:
        print(f"[train] Resuming from {args.resume} ...", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        da3.load_state_dict(ckpt["model"], strict=False)
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        best_absrel = ckpt.get("best_absrel", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"[train] Resumed from epoch {ckpt['epoch']}  global_step={global_step}  best_absrel={best_absrel:.4f}", flush=True)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[train] ===== Epoch {epoch}/{args.epochs} =====", flush=True)
        train_loss, global_step = train_one_epoch(
            da3, train_loader, criterion, optimiser, device, epoch, writer, global_step
        )
        val_loss, metrics = validate(da3, val_loader, criterion, device)

        current_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()

        # ── TensorBoard per-epoch scalars ─────────────────────────────────────
        writer.add_scalar("epoch/train_loss", train_loss,         epoch)
        writer.add_scalar("epoch/val_loss",   val_loss,           epoch)
        writer.add_scalar("epoch/absrel",     metrics["absrel"],  epoch)
        writer.add_scalar("epoch/rmse",       metrics["rmse"],    epoch)
        writer.add_scalar("epoch/delta1",     metrics["delta1"],  epoch)
        writer.add_scalar("epoch/lr",         current_lr,         epoch)
        writer.flush()

        print(
            f"[epoch {epoch}/{args.epochs}]"
            f"  train_loss={train_loss:.4f}"
            f"  val_loss={val_loss:.4f}"
            f"  AbsRel={metrics['absrel']:.4f}"
            f"  RMSE={metrics['rmse']:.4f}"
            f"  delta1={metrics['delta1']:.4f}"
            f"  lr={current_lr:.2e}",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": da3.state_dict(),
            "optimiser": optimiser.state_dict(),
            "best_absrel": best_absrel,
            "args": vars(args),
        }
        last_path = os.path.join(args.ckpt_dir, "last.pth")
        torch.save(ckpt, last_path)
        print(f"  [ckpt] Saved last checkpoint -> {last_path}", flush=True)

        if metrics["absrel"] < best_absrel:
            best_absrel = metrics["absrel"]
            best_path = os.path.join(args.ckpt_dir, "best.pth")
            torch.save(ckpt, best_path)
            print(f"  [ckpt] NEW BEST AbsRel={best_absrel:.4f} -> {best_path}", flush=True)

    writer.close()
    print(f"[train] Done. Best AbsRel: {best_absrel:.4f}", flush=True)

    # ── Upload best checkpoint to HuggingFace ──────────────────────────────────
    if not args.no_upload:
        best_path = os.path.join(args.ckpt_dir, "best.pth")
        if os.path.exists(best_path):
            print(f"[train] Uploading {best_path} to HuggingFace repo '{args.hf_repo}'...")
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=os.environ.get("HF_TOKEN"))
                api.create_repo(repo_id=args.hf_repo, exist_ok=True, private=False)
                api.upload_file(
                    path_or_fileobj=best_path,
                    path_in_repo="checkpoints/best.pth",
                    repo_id=args.hf_repo,
                    commit_message=f"LoRA best checkpoint (AbsRel={best_absrel:.4f})",
                )
                print(f"[train] Upload complete: https://huggingface.co/{args.hf_repo}")
            except Exception as e:
                print(f"[train] WARNING: HF upload failed: {e}")
        else:
            print("[train] WARNING: best.pth not found, skipping upload.")


if __name__ == "__main__":
    main()
