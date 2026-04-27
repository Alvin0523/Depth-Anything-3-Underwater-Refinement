"""
Evaluate baseline vs fine-tuned DA3 on a test set.

Works with:
  - Your Unity held-out test folder  (same rgb/ + depth/ layout)
  - FLSea benchmark  (same layout, depth in metres as .npy or uint16 PNG)
  - SQUID benchmark  (same layout)

Usage:
    # Baseline (no fine-tuning):
    python train/evaluate.py --data_root data/unity/test --model_name da3metric-large

    # Fine-tuned:
    python train/evaluate.py --data_root data/unity/test --model_name da3metric-large \
        --checkpoint checkpoints/best.pth

    # FLSea benchmark:
    python train/evaluate.py --data_root data/flsea --model_name da3metric-large \
        --checkpoint checkpoints/best.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from depth_anything_3.api import DepthAnything3

from dataset import UnityDepthDataset
from lora import inject_lora


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    mask = (gt > 0) & torch.isfinite(gt) & (pred > 0) & torch.isfinite(pred)
    if mask.sum() == 0:
        return {k: float("nan") for k in ["absrel", "sqrel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]}

    p = pred[mask]
    g = gt[mask]

    absrel  = ((p - g).abs() / g).mean().item()
    sqrel   = (((p - g) ** 2) / g).mean().item()
    rmse    = ((p - g).pow(2).mean()).sqrt().item()
    rmse_log = ((torch.log(p) - torch.log(g)).pow(2).mean()).sqrt().item()

    ratio   = torch.max(p / g, g / p)
    delta1  = (ratio < 1.25).float().mean().item()
    delta2  = (ratio < 1.25 ** 2).float().mean().item()
    delta3  = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "absrel": absrel, "sqrel": sqrel,
        "rmse": rmse, "rmse_log": rmse_log,
        "delta1": delta1, "delta2": delta2, "delta3": delta3,
    }


@torch.no_grad()
def run_evaluation(model, loader, device) -> dict[str, float]:
    model.eval()
    accum = {}
    n = 0

    for batch in loader:
        images = batch["image"].to(device)   # (B, 3, H, W)
        gt     = batch["depth"].to(device)   # (B, H, W)

        x = images.unsqueeze(1)              # (B, 1, 3, H, W)
        with torch.autocast(device_type=device.type):
            out = model.model(x)
        depth = out.depth
        if depth.dim() == 4:
            depth = depth[:, 0]

        m = compute_metrics(depth, gt)
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1

    return {k: v / max(n, 1) for k, v in accum.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  required=True, help="Test data directory (rgb/ + depth/)")
    p.add_argument("--model_name", default="da3metric-large")
    p.add_argument("--checkpoint", default=None,
                   help="Path to fine-tuned checkpoint (.pth). Omit for zero-shot baseline.")
    p.add_argument("--lora_rank",  type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--input_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}")

    if args.checkpoint:
        inject_lora(da3.model, rank=args.lora_rank, alpha=args.lora_alpha)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        da3.load_state_dict(ckpt["model"], strict=False)
        print(f"[eval] Loaded fine-tuned weights from {args.checkpoint}")
    else:
        print("[eval] Zero-shot baseline (no fine-tuning)")

    da3 = da3.to(device)

    # Dataset — use full folder as test (val_fraction=0 => all samples in 'train' split)
    ds = UnityDepthDataset(
        args.data_root,
        input_size=(args.input_size, args.input_size),
        split="train",
        val_fraction=0.0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    metrics = run_evaluation(da3, loader, device)

    label = f"Fine-tuned ({args.checkpoint})" if args.checkpoint else "Baseline (zero-shot)"
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Dataset: {args.data_root}  ({len(ds)} frames)")
    print(f"{'='*55}")
    print(f"  AbsRel   : {metrics['absrel']:.4f}")
    print(f"  SqRel    : {metrics['sqrel']:.4f}")
    print(f"  RMSE     : {metrics['rmse']:.4f}")
    print(f"  RMSE_log : {metrics['rmse_log']:.4f}")
    print(f"  delta<1.25  : {metrics['delta1']:.4f}")
    print(f"  delta<1.25^2: {metrics['delta2']:.4f}")
    print(f"  delta<1.25^3: {metrics['delta3']:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
