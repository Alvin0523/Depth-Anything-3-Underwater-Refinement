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
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from dataset import MIMIRSeaFloorDataset, UnityDepthDataset
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


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor -> uint8 HxWx3 RGB."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _save_vis(
    rgb: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
    frame_metrics: dict[str, float] | None = None,
) -> None:
    """Save a four-panel figure: RGB | predicted depth | GT depth | abs-rel error map."""
    vmax = float(gt[gt > 0].max()) if (gt > 0).any() else 1.0
    # Per-pixel abs-rel error: |pred-gt|/gt, masked to valid GT pixels
    valid = gt > 0
    err = np.zeros_like(gt)
    err[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    im1 = axes[1].imshow(pred, cmap="plasma", vmin=0, vmax=vmax)
    axes[1].set_title("Predicted depth")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(gt, cmap="plasma", vmin=0, vmax=vmax)
    axes[2].set_title("GT depth")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(err, cmap="hot", vmin=0, vmax=min(2.0, float(err[valid].max()) if valid.any() else 1.0))
    axes[3].set_title("Abs-rel error")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    if frame_metrics:
        info = "  ".join(f"{k}={v:.3f}" for k, v in frame_metrics.items())
        fig.suptitle(info, fontsize=8, y=1.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_evaluation(
    model,
    loader,
    device,
    vis_dir: Path | None = None,
    n_vis: int = 20,
    per_image_csv: Path | None = None,
) -> dict[str, float]:
    import csv

    model.eval()
    accum = {}
    n = 0
    vis_saved = 0
    vis_step = max(1, len(loader) // n_vis) if vis_dir else None

    csv_file = None
    csv_writer = None
    if per_image_csv is not None:
        per_image_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(per_image_csv, "w", newline="")
        fieldnames = ["frame", "absrel", "sqrel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    pbar = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
    global_frame = 0
    try:
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)   # (B, 3, H, W)
            gt     = batch["depth"].to(device)   # (B, H, W)
            paths  = batch.get("rgb_path", [None] * images.shape[0])

            x = images.unsqueeze(1)              # (B, 1, 3, H, W)
            with torch.autocast(device_type=device.type):
                out = model.model(x)
            depth = out.depth
            if depth.dim() == 4:
                depth = depth[:, 0]

            # Per-image metrics for CSV
            for i in range(images.shape[0]):
                im = compute_metrics(depth[i], gt[i])
                if csv_writer is not None:
                    frame_name = Path(paths[i]).name if paths[i] else str(global_frame)
                    csv_writer.writerow({"frame": frame_name, **{k: f"{v:.6f}" for k, v in im.items()}})
                global_frame += 1

            # Batch-level accumulation
            m = compute_metrics(depth, gt)
            for k, v in m.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1

            # Running metrics in tqdm postfix
            pbar.set_postfix({
                "absrel": f"{accum.get('absrel', 0)/n:.4f}",
                "d1": f"{accum.get('delta1', 0)/n:.4f}",
            })

            # Save visualisation samples
            if vis_dir is not None and vis_step is not None and batch_idx % vis_step == 0:
                rgb_np  = _denorm_rgb(images[0])
                pred_np = depth[0].cpu().float().numpy()
                gt_np   = gt[0].cpu().float().numpy()
                frame_m = compute_metrics(depth[0], gt[0])
                _save_vis(rgb_np, pred_np, gt_np, vis_dir / f"vis_{vis_saved:04d}.png", frame_metrics=frame_m)
                vis_saved += 1
    finally:
        if csv_file is not None:
            csv_file.close()

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
    p.add_argument("--dataset_type", default="unity", choices=["unity", "seafloor"],
                   help="Dataset format: 'unity' (rgb/+depth/) or 'seafloor' (MIMIR track*/auv0/...)")
    p.add_argument("--max_depth", type=float, default=None,
                   help="Max depth clip in metres (default: 10 for unity, 20 for seafloor)")
    p.add_argument("--vis_dir", default=None,
                   help="Directory to save sample visualisations (RGB | pred | GT). Created if needed.")
    p.add_argument("--n_vis", type=int, default=20,
                   help="Approx number of visualisation images to save.")
    p.add_argument("--out_json", default=None,
                   help="Path to save metrics as JSON (e.g. results/baseline.json).")
    p.add_argument("--per_image_csv", default=None,
                   help="Path to save per-image metrics as CSV (e.g. results/baseline_per_image.csv).")
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

    # Dataset
    if args.dataset_type == "seafloor":
        max_depth = args.max_depth if args.max_depth is not None else 20.0
        ds = MIMIRSeaFloorDataset(
            args.data_root,
            input_size=(args.input_size, args.input_size),
            max_depth=max_depth,
        )
    else:
        max_depth = args.max_depth if args.max_depth is not None else 10.0
        ds = UnityDepthDataset(
            args.data_root,
            input_size=(args.input_size, args.input_size),
            split="train",
            val_fraction=0.0,
            max_depth=max_depth,
        )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = Path(args.per_image_csv) if args.per_image_csv else None
    metrics = run_evaluation(
        da3, loader, device,
        vis_dir=vis_dir, n_vis=args.n_vis,
        per_image_csv=per_image_csv,
    )

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

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"label": label, "dataset": args.data_root, "n_frames": len(ds), **metrics}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval] Metrics saved to {out_path}")

    if vis_dir:
        print(f"[eval] {args.n_vis} visualisations saved to {vis_dir}")


if __name__ == "__main__":
    main()
