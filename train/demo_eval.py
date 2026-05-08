"""
Live presentation demo — runs evaluation with a pop-up window showing:

  Top row  : RGB | Predicted depth | GT depth | Abs-rel error map  (updates every N batches)
  Bottom row: Rolling line charts for AbsRel, RMSE, δ<1.25 vs frames processed

Usage (baseline):
    python train/demo_eval.py \
        --data_root /path/to/SeaFloor \
        --model_name da3metric-large \
        --dataset_type seafloor

Usage (fine-tuned):
    python train/demo_eval.py \
        --data_root /path/to/SeaFloor \
        --model_name da3metric-large \
        --checkpoint checkpoints/best.pth \
        --lora_rank 8 --lora_alpha 16.0 \
        --dataset_type seafloor

Press Ctrl-C to stop early; the window stays open until you close it.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3

from dataset import MIMIRSeaFloorDataset, UnityDepthDataset
from lora import inject_lora

# ── constants ──────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

METRIC_KEYS   = ["absrel", "sqrel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
METRIC_LABELS = ["AbsRel ↓", "SqRel ↓", "RMSE ↓", "RMSE_log ↓", "δ<1.25 ↑", "δ<1.25² ↑", "δ<1.25³ ↑"]
BETTER_LOW    = [True,       True,      True,    True,         False,      False,       False]


# ── helpers ────────────────────────────────────────────────────────────────────
def _denorm_rgb(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    mask = (gt > 0) & torch.isfinite(gt) & (pred > 0) & torch.isfinite(pred)
    if mask.sum() == 0:
        return {k: float("nan") for k in METRIC_KEYS}
    p, g = pred[mask], gt[mask]
    ratio = torch.max(p / g, g / p)
    return {
        "absrel":   ((p - g).abs() / g).mean().item(),
        "sqrel":    (((p - g) ** 2) / g).mean().item(),
        "rmse":     ((p - g).pow(2).mean()).sqrt().item(),
        "rmse_log": ((torch.log(p) - torch.log(g)).pow(2).mean()).sqrt().item(),
        "delta1":   (ratio < 1.25).float().mean().item(),
        "delta2":   (ratio < 1.25 ** 2).float().mean().item(),
        "delta3":   (ratio < 1.25 ** 3).float().mean().item(),
    }


# ── live figure ────────────────────────────────────────────────────────────────
class LiveDemo:
    """Manages the matplotlib figure for live presentation."""

    GRAPH_KEYS    = ["absrel", "rmse", "delta1"]
    GRAPH_LABELS  = ["AbsRel ↓", "RMSE ↓", "δ<1.25 ↑"]
    GRAPH_COLORS  = ["#e74c3c", "#e67e22", "#2ecc71"]
    GRAPH_BETTER  = [False, False, True]   # True = higher is better

    def __init__(self, title: str, total_frames: int):
        plt.ion()
        self.fig = plt.figure(figsize=(28, 14), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("DA3 Underwater — Live Eval Demo")
        # Maximise window (works with TkAgg)
        try:
            self.fig.canvas.manager.window.state("zoomed")
        except Exception:
            try:
                self.fig.canvas.manager.window.attributes("-zoomed", True)
            except Exception:
                pass
        self.total_frames = total_frames

        gs = gridspec.GridSpec(
            2, 4,
            figure=self.fig,
            top=0.92, bottom=0.04,
            left=0.04, right=0.97,
            hspace=0.18, wspace=0.15,
            height_ratios=[2.0, 1.4],
        )

        # ── top row: 4 image panels ──────────────────────────────────────────
        self.ax_rgb   = self.fig.add_subplot(gs[0, 0])
        self.ax_pred  = self.fig.add_subplot(gs[0, 1])
        self.ax_gt    = self.fig.add_subplot(gs[0, 2])
        self.ax_err   = self.fig.add_subplot(gs[0, 3])

        panel_titles = ["RGB Input", "Predicted Depth", "GT Depth", "Abs-Rel Error Map"]
        for ax, t in zip(
            [self.ax_rgb, self.ax_pred, self.ax_gt, self.ax_err], panel_titles
        ):
            ax.set_title(t, color="white", fontsize=30, pad=6)
            ax.axis("off")
            ax.set_facecolor("#1a1a2e")

        # Create image objects once with a correctly-sized placeholder.
        H = 518
        blank_rgb   = np.zeros((H, H, 3), dtype=np.uint8)
        blank_depth = np.zeros((H, H),    dtype=np.float32)

        self.im_rgb  = self.ax_rgb.imshow(blank_rgb,   interpolation="nearest")
        self.im_pred = self.ax_pred.imshow(blank_depth, cmap="plasma", vmin=0, vmax=1,
                                           interpolation="nearest")
        self.im_gt   = self.ax_gt.imshow(blank_depth,   cmap="plasma", vmin=0, vmax=1,
                                          interpolation="nearest")
        self.im_err  = self.ax_err.imshow(blank_depth,  cmap="hot",    vmin=0, vmax=2,
                                          interpolation="nearest")

        # Colorbars — created once, never recreated
        self.cb_pred = self.fig.colorbar(self.im_pred, ax=self.ax_pred, fraction=0.046, pad=0.04)
        self.cb_gt   = self.fig.colorbar(self.im_gt,   ax=self.ax_gt,   fraction=0.046, pad=0.04)
        self.cb_err  = self.fig.colorbar(self.im_err,  ax=self.ax_err,  fraction=0.046, pad=0.04)
        for cb in [self.cb_pred, self.cb_gt, self.cb_err]:
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=18)

        # ── bottom row: 3 metric line charts ─────────────────────────────────
        self.graph_axes = []
        self.graph_lines = []
        self.history: dict[str, list[float]] = {k: [] for k in self.GRAPH_KEYS}
        self.frame_history: list[int] = []

        for col, (key, label, color, better) in enumerate(
            zip(self.GRAPH_KEYS, self.GRAPH_LABELS, self.GRAPH_COLORS, self.GRAPH_BETTER)
        ):
            ax = self.fig.add_subplot(gs[1, col])
            ax.set_facecolor("#0f3460")
            ax.tick_params(colors="white", labelsize=22)
            ax.spines[:].set_color("#444")
            ax.set_title(label, color=color, fontsize=30, pad=4)
            ax.set_xlabel("Frames processed", color="#aaa", fontsize=24)
            (line,) = ax.plot([], [], color=color, linewidth=1.5)
            self.graph_axes.append(ax)
            self.graph_lines.append(line)

        # summary text box (bottom right panel)
        self.ax_summary = self.fig.add_subplot(gs[1, 3])
        self.ax_summary.set_facecolor("#0f3460")
        self.ax_summary.axis("off")
        self.ax_summary.set_title("Running averages", color="white", fontsize=30, pad=4)
        self.summary_text = self.ax_summary.text(
            0.05, 0.95, "", transform=self.ax_summary.transAxes,
            va="top", ha="left", fontsize=28, color="white",
            fontfamily="monospace",
        )

        self.fig.suptitle(title, color="white", fontsize=38, fontweight="bold", y=0.97)
        self._progress_text = self.fig.text(
            0.5, 0.93, "", ha="center", color="#aaa", fontsize=24
        )

        plt.pause(0.01)

    def update(
        self,
        rgb_np: np.ndarray,
        pred_np: np.ndarray,
        gt_np: np.ndarray,
        running_avg: dict[str, float],
        frames_done: int,
    ) -> None:
        valid = gt_np > 0
        vmax  = float(gt_np[valid].max()) if valid.any() else 1.0

        err = np.zeros_like(gt_np)
        if valid.any():
            err[valid] = np.abs(pred_np[valid] - gt_np[valid]) / gt_np[valid]
        err_max = min(2.0, float(err[valid].max()) if valid.any() else 1.0)

        # --- image panels: update data and clim only (no ax.cla, no new colorbars) ---
        self.im_rgb.set_data(rgb_np)

        self.im_pred.set_data(pred_np)
        self.im_pred.set_clim(0, vmax)
        self.cb_pred.mappable.set_clim(0, vmax)

        self.im_gt.set_data(gt_np)
        self.im_gt.set_clim(0, vmax)
        self.cb_gt.mappable.set_clim(0, vmax)

        self.im_err.set_data(err)
        self.im_err.set_clim(0, err_max)
        self.cb_err.mappable.set_clim(0, err_max)

        # graphs
        self.frame_history.append(frames_done)
        for key in self.GRAPH_KEYS:
            self.history[key].append(running_avg.get(key, float("nan")))

        for ax, line, key in zip(self.graph_axes, self.graph_lines, self.GRAPH_KEYS):
            vals = self.history[key]
            line.set_xdata(self.frame_history)
            line.set_ydata(vals)
            finite = [v for v in vals if not np.isnan(v)]
            if finite:
                lo, hi = min(finite), max(finite)
                pad = max((hi - lo) * 0.1, 1e-4)
                ax.set_xlim(0, max(self.total_frames, frames_done))
                ax.set_ylim(lo - pad, hi + pad)

        # summary text
        lines = []
        for key, label in zip(METRIC_KEYS, METRIC_LABELS):
            v = running_avg.get(key, float("nan"))
            lines.append(f"{label:<14s} {v:.4f}")
        self.summary_text.set_text("\n".join(lines))

        # progress
        pct = 100.0 * frames_done / max(self.total_frames, 1)
        self._progress_text.set_text(
            f"Progress: {frames_done:,} / {self.total_frames:,} frames  ({pct:.1f}%)"
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def final(self, metrics: dict[str, float]) -> None:
        lines = ["FINAL RESULTS\n"]
        for key, label, better in zip(METRIC_KEYS, METRIC_LABELS, BETTER_LOW):
            v = metrics.get(key, float("nan"))
            lines.append(f"{label:<14s} {v:.4f}")
        self.summary_text.set_text("\n".join(lines))
        self._progress_text.set_text("Evaluation complete — close window to exit")
        self.fig.canvas.draw_idle()
        plt.ioff()
        plt.show()


# ── eval loop ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_demo(model, loader, device, demo: LiveDemo, update_every: int) -> dict[str, float]:
    model.eval()
    accum: dict[str, float] = {}
    n = 0
    frames_done = 0

    pbar = tqdm(loader, desc="Evaluating", unit="batch", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        gt     = batch["depth"].to(device)

        x = images.unsqueeze(1)
        with torch.autocast(device_type=device.type):
            out = model.model(x)
        depth = out.depth
        if depth.dim() == 4:
            depth = depth[:, 0]

        m = compute_metrics(depth, gt)
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
        n += 1
        frames_done += images.shape[0]

        running_avg = {k: v / n for k, v in accum.items()}
        pbar.set_postfix({
            "absrel": f"{running_avg['absrel']:.4f}",
            "d1":     f"{running_avg['delta1']:.4f}",
        })

        if batch_idx % update_every == 0:
            rgb_np  = _denorm_rgb(images[0])
            pred_np = depth[0].cpu().float().numpy()
            gt_np   = gt[0].cpu().float().numpy()
            demo.update(rgb_np, pred_np, gt_np, running_avg, frames_done)

    return {k: v / max(n, 1) for k, v in accum.items()}


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--model_name",   default="da3metric-large")
    p.add_argument("--checkpoint",   default=None)
    p.add_argument("--lora_rank",    type=int,   default=8)
    p.add_argument("--lora_alpha",   type=float, default=16.0)
    p.add_argument("--input_size",   type=int,   default=518)
    p.add_argument("--batch_size",   type=int,   default=2)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--dataset_type", default="seafloor", choices=["unity", "seafloor"])
    p.add_argument("--max_depth",    type=float, default=None)
    p.add_argument("--update_every", type=int,   default=10,
                   help="Refresh the live window every N batches (default: 10).")
    p.add_argument("--stride", type=int, default=1,
                   help="Only evaluate every Nth sample. E.g. --stride 10 uses ~1/10 of frames (default: 1 = all).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}")

    if args.checkpoint:
        inject_lora(da3.model, rank=args.lora_rank, alpha=args.lora_alpha)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        da3.load_state_dict(ckpt["model"], strict=False)
        label = f"DA3 + LoRA fine-tuned  |  {Path(args.checkpoint).name}"
        print(f"[demo] Loaded fine-tuned weights from {args.checkpoint}")
    else:
        label = "DA3 Metric Large  |  Zero-shot baseline"
        print("[demo] Zero-shot baseline")

    da3 = da3.to(device)

    if args.dataset_type == "seafloor":
        max_depth = args.max_depth or 20.0
        ds = MIMIRSeaFloorDataset(
            args.data_root,
            input_size=(args.input_size, args.input_size),
            max_depth=max_depth,
        )
    else:
        max_depth = args.max_depth or 10.0
        ds = UnityDepthDataset(
            args.data_root,
            input_size=(args.input_size, args.input_size),
            split="train", val_fraction=0.0,
            max_depth=max_depth,
        )

    if args.stride > 1:
        indices = list(range(0, len(ds), args.stride))
        ds = torch.utils.data.Subset(ds, indices)
        print(f"[demo] Stride={args.stride}: using {len(ds)} / {len(indices)*args.stride} frames")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    demo = LiveDemo(
        title=f"COMP4471 — Underwater Depth Evaluation\n{label}",
        total_frames=len(ds),
    )

    try:
        metrics = run_demo(da3, loader, device, demo, update_every=args.update_every)
    except KeyboardInterrupt:
        print("\n[demo] Stopped early.")
        plt.ioff()
        plt.show()
        return

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Dataset: {args.data_root}  ({len(ds)} frames)")
    print(f"{'='*55}")
    for key, lbl in zip(METRIC_KEYS, METRIC_LABELS):
        print(f"  {lbl:<16s}: {metrics[key]:.4f}")
    print(f"{'='*55}\n")

    demo.final(metrics)


if __name__ == "__main__":
    main()
