"""
Minimal LoRA pipeline smoke test — no real data needed.

Run this on NSCC first to verify the environment is set up correctly
before Unity data is ready.

What it checks:
  1. DA3 loads from HuggingFace
  2. LoRA injection works
  3. Forward pass runs
  4. Loss computes and backward pass works
  5. Checkpoint save/load works

Usage:
    python train/test_lora_pipeline.py

Expected output (no errors, ends with "ALL CHECKS PASSED"):
    [1/5] Loading da3metric-large from HuggingFace...  OK
    [2/5] Injecting LoRA (rank=8)...  OK  -- trainable: X / Y (Z%)
    [3/5] Forward pass with dummy batch...  OK  -- depth shape: torch.Size([2, 518, 518])
    [4/5] Backward pass (loss + optimiser step)...  OK  -- loss: X.XXXX
    [5/5] Checkpoint save + reload...  OK
    ALL CHECKS PASSED
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from depth_anything_3.api import DepthAnything3
from lora import inject_lora, freeze_backbone, count_trainable, merge_lora_weights
from losses import UnderwaterDepthLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 2
H, W   = 518, 518   # must be multiple of 14


def make_dummy_batch():
    images = torch.randn(BATCH, 3, H, W, device=DEVICE)
    depth  = torch.rand(BATCH, H, W, device=DEVICE) * 8.0 + 0.5  # 0.5–8.5 m
    return images, depth


def check(step, desc):
    print(f"[{step}] {desc}...", end="  ", flush=True)


def ok(msg=""):
    print(f"OK  {msg}")


# ── 1. Load model ──────────────────────────────────────────────────────────────
check("1/5", "Loading da3metric-large from HuggingFace")
da3 = DepthAnything3.from_pretrained("depth-anything/da3metric-large")
da3 = da3.to(DEVICE)
ok()

# ── 2. Inject LoRA ─────────────────────────────────────────────────────────────
check("2/5", "Injecting LoRA (rank=8)")
inject_lora(da3.model, rank=8, alpha=16.0)
freeze_backbone(da3)
trainable, total = count_trainable(da3)
ok(f"-- trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── 3. Forward pass ────────────────────────────────────────────────────────────
check("3/5", "Forward pass with dummy batch")
images, gt_depth = make_dummy_batch()
with torch.autocast(device_type=DEVICE.type):
    out = da3.model(images.unsqueeze(1))   # (B,1,3,H,W)
depth = out.depth
if depth.dim() == 4:
    depth = depth[:, 0]
assert depth.shape == (BATCH, H, W), f"Unexpected depth shape: {depth.shape}"
ok(f"-- depth shape: {depth.shape}")

# ── 4. Backward pass ───────────────────────────────────────────────────────────
check("4/5", "Backward pass (loss + optimiser step)")
criterion = UnderwaterDepthLoss().to(DEVICE)
optimiser = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, da3.parameters()), lr=1e-4
)

da3.train()
images, gt_depth = make_dummy_batch()
with torch.autocast(device_type=DEVICE.type):
    out = da3.model(images.unsqueeze(1))
pred = out.depth[:, 0] if out.depth.dim() == 4 else out.depth
losses = criterion(pred, gt_depth)
optimiser.zero_grad()
losses["loss"].backward()
optimiser.step()
ok(f"-- loss: {losses['loss'].item():.4f}")

# ── 5. Checkpoint save + reload ────────────────────────────────────────────────
check("5/5", "Checkpoint save + reload")
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "test.pth")
    torch.save({"model": da3.state_dict(), "epoch": 1}, ckpt_path)

    da3_reload = DepthAnything3.from_pretrained("depth-anything/da3metric-large", local_files_only=True).to(DEVICE)
    inject_lora(da3_reload.model, rank=8, alpha=16.0)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    da3_reload.load_state_dict(ckpt["model"], strict=False)
ok()

print("\nALL CHECKS PASSED")
print(f"Device: {DEVICE}  |  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
