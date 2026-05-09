# 🔬 Implementation

What every file in the `train/` directory implements, and how the components fit together.

---

## Overview

```
train/
├── dataset.py             # data loading + preprocessing
├── lora.py                # LoRA adapter injection
├── losses.py              # training objective
├── train.py               # main training loop
├── evaluate.py            # offline benchmarking
├── demo_eval.py           # live visualisation demo
└── test_lora_pipeline.py  # smoke test (no data needed)
```

---

## `train/dataset.py`

Loads MIMIR-UW RGB+depth pairs and applies physics-aware preprocessing.

### `UnityDepthDataset`

Training dataset for the SeaFloor Algae environment (NPY inverse-depth files).

| Method | Description |
|---|---|
| `__init__()` | Scans `rgb/` and `depth/` subdirs, builds paired file list, applies 80/20 split at seed 42 |
| `_load_depth()` | Loads `.npy` as float32 inverse-depth → inverts to metres; or `.png` uint16 ÷ 1000 |
| `__getitem__()` | Loads pair → gray world WB → histogram stretch → resize to 518×518 → ImageNet normalise → depth clip to (0, max_depth] → tensors |
| `gray_world_white_balance()` | Per-channel multiplicative scaling to equalise channel means |
| `histogram_stretch()` | Per-channel 2nd–98th percentile linear stretch |

### `MIMIRSeaFloorDataset`

Benchmark dataset for the held-out SeaFloor environment (EXR inverse-depth, multi-track/multi-cam layout).

Walks `root/track*/auv0/rgb/cam{0,1}/data/*.png` and corresponding EXR depth files. Applies the same preprocessing pipeline as `UnityDepthDataset`.

---

## `train/lora.py`

Injects LoRA adapters into the DINOv2-L backbone.

### `LoRALinear`

Drop-in `nn.Linear` replacement that adds a low-rank side-path:

```
output = W_frozen(x) + B(A(x)) * (alpha / rank)
```

- `lora_A`: `(in_features, rank)` — initialised kaiming uniform
- `lora_B`: `(rank, out_features)` — initialised zeros (delta = 0 at init)
- `base.weight` and `base.bias` are frozen (`.requires_grad_(False)`)

### Key Functions

| Function | Description |
|---|---|
| `inject_lora(model, rank=8, alpha=16.0)` | Walks all `Attention` blocks in the DINOv2 backbone and replaces `qkv` and `proj` Linear layers with `LoRALinear` |
| `freeze_backbone(model)` | Freezes all backbone parameters except `lora_A` and `lora_B` matrices; DPT prediction head stays fully trainable |
| `count_trainable(model)` | Returns `(trainable_params, total_params)` — expect ~9.24% trainable |
| `merge_lora_weights(model)` | Bakes `B @ A * scaling` into the base weight matrix in-place — produces a plain `nn.Linear` for TensorRT export |

### Trainable Parameters

| Component | Params | Trainable |
|---|---|---|
| DINOv2-L backbone | ~300M | LoRA A+B only (~3M, ~1%) |
| DPT prediction head | ~25M | ✅ Fully trainable (~8.24%) |
| **Total trainable** | | **~9.24%** |

---

## `train/losses.py`

### `SILogLoss`

Scale-Invariant Log loss (Eigen et al., 2014) with a scale anchor term:

```
g_i = log(pred_i) - log(gt_i)
L_silog = mean(g²) - 0.85 * mean(g)² + 0.1 * |mean(g)|
```

The third term (`scale_weight=0.1`) is a **scale anchor penalty**. Without it, the variance-focus term `0.85 * mean(g)²` cancels the global scale penalty, allowing the model to learn arbitrary depth scale offsets while SILog stays low — AbsRel diverges. The scale anchor was added specifically to fix training instability observed at `lr=1e-4`.

### `GradientLoss`

L1 loss on Sobel-filtered depth maps to enforce edge sharpness:

```
L_grad = mean(|∇x_pred - ∇x_gt| + |∇y_pred - ∇y_gt|)
```

Sobel kernels are registered as non-trainable buffers.

### `UnderwaterDepthLoss`

Combined objective:

```
L = L_silog + 0.5 * L_grad
```

Returns a dict with keys `loss`, `silog`, `grad` for TensorBoard logging.

---

## `train/train.py`

Main training loop.

### What It Does

1. Loads `da3metric-large` from HuggingFace
2. Calls `inject_lora()` then `freeze_backbone()`, moves model to GPU
3. Builds `UnityDepthDataset` train/val splits
4. Trains with AdamW + CosineAnnealingLR for N epochs
5. After each epoch: validates, logs to TensorBoard, saves `best.pth` and `last.pth`
6. Supports `--resume checkpoints/last.pth` for HPC job continuation

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_root` | — | Path to directory with `rgb/` and `depth/` subdirs |
| `--model_name` | `da3metric-large` | DA3 config name |
| `--epochs` | `30` | Total training epochs |
| `--batch_size` | `16` | Batch size (A100 40GB handles 16 @ 518×518) |
| `--lr` | `2e-5` | Initial learning rate |
| `--lora_rank` | `8` | LoRA rank |
| `--lora_alpha` | `16.0` | LoRA alpha (scaling = alpha/rank = 2.0) |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--grad_clip` | `1.0` | Max gradient norm |
| `--ckpt_dir` | `checkpoints/` | Checkpoint output directory |
| `--resume` | `None` | Path to checkpoint to resume from |

### Training Configuration (Used for Published Checkpoint)

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| LR schedule | Cosine annealing → 1e-6 |
| Optimizer | AdamW, weight_decay=1e-4 |
| Gradient clip | 1.0 |
| Loss | SILog (scale_weight=0.1) + 0.5 × Sobel gradient |
| Best epoch | 27 |

---

## `train/evaluate.py`

Offline benchmarking against MIMIR-UW SeaFloor.

### Metrics Computed

| Metric | Formula | Better |
|---|---|---|
| AbsRel | mean(\|pred−gt\| / gt) | ↓ |
| SqRel | mean((pred−gt)² / gt) | ↓ |
| RMSE | √mean((pred−gt)²) | ↓ |
| RMSE_log | √mean((log pred − log gt)²) | ↓ |
| δ<1.25 | % pixels where max(pred/gt, gt/pred) < 1.25 | ↑ |
| δ<1.25² | same with threshold 1.5625 | ↑ |
| δ<1.25³ | same with threshold 1.953 | ↑ |

### Outputs

| Output | Description |
|---|---|
| `--out_json` | Aggregate metrics across all frames |
| `--per_image_csv` | Per-frame metrics — sort by AbsRel to find worst cases |
| `--vis_dir` | 30 sampled four-panel figures: RGB \| predicted depth \| GT depth \| abs-rel error map |

---

## `train/demo_eval.py`

Live matplotlib evaluation demo — updates a figure window as evaluation proceeds. Useful for presentations.

**Layout (dark theme, 28×14 figure):**
- **Top row:** RGB input | Predicted depth (plasma) | GT depth (plasma) | Abs-rel error (hot)
- **Bottom row:** AbsRel, RMSE, δ<1.25 running charts | Summary stats box

| Argument | Default | Description |
|---|---|---|
| `--stride` | `1` | Evaluate every Nth sample (`10` → ~3 min for full benchmark) |
| `--update_every` | `10` | Refresh window every N batches |

---

## `train/test_lora_pipeline.py`

Five-step smoke test using random dummy data — no dataset or HuggingFace access required:

| Check | Tests |
|---|---|
| 1 — Load model | DA3 loads from HuggingFace without errors |
| 2 — Inject LoRA | LoRA wraps qkv+proj in all attention blocks; correct % trainable |
| 3 — Forward pass | Model runs on (2, 3, 518, 518) dummy input; output shape is (2, 518, 518) |
| 4 — Backward pass | Gradients flow through LoRA A+B; optimizer step runs |
| 5 — Checkpoint | `state_dict` saves and reloads with LoRA weights intact |

---

## `src/depth_anything_3/` (upstream DA3)

| File | Role |
|---|---|
| `model/da3.py` — `DepthAnything3Net` | Top-level model: DINOv2-L encoder + DPT/DualDPT decoder |
| `model/dinov2/dinov2.py` | DINOv2 ViT backbone |
| `model/dinov2/layers/attention.py` | Attention block with `qkv` and `proj` Linear layers — **LoRA targets** |
| `model/dpt.py` / `model/dualdpt.py` | DPT prediction head — **fully trainable during fine-tuning** |
| `api.py` — `DepthAnything3` | High-level API: `from_pretrained()`, forward pass, output post-processing |
| `configs/da3metric-large.yaml` | Config for DA3 Mono Metric Large — the variant used for training |
