# ⚙️ Operation Guide

Full workflow from environment setup to trained checkpoint — including NSCC HPC job submission.

---

## Environment

All commands use `pixi run python ...` — no `pip` anywhere. The Pixi environment is defined in `pyproject.toml` and is fully reproducible across machines.

```bash
# Install Pixi (one-time)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies
pixi install

# Verify GPU and environment
pixi run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Step 1 — Verify the Pipeline (Smoke Test)

Always run this first on a new machine or after pulling changes:

```bash
pixi run python train/test_lora_pipeline.py
```

This uses random dummy data — no dataset or GPU required (though GPU is tested if available). Expected: `ALL CHECKS PASSED`.

---

## Step 2 — Prepare the Dataset

Download the MIMIR-UW SeaFloor Algae environment from HuggingFace:

```bash
pixi run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='remaro-network/MIMIR-UW',
    repo_type='dataset',
    local_dir='/path/to/data/MIMIR-UW',
)
"
```

Organise the SeaFloor Algae subset into the expected layout:

```
/path/to/data/SeaFloorAlgae/
├── rgb/
│   ├── 0000000001.png
│   ├── 0000000002.png
│   └── ...
└── depth/
    ├── 0000000001.npy   # float32 inverse-depth (1/metres)
    ├── 0000000002.npy
    └── ...
```

> **Note:** MIMIR-UW NPY files store inverse depth (1/metres). The dataset loader (`train/dataset.py`) inverts them automatically — do not pre-process them.

Sanity check a few depth files:

```bash
pixi run python -c "
import numpy as np, glob
files = sorted(glob.glob('/path/to/data/SeaFloorAlgae/depth/*.npy'))[:5]
for f in files:
    d = np.load(f)
    inv_min, inv_max = d[d>0].min(), d.max()
    print(f'{f}: inv_depth [{inv_min:.4f}, {inv_max:.4f}] → metres [{1/inv_max:.1f}, {1/inv_min:.1f}]')
"
```

---

## Step 3 — Run Baseline Evaluation (Zero-Shot)

Before training, record the zero-shot baseline to establish a comparison point:

```bash
pixi run python train/evaluate.py \
  --data_root /path/to/SeaFloor \
  --model_name da3metric-large \
  --dataset_type seafloor \
  --batch_size 2 --num_workers 2 \
  --vis_dir results/vis_baseline \
  --out_json results/baseline.json \
  --per_image_csv results/baseline_per_image.csv
```

Expected baseline on MIMIR-UW SeaFloor: AbsRel ≈ 0.77, δ<1.25 ≈ 0.94%.

---

## Step 4 — Train (Local GPU)

For a local GPU with sufficient VRAM (≥24 GB recommended for batch size 16):

```bash
pixi run python train/train.py \
  --data_root /path/to/data/SeaFloorAlgae \
  --model_name da3metric-large \
  --epochs 30 \
  --batch_size 16 \
  --lr 2e-5 \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --ckpt_dir checkpoints
```

Reduce `--batch_size` to 4–8 if you hit OOM. TensorBoard logs are written to `checkpoints/tb_logs/`:

```bash
pixi run tensorboard --logdir checkpoints/tb_logs
```

---

## Step 5 — Train on NSCC HPC (A100)

### Environment Variables

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

### PBS Job Script

Save as `train_lora.pbs`:

```bash
#!/bin/bash
#PBS -N da3_underwater
#PBS -q normal
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=12:00:00
#PBS -P personal

cd $PBS_O_WORKDIR

export HF_TOKEN=hf_YOUR_TOKEN_HERE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

.pixi/envs/default/bin/python train/train.py \
  --data_root /path/to/scratch/SeaFloorAlgae \
  --model_name da3metric-large \
  --epochs 30 \
  --batch_size 16 \
  --lr 2e-5 \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --ckpt_dir /path/to/scratch/da3_checkpoints
```

Submit and monitor:

```bash
qsub train_lora.pbs
qstat -u $USER
tail -f da3_underwater.o<JOB_ID>
```

### Resume a Failed/Preempted Job

```bash
.pixi/envs/default/bin/python train/train.py \
  --data_root /path/to/scratch/SeaFloorAlgae \
  --resume /path/to/scratch/da3_checkpoints/last.pth \
  --epochs 30 \
  --batch_size 16
```

---

## Step 6 — Evaluate Fine-Tuned Model

```bash
pixi run python train/evaluate.py \
  --data_root /path/to/SeaFloor \
  --model_name da3metric-large \
  --checkpoint checkpoints/best.pth \
  --lora_rank 8 --lora_alpha 16.0 \
  --dataset_type seafloor \
  --batch_size 2 --num_workers 2 \
  --vis_dir results/vis_finetuned \
  --out_json results/finetuned.json \
  --per_image_csv results/finetuned_per_image.csv
```

Expected results after 30 epochs on SeaFloor benchmark:

| Metric | Baseline | Fine-tuned | Improvement |
|---|---|---|---|
| AbsRel ↓ | 0.7744 | **0.3196** | −59% |
| RMSE ↓ | 8.3562 m | **6.4019 m** | −23% |
| δ<1.25 ↑ | 0.94% | **56.8%** | +60× |

---

## Step 7 — Live Demo

```bash
# Fine-tuned model, every 10th frame (~1,480 frames, ~3 min)
pixi run python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --checkpoint checkpoints/best.pth \
  --lora_rank 8 --lora_alpha 16.0 \
  --stride 10 --update_every 5

# Zero-shot baseline
pixi run python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --stride 10 --update_every 5
```

Requires a display (X11 or Wayland). Use `DISPLAY=:0` on headless servers with X forwarding.

---

## Checkpoint Upload to HuggingFace

```bash
pixi run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='checkpoints/best.pth',
    path_in_repo='checkpoints/best.pth',
    repo_id='YOUR_HF_USERNAME/YOUR_REPO',
    repo_type='model',
)
print('Upload complete')
"
```

---

## Known Issues

| Issue | Cause | Fix |
|---|---|---|
| `AbsRel` diverges while `SILog` stays low | Variance-focus term cancels global scale offset | Scale anchor (`+0.1·\|mean(g)\|`) in `losses.py` is already applied; do not reduce `scale_weight` below 0.05 |
| LoRA weights on CPU after injection | `inject_lora()` creates new `nn.Linear` on CPU | `train.py` calls `da3.to(device)` after injection — verify this if modifying the training script |
| `bfloat16` autocast mismatch | Sobel buffers are float32, autocast returns bfloat16 | `losses.py` casts to `.float()` before Sobel ops — do not remove |
| SSH port 22 blocked on NSCC | Network firewall | Use HTTPS git remotes: `git remote set-url origin https://github.com/...` |
| HuggingFace rate limit | Concurrent downloads | Use `snapshot_download` with `local_dir` rather than per-file `hf_hub_download` |

---

## Storage Layout (NSCC Scratch)

```
/scratch/
└── da3_checkpoints/
    ├── best.pth        ← best validation AbsRel checkpoint
    ├── last.pth        ← latest epoch (for resume)
    └── tb_logs/        ← TensorBoard event files
/scratch/MIMIR-UW/
    ├── SeaFloorAlgae/
    │   ├── rgb/
    │   └── depth/
    └── SeaFloor/       ← benchmark (held-out)
        ├── track0/
        ├── track1/
        └── track2/
```

Total dataset size: ~34 GB (SeaFloor Algae) + ~42 GB (full MIMIR-UW).
