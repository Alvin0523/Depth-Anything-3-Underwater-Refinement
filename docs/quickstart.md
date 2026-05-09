# 🚀 Quick Start

Get the full LoRA fine-tuning pipeline running from a clean clone in three steps.

---

## Prerequisites

| Requirement | Version |
|---|---|
| [Pixi](https://pixi.sh/) | latest |
| CUDA | 12.4 (for GPU training) |
| HuggingFace token | gated access to `depth-anything/da3metric-large` |

---

## 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
# Restart your shell, or:
source ~/.bashrc
```

---

## 2. Clone and Install

```bash
git clone https://github.com/Alvin0523/Depth-Anything-3-Underwater-Refinement-.git
cd Depth-Anything-3-Underwater-Refinement
pixi install
```

This installs all dependencies (PyTorch, torchvision, xformers, OpenCV, TensorBoard, Zensical, etc.) into an isolated `.pixi/` environment. No `pip install` needed anywhere.

---

## 3. Set HuggingFace Token

The DA3 Mono Metric Large weights are behind a HuggingFace gate:

```bash
# Store your token (one-time)
mkdir -p ~/.huggingface
echo "hf_YOUR_TOKEN_HERE" > ~/.huggingface/token

# Or export for the current session
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

---

## 4. Smoke Test (no data needed)

Verifies the full LoRA pipeline — model download, injection, forward/backward pass, and checkpoint save — using random dummy data:

```bash
pixi run python train/test_lora_pipeline.py
```

Expected output:
```
[1/5] Loading da3metric-large from HuggingFace...  OK
[2/5] Injecting LoRA (rank=8)...  OK  -- trainable: X / Y (9.24%)
[3/5] Forward pass with dummy batch...  OK  -- depth shape: torch.Size([2, 518, 518])
[4/5] Backward pass (loss + optimiser step)...  OK  -- loss: X.XXXX
[5/5] Checkpoint save + reload...  OK
ALL CHECKS PASSED
Device: cuda  |  GPU: NVIDIA A100 40GB
```

---

## 5. Run the Fine-Tuned Model

Download the checkpoint from HuggingFace and run evaluation against the MIMIR-UW SeaFloor benchmark:

```bash
# Download checkpoint
pixi run python -c "
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download('Frieddeli/COMP4471', 'checkpoints/best.pth')
os.makedirs('checkpoints', exist_ok=True)
shutil.copy(path, 'checkpoints/best.pth')
print('Downloaded to checkpoints/best.pth')
"

# Run evaluation
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

---

## 6. Live Evaluation Demo

Opens a live matplotlib window updating as evaluation proceeds (requires a display):

```bash
pixi run python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --checkpoint checkpoints/best.pth \
  --lora_rank 8 --lora_alpha 16.0 \
  --stride 10 --update_every 5
```

---

## Next Steps

- **Train your own model** → [Operation Guide](operation-guide.md)
- **Understand the dataset format** → [Dataset](dataset.md)
- **See what each file does** → [Implementation](implementation.md)
