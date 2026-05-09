---
icon: lucide/waves
---

# Underwater DA3 — Documentation

LoRA fine-tuning of **Depth Anything 3 Mono Metric Large** for underwater metric depth estimation, integrated into a full ROS 2 AUV localisation pipeline.

---

## 📚 Guides

| Guide | What it covers |
|---|---|
| [Quick Start](quickstart.md) | Install Pixi, clone the repo, run the smoke test and first evaluation |
| [Dataset](dataset.md) | MIMIR-UW format, inverse-depth NPY files, preprocessing pipeline, dataset splits |
| [Implementation](implementation.md) | What every file in `train/` does — dataset loader, LoRA injection, losses, training loop, evaluation |
| [Operation Guide](operation-guide.md) | Full HPC workflow: baseline eval → training on NSCC A100 → resume → evaluate → upload checkpoint |

---

## 📖 Original DA3 Reference

| Reference | What it covers |
|---|---|
| [CLI](CLI.md) | Original DA3 command-line interface (`auto`, `image`, `video`, `colmap`, `gradio`, …) |
| [API](API.md) | Original DA3 Python API — `DepthAnything3`, `InferenceService`, batch inference |
| [Benchmark](BENCHMARK.md) | Original DA3 benchmark evaluation on standard terrestrial depth datasets |

---

## 🔬 Project Summary

**Problem:** DA3 achieves state-of-the-art metric depth on terrestrial benchmarks but fails dramatically underwater — wavelength-dependent light attenuation causes a blue-green colour cast, and backscatter reduces contrast.

**Approach:** Rank-8 LoRA on all DINOv2-L attention projections (q/k/v/out), trained on the [MIMIR-UW SeaFloor Algae](https://github.com/remaro-network/MIMIR-UW) synthetic underwater dataset with physics-aware preprocessing (Gray World white balance + histogram stretching) and a SILog + Sobel gradient combined loss with a scale anchor term.

**Result:** 59% reduction in AbsRel (0.7744 → 0.3196); δ<1.25 accuracy from 0.94% → 56.8% on the held-out MIMIR-UW SeaFloor benchmark.

**Checkpoint:** [`Frieddeli/COMP4471`](https://huggingface.co/Frieddeli/COMP4471) on HuggingFace.

---

## 🚀 Quickest Start

```bash
# 1. Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone and install
git clone https://github.com/Alvin0523/Depth-Anything-3-Underwater-Refinement-.git
cd Depth-Anything-3-Underwater-Refinement
pixi install

# 3. Smoke test
pixi run python train/test_lora_pipeline.py
# Expected: ALL CHECKS PASSED
```

See [Quick Start](quickstart.md) for the full setup and evaluation walkthrough.

