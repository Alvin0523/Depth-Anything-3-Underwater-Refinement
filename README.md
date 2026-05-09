# Underwater DA3: LoRA Fine-Tuning of Depth Anything 3 for AUV Depth Estimation

<p align="center">

<img src="assets/images/demo320-2.gif" alt="DA3 Underwater Demo" width="70%"/>

</p>

<p align="center">

<a href="https://arxiv.org/abs/2511.10647"><img src="https://img.shields.io/badge/arXiv-Depth%20Anything%203-red" alt="Paper PDF"/></a> <a href="https://depth-anything-3.github.io"><img src="https://img.shields.io/badge/Project_Page-Depth%20Anything%203-green" alt="Project Page"/></a> <a href="https://huggingface.co/Frieddeli/COMP4471"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-orange" alt="HuggingFace Checkpoint"/></a> <a href="https://github.com/remaro-network/MIMIR-UW"><img src="https://img.shields.io/badge/Dataset-MIMIR--UW-blue" alt="MIMIR-UW Dataset"/></a> <img src="https://img.shields.io/badge/Platform-NSCC%20A100-76b900?logo=nvidia" alt="NSCC A100"/> <img src="https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros" alt="ROS2"/> <img src="https://img.shields.io/badge/Course-HKUST%20COMP4471-003366" alt="HKUST COMP4471"/>

</p>

<p align="center">

<strong>Fine-tune DA3 for underwater metric depth estimation via LoRA — part of a full AUV target localisation pipeline.</strong>

</p>

> **This is a fork of the [original DA3 repo](https://github.com/ByteDance-Seed/Depth-Anything-3).** We fine-tune DA3 Mono Metric Large for underwater depth estimation using LoRA, as part of our COMP4471 course project on Autonomous Underwater Vehicles (AUVs).

------------------------------------------------------------------------

## About

DA3 was trained on terrestrial data and performs poorly underwater due to the domain gap — wavelength-dependent light attenuation produces a strong blue-green colour cast, and backscatter degrades contrast. This project adapts DA3 to the underwater domain and integrates it into a full ROS 2 target localisation pipeline.

| Component | Role |
|----|----|
| `train/` | LoRA fine-tuning pipeline for DA3 on MIMIR-UW |
| YOLOv26-seg | Instance segmentation — produces per-object binary masks |
| DA3 Metric Large | Per-pixel metric depth estimation from a single RGB image |
| Mask × Depth fusion | Median depth within each instance mask → distance estimate |

### Hugging Face

+-----------------------------------+-----------------------------------------------------------------+
| Type                              | Link                                                            |
+===================================+=================================================================+
| Fine-tuned Checkpoint             | [Frieddeli/COMP4471](https://huggingface.co/Frieddeli/COMP4471) |
+-----------------------------------+-----------------------------------------------------------------+

------------------------------------------------------------------------

## What We Did

DA3 was trained on terrestrial data and performs poorly underwater due to the domain gap — wavelength-dependent light attenuation causes a strong blue-green colour cast, and backscatter degrades contrast. We are adapting DA3 to the underwater domain via:

- **LoRA fine-tuning** of the DINOv2-L backbone (rank=8, α=16, \~1% trainable params of \~300M total)
- **MIMIR-UW training data** — 39,943 synchronized RGB+metric depth pairs from a photorealistic Unreal Engine 4 AUV simulation, spanning 4 underwater environments (SeaFloor, SeaFloor Algae, OceanFloor, SandPipe)
- **Physics-aware preprocessing** — Gray World white balance + percentile histogram stretching, applied consistently at both train and inference time
- **Combined loss** — SILog + Sobel gradient loss for metric accuracy and edge sharpness
- **Full localisation pipeline** — YOLOv26-seg instance masks + DA3 depth map → median depth per object → distance estimate published over ROS 2 with TensorRT optimisation

The fine-tuned model feeds into a full target localisation pipeline: YOLOv26-seg instance masks + DA3 depth map → median depth per object → distance estimate published over ROS 2.

------------------------------------------------------------------------

## Fine-Tuning Code

All training code lives in [`train/`](train/):

| File | Purpose |
|----|----|
| `train/lora.py` | LoRA injection into DINOv2 attention blocks |
| `train/losses.py` | SILog + Sobel gradient loss |
| `train/dataset.py` | MIMIR-UW RGB+depth dataset loader with preprocessing |
| `train/train.py` | Main training script |
| `train/evaluate.py` | Benchmarking (AbsRel, RMSE, δ\<1.25) |
| `train/demo_eval.py` | Live matplotlib evaluation demo |
| `train/test_lora_pipeline.py` | Smoke test — run on NSCC first to verify env |

For the full project plan, NSCC setup, job scripts, and per-person TODO list, see [`comp4471.md`](comp4471.md).

### Quick start (NSCC env check — no data needed)

``` bash
pip install -e .
python train/test_lora_pipeline.py
# Expected: ALL CHECKS PASSED
```

### Train (once MIMIR-UW data is ready)

``` bash
python train/train.py \
  --data_root /path/to/MIMIR-UW \
  --epochs 30 \
  --batch_size 16
```

------------------------------------------------------------------------

## Benchmark — MIMIR-UW SeaFloor Dataset

We evaluate on the [MIMIR-UW SeaFloor](https://github.com/remaro-network/MIMIR-UW) benchmark — a photorealistic AUV simulation dataset with metric ground-truth depth stored as inverse-depth EXR files.

**Dataset:** 3 tracks × 2 front cameras (cam0 = front-left, cam1 = front-right). cam2 (straight-down) is excluded as the model was not trained on nadir views. 14,828 frames total with metric ground-truth depth stored as inverse-depth EXR files.

| Tracks    | Frames                 |
|-----------|------------------------|
| track0    | 2,847 × 2 cams = 5,694 |
| track1    | 2,030 × 2 cams = 4,060 |
| track2    | 2,537 × 2 cams = 5,074 |
| **Total** | **14,828**             |

**Run baseline (0 shot, no checkpoints)**

``` bash
.pixi/envs/default/bin/python train/evaluate.py \
  --data_root /path/to/SeaFloor \
  --model_name da3metric-large \
  --dataset_type seafloor \
  --batch_size 2 --num_workers 2 \
  --vis_dir results/vis_baseline \
  --out_json results/baseline.json \
  --per_image_csv results/baseline_per_image.csv
```

**Run fine-tuned**

``` bash
.pixi/envs/default/bin/python train/evaluate.py \
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

Outputs per run:

- `results/*.json` — aggregate metrics

- `results/*_per_image.csv` — per-frame metrics for all 14,828 frames (sort by absrel to find worst cases)

- `results/vis_*/vis_NNNN.png` — 30 four-panel figures: RGB \| predicted depth \| GT depth \| abs-rel error map

### Benchmarking Results

| Model | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSE_log ↓ | δ\<1.25 ↑ | δ\<1.25² ↑ | δ\<1.25³ ↑ |
|----|----|----|----|----|----|----|----|
| DA3 Metric Large (zero-shot) | 0.7744 | 5.9656 | 8.3562 | 1.6430 | 0.0094 | 0.0216 | 0.0381 |
| **DA3 + LoRA fine-tuned (ours)** | **0.3196** | **2.7898** | **6.4019** | **1.1234** | **0.5680** | **0.6958** | **0.7555** |

> LoRA fine-tuning achieves a **59% reduction in AbsRel** over the zero-shot baseline, with δ\<1.25 accuracy improving from 0.94% to 56.8%.
>
> **Checkpoint:** [`Frieddeli/COMP4471`](https://huggingface.co/Frieddeli/COMP4471) on HuggingFace.

### Live evaluation demo (`train/demo_eval.py`)

`train/demo_eval.py` runs inference and opens a **live matplotlib popup** that updates as evaluation proceeds — useful for presentations.

**Layout (dark theme, 28×14 figure):**

- **Top row:** RGB input \| Predicted depth (plasma) \| GT depth (plasma) \| Abs-rel error map (hot)

- **Bottom row:** AbsRel \| RMSE \| δ\<1.25 line charts (one data point per evaluated batch) \| Running-average summary box

| Argument | Default | Description |
|----|----|----|
| `--data_root` | — | Path to SeaFloor dataset root |
| `--model_name` | `da3metric-large` | DA3 model config name |
| `--checkpoint` | `None` | Path to LoRA checkpoint (omit for zero-shot baseline) |
| `--lora_rank` | `8` | LoRA rank (must match checkpoint) |
| `--lora_alpha` | `16.0` | LoRA alpha |
| `--stride` | `1` | Evaluate every Nth sample (e.g. `10` → \~1,480 frames, \~3 min) |
| `--update_every` | `10` | Refresh the window every N batches |
| `--batch_size` | `2` | Inference batch size |
| `--num_workers` | `2` | DataLoader workers |

``` bash
# Run fine-tuned model
.pixi/envs/default/bin/python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --checkpoint checkpoints/best.pth \
  --lora_rank 8 --lora_alpha 16.0 \
  --stride 10 --update_every 5

# Zero-shot baseline (no checkpoint)
.pixi/envs/default/bin/python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --stride 10 --update_every 5
```

> **Note:** requires a display (X11/Wayland). The window will auto-maximise on TkAgg backends. Requires `matplotlib`, `tqdm`, and the same pixi env used for training.

------------------------------------------------------------------------

## Team

**HKUST COMP4471 — Course Project**

+-------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------+
|                                                                                                             |                                                                                                           |                                                                                         |
+=============================================================================================================+===========================================================================================================+=========================================================================================+
| [![Shao Ying Zhan](https://github.com/frieddeli.png){alt="Shao Ying Zhan"} ](https://github.com/frieddeli)\ | [![Wong Wei Ming](https://github.com/Alvin0523.png){alt="Wong Wei Ming"} ](https://github.com/Alvin0523)\ | [![Dana Yak](https://github.com/dnyk7.png){alt="Dana Yak"} ](https://github.com/dnyk7)\ |
| ~**Shao\ Ying\ Zhan**~                                                                                      | ~**Wong\ Wei\ Ming**~                                                                                     | ~**Dana\ Yak**~                                                                         |
+-------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------+

| Name           | Email                    |
|----------------|--------------------------|
| Shao Ying Zhan | yshaoau\@connect.ust.hk  |
| Wong Wei Ming  | wmwongap\@connect.ust.hk |
| Dana Yak       | dyak\@connect.ust.hk     |

------------------------------------------------------------------------

## DA3 Quick Start in our repository

### Installation

``` bash
pixi install
```

The model architecture is defined in [`DepthAnything3Net`](src/depth_anything_3/model/da3.py), and specified with a Yaml config file located at [`src/depth_anything_3/configs`](src/depth_anything_3/configs). The input and output processing are handled by [`DepthAnything3`](src/depth_anything_3/api.py). To customize the model architecture, simply create a new config file and reference it as:

``` python
from depth_anything_3.cfg import create_object, load_config
Model = create_object(load_config("path/to/new/config"))
```

## Useful Documentation

- [Command Line Interface](docs/CLI.md)
- [Python API](docs/API.md)
- [Benchmark Evaluation](docs/BENCHMARK.md)
