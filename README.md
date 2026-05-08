# Underwater DA3: LoRA Fine-Tuning of Depth Anything 3 for AUV Depth Estimation

<p align="center">
  <img src="assets/images/demo320-2.gif" alt="DA3 Underwater Demo" width="70%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.10647"><img src="https://img.shields.io/badge/arXiv-Depth%20Anything%203-red" alt="Paper PDF"></a>
  <a href="https://depth-anything-3.github.io"><img src="https://img.shields.io/badge/Project_Page-Depth%20Anything%203-green" alt="Project Page"></a>
  <a href="https://huggingface.co/Frieddeli/COMP4471"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-orange" alt="HuggingFace Checkpoint"></a>
  <a href="https://github.com/remaro-network/MIMIR-UW"><img src="https://img.shields.io/badge/Dataset-MIMIR--UW-blue" alt="MIMIR-UW Dataset"></a>
  <img src="https://img.shields.io/badge/Platform-NSCC%20A100-76b900?logo=nvidia" alt="NSCC A100">
  <img src="https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros" alt="ROS2">
  <img src="https://img.shields.io/badge/Course-HKUST%20COMP4471-003366" alt="HKUST COMP4471">
</p>

<p align="center">
  <strong>Fine-tune DA3 for underwater metric depth estimation via LoRA — part of a full AUV target localisation pipeline.</strong>
</p>

> **This is a fork of the [original DA3 repo](https://github.com/ByteDance-Seed/Depth-Anything-3).** We fine-tune DA3 Mono Metric Large for underwater depth estimation using LoRA, as part of our COMP4471 course project on Autonomous Underwater Vehicles (AUVs).

---

## About

DA3 was trained on terrestrial data and performs poorly underwater due to the domain gap — wavelength-dependent light attenuation produces a strong blue-green colour cast, and backscatter degrades contrast. This project adapts DA3 to the underwater domain and integrates it into a full ROS 2 target localisation pipeline.

| Component | Role |
|---|---|
| `train/` | LoRA fine-tuning pipeline for DA3 on MIMIR-UW |
| YOLOv26-seg | Instance segmentation — produces per-object binary masks |
| DA3 Metric Large | Per-pixel metric depth estimation from a single RGB image |
| Mask × Depth fusion | Median depth within each instance mask → distance estimate |

### Hugging Face

| Type | Link |
|---|---|
| Fine-tuned Checkpoint | [Frieddeli/COMP4471](https://huggingface.co/Frieddeli/COMP4471) |

---

## What We Did

- **LoRA fine-tuning** of the DINOv2-L backbone (rank=8, α=16, ~1% trainable params of ~300M total)
- **MIMIR-UW training data** — 39,943 synchronized RGB+metric depth pairs from a photorealistic Unreal Engine 4 AUV simulation, spanning 4 underwater environments (SeaFloor, SeaFloor Algae, OceanFloor, SandPipe)
- **Physics-aware preprocessing** — Gray World white balance + percentile histogram stretching, applied consistently at both train and inference time
- **Combined loss** — SILog + Sobel gradient loss for metric accuracy and edge sharpness
- **Full localisation pipeline** — YOLOv26-seg instance masks + DA3 depth map → median depth per object → distance estimate published over ROS 2 with TensorRT optimisation

---

## Results

Evaluated on the **MIMIR-UW SeaFloor** benchmark — 3 tracks × 2 front cameras (cam0, cam1), 14,828 frames total with metric ground-truth depth stored as inverse-depth EXR files.

| Model | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSE_log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|---|---|---|---|---|---|---|---|
| DA3 Metric Large (zero-shot) | 0.7744 | 5.9656 | 8.3562 | 1.6430 | 0.0094 | 0.0216 | 0.0381 |
| **DA3 + LoRA fine-tuned (ours)** | **0.3196** | **2.7898** | **6.4019** | **1.1234** | **0.5680** | **0.6958** | **0.7555** |

> LoRA fine-tuning achieves a **59% reduction in AbsRel** over the zero-shot baseline, with δ<1.25 accuracy improving from 0.94% to 56.8%.

---

## Fine-Tuning Code

All training code lives in [`train/`](train/):

| File | Purpose |
|------|---------|
| `train/lora.py` | LoRA injection into DINOv2 attention blocks |
| `train/losses.py` | SILog + Sobel gradient loss |
| `train/dataset.py` | MIMIR-UW RGB+depth dataset loader with preprocessing |
| `train/train.py` | Main training script |
| `train/evaluate.py` | Benchmarking (AbsRel, RMSE, δ<1.25) |
| `train/demo_eval.py` | Live matplotlib evaluation demo |
| `train/test_lora_pipeline.py` | Smoke test — run on NSCC first to verify env |

For the full project plan, NSCC setup, job scripts, and per-person TODO list, see [`comp4471.md`](comp4471.md).

### Quick start (smoke test — no data needed)

```bash
pip install -e .
python train/test_lora_pipeline.py
# Expected: ALL CHECKS PASSED
```

### Train (once MIMIR-UW data is ready)

```bash
python train/train.py \
  --data_root /path/to/MIMIR-UW \
  --epochs 30 \
  --batch_size 16
```

---

## Benchmark — MIMIR-UW SeaFloor Dataset

We evaluate on the [MIMIR-UW SeaFloor](https://github.com/remaro-network/MIMIR-UW) benchmark — a photorealistic AUV simulation dataset with metric ground-truth depth stored as inverse-depth EXR files.

**Dataset:** 3 tracks × 2 front cameras (cam0 = front-left, cam1 = front-right). cam2 (straight-down) is excluded as the model was not trained on nadir views.

| | Frames |
|---|---|
| track0 | 2,847 × 2 cams = 5,694 |
| track1 | 2,030 × 2 cams = 4,060 |
| track2 | 2,537 × 2 cams = 5,074 |
| **Total** | **14,828** |

**Checkpoint:** [`Frieddeli/COMP4471`](https://huggingface.co/Frieddeli/COMP4471) on HuggingFace.

### Run baseline

```bash
.pixi/envs/default/bin/python train/evaluate.py \
  --data_root /path/to/SeaFloor \
  --model_name da3metric-large \
  --dataset_type seafloor \
  --batch_size 2 --num_workers 2 \
  --vis_dir results/vis_baseline \
  --out_json results/baseline.json \
  --per_image_csv results/baseline_per_image.csv
```

### Run fine-tuned

```bash
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
- `results/vis_*/vis_NNNN.png` — 30 four-panel figures: RGB | predicted depth | GT depth | abs-rel error map

### Live evaluation demo (`train/demo_eval.py`)

`train/demo_eval.py` runs inference and opens a **live matplotlib popup** that updates as evaluation proceeds — useful for presentations.

**Layout (dark theme, 28×14 figure):**
- **Top row:** RGB input · Predicted depth (plasma) · GT depth (plasma) · Abs-rel error map (hot)
- **Bottom row:** AbsRel · RMSE · δ<1.25 line charts (one data point per evaluated batch) · Running-average summary box

| Argument | Default | Description |
|---|---|---|
| `--data_root` | — | Path to SeaFloor dataset root |
| `--model_name` | `da3metric-large` | DA3 model config name |
| `--checkpoint` | `None` | Path to LoRA checkpoint (omit for zero-shot baseline) |
| `--lora_rank` | `8` | LoRA rank (must match checkpoint) |
| `--lora_alpha` | `16.0` | LoRA alpha |
| `--stride` | `1` | Evaluate every Nth sample (e.g. `10` → ~1,480 frames, ~3 min) |
| `--update_every` | `10` | Refresh the window every N batches |
| `--batch_size` | `2` | Inference batch size |
| `--num_workers` | `2` | DataLoader workers |

```bash
# Fine-tuned model
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

> **Note:** requires a display (X11/Wayland). The window will auto-maximise on TkAgg backends.

---

## Team

**HKUST COMP4471 — Course Project**

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/frieddeli">
        <img src="https://github.com/frieddeli.png" width="80" style="border-radius:50%" alt="Shao Ying Zhan">
      </a>
      <br><sub><b>Shao Ying Zhan</b></sub>
    </td>
    <td align="center">
      <a href="https://github.com/Alvin0523">
        <img src="https://github.com/Alvin0523.png" width="80" style="border-radius:50%" alt="Wong Wei Ming">
      </a>
      <br><sub><b>Wong Wei Ming</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/identicons/danayak.png" width="80" style="border-radius:50%" alt="Dana Yak">
      <br><sub><b>Dana Yak</b></sub>
    </td>
  </tr>
</table>

| Name | Email |
|---|---|
| Shao Ying Zhan | yshaoau@connect.ust.hk |
| Wong Wei Ming | wmwongap@connect.ust.hk |
| Dana Yak | dyak@connect.ust.hk |

---
---

# Depth Anything 3: Recovering the Visual Space from Any Views

> The original upstream README follows below, preserved in full.

<div align="center">

[**Haotong Lin**](https://haotongl.github.io/)<sup>&ast;</sup> · [**Sili Chen**](https://github.com/SiliChen321)<sup>&ast;</sup> · [**Jun Hao Liew**](https://liewjunhao.github.io/)<sup>&ast;</sup> · [**Donny Y. Chen**](https://donydchen.github.io)<sup>&ast;</sup> · [**Zhenyu Li**](https://zhyever.github.io/) · [**Guang Shi**](https://scholar.google.com/citations?user=MjXxWbUAAAAJ&hl=en) · [**Jiashi Feng**](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)
<br>
[**Bingyi Kang**](https://bingyikang.com/)<sup>&ast;&dagger;</sup>

&dagger;project lead&emsp;&ast;Equal Contribution

<a href="https://arxiv.org/abs/2511.10647"><img src='https://img.shields.io/badge/arXiv-Depth Anything 3-red' alt='Paper PDF'></a>
<a href='https://depth-anything-3.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything 3-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

</div>

---

This work presents **Depth Anything 3 (DA3)**, a model that predicts spatially consistent geometry from
arbitrary visual inputs, with or without known camera poses.
In pursuit of minimal modeling, DA3 yields two key insights:
- A **single plain transformer** (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization,
- A singular **depth-ray representation** obviates the need for complex multi-task learning.

DA3 significantly outperforms
[DA2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation,
and [VGGT](https://github.com/facebookresearch/vggt) for multi-view depth estimation and pose estimation.
All models are trained exclusively on **public academic datasets**.

<p align="center">
  <img src="assets/images/demo320-2.gif" alt="Depth Anything 3" width="70%">
</p>
<p align="center">
  <img src="assets/images/da3_radar.png" alt="Depth Anything 3" width="100%">
</p>

## News
- **11-12-2025:** New models and [**DA3-Streaming**](da3_streaming/README.md) released! Handle ultra-long video sequence inference with less than 12GB GPU memory via sliding-window streaming inference. Special thanks to [Kai Deng](https://github.com/DengKaiCQ) for his contribution to DA3-Streaming!
- **08-12-2025:** [Benchmark evaluation pipeline](docs/BENCHMARK.md) released! Evaluate pose estimation & 3D reconstruction on 5 datasets.
- **30-11-2025:** Add [`use_ray_pose`](#use-ray-pose) and [`ref_view_strategy`](docs/funcs/ref_view_strategy.md) (reference view selection for multi-view inputs).
- **25-11-2025:** Add [Awesome DA3 Projects](#-awesome-da3-projects), a community-driven section featuring DA3-based applications.
- **14-11-2025:** Paper, project page, code and models are all released.

## Highlights

### Model Zoo
We release three series of models, each tailored for specific use cases in visual geometry.

- **DA3 Main Series** (`DA3-Giant`, `DA3-Large`, `DA3-Base`, `DA3-Small`) These are our flagship foundation models, trained with a unified depth-ray representation. By varying the input configuration, a single model can perform a wide range of tasks:
  + **Monocular Depth Estimation**: Predicts a depth map from a single RGB image.
  + **Multi-View Depth Estimation**: Generates consistent depth maps from multiple images for high-quality fusion.
  + **Pose-Conditioned Depth Estimation**: Achieves superior depth consistency when camera poses are provided as input.
  + **Camera Pose Estimation**: Estimates camera extrinsics and intrinsics from one or more images.
  + **3D Gaussian Estimation**: Directly predicts 3D Gaussians, enabling high-fidelity novel view synthesis.

- **DA3 Metric Series** (`DA3Metric-Large`) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

- **DA3 Monocular Series** (`DA3Mono-Large`). A dedicated model for high-quality relative monocular depth estimation. Unlike disparity-based models (e.g., [Depth Anything 2](https://github.com/DepthAnything/Depth-Anything-V2)), it directly predicts depth, resulting in superior geometric accuracy.

🔗 Leveraging these available models, we developed a **nested series** (`DA3Nested-Giant-Large`). This series combines a any-view giant model with a metric model to reconstruct visual geometry at a real-world metric scale.

### Codebase Features
- **Interactive Web UI & Gallery**: Visualize model outputs and compare results with an easy-to-use Gradio-based web interface.
- **Flexible Command-Line Interface (CLI)**: Powerful and scriptable CLI for batch processing and integration into custom workflows.
- **Multiple Export Formats**: Save your results in various formats, including `glb`, `npz`, depth images, `ply`, 3DGS videos, etc.
- **Extensible and Modular Design**: The codebase is structured to facilitate future research and the integration of new models or functionalities.

## Quick Start

### Installation

```bash
pip install xformers torch>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
```

### Basic Usage

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(images)
# prediction.processed_images : [N, H, W, 3] uint8   array
# prediction.depth            : [N, H, W]    float32 array
# prediction.conf             : [N, H, W]    float32 array
# prediction.extrinsics       : [N, 3, 4]    float32 array
# prediction.intrinsics       : [N, 3, 3]    float32 array
```

```bash
export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR}
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend

# CLI video processing with feature visualization
da3 video assets/examples/robot_unitree.mp4 \
    --fps 15 \
    --use-backend \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/robo \
    --export-format glb-feat_vis \
    --feat-vis-fps 15 \
    --process-res-method lower_bound_resize \
    --export-feat "11,21,31"

# CLI auto mode without backend reuse
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_CLI/SOH \
    --model-dir ${MODEL_DIR}
```

The model architecture is defined in [`DepthAnything3Net`](src/depth_anything_3/model/da3.py), and specified with a Yaml config file located at [`src/depth_anything_3/configs`](src/depth_anything_3/configs). The input and output processing are handled by [`DepthAnything3`](src/depth_anything_3/api.py). To customize the model architecture, simply create a new config file and reference it as:

```python
from depth_anything_3.cfg import create_object, load_config
Model = create_object(load_config("path/to/new/config"))
```

## Useful Documentation

- [Command Line Interface](docs/CLI.md)
- [Python API](docs/API.md)
- [Benchmark Evaluation](docs/BENCHMARK.md)

## 🗂️ Model Cards

Generally, you should observe that DA3-LARGE achieves comparable results to VGGT.

The Nested series uses an Any-view model to estimate pose and depth, and a monocular metric depth estimator for scaling.

⚠️ Models with the `-1.1` suffix are retrained after fixing a training bug; prefer these refreshed checkpoints. The original `DA3NESTED-GIANT-LARGE`, `DA3-GIANT`, and `DA3-LARGE` remain available but are deprecated.

| 🗃️ Model Name | 📏 Params | 📊 Rel. Depth | 📷 Pose Est. | 🧭 Pose Cond. | 🎨 GS | 📐 Met. Depth | ☁️ Sky Seg | 📄 License |
|-------------------------------|-----------|---------------|--------------|---------------|-------|---------------|-----------|----------------|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) | 1.40B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CC BY-NC 4.0 |
| [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE) | 1.40B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | CC BY-NC 4.0 |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1) | 1.15B | ✅ | ✅ | ✅ | ✅ | | | CC BY-NC 4.0 |
| [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT) | 1.15B | ✅ | ✅ | ✅ | ✅ | | | CC BY-NC 4.0 |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | 0.35B | ✅ | ✅ | ✅ | | | | CC BY-NC 4.0 |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE) | 0.35B | ✅ | ✅ | ✅ | | | | CC BY-NC 4.0 |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | 0.12B | ✅ | ✅ | ✅ | | | | Apache 2.0 |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | 0.08B | ✅ | ✅ | ✅ | | | | Apache 2.0 |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | 0.35B | ✅ | | | | ✅ | ✅ | Apache 2.0 |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE) | 0.35B | ✅ | | | | | ✅ | Apache 2.0 |

## ❓ FAQ

- **Monocular Metric Depth**: To obtain metric depth in meters from `DA3METRIC-LARGE`, use `metric_depth = focal * net_output / 300.`, where `focal` is the focal length in pixels (typically the average of fx and fy from the camera intrinsic matrix K). Note that the output from `DA3NESTED-GIANT-LARGE` is already in meters.

- <a id="use-ray-pose"></a>**Ray Head (`use_ray_pose`)**: Our API and CLI support `use_ray_pose` arg, which means that the model will derive camera pose from ray head, which is generally slightly slower, but more accurate. Note that the default is `False` for faster inference speed.
  <details>
  <summary>AUC3 Results for DA3NESTED-GIANT-LARGE</summary>

  | Model | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ |
  |-------|------|-------|-----|---------|-----------|
  | `ray_head` | 84.4 | 52.6 | 93.9 | 29.5 | 89.4 |
  | `cam_head` | 80.3 | 48.4 | 94.1 | 28.5 | 85.0 |

  </details>

- **Older GPUs without XFormers support**: See [Issue #11](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11). Thanks to [@S-Mahoney](https://github.com/S-Mahoney) for the solution!

## 🏢 Awesome DA3 Projects

A community-curated list of Depth Anything 3 integrations across 3D tools, creative pipelines, robotics, and web/VR viewers. You are welcome to submit your DA3-based project via PR.

- [DA3-blender](https://github.com/xy-gao/DA3-blender): Blender addon for DA3-based 3D reconstruction from a set of images.
- [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3): ComfyUI nodes for Depth Anything 3, supporting single/multi-view and video-consistent depth with optional point-cloud export.
- [DA3-ROS2-Wrapper](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper): Real-time DA3 depth in ROS2 with multi-camera support.
- [DA3-ROS2-CPP-TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt): DA3 ROS2 C++ TensorRT Inference Node for real-time inference.
- [VideoDepthViewer3D](https://github.com/amariichi/VideoDepthViewer3D): Streaming videos with DA3 metric depth to a Three.js/WebXR 3D viewer for VR/stereo playback.

## 🧑‍💻 Official Codebase Core Contributors and Maintainers

<table>
  <tr>
    <td align="center">
      <a href="https://bingykang.github.io/">
        <img src="https://images.weserv.nl/?url=https://bingykang.github.io/images/bykang_homepage.jpeg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
      <br /><sub><b>Bingyi Kang</b></sub>
    </td>
    <td align="center">
      <a href="https://haotongl.github.io/">
        <img src="https://images.weserv.nl/?url=https://haotongl.github.io/assets/img/prof_pic.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
      <br /><sub>Haotong Lin</sub>
    </td>
    <td align="center">
      <a href="https://github.com/SiliChen321">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/195901058?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
      <br /><sub>Sili Chen</sub>
    </td>
    <td align="center">
      <a href="https://liewjunhao.github.io/">
        <img src="https://images.weserv.nl/?url=https://liewjunhao.github.io/images/liewjunhao.png?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
      <br /><sub>Jun Hao Liew</sub>
    </td>
    <td align="center">
      <a href="https://donydchen.github.io/">
        <img src="https://images.weserv.nl/?url=https://donydchen.github.io/assets/img/profile.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
      <br /><sub>Donny Y. Chen</sub>
    </td>
    <td align="center">
      <a href="https://github.com/DengKaiCQ">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/59907452?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
      <br /><sub>Kai Deng</sub>
    </td>
  </tr>
</table>

## 📝 Citations

If you find Depth Anything 3 useful in your research or projects, please cite our work:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
