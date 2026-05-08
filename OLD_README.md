<div align="center">
<h1 style="border-bottom: none; margin-bottom: 0px ">Depth Anything 3: Recovering the Visual Space from Any Views</h1>
<!-- <h2 style="border-top: none; margin-top: 3px;">Recovering the Visual Space from Any Views</h2> -->


[**Haotong Lin**](https://haotongl.github.io/)<sup>&ast;</sup> · [**Sili Chen**](https://github.com/SiliChen321)<sup>&ast;</sup> · [**Jun Hao Liew**](https://liewjunhao.github.io/)<sup>&ast;</sup> · [**Donny Y. Chen**](https://donydchen.github.io)<sup>&ast;</sup> · [**Zhenyu Li**](https://zhyever.github.io/) · [**Guang Shi**](https://scholar.google.com/citations?user=MjXxWbUAAAAJ&hl=en) · [**Jiashi Feng**](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)
<br>
[**Bingyi Kang**](https://bingyikang.com/)<sup>&ast;&dagger;</sup>

&dagger;project lead&emsp;&ast;Equal Contribution

<a href="https://arxiv.org/abs/2511.10647"><img src='https://img.shields.io/badge/arXiv-Depth Anything 3-red' alt='Paper PDF'></a>
<a href='https://depth-anything-3.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything 3-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-3'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<!-- <a href='https://huggingface.co/datasets/depth-anything/VGB'><img src='https://img.shields.io/badge/Benchmark-VisGeo-yellow' alt='Benchmark'></a> -->
<!-- <a href='https://huggingface.co/datasets/depth-anything/data'><img src='https://img.shields.io/badge/Benchmark-xxx-yellow' alt='Data'></a> -->

</div>

---

## 🌊 COMP4471 Project — Underwater Fine-Tuning (HKUST)

> **This is a fork of the original DA3 repo.** We are fine-tuning DA3 Mono Metric Large for underwater depth estimation using LoRA, as part of our COMP4471 course project on Autonomous Underwater Vehicles (AUVs).

**Team:** Dana Yak · Shao Ying Zhan · Wong Wei Ming

### What we are doing
DA3 was trained on terrestrial data and performs poorly underwater due to the domain gap — wavelength-dependent light attenuation causes a strong blue-green colour cast, and backscatter degrades contrast. We are adapting DA3 to the underwater domain via:

- **LoRA fine-tuning** of the DINOv2 backbone (rank=8, ~1% trainable params)
- **Synthetic training data** generated from a Unity AUV simulation environment (paired RGB + metric depth)
- **Physics-aware preprocessing** — Gray World white balance + histogram stretching applied at both train and inference time
- **Combined loss** — SILog + Sobel gradient loss

The fine-tuned model feeds into a full target localisation pipeline: YOLOv26-seg instance masks + DA3 depth map → median depth per object → distance estimate published over ROS 2.

### Project guide and TODO list
See [`comp4471.md`](comp4471.md) for:
- Full project plan and status
- Explanation of every file in `train/`
- Step-by-step instructions for Unity data export, NSCC training, and benchmarking
- Per-person TODO task list
- NSCC environment setup and job script

### Fine-tuning code
All training code lives in [`train/`](train/):

| File | Purpose |
|------|---------|
| `train/lora.py` | LoRA injection into DINOv2 attention blocks |
| `train/losses.py` | SILog + Sobel gradient loss |
| `train/dataset.py` | Unity RGB+depth dataset loader with preprocessing |
| `train/train.py` | Main training script |
| `train/evaluate.py` | Benchmarking (AbsRel, RMSE, δ<1.25) |
| `train/test_lora_pipeline.py` | Smoke test — run on NSCC first to verify env |

**Quick start (NSCC env check — no data needed):**
```bash
pip install -e .
python train/test_lora_pipeline.py
# Expected: ALL CHECKS PASSED
```

**Train (once Unity data is ready):**
```bash
python train/train.py --data_root data/unity --epochs 30 --batch_size 4
```

### Benchmark — MIMIR SeaFloor dataset

We evaluate on the [MIMIR SeaFloor](https://github.com/remaro-network/MIMIR-UW) benchmark — a photorealistic AUV simulation dataset with metric ground-truth depth stored as inverse-depth EXR files.

**Dataset:** 3 tracks × 2 front cameras (cam0 = front-left, cam1 = front-right). cam2 (straight-down / `bottom_center`) is excluded as the model was not trained on nadir views.

| | Frames |
|---|---|
| track0 | 2,847 × 2 cams = 5,694 |
| track1 | 2,030 × 2 cams = 4,060 |
| track2 | 2,537 × 2 cams = 5,074 |
| **Total** | **14,828** |

**Checkpoint:** [`Frieddeli/COMP4471`](https://huggingface.co/Frieddeli/COMP4471) on HuggingFace.

#### Results

| Model | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSE_log ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|---|---|---|---|---|---|---|---|
| DA3 Metric Large (zero-shot) | 0.7744 | 5.9656 | 8.3562 | 1.6430 | 0.0094 | 0.0216 | 0.0381 |
| DA3 + LoRA fine-tuned (ours) | **0.3196** | **2.7898** | **6.4019** | **1.1234** | **0.5680** | **0.6958** | **0.7555** |

**Run baseline:**
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

**Run fine-tuned:**
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

#### Live evaluation demo (`train/demo_eval.py`)

`train/demo_eval.py` runs inference and opens a **live matplotlib popup** that updates as evaluation proceeds — useful for presentations.

**Layout (dark theme, 28×14 figure):**
- **Top row:** RGB input · Predicted depth (plasma) · GT depth (plasma) · Abs-rel error map (hot)
- **Bottom row:** AbsRel · RMSE · δ<1.25 line charts (one data point per evaluated batch) · Running-average summary box

**Arguments:**

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

**Run fine-tuned demo:**
```bash
.pixi/envs/default/bin/python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --model_name da3metric-large \
  --checkpoint checkpoints/best.pth \
  --lora_rank 8 --lora_alpha 16.0 \
  --stride 10 --update_every 5
```

**Run baseline demo (zero-shot, no checkpoint):**
```bash
.pixi/envs/default/bin/python train/demo_eval.py \
  --data_root /path/to/SeaFloor \
  --model_name da3metric-large \
  --stride 10 --update_every 5
```

> **Note:** requires a display (X11/Wayland). The window will auto-maximise on TkAgg backends. Requires `matplotlib`, `tqdm`, and the same pixi env used for training.



This work presents **Depth Anything 3 (DA3)**, a model that predicts spatially consistent geometry from
arbitrary visual inputs, with or without known camera poses.
In pursuit of minimal modeling, DA3 yields two key insights:
- 💎 A **single plain transformer** (e.g., vanilla DINO encoder) is sufficient as a backbone without architectural specialization,
- ✨ A singular **depth-ray representation** obviates the need for complex multi-task learning.

🏆 DA3 significantly outperforms
[DA2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation,
and [VGGT](https://github.com/facebookresearch/vggt) for multi-view depth estimation and pose estimation.
All models are trained exclusively on **public academic datasets**.

<!-- <p align="center">
  <img src="assets/images/da3_teaser.png" alt="Depth Anything 3" width="100%">
</p> -->
<p align="center">
  <img src="assets/images/demo320-2.gif" alt="Depth Anything 3 - Left" width="70%">
</p>
<p align="center">
  <img src="assets/images/da3_radar.png" alt="Depth Anything 3" width="100%">
</p>


## 📰 News
- **11-12-2025:** 🚀 New models and [**DA3-Streaming**](da3_streaming/README.md) released! Handle ultra-long video sequence inference with less than 12GB GPU memory via sliding-window streaming inference. Special thanks to [Kai Deng](https://github.com/DengKaiCQ) for his contribution to DA3-Streaming!
- **08-12-2025:** 📊 [Benchmark evaluation pipeline](docs/BENCHMARK.md) released! Evaluate pose estimation & 3D reconstruction on 5 datasets.
- **30-11-2025:** Add [`use_ray_pose`](#use-ray-pose) and [`ref_view_strategy`](docs/funcs/ref_view_strategy.md) (reference view selection for multi-view inputs).   
- **25-11-2025:** Add [Awesome DA3 Projects](#-awesome-da3-projects), a community-driven section featuring DA3-based applications.
- **14-11-2025:** Paper, project page, code and models are all released.

## ✨ Highlights

### 🏆 Model Zoo
We release three series of models, each tailored for specific use cases in visual geometry.

- 🌟 **DA3 Main Series** (`DA3-Giant`, `DA3-Large`, `DA3-Base`, `DA3-Small`) These are our flagship foundation models, trained with a unified depth-ray representation. By varying the input configuration, a single model can perform a wide range of tasks:
  + 🌊 **Monocular Depth Estimation**: Predicts a depth map from a single RGB image.
  + 🌊 **Multi-View Depth Estimation**: Generates consistent depth maps from multiple images for high-quality fusion.
  + 🎯 **Pose-Conditioned Depth Estimation**: Achieves superior depth consistency when camera poses are provided as input.
  + 📷 **Camera Pose Estimation**:  Estimates camera extrinsics and intrinsics from one or more images.
  + 🟡 **3D Gaussian Estimation**: Directly predicts 3D Gaussians, enabling high-fidelity novel view synthesis.

- 📐 **DA3 Metric Series** (`DA3Metric-Large`) A specialized model fine-tuned for metric depth estimation in monocular settings, ideal for applications requiring real-world scale.

- 🔍 **DA3 Monocular Series** (`DA3Mono-Large`). A dedicated model for high-quality relative monocular depth estimation. Unlike disparity-based models (e.g.,  [Depth Anything 2](https://github.com/DepthAnything/Depth-Anything-V2)), it directly predicts depth, resulting in superior geometric accuracy.

🔗 Leveraging these available models, we developed a **nested series** (`DA3Nested-Giant-Large`). This series combines a any-view giant model with a metric model to reconstruct visual geometry at a real-world metric scale.

### 🛠️ Codebase Features
Our repository is designed to be a powerful and user-friendly toolkit for both practical application and future research.
- 🎨 **Interactive Web UI & Gallery**: Visualize model outputs and compare results with an easy-to-use Gradio-based web interface.
- ⚡ **Flexible Command-Line Interface (CLI)**: Powerful and scriptable CLI for batch processing and integration into custom workflows.
- 💾 **Multiple Export Formats**: Save your results in various formats, including `glb`, `npz`, depth images, `ply`, 3DGS videos, etc, to seamlessly connect with other tools.
- 🔧 **Extensible and Modular Design**: The codebase is structured to facilitate future research and the integration of new models or functionalities.


<!-- ### 🎯 Visual Geometry Benchmark
We introduce a new benchmark to rigorously evaluate geometry prediction models on three key tasks: pose estimation, 3D reconstruction, and visual rendering (novel view synthesis) quality.

- 🔄 **Broad Model Compatibility**: Our benchmark is designed to be versatile, supporting the evaluation of various models, including both monocular and multi-view depth estimation approaches.
- 🔬 **Robust Evaluation Pipeline**: We provide a standardized pipeline featuring RANSAC-based pose alignment, TSDF fusion for dense reconstruction, and a principled view selection strategy for novel view synthesis.
- 📊 **Standardized Metrics**: Performance is measured using established metrics: AUC for pose accuracy, F1-score and Chamfer Distance for reconstruction, and PSNR/SSIM/LPIPS for rendering quality.
- 🌍 **Diverse and Challenging Datasets**: The benchmark spans a wide range of scenes from datasets like HiRoom, ETH3D, DTU, 7Scenes, ScanNet++, DL3DV, Tanks and Temples, and MegaDepth. -->


## 🚀 Quick Start

### 📦 Installation

```bash
pip install xformers torch\>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
```

For detailed model information, please refer to the [Model Cards](#-model-cards) section below.

### 💻 Basic Usage

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))
prediction = model.inference(
    images,
)
# prediction.processed_images : [N, H, W, 3] uint8   array
print(prediction.processed_images.shape)
# prediction.depth            : [N, H, W]    float32 array
print(prediction.depth.shape)  
# prediction.conf             : [N, H, W]    float32 array
print(prediction.conf.shape)  
# prediction.extrinsics       : [N, 3, 4]    float32 array # opencv w2c or colmap format
print(prediction.extrinsics.shape)
# prediction.intrinsics       : [N, 3, 3]    float32 array
print(prediction.intrinsics.shape)
```

```bash

export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
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

The model architecture is defined in [`DepthAnything3Net`](src/depth_anything_3/model/da3.py), and specified with a Yaml config file located at [`src/depth_anything_3/configs`](src/depth_anything_3/configs). The input and output processing are handled by [`DepthAnything3`](src/depth_anything_3/api.py). To customize the model architecture, simply create a new config file (*e.g.*, `path/to/new/config`) as:

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params

  name: vitb
  out_layers: [5, 7, 9, 11]
  alt_start: 4
  qknorm_start: 4
  rope_start: 4
  cat_token: True

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params

  dim_in: &head_dim_in 1536
  output_dim: 2
  features: &head_features 128
  out_channels: &head_out_channels [96, 192, 384, 768]
```

Then, the model can be created with the following code snippet.
```python
from depth_anything_3.cfg import create_object, load_config

Model = create_object(load_config("path/to/new/config"))
```



## 📚 Useful Documentation

- 🖥️ [Command Line Interface](docs/CLI.md)
- 📑 [Python API](docs/API.md)
- 📊 [Benchmark Evaluation](docs/BENCHMARK.md)

## 🗂️ Model Cards

Generally, you should observe that DA3-LARGE achieves comparable results to VGGT.

The Nested series uses an Any-view model to estimate pose and depth, and a monocular metric depth estimator for scaling. 

⚠️ Models with the `-1.1` suffix are retrained after fixing a training bug; prefer these refreshed checkpoints. The original `DA3NESTED-GIANT-LARGE`, `DA3-GIANT`, and `DA3-LARGE` remain available but are deprecated. You could expect much better performance for street scenes with the `-1.1` models.

| 🗃️ Model Name                  | 📏 Params | 📊 Rel. Depth | 📷 Pose Est. | 🧭 Pose Cond. | 🎨 GS | 📐 Met. Depth | ☁️ Sky Seg | 📄 License     |
|-------------------------------|-----------|---------------|--------------|---------------|-------|---------------|-----------|----------------|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE)                     | 0.12B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL)                     | 0.08B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)              | 0.35B     | ✅             |              |               |       | ✅             | ✅         | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE)                | 0.35B     | ✅             |              |               |               |       | ✅         | Apache 2.0     |


## ❓ FAQ

- **Monocular Metric Depth**: To obtain metric depth in meters from `DA3METRIC-LARGE`, use `metric_depth = focal * net_output / 300.`, where `focal` is the focal length in pixels (typically the average of fx and fy from the camera intrinsic matrix K). Note that the output from `DA3NESTED-GIANT-LARGE` is already in meters.

- <a id="use-ray-pose"></a>**Ray Head (`use_ray_pose`)**:  Our API and CLI support `use_ray_pose` arg, which means that the model will derive camera pose from ray head, which is generally slightly slower, but more accurate. Note that the default is `False` for faster inference speed. 
  <details>
  <summary>AUC3 Results for DA3NESTED-GIANT-LARGE</summary>
  
  | Model | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ | 
  |-------|------|-------|-----|---------|-----------|
  | `ray_head` | 84.4 | 52.6 | 93.9 | 29.5 | 89.4 |
  | `cam_head` | 80.3 | 48.4 | 94.1 | 28.5 | 85.0 |

  </details>




- **Older GPUs without XFormers support**: See [Issue #11](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/11). Thanks to [@S-Mahoney](https://github.com/S-Mahoney) for the solution!


## 🏢 Awesome DA3 Projects

A community-curated list of Depth Anything 3 integrations across 3D tools, creative pipelines, robotics, and web/VR viewers, including but not limited to these. You are welcome to submit your DA3-based project via PR, and we will review and feature it if applicable.

- [DA3-blender](https://github.com/xy-gao/DA3-blender): Blender addon for DA3-based 3D reconstruction from a set of images. 

- [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3): ComfyUI nodes for Depth Anything 3, supporting single/multi-view and video-consistent depth with optional point‑cloud export.

- [DA3-ROS2-Wrapper](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper): Real-time DA3 depth in ROS2 with multi-camera support. 

- [DA3-ROS2-CPP-TensorRT](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt): DA3 ROS2 C++ TensorRT Inference Node: a ROS2 node for DA3 depth estimation using TensorRT for real-time inference.

- [VideoDepthViewer3D](https://github.com/amariichi/VideoDepthViewer3D): Streaming videos with DA3 metric depth to a Three.js/WebXR 3D viewer for VR/stereo playback.


## 🧑‍💻 Official Codebase Core Contributors and Maintainers

<table>
  <tr>
    <td align="center">
      <a href="https://bingykang.github.io/">
        <img src="https://images.weserv.nl/?url=https://bingykang.github.io/images/bykang_homepage.jpeg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub><b>Bingyi Kang</b></sub>
    </td>
    <td align="center">
      <a href="https://haotongl.github.io/">
        <img src="https://images.weserv.nl/?url=https://haotongl.github.io/assets/img/prof_pic.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Haotong Lin</sub>
    </td>
    <td align="center">
      <a href="https://github.com/SiliChen321">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/195901058?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Sili Chen</sub>
    </td>
    <td align="center">
      <a href="https://liewjunhao.github.io/">
        <img src="https://images.weserv.nl/?url=https://liewjunhao.github.io/images/liewjunhao.png?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
       </a>
        <br />
        <sub>Jun Hao Liew</sub>
    </td>
    <td align="center">
      <a href="https://donydchen.github.io/">
        <img src="https://images.weserv.nl/?url=https://donydchen.github.io/assets/img/profile.jpg?h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Donny Y. Chen</sub>
    </td>
    <td align="center">
      <a href="https://github.com/DengKaiCQ">
        <img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/59907452?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" width="100px;" alt=""/>
      </a>
        <br />
        <sub>Kai Deng</sub>
    </td>
  </tr>
</table>

## 📝 Citations
If you find Depth Anything 3 useful in your research or projects, please cite our work:

```
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
