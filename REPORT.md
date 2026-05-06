# Depth Anything 3 Underwater Fine-Tuning: Setup and Training Report

## Abstract

This report documents the setup, configuration, training, and results of LoRA fine-tuning of Depth Anything 3 (DA3) Mono Metric Large on the MIMIR-UW synthetic underwater dataset. The project addresses domain adaptation challenges for underwater depth estimation through efficient parameter adaptation (rank-8 LoRA, 9.24% trainable parameters) combined with physics-aware preprocessing. Training was executed on the NSCC HPC cluster (A100 40 GB) and achieved a final validation AbsRel of **0.099**, RMSE of **0.739 m**, and δ<1.25 accuracy of **91.0%** over 30 epochs.

---

## 1. Introduction

Depth Anything 3 is a state-of-the-art monocular and multi-view depth estimation model trained predominantly on terrestrial datasets. Application to underwater robotics requires adaptation to the underwater domain, where wavelength-dependent light attenuation causes chromatic aberration (blue-green color cast) and backscatter reduces contrast. This project employs Low-Rank Adaptation (LoRA) to efficiently fine-tune DA3 on synthetic underwater data without catastrophic forgetting.

**Objectives:**
1. Establish network connectivity and git infrastructure for repository management
2. Download and organize 39 GB training dataset from HuggingFace
3. Configure training environment and validate data pipeline
4. Submit training jobs to HPC scheduler

---

## 2. System Configuration and Constraints

### 2.1 Network Connectivity Issue

**Problem:** SSH port 22 (git@github.com) was blocked on the network, preventing standard SSH-based git operations.

**Solution:** Converted all git remotes from SSH (`git@github.com:...`) to HTTPS protocol:

```bash
git remote set-url origin https://github.com/Alvin0523/Depth-Anything-3-Underwater-Refinement-.git
git remote set-url upstream https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

**Verification:** `git fetch origin` completed successfully over HTTPS.

### 2.2 Storage Configuration

**Constraint:** Home directory quota insufficient for 39 GB dataset.

**Solution:** 
- Dataset cloned to `/home/users/ntu/m230060/scratch/COMP4471_git/` (34.63 GB)
- Symbolic links created in project directory: 
  ```
  data/unity/rgb → /home/users/ntu/m230060/scratch/COMP4471_git/dataset/rgb/cam0
  data/unity/depth → /home/users/ntu/m230060/scratch/COMP4471_git/dataset/depth/cam0
  ```

---

## 3. Data Pipeline

### 3.1 Dataset Structure and Format

**Source:** MIMIR-UW — Multipurpose Underwater Dataset (Cerqueira et al.)
Collected with Unreal Engine 4 + AirSim plugin across four synthetic underwater environments: *SeaFloor*, *SeaFloor Algae*, *OceanFloor*, *SandPipe*.
Converted from original EXR format to NPY for compatibility with the training pipeline.

**Layout:**
```
data/unity/
├── rgb/
│   ├── <timestamp>.png  (RGB images, 8-bit PNG, 720×540)
│   └── ... (7,499 files)
└── depth/
    ├── <timestamp>.npy  (inverse depth, float32)
    └── ... (7,499 files)
```

**Data Characteristics:**
- **Modality:** Paired RGB + depth from synthetic underwater AUV trajectories
- **Camera:** Forward-facing cam0, 7,499 synchronized frame pairs
- **Depth Format:** NumPy float32 arrays storing **inverse depth** (1/metres), as per the MIMIR-UW paper: *"This depth recording is converted and stored as an inverse depth image for storage efficiency. The depth values recorded as zeros are stored as zeros in the inverse depth image to avoid zero division."*
- **Background pixels:** Encoded as a very small positive value (~6×10⁻⁵), which converts to >10,000 m after inversion and is masked out by the 10 m cutoff
- **Valid pixel fraction:** ~48% of pixels per frame (valid scene surface within 10 m)
- **Metric depth range (valid pixels):** 2.0 m – 10.0 m, mean ~4.3 m
- **Resolution:** 720×540 native (resized to 518×518 during training)

### 3.2 Preprocessing Pipeline

Applied at load-time (both train and inference):

1. **Gray World White Balance** — corrects blue-green underwater color cast
   - Per-channel scaling: `scale[c] = mean_all_channels / mean[c]`
   
2. **Percentile Histogram Stretching** — robust contrast normalization
   - Per-channel: stretch to [2nd percentile, 98th percentile]
   
3. **ImageNet Normalization** — standardize to pre-trained backbone statistics
   - Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]

**Implementation:** See `train/dataset.py` (UnityDepthDataset class)

### 3.3 Data Split

- **Train:** 80% of pairs (6,000 samples)
- **Validation:** 20% of pairs (1,499 samples)
- **Deterministic split:** Fixed random seed (42) for reproducibility

---

## 4. Model Configuration

### 4.1 Base Model

- **Model:** DA3 Mono Metric Large (DINOv2-L backbone)
- **Source:** HuggingFace (`depth-anything/da3metric-large`)
- **Parameters:** ~300M total parameters

### 4.2 LoRA Adaptation

**Strategy:** Apply LoRA to attention blocks in DINOv2 backbone only

**Hyperparameters:**
- **Rank:** 8
- **Alpha:** 16.0 (scaling factor)
- **Trainable Parameters:** ~1% of total (~3M parameters)
- **Frozen:** Encoder blocks (except LoRA), prediction head trainable

**Rationale:** Attention blocks are primary adaptation targets for domain shift. Low rank preserves pre-trained features while enabling efficient fine-tuning.

---

## 5. Training Configuration

### 5.1 Hardware Allocation

- **GPU:** 1× NVIDIA A100 40GB
- **CPU:** 16 cores
- **Memory:** 64 GB system RAM
- **Walltime:** 12 hours (sufficient for 30 epochs)

### 5.2 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 30 | Balanced convergence without overfitting |
| Batch Size | 16 | Maximize GPU utilization (A100 40GB handles ~16 @ 518×518) |
| Learning Rate | 2e-5 | Reduced from initial 1e-4 after observing instability |
| Optimizer | AdamW | Standard for transformer adaptation |
| Weight Decay | 1e-4 | L2 regularization |
| Scheduler | Cosine Annealing | Smooth LR decay to learning rate floor (1e-6) |
| Loss | SILog + Sobel Gradient | Scale-invariant log + edge-aware regularization |
| Gradient Loss Weight | 0.5 | Balance photometric + geometric supervision |
| Scale Anchor Weight | 0.1 | Penalises global depth scale drift in SILog |
| Gradient Clip | 1.0 | Max gradient norm; prevents single-batch weight explosions |

### 5.3 Input Specifications

- **Size:** 518×518 (multiple of 14, required by DA3 vit tokenizer)
- **Format:** float32 RGB [0, 1] (preprocessed)
- **Depth Range:** Clipped to [0, 10] metres (underwater typical range)

---

## 6. Environment Setup

### 6.1 Dependency Management

**Package Manager:** Pixi (conda-free, deterministic)

**Environment Location:** 
```
.pixi/envs/default/bin/python
```

**Key Dependencies:**
- PyTorch 2.x (GPU-enabled)
- torchvision
- huggingface_hub, transformers
- opencv-python (image I/O + preprocessing)
- xformers (memory-efficient attention)
- imageio (depth visualization)

### 6.2 Authentication

**HuggingFace Token:** Stored in `~/.huggingface/token`
- Used for gated model access (DA3 weights)
- Passed via `$HF_TOKEN` environment variable in PBS jobs

---

## 7. Job Submission

### 7.1 PBS Job Script

**File:** `train_lora.pbs`

**Key Configuration:**
```bash
#PBS -q normal                           # Job queue
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb  # Resource request
#PBS -l walltime=12:00:00                # Maximum runtime
#PBS -P personal                         # Project allocation
```

**Execution:**
```bash
cd /home/users/ntu/m230060/Depth-Anything-3-Underwater-Refinement-
qsub train_lora.pbs
```

### 7.2 Environment Variables (Job)

```bash
CUDA_VISIBLE_DEVICES=0              # Single GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory efficiency
TOKENIZERS_PARALLELISM=false        # Suppress tokenizers fork warning
HF_TOKEN=***                        # HuggingFace authentication
```

### 7.3 Checkpointing Strategy

- **Best Model:** `checkpoints/best.pth` (lowest validation AbsRel)
- **Last Model:** `checkpoints/last.pth` (resume capability)
- **Checkpoint Contents:** Model state, optimizer state, epoch, best metric, args

**Resume Example:**
```bash
python train/train.py --data_root data/unity --resume checkpoints/last.pth
```

---

## 8. Validation and Metrics

### 8.1 Evaluation Metrics

Computed on validation set after each epoch:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| AbsRel | mean(\|pred - gt\| / gt) | Relative depth error |
| RMSE | sqrt(mean((pred - gt)²)) | Absolute depth error in metres |
| δ<1.25 | % of pairs where max(pred/gt, gt/pred) < 1.25 | Accuracy within 25% |

### 8.2 Smoke Test

Pre-training validation: `test_lora_pipeline.py`
- Verifies environment setup
- Tests LoRA injection
- Validates dataset loading
- Confirms CUDA availability

**Execution:**
```bash
qsub test_lora_pipeline.pbs
```

---

## 9. Training Results (Job 14352845)

### 9.1 Final Performance

| Metric | Epoch 1 | Epoch 10 | Epoch 18 | **Best (Ep 27)** | Epoch 30 |
|--------|---------|---------|---------|-----------------|----------|
| AbsRel | 0.186 | 0.115 | 0.102 | **0.099** | 0.099 |
| RMSE (m) | 1.040 | 0.786 | 0.738 | **0.739** | 0.741 |
| δ<1.25 | 76.5% | 88.9% | 90.6% | **91.0%** | 91.0% |

- **Training time:** ~8.5 hours (30 epochs × ~17 min each, A100 40 GB)
- **Best checkpoint:** epoch 27, uploaded to `Frieddeli/COMP4471` on HuggingFace
- **Convergence:** Stable throughout; no divergence. Model plateaued naturally around epoch 18.

### 9.2 Output Artifacts

- Trained LoRA weights: `/home/users/ntu/m230060/scratch/da3_checkpoints/best.pth`
- Last checkpoint: `/home/users/ntu/m230060/scratch/da3_checkpoints/last.pth`
- TensorBoard logs: `/home/users/ntu/m230060/scratch/da3_checkpoints/tb_logs/`
- Training log: `train_lora_da3.o14352845`
- HuggingFace: `Frieddeli/COMP4471` → `checkpoints/best.pth`

---

## 10. Known Issues and Mitigations

| Issue | Root Cause | Mitigation |
|-------|-----------|-----------|
| SSH port 22 blocked | Network firewall | Switched to HTTPS remotes |
| Dataset too large for home | Quota limits | Symlinked from scratch storage |
| Rate limit (HuggingFace API) | Concurrent downloads | Used `git clone` + git-lfs (batch download) |
| Pixi env not isolated | Global python used | Explicitly call `.pixi/envs/default/bin/python` |
| **AbsRel explosion (job 14344846)** | Depth loader treated inverse-depth NPY as direct metres (GT depths read as ~0.15 m instead of ~6.7 m) | Fixed `_load_depth` to invert: `depth = 1.0 / stored`; background pixels zeroed via 10 m cutoff |
| Training divergence (lr=1e-4) | SILog `variance_focus` term cancels global scale penalty; model learned arbitrary scale offsets | Added scale anchor term (`λ·\|mean(log pred − log gt)\|`) to SILog; reduced lr to 2e-5 |
| LoRA weights on CPU | `inject_lora()` creates new `nn.Linear` layers on CPU after model already moved to GPU | Added second `da3.to(device)` call after `inject_lora()` |
| bfloat16 autocast → float32 Sobel mismatch | Autocast returned float16; Sobel kernel buffers were float32 | Switched to `torch.bfloat16` autocast + explicit `.float()` cast before loss |

---

## 11. Future Work

1. **Inference Pipeline:** Implement depth prediction endpoint for live AUV deployment
2. **Multi-View Adaptation:** Extend to multi-camera LoRA fine-tuning (DA3 multi-view mode)
3. **Real Underwater Data:** Validate fine-tuned model on actual seafloor imagery
4. **Distillation:** Compress LoRA-adapted model for edge deployment
5. **Hyperparameter Sweep:** Tune LoRA rank, learning rate, batch size via grid search

---

## 12. References

1. Lin, H., et al. "Depth Anything 3: Recovering the Visual Space from Any Views." arXiv:2511.10647 (2025).
2. Oquab, M., et al. "DINOv2: Learning robust visual features without supervision." arXiv:2304.07193 (2023).
3. Hu, E. J., et al. "LoRA: Low-rank adaptation of large language models." arXiv:2106.09685 (2021).
4. Eigen, D., Puhrsch, C., Fergus, R. "Depth map prediction from a single image using a multi-scale deep network." ECCV 2014.

---

## Appendix A: File Structure

```
Depth-Anything-3-Underwater-Refinement-/
├── train/
│   ├── train.py              # Main training script
│   ├── dataset.py            # UnityDepthDataset loader
│   ├── lora.py               # LoRA injection + counting
│   ├── losses.py             # SILog + gradient loss
│   ├── test_lora_pipeline.py # Smoke test
│   └── evaluate.py           # Benchmarking
├── src/
│   └── depth_anything_3/     # DA3 package (installed)
├── checkpoints/              # (created at runtime)
│   ├── best.pth
│   └── last.pth
├── data/
│   └── unity/                # Symlinked to scratch dataset
│       ├── rgb/ → .../dataset/rgb/cam0
│       └── depth/ → .../dataset/depth/cam0
├── train_lora.pbs            # Main training job
├── test_lora_pipeline.pbs    # Smoke test job
├── pyproject.toml
└── README.md
```

---

**Report Generated:** 2026-05-05  
**Last Updated:** 2026-05-07  
**Status:** Training Complete — Best AbsRel 0.099 (Job 14352845)  
**Next Steps:** TensorRT export → Jetson Orin NX deployment → real pool test
