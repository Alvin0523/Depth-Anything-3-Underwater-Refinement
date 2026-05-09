# 📦 Dataset

Details on the training and benchmark datasets, their file formats, preprocessing, and splits.

---

## MIMIR-UW

[MIMIR-UW](https://github.com/remaro-network/MIMIR-UW) is a synthetic underwater dataset rendered in Unreal Engine 4 with the AirSim plugin. It provides synchronized RGB + metric depth pairs across four distinct underwater environments modelling physics-accurate light attenuation, backscatter, and caustics.

| Environment | Frames | Conditions |
|---|---|---|
| SeaFloor | ~10k | Rocky terrain, medium visibility |
| **SeaFloor Algae** | **9,987** | **Dynamic algae occlusions, high texture — used for training** |
| OceanFloor | ~10k | Flat sandy bottom, varying depth |
| SandPipe | ~10k | Pipeline inspection, low visibility |

**Total:** 39,943 synchronized RGB+depth pairs across all environments.

---

## Training Subset: SeaFloor Algae

We train exclusively on the SeaFloor Algae environment (9,987 frames) — a challenging mid-complexity scenario with dynamic algae occlusions and high texture complexity, which provides robust depth cues without the confounding factors of extreme darkness (OceanFloor) or featureless sand (SandPipe).

### Split

| Set | Samples | Fraction |
|---|---|---|
| Train | 7,990 | 80% |
| Validation | 1,997 | 20% |

Split is deterministic with `random.seed(42)`.

---

## Benchmark Dataset: MIMIR-UW SeaFloor (held-out)

The zero-shot baseline and fine-tuned model are benchmarked on the held-out **SeaFloor** environment — a different environment from SeaFloor Algae used for training, testing cross-environment generalisation.

### Frame Breakdown

| Track | Cameras | Frames |
|---|---|---|
| track0 | cam0 + cam1 | 2,847 × 2 = 5,694 |
| track1 | cam0 + cam1 | 2,030 × 2 = 4,060 |
| track2 | cam0 + cam1 | 2,537 × 2 = 5,074 |
| **Total** | | **14,828** |

> cam2 (straight-down nadir view) is excluded — the model was not trained on nadir views.

---

## File Format

### RGB Images

- Format: 8-bit PNG
- Resolution: approximately 1280×720 (native), resized to **518×518** for training
- 518 = 37 × 14 — satisfies the ViT patch tokeniser requirement (multiples of 14)

### Depth Ground Truth — ⚠️ Inverse Depth

**MIMIR-UW stores depth as float32 inverse-depth (1/metres)** for storage efficiency, not direct metres.

```
stored_value = 1 / depth_in_metres
```

The `_load_depth()` function in `train/dataset.py` handles this automatically:

```python
stored = np.load(path).astype(np.float32)
depth = np.where(stored > 0, 1.0 / stored, 0.0)
```

**Background pixels** (sky, infinite distance) are encoded as a very small positive value (~6×10⁻⁵), which after inversion maps to depths >10,000 m. These are zeroed by the 10 m max-depth cutoff.

### SeaFloor Benchmark Format

The held-out SeaFloor benchmark stores depth as float32 **inverse-depth EXR files** — same format, handled by `train/dataset.py`'s `MIMIRSeaFloorDataset` class.

---

## Depth Range

| Parameter | Value | Source |
|---|---|---|
| Min clip | `> 0` (strict positive) | `train/dataset.py` line 139 |
| Max clip | `10.0 m` | configured via `--max_depth` |
| Observed valid range | ~0–10 m | SeaFloor Algae scene geometry |

---

## Physics-Aware Preprocessing

Applied at load time identically during **both training and inference**.

### 1. Gray World White Balance

Corrects wavelength-dependent light attenuation that produces a blue-green cast. Per-channel multiplicative scaling:

```
scale_c = mean(all_channels) / mean(channel_c)
```

Equalises all channel means, neutralising the colour cast without requiring per-image calibration.

### 2. Percentile Histogram Stretching

Robust contrast normalisation — each channel is linearly stretched so the 2nd percentile maps to 0 and the 98th percentile maps to 1:

```
stretched = (pixel - p2) / (p98 - p2)
```

Avoids sensitivity to outlier pixel intensities from specular highlights or backscatter hot spots.

### 3. ImageNet Normalisation

Final normalisation to match the pretrained DINOv2 backbone's expected input distribution:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## Dataset Loader

All loading and preprocessing is implemented in `train/dataset.py`:

| Class | Dataset |
|---|---|
| `UnityDepthDataset` | SeaFloor Algae training data (NPY inverse-depth + PNG RGB) |
| `MIMIRSeaFloorDataset` | SeaFloor benchmark (EXR inverse-depth, multi-track/multi-cam) |

See [Implementation](implementation.md#traindatasetpy) for full details.
