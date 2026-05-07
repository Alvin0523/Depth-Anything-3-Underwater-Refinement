"""
Dataset loader for Unity-exported RGB + metric depth pairs.

Expected folder layout:
    data/unity/
        rgb/    <frame_id>.png
        depth/  <frame_id>.npy   (float32, metric metres)
                   OR
                <frame_id>.png   (uint16, depth_mm / 1000 = metres)

Preprocessing applied at load time (train AND inference):
    1. Gray World white balance  (corrects blue-green underwater cast)
    2. Percentile histogram stretching  (2nd–98th percentile per channel)
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """img: float32 HxWx3 in [0,1]. Returns corrected float32 HxWx3."""
    means = img.mean(axis=(0, 1))          # (3,)
    ref = means.mean()                     # scalar
    scale = np.where(means > 0, ref / means, 1.0)
    return np.clip(img * scale, 0.0, 1.0)


def histogram_stretch(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile stretch per channel. img: float32 HxWx3 in [0,1]."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        p_lo = np.percentile(img[:, :, c], lo)
        p_hi = np.percentile(img[:, :, c], hi)
        denom = p_hi - p_lo if p_hi > p_lo else 1e-6
        out[:, :, c] = np.clip((img[:, :, c] - p_lo) / denom, 0.0, 1.0)
    return out


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline. Input: uint8 HxWx3 BGR. Output: float32 HxWx3 RGB in [0,1]."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = gray_world_white_balance(img)
    img = histogram_stretch(img)
    return img


class UnityDepthDataset(Dataset):
    """
    Loads paired RGB + depth from a Unity capture directory.

    Args:
        root:        Path to directory containing rgb/ and depth/ subdirs.
        input_size:  (H, W) to resize inputs. DA3 expects multiples of 14.
        split:       'train' or 'val'.
        val_fraction: Fraction of data held out for validation.
        seed:        Random seed for split reproducibility.
        max_depth:   Clip depth values beyond this (metres). Avoids inf GT.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root: str,
        input_size: tuple[int, int] = (518, 518),
        split: str = "train",
        val_fraction: float = 0.2,
        seed: int = 42,
        max_depth: float = 10.0,
    ):
        self.root = Path(root)
        self.input_size = input_size
        self.max_depth = max_depth

        rgb_dir = self.root / "rgb"
        depth_dir = self.root / "depth"

        rgb_files = sorted(rgb_dir.glob("*.png"))
        self.samples = []
        for rgb_path in rgb_files:
            stem = rgb_path.stem
            depth_npy = depth_dir / f"{stem}.npy"
            depth_png = depth_dir / f"{stem}.png"
            if depth_npy.exists():
                self.samples.append((rgb_path, depth_npy))
            elif depth_png.exists():
                self.samples.append((rgb_path, depth_png))

        assert len(self.samples) > 0, f"No paired samples found under {root}"

        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx = set(indices[:n_val])

        if split == "val":
            self.samples = [self.samples[i] for i in indices[:n_val]]
        else:
            self.samples = [self.samples[i] for i in indices[n_val:]]

        print(f"[Dataset] {split}: {len(self.samples)} samples from {root}")

    def _load_depth(self, path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            stored = np.load(path).astype(np.float32)
            # MIMIR-UW stores inverse depth (1/metres) for compression.
            # Zeros in the stored array represent invalid/infinite distance.
            depth = np.where(stored > 0, 1.0 / stored, 0.0)
        else:
            # uint16 PNG: assume millimetres stored as uint16
            depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            depth = depth_raw.astype(np.float32) / 1000.0
        return depth

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb_path, depth_path = self.samples[idx]

        img_bgr = cv2.imread(str(rgb_path))
        img = preprocess_image(img_bgr)                        # float32 HxWx3 [0,1]
        depth = self._load_depth(depth_path)                   # float32 HxW metres

        H, W = self.input_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # Zero out background pixels beyond max_depth (they are invalid, not at max_depth m)
        depth = np.where((depth > 0) & (depth <= self.max_depth), depth, 0.0)

        # ImageNet normalise then to CHW tensor
        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)
        depth_tensor = torch.from_numpy(depth)                 # (H, W)

        return {"image": img_tensor, "depth": depth_tensor, "rgb_path": str(rgb_path)}


class MIMIRSeaFloorDataset(Dataset):
    """
    Loads paired RGB + inverse-depth from the MIMIR SeaFloor benchmark dataset.

    Expected layout (one or more tracks):
        root/
            track0/auv0/rgb/cam0/data/*.png
            track0/auv0/depth/cam0/data/*.exr   (float32 inverse-depth, 1/metres)
            track1/...
            track2/...

    Args:
        root:        Path to the SeaFloor root directory (contains track* folders).
        input_size:  (H, W) to resize to. DA3 expects multiples of 14.
        max_depth:   Clip depths beyond this value (metres).
        cams:        List of cameras to include. Defaults to ['cam0', 'cam1'] (front
                     cameras only — cam2 is the straight-down bottom camera and is
                     excluded by default).
        agent:       Agent folder name (default 'auv0').
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        root: str,
        input_size: tuple[int, int] = (518, 518),
        max_depth: float = 20.0,
        cams: list[str] | None = None,
        agent: str = "auv0",
    ):
        self.input_size = input_size
        self.max_depth = max_depth
        # Default: front cameras only (cam2 = bottom_center, excluded)
        if cams is None:
            cams = ["cam0", "cam1"]

        root_path = Path(root)
        self.samples: list[tuple[Path, Path]] = []

        for track_dir in sorted(root_path.glob("track*")):
            for cam in cams:
                rgb_dir = track_dir / agent / "rgb" / cam / "data"
                depth_dir = track_dir / agent / "depth" / cam / "data"
                if not rgb_dir.is_dir() or not depth_dir.is_dir():
                    continue
                for rgb_path in sorted(rgb_dir.glob("*.png")):
                    depth_path = depth_dir / f"{rgb_path.stem}.exr"
                    if depth_path.exists():
                        self.samples.append((rgb_path, depth_path))

        assert len(self.samples) > 0, f"No paired EXR samples found under {root} for cameras {cams}"
        print(f"[MIMIRSeaFloorDataset] {len(self.samples)} samples from {root} (cams: {cams})")

    def _load_depth(self, path: Path) -> np.ndarray:
        # EXR stores inverse depth (1/metres); convert to metres.
        stored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        return np.where(stored > 0, 1.0 / stored, 0.0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb_path, depth_path = self.samples[idx]

        img_bgr = cv2.imread(str(rgb_path))
        img = preprocess_image(img_bgr)           # float32 HxWx3 [0,1]
        depth = self._load_depth(depth_path)      # float32 HxW metres

        H, W = self.input_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        depth = np.where((depth > 0) & (depth <= self.max_depth), depth, 0.0)

        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)
        depth_tensor = torch.from_numpy(depth)                 # (H, W)

        return {"image": img_tensor, "depth": depth_tensor, "rgb_path": str(rgb_path)}
