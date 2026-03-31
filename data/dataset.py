"""
Sen12MS-CR DataLoader
---------------------
Loads triplets of (SAR, Cloudy Optical, Clear Optical) patches from the
Sen12MS-CR dataset. Each sample is a 256x256 patch.

Expected directory layout:
    data_root/
        s1/          <- Sentinel-1 SAR .tif files (2 bands: VV, VH)
        s2_cloudy/   <- Sentinel-2 cloud-covered .tif files (13 bands)
        s2_clear/    <- Sentinel-2 cloud-free .tif files (13 bands)

Each triplet shares the same filename stem (e.g., ROI_0001_patch_0042.tif).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path
from typing import Tuple, Optional


class Sen12MSCRDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",           # "train" or "test"
        patch_size: int = 256,
        normalize: bool = True,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size
        self.normalize = normalize
        self.augment = augment and (split == "train")

        self.sar_dir = self.data_root / split / "s1"
        self.cloudy_dir = self.data_root / split / "s2_cloudy"
        self.clear_dir = self.data_root / split / "s2_clear"

        self.samples = sorted([
            f.stem for f in self.sar_dir.glob("*.tif")
        ])
        assert len(self.samples) > 0, f"No .tif files found in {self.sar_dir}"
        print(f"[Sen12MS-CR] {split}: {len(self.samples)} triplets loaded.")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_tif(self, path: Path) -> np.ndarray:
        """Read a GeoTIFF and return a float32 numpy array (C, H, W)."""
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        return data

    def _normalize(self, img: np.ndarray, clip_max: float = 10000.0) -> np.ndarray:
        """Normalize Sentinel-2 reflectance to [0, 1]."""
        return np.clip(img / clip_max, 0.0, 1.0)

    def _normalize_sar(self, sar: np.ndarray) -> np.ndarray:
        """Normalize SAR backscatter (dB) to [0, 1] using typical SAR range."""
        db_min, db_max = -25.0, 0.0
        return np.clip((sar - db_min) / (db_max - db_min), 0.0, 1.0)

    def _augment(
        self,
        sar: np.ndarray,
        cloudy: np.ndarray,
        clear: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random horizontal/vertical flip — applied identically to all three modalities."""
        if np.random.rand() > 0.5:
            sar = np.flip(sar, axis=2).copy()
            cloudy = np.flip(cloudy, axis=2).copy()
            clear = np.flip(clear, axis=2).copy()
        if np.random.rand() > 0.5:
            sar = np.flip(sar, axis=1).copy()
            cloudy = np.flip(cloudy, axis=1).copy()
            clear = np.flip(clear, axis=1).copy()
        return sar, cloudy, clear

    def __getitem__(self, idx: int) -> dict:
        stem = self.samples[idx]

        sar = self._read_tif(self.sar_dir / f"{stem}.tif")         # (2, H, W)
        cloudy = self._read_tif(self.cloudy_dir / f"{stem}.tif")   # (13, H, W)
        clear = self._read_tif(self.clear_dir / f"{stem}.tif")     # (13, H, W)

        if self.normalize:
            sar = self._normalize_sar(sar)
            cloudy = self._normalize(cloudy)
            clear = self._normalize(clear)

        if self.augment:
            sar, cloudy, clear = self._augment(sar, cloudy, clear)

        # Derive cloud mask from Sentinel-2 B1 (coastal) and B10 (cirrus)
        cloud_mask = self._derive_cloud_mask(cloudy)               # (1, H, W)

        return {
            "sar": torch.from_numpy(sar),               # (2, H, W)
            "cloudy": torch.from_numpy(cloudy),         # (13, H, W)
            "clear": torch.from_numpy(clear),           # (13, H, W)
            "cloud_mask": torch.from_numpy(cloud_mask), # (1, H, W)  1=cloud, 0=clear
            "sample_id": stem,
        }

    def _derive_cloud_mask(self, optical: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Simple cloud mask from coastal aerosol (B1, idx=0) and cirrus (B10, idx=10).
        Returns binary mask: 1 = cloud-occluded, 0 = clear.
        """
        coastal = optical[0]    # B1
        cirrus = optical[10]    # B10
        mask = ((coastal > threshold) | (cirrus > threshold)).astype(np.float32)
        return mask[np.newaxis, ...]  # (1, H, W)


def build_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    train_dataset = Sen12MSCRDataset(
        data_root=config.data_root,
        split="train",
        patch_size=config.patch_size,
        normalize=True,
        augment=True,
    )
    test_dataset = Sen12MSCRDataset(
        data_root=config.data_root,
        split="test",
        patch_size=config.patch_size,
        normalize=True,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
