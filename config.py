from dataclasses import dataclass, field
from typing import List


@dataclass
class AMBConfig:
    # --- Data ---
    data_root: str = "./data/sen12ms_cr"        # Root directory of Sen12MS-CR dataset
    patch_size: int = 256
    sar_bands: int = 2                          # Sentinel-1: VV, VH
    optical_bands: int = 13                     # Sentinel-2: all 13 bands
    num_workers: int = 4
    batch_size: int = 8

    # --- SAR Pre-Processing Stem ---
    sar_hidden_dim: int = 64                    # NAFBlock feature channels
    dcn_groups: int = 4                         # DCNv2 deformable groups

    # --- Vision Mamba Backbone ---
    mamba_dim: int = 256                        # SSM state dimension
    mamba_depth: int = 12                       # Number of Mamba blocks
    mamba_expand: int = 2                       # Inner expansion ratio

    # --- Diffusion Bridge (OT-ODE) ---
    num_timesteps: int = 1000                   # Training diffusion steps
    num_inference_steps: int = 10               # OT-ODE inference steps (5–10)
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # --- Cloud-Aware Adaptive Loss ---
    cloud_loss_weight: float = 10.0             # Weight on cloud-occluded regions
    clear_loss_weight: float = 1.0              # Weight on cloud-free regions
    # Sentinel-2 band indices for cloud mask derivation
    cirrus_band_idx: int = 10                   # B10 (cirrus)
    coastal_band_idx: int = 0                   # B1 (coastal aerosol)
    cloud_threshold: float = 0.1               # Reflectance threshold for cloud detection

    # --- Training ---
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    save_every: int = 5                         # Save checkpoint every N epochs
    log_every: int = 100                        # Log metrics every N steps
    checkpoint_dir: str = "./checkpoints"

    # --- Validation (Frozen Judge Protocol) ---
    frozen_classifier_path: str = ""            # Path to pre-trained land cover segmentation model
    num_classes: int = 9                        # Land cover classes (e.g. DFC2020 labels)

    # --- Hardware ---
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42
