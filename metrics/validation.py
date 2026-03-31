"""
Validation Metrics — The "Frozen Judge" Protocol
--------------------------------------------------
Two-tiered evaluation strategy to prove scientific validity:

Tier 1 — Full-Reference Mathematical Metrics:
    - PSNR  (Peak Signal-to-Noise Ratio)    — absolute pixel accuracy
    - SSIM  (Structural Similarity Index)   — structural & edge alignment
    - SAM   (Spectral Angle Mapper)         — multi-spectral fidelity

Tier 2 — Downstream Task Validation (Frozen Judge):
    - A pre-trained land cover segmentation model is frozen (requires_grad=False)
    - mIoU is measured on:  (a) true clear images [upper bound]
                            (b) AMB-generated images [our score]
    - Minimal accuracy drop = spectral fidelity proven scientifically useful
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Tier 1A: PSNR — Peak Signal-to-Noise Ratio
# ---------------------------------------------------------------------------
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Args:
        pred:   (B, C, H, W) — reconstructed image, in [0, max_val]
        target: (B, C, H, W) — ground truth, in [0, max_val]
    Returns:
        psnr: (B,) — per-sample PSNR in dB
    """
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
    mse = mse.clamp(min=1e-10)
    psnr = 10.0 * torch.log10(max_val ** 2 / mse)
    return psnr


# ---------------------------------------------------------------------------
# Tier 1B: SSIM — Structural Similarity Index
# ---------------------------------------------------------------------------
def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Computes SSIM averaged over all channels.

    Args:
        pred:   (B, C, H, W)
        target: (B, C, H, W)
    Returns:
        ssim: (B,) — per-sample mean SSIM
    """
    kernel = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)  # (1, 1, k, k)

    def _pool(x):
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        return F.conv2d(x, kernel, padding=window_size // 2).reshape(B, C, H, W)

    mu_x = _pool(pred)
    mu_y = _pool(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = _pool(pred * pred) - mu_x2
    sigma_y2 = _pool(target * target) - mu_y2
    sigma_xy = _pool(pred * target) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den.clamp(min=1e-8)
    return ssim_map.mean(dim=[1, 2, 3])


def _gaussian_kernel(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Tier 1C: SAM — Spectral Angle Mapper
# ---------------------------------------------------------------------------
def compute_sam(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spectral Angle Mapper — measures the angle between predicted and target
    spectral vectors at each pixel. Smaller angle = better spectral fidelity.

    Critical for proving multi-spectral chemical reflectance accuracy
    (e.g., vegetation vs. water classification won't work if SAM is high).

    Args:
        pred:   (B, C, H, W) — C spectral bands
        target: (B, C, H, W)
    Returns:
        sam: (B,) — mean spectral angle in radians per sample
    """
    # Reshape to (B, C, N) where N = H*W
    B, C, H, W = pred.shape
    pred_flat = pred.reshape(B, C, -1)
    target_flat = target.reshape(B, C, -1)

    # Dot product and norms along channel dim
    dot = (pred_flat * target_flat).sum(dim=1)      # (B, N)
    norm_pred = pred_flat.norm(dim=1).clamp(min=1e-8)
    norm_tgt = target_flat.norm(dim=1).clamp(min=1e-8)

    cos_angle = (dot / (norm_pred * norm_tgt)).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_angle)                   # (B, N) in radians
    return angle.mean(dim=1)                        # (B,)


# ---------------------------------------------------------------------------
# Tier 2: Frozen Judge — Downstream mIoU Validation
# ---------------------------------------------------------------------------
class FrozenJudgeValidator:
    """
    Loads a pre-trained land cover segmentation model and evaluates
    Mean IoU (mIoU) on both ground truth and AMB-generated images.

    The gap between truth_miou and generated_miou proves spectral fidelity.
    A small gap (<2%) proves the generated data is scientifically actionable.

    Usage:
        validator = FrozenJudgeValidator(model_path, num_classes=9)
        results = validator.evaluate(generated_images, true_clear_images)
        print(results)  # {"miou_truth": 0.82, "miou_generated": 0.80, "gap": 0.02}
    """
    def __init__(self, model_path: str, num_classes: int = 9, device: str = "cuda"):
        self.num_classes = num_classes
        self.device = device
        self.model = self._load_frozen_model(model_path)

    def _load_frozen_model(self, model_path: str) -> Optional[nn.Module]:
        if not model_path:
            print("[FrozenJudge] No model path provided — Tier 2 validation skipped.")
            return None
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False             # FREEZE all weights
            print(f"[FrozenJudge] Loaded and frozen classifier from {model_path}")
            return model
        except Exception as e:
            print(f"[FrozenJudge] Failed to load model: {e}")
            return None

    @torch.no_grad()
    def compute_miou(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Args:
            images: (B, C, H, W) — optical images (generated or true)
            labels: (B, H, W) — ground truth land cover labels (long tensor)
        Returns:
            miou: float — mean IoU across all classes
        """
        if self.model is None:
            return float("nan")

        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(images)                         # (B, num_classes, H, W)
        preds = logits.argmax(dim=1)                        # (B, H, W)

        iou_per_class = []
        for cls in range(self.num_classes):
            pred_cls = (preds == cls)
            true_cls = (labels == cls)
            intersection = (pred_cls & true_cls).sum().float()
            union = (pred_cls | true_cls).sum().float()
            if union == 0:
                continue
            iou_per_class.append((intersection / union).item())

        return float(np.mean(iou_per_class)) if iou_per_class else float("nan")

    @torch.no_grad()
    def evaluate(
        self,
        generated: torch.Tensor,
        truth: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Full Frozen Judge evaluation.

        Args:
            generated: (B, C, H, W) — AMB-reconstructed images
            truth:     (B, C, H, W) — true clear images (upper bound)
            labels:    (B, H, W)    — land cover ground truth labels
        Returns:
            dict with miou_truth, miou_generated, gap
        """
        miou_truth = self.compute_miou(truth, labels)
        miou_generated = self.compute_miou(generated, labels)
        gap = miou_truth - miou_generated if not (np.isnan(miou_truth) or np.isnan(miou_generated)) else float("nan")
        return {
            "miou_truth": miou_truth,
            "miou_generated": miou_generated,
            "gap": gap,
        }


# ---------------------------------------------------------------------------
# Convenience: compute all Tier 1 metrics in one call
# ---------------------------------------------------------------------------
def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloud_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute PSNR, SSIM, SAM for a batch and return mean values.
    Optionally also computes cloud-region-only metrics.

    Args:
        pred:       (B, C, H, W)
        target:     (B, C, H, W)
        cloud_mask: (B, 1, H, W) optional — if provided, also reports cloud-only metrics
    Returns:
        dict of metric name → mean float value
    """
    metrics = {
        "psnr": compute_psnr(pred, target).mean().item(),
        "ssim": compute_ssim(pred, target).mean().item(),
        "sam":  compute_sam(pred, target).mean().item(),
    }

    if cloud_mask is not None:
        # Cloud-region only metrics
        mask = cloud_mask.expand_as(pred).bool()
        pred_cloud = pred.clone()
        pred_cloud[~mask] = target[~mask]      # zero out clear regions
        target_cloud = target.clone()

        metrics["psnr_cloud"] = compute_psnr(pred_cloud, target_cloud).mean().item()
        metrics["ssim_cloud"] = compute_ssim(pred_cloud, target_cloud).mean().item()
        metrics["sam_cloud"] = compute_sam(pred_cloud, target_cloud).mean().item()

    return metrics
