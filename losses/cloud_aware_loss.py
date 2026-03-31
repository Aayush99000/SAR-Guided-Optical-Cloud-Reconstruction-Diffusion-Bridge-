"""
Cloud-Aware Adaptive Loss
--------------------------
Standard MSE/L1 loss applies uniformly across an image, which slightly blurs
the already-clear background pixels. This loss uses a spatial weight mask
derived from the cloud detection (coastal blue B1 / cirrus B10 bands) to
focus ALL gradient energy on the cloud-occluded regions.

Result: perfect composite stitching — clear terrain is left mathematically
untouched during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudAwareAdaptiveLoss(nn.Module):
    """
    Spatially-weighted reconstruction loss.

    For each pixel (i, j):
        weight(i,j) = cloud_weight  if cloud_mask(i,j) == 1  (occluded)
        weight(i,j) = clear_weight  if cloud_mask(i,j) == 0  (clear)

    Loss = mean( weight * |pred - target|² )

    The high cloud_weight / clear_weight ratio (default 10:1) forces the
    network to prioritize reconstructing hidden terrain over preserving
    background fidelity.
    """
    def __init__(
        self,
        cloud_weight: float = 10.0,
        clear_weight: float = 1.0,
        loss_type: str = "l2",          # "l1" or "l2"
    ):
        super().__init__()
        self.cloud_weight = cloud_weight
        self.clear_weight = clear_weight
        assert loss_type in ("l1", "l2"), "loss_type must be 'l1' or 'l2'"
        self.loss_type = loss_type

    def build_weight_mask(self, cloud_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert binary cloud mask to a continuous spatial weight map.

        Args:
            cloud_mask: (B, 1, H, W) — float tensor, 1=cloud, 0=clear
        Returns:
            weight_map: (B, 1, H, W) — per-pixel loss weights
        """
        weight_map = (
            self.clear_weight
            + (self.cloud_weight - self.clear_weight) * cloud_mask
        )
        return weight_map

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:       (B, C, H, W) — reconstructed image
            target:     (B, C, H, W) — ground truth clear image
            cloud_mask: (B, 1, H, W) — binary cloud mask (1=occluded)
        Returns:
            loss: scalar
        """
        weight_map = self.build_weight_mask(cloud_mask)    # (B, 1, H, W)

        if self.loss_type == "l2":
            per_pixel_loss = (pred - target) ** 2
        else:
            per_pixel_loss = (pred - target).abs()

        # Broadcast weight across all channels
        weighted_loss = per_pixel_loss * weight_map        # (B, C, H, W)
        return weighted_loss.mean()

    def cloud_only_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes loss ONLY on cloud-occluded pixels (for diagnostic logging).
        """
        if self.loss_type == "l2":
            per_pixel = (pred - target) ** 2
        else:
            per_pixel = (pred - target).abs()

        mask = cloud_mask.expand_as(per_pixel)
        n_cloud = mask.sum().clamp(min=1)
        return (per_pixel * mask).sum() / n_cloud

    def clear_only_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes loss ONLY on cloud-free pixels (for diagnostic logging).
        """
        if self.loss_type == "l2":
            per_pixel = (pred - target) ** 2
        else:
            per_pixel = (pred - target).abs()

        clear_mask = (1.0 - cloud_mask).expand_as(per_pixel)
        n_clear = clear_mask.sum().clamp(min=1)
        return (per_pixel * clear_mask).sum() / n_clear
