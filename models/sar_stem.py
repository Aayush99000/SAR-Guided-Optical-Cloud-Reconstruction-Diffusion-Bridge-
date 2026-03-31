"""
SAR Pre-Processing Stem: NAFBlock + DCNv2
-----------------------------------------
Translates raw SAR data (speckle noise + geometric layover) into a
clean, optically-aligned feature representation before diffusion fusion.

Pipeline:
    SAR tensor (2, H, W)
        → NAFBlock stack  (despeckle via nonlinear activation-free layers)
        → DCNv2           (geometric alignment: radar layover → top-down optical view)
        → SAR features    (C, H, W) ready for cross-attention with optical branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ---------------------------------------------------------------------------
# SimpleGate — the core nonlinearity in NAFNet (replaces GELU/ReLU entirely)
# ---------------------------------------------------------------------------
class SimpleGate(nn.Module):
    """Splits channels in half and uses the second half to gate the first."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ---------------------------------------------------------------------------
# NAFBlock — Nonlinear Activation-Free Block for SAR Despeckling
# ---------------------------------------------------------------------------
class NAFBlock(nn.Module):
    """
    From: Simple Baselines for Image Restoration (NAFNet, ECCV 2022).
    Replaces all nonlinear activations with SimpleGate and SKFF attention.
    Highly effective at removing multiplicative speckle noise.
    """
    def __init__(self, channels: int, ffn_expand: int = 2, dropout: float = 0.0):
        super().__init__()
        dw_channels = channels * 2      # doubled for SimpleGate split

        # Depth-wise conv path
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.gate = SimpleGate()

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )

        # FFN path
        ffn_channels = channels * ffn_expand * 2  # x2 for SimpleGate
        self.ffn_conv1 = nn.Conv2d(channels, ffn_channels, 1)
        self.ffn_conv2 = nn.Conv2d(ffn_channels // 2, channels, 1)

        self.norm1 = nn.LayerNorm([channels, 1, 1], elementwise_affine=True)
        self.norm2 = nn.LayerNorm([channels, 1, 1], elementwise_affine=True)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scaling
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise attention branch
        residual = x
        x = self._layer_norm(x, self.norm1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)             # (C*2 → C) via SimpleGate
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = residual + x * self.beta

        # FFN branch
        residual = x
        x = self._layer_norm(x, self.norm2)
        x = self.ffn_conv1(x)
        x = self.gate(x)             # (C*4 → C*2) via SimpleGate
        x = self.ffn_conv2(x)
        x = self.dropout(x)
        x = residual + x * self.gamma
        return x

    @staticmethod
    def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        # Apply LayerNorm over channel dim while keeping spatial dims
        b, c, h, w = x.shape
        x = x.reshape(b, c, 1, 1)
        x = norm(x)
        return x.reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# DCNv2 Wrapper — Deformable Convolution for Geometric Alignment
# ---------------------------------------------------------------------------
class DeformableConvBlock(nn.Module):
    """
    Deformable Convolution v2 block.
    Learns spatial offsets to geometrically warp radar signatures
    (which lean due to side-looking SAR geometry) into a top-down
    optical reference frame before cross-attention fusion.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Offset prediction: 2 * k * k values (dx, dy per kernel position)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        # Modulation mask: k * k values in [0, 1]
        self.mask_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=self.padding,
        )
        # Main deformable weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.groups = groups
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.ones_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        out = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=self.padding,
            mask=mask,
        )
        return self.act(self.norm(out))


# ---------------------------------------------------------------------------
# Full SAR Pre-Processing Stem
# ---------------------------------------------------------------------------
class SARPreProcessingStem(nn.Module):
    """
    Complete SAR pre-processing pipeline:
        1. Project 2-channel SAR input → hidden_dim features
        2. Stack of NAFBlocks for speckle removal
        3. DCNv2 block for geometric layover correction
        4. Project to output_dim for cross-attention fusion

    Args:
        in_channels:  Number of SAR input bands (2 for VV/VH)
        hidden_dim:   Internal feature channels (default: 64)
        out_channels: Output channels matching diffusion bridge input
        num_naf_blocks: Depth of the despeckling stack
    """
    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        out_channels: int = 256,
        num_naf_blocks: int = 4,
        dcn_groups: int = 4,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        # NAFBlock stack — despeckling
        self.naf_blocks = nn.Sequential(*[
            NAFBlock(hidden_dim) for _ in range(num_naf_blocks)
        ])

        # DCNv2 block — geometric alignment
        self.dcn = DeformableConvBlock(hidden_dim, hidden_dim, kernel_size=3, groups=dcn_groups)

        # Output projection → match diffusion bridge feature dim
        self.output_proj = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar: (B, 2, H, W) — raw Sentinel-1 VV/VH backscatter
        Returns:
            features: (B, out_channels, H, W) — clean, geo-aligned SAR features
        """
        x = self.input_proj(sar)     # (B, hidden_dim, H, W)
        x = self.naf_blocks(x)       # despeckle
        x = self.dcn(x)              # geometric alignment
        x = self.output_proj(x)      # (B, out_channels, H, W)
        return x
