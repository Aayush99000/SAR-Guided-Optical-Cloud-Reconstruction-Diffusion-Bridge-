"""
Vision Mamba (Vim) Backbone
----------------------------
Replaces the traditional U-Net / ViT backbone with Bidirectional State Space Models (SSMs).
Operates at O(N) complexity — eliminating the O(N²) transformer memory bottleneck —
allowing full-resolution satellite swaths without patch cropping.

Architecture:
    Optical latent tokens (flattened spatial sequence)
        → Patch Embedding
        → Bidirectional Mamba Blocks (forward + backward SSM scan)
        → Reconstruct spatial feature map
        → Output for ODE Bridge solver
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ---------------------------------------------------------------------------
# Selective State Space Model (S6) — Core Mamba Operation
# ---------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """
    Simplified Selective State Space Model (S6) from Mamba (Gu & Dao, 2023).
    The key innovation: A, B, C matrices are input-dependent (selective),
    enabling the model to focus on relevant spatial context.
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input-dependent projections (the "selective" part)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — sequence of spatial tokens
        Returns:
            y: (B, L, d_model)
        """
        b, l, d = x.shape

        # Compute input-dependent SSM parameters
        x_dbl = self.x_proj(x)                          # (B, L, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))               # (B, L, d_model)

        A = -torch.exp(self.A_log.float())               # (d_model, d_state)

        # Discretize A, B via Zero-Order Hold (ZOH)
        dA = torch.exp(dt.unsqueeze(-1) * A)            # (B, L, d_model, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)          # (B, L, d_model, d_state)

        # Sequential SSM scan (recurrent form for simplicity)
        h = torch.zeros(b, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)   # state update
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)                # output
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                      # (B, L, d_model)
        y = y + x * self.D                              # skip connection
        return y


# ---------------------------------------------------------------------------
# Bidirectional Mamba Block — Core Vision Mamba Building Block
# ---------------------------------------------------------------------------
class BidirectionalMambaBlock(nn.Module):
    """
    Processes spatial tokens in BOTH forward and backward directions,
    then fuses the two scans. Captures long-range context from both
    ends of the flattened 2D feature map.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)  # x2: one for gate

        # Forward and backward SSMs
        self.ssm_fwd = SelectiveSSM(d_inner, d_state)
        self.ssm_bwd = SelectiveSSM(d_inner, d_state)

        self.conv = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x: (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                              # (B, L, d_inner*2)
        x_in, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        # 1D depthwise conv over sequence
        x_in = self.conv(x_in.transpose(1, 2)).transpose(1, 2)
        x_in = self.act(x_in)

        # Bidirectional SSM scan
        y_fwd = self.ssm_fwd(x_in)
        y_bwd = self.ssm_bwd(x_in.flip(1)).flip(1)       # reverse → scan → un-reverse
        y = y_fwd + y_bwd

        # Gated output
        y = y * self.act(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return residual + y


# ---------------------------------------------------------------------------
# Patch Embedding & Unembedding for 2D feature maps
# ---------------------------------------------------------------------------
class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W) → (B, L, embed_dim)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class PatchUnembed2D(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, out_channels, 1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, L, embed_dim) → (B, out_channels, H, W)
        B, L, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Cross-Attention: Optical tokens attend to SAR features
# ---------------------------------------------------------------------------
class SARCrossAttention(nn.Module):
    """
    Optical latent tokens (queries) attend to SAR feature tokens (keys/values).
    Enables flexible spatial correspondence between SAR and optical modalities.
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        optical_tokens: torch.Tensor,   # (B, L, d_model) — queries
        sar_tokens: torch.Tensor,        # (B, L_sar, d_model) — keys & values
    ) -> torch.Tensor:
        out, _ = self.attn(optical_tokens, sar_tokens, sar_tokens)
        return self.norm(optical_tokens + out)


# ---------------------------------------------------------------------------
# Vision Mamba Backbone
# ---------------------------------------------------------------------------
class VisionMambaBackbone(nn.Module):
    """
    Full Vision Mamba backbone for the AMB diffusion bridge.

    Accepts:
        - Noisy optical latent z_t   (B, C_opt, H, W)
        - SAR features from stem     (B, C_sar, H, W)
        - Timestep embedding t       (B, d_model)
        - Cloud mask                 (B, 1, H, W)

    Returns:
        - Predicted noise / velocity (B, C_opt, H, W)
    """
    def __init__(
        self,
        in_channels: int = 13,
        sar_channels: int = 256,
        d_model: int = 256,
        depth: int = 12,
        d_state: int = 16,
        expand: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Patch embedding for optical latent
        self.patch_embed = PatchEmbed2D(in_channels + 1, d_model)   # +1 for cloud mask

        # SAR cross-attention projection
        self.sar_proj = nn.Linear(sar_channels, d_model)

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Alternating Mamba + Cross-Attention blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.ModuleDict({
                "mamba": BidirectionalMambaBlock(d_model, d_state, expand, dropout),
                "cross_attn": SARCrossAttention(d_model, num_heads) if (i % 3 == 0) else None,
            }))

        self.norm_out = nn.LayerNorm(d_model)
        self.patch_unembed = PatchUnembed2D(d_model, in_channels)

    def forward(
        self,
        z_t: torch.Tensor,
        sar_features: torch.Tensor,
        t_emb: torch.Tensor,
        cloud_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t:          (B, C, H, W) — noisy optical at timestep t
            sar_features: (B, C_sar, H, W) — output of SARPreProcessingStem
            t_emb:        (B, d_model) — sinusoidal timestep embedding
            cloud_mask:   (B, 1, H, W) — binary cloud mask
        Returns:
            pred:         (B, C, H, W) — predicted velocity / noise
        """
        # Concatenate cloud mask as extra channel
        x = torch.cat([z_t, cloud_mask], dim=1)           # (B, C+1, H, W)

        # Tokenize
        tokens, H, W = self.patch_embed(x)                # (B, L, d_model)

        # Timestep conditioning: add to every token
        t = self.time_mlp(t_emb).unsqueeze(1)             # (B, 1, d_model)
        tokens = tokens + t

        # SAR tokens for cross-attention
        sar_tokens = self.sar_proj(
            sar_features.flatten(2).transpose(1, 2)       # (B, L, d_model)
        )

        # Mamba blocks with interleaved SAR cross-attention
        for block in self.blocks:
            tokens = block["mamba"](tokens)
            if block["cross_attn"] is not None:
                tokens = block["cross_attn"](tokens, sar_tokens)

        tokens = self.norm_out(tokens)
        pred = self.patch_unembed(tokens, H, W)            # (B, C, H, W)
        return pred


# ---------------------------------------------------------------------------
# Sinusoidal Timestep Embedding
# ---------------------------------------------------------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding for diffusion timesteps.
    Args:
        t:   (B,) integer timesteps
        dim: embedding dimension
    Returns:
        emb: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb
