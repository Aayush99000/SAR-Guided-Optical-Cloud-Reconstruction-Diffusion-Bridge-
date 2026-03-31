"""
Diffusion Bridge — Optimal Transport ODE (DB-CR)
-------------------------------------------------
Instead of starting from pure Gaussian noise N(0, I), the bridge builds a
direct deterministic trajectory from the CLOUDY image distribution to the
CLEAR image distribution via an Optimal Transport ODE.

This reduces inference from ~1,000 steps (standard DDPM) to just 5–10 steps,
making the model viable for rapid disaster-response mapping.

Key insight:
    - Standard diffusion:  x_T ~ N(0, I)  →  x_0 (clean)   [~1000 steps]
    - Diffusion bridge:    x_T = x_cloudy →  x_0 (clean)   [5–10 steps]

The bridge interpolation at any timestep t ∈ [0, 1]:
    z_t = (1 - t) * z_cloudy + t * z_clean + σ_t * ε,   ε ~ N(0, I)
where σ_t is a small perturbation schedule that peaks at t=0.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Optional


# ---------------------------------------------------------------------------
# Noise Schedule — Bridge perturbation kernel
# ---------------------------------------------------------------------------
class BridgeNoiseSchedule:
    """
    Symmetric noise schedule for the diffusion bridge.
    σ_t peaks at t=0.5 (maximum uncertainty) and is 0 at endpoints.

    σ_t = σ_max * sqrt(t * (1 - t))
    """
    def __init__(self, num_timesteps: int = 1000, sigma_max: float = 0.5):
        self.num_timesteps = num_timesteps
        self.sigma_max = sigma_max

        t = torch.linspace(0, 1, num_timesteps + 1)
        self.t_continuous = t
        self.sigma = sigma_max * torch.sqrt(t * (1 - t) + 1e-8)

    def get_sigma(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Get noise level for integer timestep indices."""
        return self.sigma[t_idx].to(t_idx.device)

    def get_t_continuous(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Convert integer timestep to continuous t ∈ [0, 1]."""
        return self.t_continuous[t_idx].to(t_idx.device)


# ---------------------------------------------------------------------------
# Bridge Forward Process — q(z_t | z_cloudy, z_clean)
# ---------------------------------------------------------------------------
def bridge_forward(
    z_clean: torch.Tensor,
    z_cloudy: torch.Tensor,
    t: torch.Tensor,
    sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a bridge intermediate state z_t given the two endpoints.

    Args:
        z_clean:  (B, C, H, W) — target clean optical latent (x_0)
        z_cloudy: (B, C, H, W) — source cloudy optical latent (x_T)
        t:        (B,) — continuous time in [0, 1]
        sigma:    (B,) — noise level at time t
    Returns:
        z_t:    (B, C, H, W) — noisy bridge sample at time t
        noise:  (B, C, H, W) — the injected noise ε
    """
    t = t.view(-1, 1, 1, 1)
    sigma = sigma.view(-1, 1, 1, 1)

    # Linear interpolation between cloudy (t=0) and clean (t=1)
    mean = (1 - t) * z_cloudy + t * z_clean
    noise = torch.randn_like(mean)
    z_t = mean + sigma * noise
    return z_t, noise


# ---------------------------------------------------------------------------
# OT-ODE Solver — Deterministic inference (5–10 steps)
# ---------------------------------------------------------------------------
class OTODESolver:
    """
    Optimal Transport ODE solver for inference.
    Integrates the learned velocity field from z_cloudy → z_clean
    using simple Euler steps (or Heun's method for higher accuracy).

    At each step, the network predicts the velocity v_θ(z_t, t) ≈ dz/dt.
    """
    def __init__(self, schedule: BridgeNoiseSchedule):
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        network: nn.Module,
        z_cloudy: torch.Tensor,
        sar_features: torch.Tensor,
        cloud_mask: torch.Tensor,
        num_steps: int = 10,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Run OT-ODE inference from z_cloudy (t=0) → z_clean (t=1).

        Args:
            network:      Velocity prediction network (VisionMambaBackbone)
            z_cloudy:     (B, C, H, W) — cloudy optical latent
            sar_features: (B, C_sar, H, W) — preprocessed SAR features
            cloud_mask:   (B, 1, H, W) — binary cloud mask
            num_steps:    Number of ODE integration steps (5–10)
            method:       "euler" or "heun" (Heun = 2nd-order Runge-Kutta)
        Returns:
            z_pred:       (B, C, H, W) — predicted clean latent
        """
        device = z_cloudy.device
        z = z_cloudy.clone()

        # Time steps: uniform from 0 → 1
        ts = torch.linspace(0, 1, num_steps + 1, device=device)

        for i in range(num_steps):
            t0 = ts[i]
            t1 = ts[i + 1]
            dt = t1 - t0

            t_batch = t0.expand(z.shape[0])
            t_idx = (t0 * self.schedule.num_timesteps).long().clamp(0, self.schedule.num_timesteps - 1)
            t_emb = self._get_time_emb(t_batch, network)

            if method == "euler":
                v = network(z, sar_features, t_emb, cloud_mask)
                z = z + dt * v

            elif method == "heun":
                # Heun's predictor-corrector
                v0 = network(z, sar_features, t_emb, cloud_mask)
                z_pred = z + dt * v0
                t1_batch = t1.expand(z.shape[0])
                t1_emb = self._get_time_emb(t1_batch, network)
                v1 = network(z_pred, sar_features, t1_emb, cloud_mask)
                z = z + dt * (v0 + v1) / 2.0

        return z

    def _get_time_emb(self, t_continuous: torch.Tensor, network: nn.Module) -> torch.Tensor:
        """Convert continuous t to sinusoidal embedding."""
        from models.vision_mamba import sinusoidal_timestep_embedding
        d_model = network.time_mlp[0].in_features
        t_idx = (t_continuous * 1000).long().clamp(0, 999)
        return sinusoidal_timestep_embedding(t_idx, d_model)


# ---------------------------------------------------------------------------
# Bridge Training Loss — velocity matching objective
# ---------------------------------------------------------------------------
class BridgeVelocityLoss(nn.Module):
    """
    The training objective for the diffusion bridge.
    The network learns to predict the velocity field:
        v_target = z_clean - z_cloudy   (direction from cloudy → clean)

    Combined with the cloud-aware adaptive loss mask.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        v_pred: torch.Tensor,       # (B, C, H, W) — network prediction
        z_clean: torch.Tensor,       # (B, C, H, W) — clean target
        z_cloudy: torch.Tensor,      # (B, C, H, W) — cloudy source
        loss_weight_mask: Optional[torch.Tensor] = None,  # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        MSE between predicted velocity and target velocity,
        optionally weighted by cloud-aware spatial mask.
        """
        v_target = z_clean - z_cloudy                      # true velocity direction

        if loss_weight_mask is not None:
            loss = ((v_pred - v_target) ** 2) * loss_weight_mask
        else:
            loss = (v_pred - v_target) ** 2

        return loss.mean()
