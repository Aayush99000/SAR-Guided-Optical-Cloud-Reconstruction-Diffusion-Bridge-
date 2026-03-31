"""
Adaptive Mamba-Bridge (AMB) — Full Model
-----------------------------------------
Integrates all four components into a single forward pass:

    1. SARPreProcessingStem    — NAFBlock despeckle + DCNv2 geo-align
    2. VisionMambaBackbone     — Bidirectional SSM velocity predictor
    3. DiffusionBridge         — OT-ODE: cloudy → clean trajectory
    4. Hard Mask Composite     — Preserve cloud-free pixels exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sar_stem import SARPreProcessingStem
from models.vision_mamba import VisionMambaBackbone, sinusoidal_timestep_embedding
from models.diffusion_bridge import BridgeNoiseSchedule, bridge_forward, OTODESolver, BridgeVelocityLoss
from losses.cloud_aware_loss import CloudAwareAdaptiveLoss


class AdaptiveMambaBridge(nn.Module):
    """
    Full AMB model.

    Training forward pass:
        - Samples a random bridge timestep t
        - Corrupts the clean latent into z_t via bridge_forward
        - Predicts velocity v_θ(z_t, SAR, t, mask)
        - Computes cloud-aware velocity loss

    Inference:
        - Runs OT-ODE solver for num_inference_steps (5–10)
        - Applies hard mask composite to preserve clear regions
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- Component 1: SAR Pre-Processing Stem ---
        self.sar_stem = SARPreProcessingStem(
            in_channels=config.sar_bands,
            hidden_dim=config.sar_hidden_dim,
            out_channels=config.mamba_dim,
            dcn_groups=config.dcn_groups,
        )

        # --- Component 2: Vision Mamba Backbone ---
        self.backbone = VisionMambaBackbone(
            in_channels=config.optical_bands,
            sar_channels=config.mamba_dim,
            d_model=config.mamba_dim,
            depth=config.mamba_depth,
            expand=config.mamba_expand,
        )

        # --- Component 3: Diffusion Bridge Schedule ---
        self.schedule = BridgeNoiseSchedule(
            num_timesteps=config.num_timesteps,
            sigma_max=0.5,
        )
        self.bridge_loss = BridgeVelocityLoss()

        # --- Component 4: Cloud-Aware Adaptive Loss ---
        self.cloud_loss = CloudAwareAdaptiveLoss(
            cloud_weight=config.cloud_loss_weight,
            clear_weight=config.clear_loss_weight,
        )

        # ODE Solver for inference
        self.ode_solver = OTODESolver(self.schedule)

    def forward(self, batch: dict) -> dict:
        """
        Training forward pass.

        Args:
            batch: dict with keys:
                "sar":        (B, 2, H, W)
                "cloudy":     (B, 13, H, W)
                "clear":      (B, 13, H, W)
                "cloud_mask": (B, 1, H, W)
        Returns:
            dict with "loss", "loss_velocity", "loss_pixel"
        """
        device = batch["sar"].device
        B = batch["sar"].shape[0]

        # === Step 1: SAR Pre-Processing ===
        sar_features = self.sar_stem(batch["sar"])          # (B, mamba_dim, H, W)

        # === Step 2: Sample random bridge timestep ===
        t_idx = torch.randint(1, self.config.num_timesteps, (B,), device=device)
        t_cont = self.schedule.get_t_continuous(t_idx)      # (B,) in [0, 1]
        sigma = self.schedule.get_sigma(t_idx)              # (B,)

        # === Step 3: Bridge forward — corrupt clean into z_t ===
        z_t, noise = bridge_forward(
            z_clean=batch["clear"],
            z_cloudy=batch["cloudy"],
            t=t_cont,
            sigma=sigma,
        )

        # === Step 4: Predict velocity via Vision Mamba ===
        t_emb = sinusoidal_timestep_embedding(t_idx, self.config.mamba_dim)
        v_pred = self.backbone(z_t, sar_features, t_emb, batch["cloud_mask"])

        # === Step 5: Compute cloud-aware loss mask ===
        weight_mask = self.cloud_loss.build_weight_mask(batch["cloud_mask"])

        # === Step 6: Velocity loss (bridge objective) ===
        loss_velocity = self.bridge_loss(
            v_pred=v_pred,
            z_clean=batch["clear"],
            z_cloudy=batch["cloudy"],
            loss_weight_mask=weight_mask,
        )

        # === Step 7: Pixel reconstruction loss on inferred clean ===
        # Approximate clean from v_pred using Euler step from t → 1
        dt = (1.0 - t_cont).view(-1, 1, 1, 1)
        z_clean_pred = z_t + dt * v_pred
        loss_pixel = self.cloud_loss(z_clean_pred, batch["clear"], batch["cloud_mask"])

        total_loss = loss_velocity + 0.5 * loss_pixel

        return {
            "loss": total_loss,
            "loss_velocity": loss_velocity.detach(),
            "loss_pixel": loss_pixel.detach(),
        }

    @torch.no_grad()
    def reconstruct(
        self,
        sar: torch.Tensor,
        cloudy: torch.Tensor,
        cloud_mask: torch.Tensor,
        num_steps: int = 10,
        method: str = "heun",
    ) -> torch.Tensor:
        """
        Full inference: cloudy optical → reconstructed clear optical.

        Args:
            sar:        (B, 2, H, W)
            cloudy:     (B, 13, H, W)
            cloud_mask: (B, 1, H, W)  — 1=cloud, 0=clear
            num_steps:  ODE integration steps (5–10)
            method:     "euler" or "heun"
        Returns:
            output: (B, 13, H, W) — reconstructed clear image
        """
        # SAR pre-processing
        sar_features = self.sar_stem(sar)

        # OT-ODE: run bridge from cloudy → clean
        z_pred = self.ode_solver.sample(
            network=self.backbone,
            z_cloudy=cloudy,
            sar_features=sar_features,
            cloud_mask=cloud_mask,
            num_steps=num_steps,
            method=method,
        )

        # Hard mask composite: keep clear pixels exactly from input
        # output = mask * generated + (1 - mask) * original_cloudy
        output = cloud_mask * z_pred + (1.0 - cloud_mask) * cloudy
        return output
