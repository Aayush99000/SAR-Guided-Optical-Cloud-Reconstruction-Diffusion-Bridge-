"""
AMB Training Script
--------------------
Main training loop for the Adaptive Mamba-Bridge model.

Usage:
    python train.py

Checkpoints are saved to config.checkpoint_dir every config.save_every epochs.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import AMBConfig
from data.dataset import build_dataloaders
from models.amb import AdaptiveMambaBridge
from metrics.validation import compute_all_metrics, FrozenJudgeValidator


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(model: nn.Module, optimizer, path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]


@torch.no_grad()
def validate(model: nn.Module, loader, device: str, frozen_judge: FrozenJudgeValidator = None):
    model.eval()
    all_psnr, all_ssim, all_sam = [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        pred = model.reconstruct(
            sar=batch["sar"],
            cloudy=batch["cloudy"],
            cloud_mask=batch["cloud_mask"],
            num_steps=10,
            method="heun",
        )
        m = compute_all_metrics(pred, batch["clear"], batch["cloud_mask"])
        all_psnr.append(m["psnr"])
        all_ssim.append(m["ssim"])
        all_sam.append(m["sam"])

    results = {
        "val/psnr": np.mean(all_psnr),
        "val/ssim": np.mean(all_ssim),
        "val/sam":  np.mean(all_sam),
    }
    model.train()
    return results


def train(config: AMBConfig):
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[AMB] Training on device: {device}")

    # --- Data ---
    train_loader, val_loader = build_dataloaders(config)
    print(f"[AMB] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model = AdaptiveMambaBridge(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[AMB] Model parameters: {n_params:,}")

    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Warmup + cosine decay schedule
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.epochs)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[config.warmup_steps])

    # Mixed precision
    scaler = GradScaler(enabled=config.mixed_precision)

    # Frozen Judge for Tier 2 validation
    frozen_judge = FrozenJudgeValidator(config.frozen_classifier_path, config.num_classes, device=str(device))

    # --- Training Loop ---
    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_v_loss = 0.0
        epoch_p_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.mixed_precision):
                outputs = model(batch)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_v_loss += outputs["loss_velocity"].item()
            epoch_p_loss += outputs["loss_pixel"].item()
            global_step += 1

            if global_step % config.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:03d} | Step {global_step:06d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"V-Loss: {outputs['loss_velocity'].item():.4f} | "
                    f"P-Loss: {outputs['loss_pixel'].item():.4f} | "
                    f"LR: {lr:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n[Epoch {epoch:03d}] Avg Loss: {avg_loss:.4f} | "
              f"V: {epoch_v_loss/len(train_loader):.4f} | "
              f"P: {epoch_p_loss/len(train_loader):.4f}")

        # --- Validation ---
        val_metrics = validate(model, val_loader, str(device))
        print(f"  [Val] PSNR: {val_metrics['val/psnr']:.2f} dB | "
              f"SSIM: {val_metrics['val/ssim']:.4f} | "
              f"SAM: {val_metrics['val/sam']:.4f} rad")

        # --- Checkpoint ---
        if epoch % config.save_every == 0 or epoch == config.epochs:
            ckpt_path = os.path.join(config.checkpoint_dir, f"amb_epoch_{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    print("\n[AMB] Training complete.")


if __name__ == "__main__":
    config = AMBConfig()
    train(config)
