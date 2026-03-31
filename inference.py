"""
AMB Inference Script
---------------------
Runs a trained AMB checkpoint on a single sample or full test set.

Usage:
    # Single sample
    python inference.py --checkpoint checkpoints/amb_epoch_100.pt \
                        --sar path/to/sar.tif \
                        --cloudy path/to/cloudy.tif \
                        --output output/reconstructed.tif

    # Full test set evaluation
    python inference.py --checkpoint checkpoints/amb_epoch_100.pt \
                        --eval_all
"""

import os
import argparse
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds

from config import AMBConfig
from models.amb import AdaptiveMambaBridge
from data.dataset import Sen12MSCRDataset, build_dataloaders
from metrics.validation import compute_all_metrics, FrozenJudgeValidator


def load_model(checkpoint_path: str, config: AMBConfig, device: str) -> AdaptiveMambaBridge:
    model = AdaptiveMambaBridge(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Inference] Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def read_tif_as_tensor(path: str, device: str) -> torch.Tensor:
    """Read a .tif file and return a (1, C, H, W) tensor."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    tensor = torch.from_numpy(data).unsqueeze(0).to(device)
    return tensor


def save_tif(array: np.ndarray, output_path: str, reference_path: str = None):
    """Save a numpy array (C, H, W) as a GeoTIFF."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    C, H, W = array.shape

    if reference_path:
        with rasterio.open(reference_path) as ref:
            profile = ref.profile.copy()
        profile.update(count=C, dtype=rasterio.float32)
    else:
        profile = {
            "driver": "GTiff",
            "dtype": rasterio.float32,
            "count": C,
            "height": H,
            "width": W,
        }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(np.float32))
    print(f"[Inference] Saved → {output_path}")


@torch.no_grad()
def run_single(args, config: AMBConfig):
    """Reconstruct a single SAR + cloudy optical pair."""
    device = config.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, config, device)

    # Load inputs
    sar = read_tif_as_tensor(args.sar, device)                # (1, 2, H, W)
    cloudy = read_tif_as_tensor(args.cloudy, device)          # (1, 13, H, W)

    # Derive cloud mask from input
    dataset = Sen12MSCRDataset.__new__(Sen12MSCRDataset)
    cloud_mask_np = dataset._derive_cloud_mask(
        cloudy.squeeze(0).cpu().numpy()
    )
    cloud_mask = torch.from_numpy(cloud_mask_np).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Normalize
    sar_norm = torch.clamp((sar - (-25.0)) / (0.0 - (-25.0)), 0, 1)
    cloudy_norm = torch.clamp(cloudy / 10000.0, 0, 1)

    # Reconstruct
    output = model.reconstruct(
        sar=sar_norm,
        cloudy=cloudy_norm,
        cloud_mask=cloud_mask,
        num_steps=10,
        method="heun",
    )

    # Save output
    out_np = (output.squeeze(0).cpu().numpy() * 10000.0).astype(np.float32)
    save_tif(out_np, args.output, reference_path=args.cloudy)


@torch.no_grad()
def run_eval_all(args, config: AMBConfig):
    """Run full test set evaluation with Tier 1 + Tier 2 metrics."""
    device = config.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, config, device)

    _, val_loader = build_dataloaders(config)

    frozen_judge = FrozenJudgeValidator(
        model_path=config.frozen_classifier_path,
        num_classes=config.num_classes,
        device=device,
    )

    all_psnr, all_ssim, all_sam = [], [], []

    for i, batch in enumerate(val_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        pred = model.reconstruct(
            sar=batch["sar"],
            cloudy=batch["cloudy"],
            cloud_mask=batch["cloud_mask"],
            num_steps=10,
            method="heun",
        )

        metrics = compute_all_metrics(pred, batch["clear"], batch["cloud_mask"])
        all_psnr.append(metrics["psnr"])
        all_ssim.append(metrics["ssim"])
        all_sam.append(metrics["sam"])

        if i % 20 == 0:
            print(f"  Batch {i+1}/{len(val_loader)} | "
                  f"PSNR: {metrics['psnr']:.2f} | "
                  f"SSIM: {metrics['ssim']:.4f} | "
                  f"SAM: {metrics['sam']:.4f}")

    print("\n=== Tier 1 Results (Full Test Set) ===")
    print(f"  PSNR:  {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"  SSIM:  {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print(f"  SAM:   {np.mean(all_sam):.4f} ± {np.std(all_sam):.4f} rad")


def main():
    parser = argparse.ArgumentParser(description="AMB Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate on full test set")
    parser.add_argument("--sar", type=str, help="Path to SAR .tif (single sample)")
    parser.add_argument("--cloudy", type=str, help="Path to cloudy optical .tif (single sample)")
    parser.add_argument("--output", type=str, default="output/reconstructed.tif")
    args = parser.parse_args()

    config = AMBConfig()

    if args.eval_all:
        run_eval_all(args, config)
    else:
        if not args.sar or not args.cloudy:
            parser.error("--sar and --cloudy are required for single sample inference")
        run_single(args, config)


if __name__ == "__main__":
    main()
