# SAR-Guided Optical Cloud Reconstruction via Latent Diffusion Bridge

Reconstructing cloud-occluded regions in optical satellite imagery (Sentinel-2) by leveraging SAR data (Sentinel-1) as structural guidance through a diffusion bridge framework operating in latent space.

## The Problem

Optical satellite imagery is essential for agriculture monitoring, disaster response, and environmental science — but up to 67% of the Earth's surface is cloud-covered at any given time. Existing approaches to cloud removal either produce blurry outputs (deterministic regression) or hallucinate unrealistic content (standard generative models).

## Why Existing Solutions Fall Short

**Hallucination in generative models.** Standard diffusion models generate from pure noise, giving the network freedom to invent plausible-looking but factually incorrect scene content — phantom roads, wrong crop types, fabricated structures.

**Compute bottleneck on high-resolution imagery.** Satellite data comes as multi-band GeoTIFFs at resolutions far beyond the 256×256 patches most diffusion models are designed for. Running reverse diffusion at full resolution is prohibitively expensive.

**Temporal misalignment between sensors.** SAR and optical acquisitions rarely coincide. The radar pass may occur days before or after the optical capture, meaning land cover can change between observations.

## Our Approach

We use a **latent diffusion bridge** (based on I²SB) that learns a direct transport between cloudy and clean optical distributions, constrained by SAR structural guidance at every step.

### Architecture Overview

```
Inputs: Cloudy Optical (Sentinel-2) + SAR (Sentinel-1 VV/VH) + Cloud Mask
                                    │
                          ┌─────────▼──────────┐
                          │  Pretrained Encoder  │
                          │  (pixel → latent)    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  I²SB Latent Bridge  │ ◄── SAR features (cross-attention)
                          │  z_cloudy → z_clean  │ ◄── Cloud mask M
                          │  (5–20 bridge steps) │ ◄── Timestep embedding
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Pretrained Decoder  │
                          │  (latent → pixel)    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Hard mask composite │
                          │  M·output + (1-M)·in │
                          └────────────────────┘
                                    │
                              Clean Optical
```

### How We Address Each Bottleneck

**Anti-hallucination.** The diffusion bridge never passes through pure noise — intermediate states are always a blend of the clean and cloudy endpoints, preserving scene structure. A hard mask composite ensures cloud-free pixels pass through untouched.

**Compute efficiency.** All diffusion steps operate in a compressed latent space (4× spatial downsample via a pretrained autoencoder). Combined with only 5–20 bridge steps (vs. 50–1000 in standard DDPM), this makes high-resolution inference tractable.

**Temporal robustness.** SAR features are injected via cross-attention rather than pixel-level concatenation, allowing the network to learn flexible spatial correspondences rather than assuming exact pixel alignment between sensor passes.

### Network Backbone

The bridge network is a UNet with:

- **ResBlocks + self-attention** at each resolution level
- **Cross-attention layers** where SAR features serve as keys/values
- **Channel concatenation** of downsampled SAR at early layers for coarse alignment
- **Sinusoidal timestep embeddings** added to each ResBlock

## Data

- **Optical:** Sentinel-2 L2A (B2, B3, B4, B8 — RGB + NIR)
- **SAR:** Sentinel-1 GRD (VV/VH polarizations)
- **Cloud masks:** Generated via a pretrained UNet segmentation model
- **Candidate datasets:** SEN12MS-CR, SEN12MS-CR-TS (multi-temporal)

## Project Status

🚧 **Work in progress**

- [x] Problem formulation and architecture design
- [ ] Data pipeline (Sentinel-1/2 pair acquisition and preprocessing)
- [ ] Autoencoder training / adaptation for satellite bands
- [ ] I²SB bridge network implementation
- [ ] SAR cross-attention conditioning module
- [ ] Training loop and evaluation metrics (PSNR, SSIM, SAM, LPIPS)
- [ ] Patch-based inference with tile blending for full-scene reconstruction
- [ ] Ablation studies and benchmarking

## References

- Liu et al., _I²SB: Image-to-Image Schrödinger Bridge_ (ICML 2023)
- Rombach et al., _High-Resolution Image Synthesis with Latent Diffusion Models_ (CVPR 2022)
- Ebel et al., _SEN12MS-CR-TS: A Multi-Temporal Cloud Removal Dataset_ (IEEE TGRS 2022)

## License

TBD
