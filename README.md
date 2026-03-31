# Adaptive Mamba-Bridge (AMB)

**Structure-Aware Diffusion with State Space Models for SAR-Guided Optical Cloud Reconstruction**

Reconstructing cloud-occluded regions in optical satellite imagery (Sentinel-2) by fusing SAR data (Sentinel-1) through a novel multimodal architecture combining Diffusion Bridges, Vision Mamba, and SAR-aware pre-processing.

## The Problem

Optical Earth Observation (EO) data is frequently rendered unusable by cloud cover — up to 67% of Earth's surface is obscured at any given time. Traditional approaches rely on discriminative masking (leaving data gaps) or GANs, which suffer from mode collapse and hallucinate spectral signatures. While recent Latent Diffusion Models (LDMs) produce high-fidelity reconstructions, they face three severe bottlenecks:

1. **Computational Complexity:** Transformer-based diffusion scales quadratically O(N²), causing GPU memory exhaustion on high-resolution multi-spectral TIFFs.
2. **Sampling Inefficiency:** Standard diffusion destroys data into pure Gaussian noise, requiring hundreds of sampling steps to reverse.
3. **The "Pretty vs. Useful" Paradox:** Generative models prioritize perceptual aesthetics over the chemical and structural accuracy required for downstream scientific tasks.

## The AMB Architecture

The proposed model synthesizes three distinct, state-of-the-art methodologies into a single, novel pipeline.

### A. Generation Paradigm: Diffusion Bridge (DB-CR)

Instead of using standard SDEs that start from pure noise N(0, I), the model utilizes an **Optimal Transport ODE** that builds a direct, deterministic mathematical trajectory from the cloudy image distribution to the clear image distribution. This reduces inference from ~1,000 steps down to **5–10 steps**, making the model viable for rapid disaster-response mapping.

### B. Backbone: Vision Mamba (Vim)

Replaces the traditional U-Net or Vision Transformer (ViT) with **Bidirectional State Space Models (SSMs)**. The model processes flattened spatial tokens sequentially, operating at **linear time complexity O(N)** — completely eliminating the VRAM memory bottleneck and allowing the model to ingest massive, continent-scale satellite swaths without cropping them into contextless 256×256 patches.

### C. SAR Pre-Processing Stem: NAFBlock + DCNv2

SAR data inherently contains multiplicative speckle noise and geometric layover (leaning buildings due to side-looking radar physics). Before cross-attention fusion, the raw SAR tensor is passed through:

- **NAFBlock** (Nonlinear Activation-Free Network) — to despeckle the image
- **DCNv2** (Deformable Convolutions) — to geometrically align leaning radar signatures into a top-down optical projection

### D. Optimization: Cloud-Aware Adaptive Loss

A spatial weight mask (derived from coastal blue/cirrus bands) dynamically scales the loss function, focusing all computational energy strictly on the cloud-occluded pixels. The network mathematically ignores clear regions during training, guaranteeing perfect composite stitching and leaving clear terrain untouched.

### Architecture Overview

```
Inputs: Cloudy Optical (Sentinel-2) + SAR (Sentinel-1 VV/VH) + Cloud Mask
                                    │
              ┌─────────────────────┴────────────────────┐
              │                                          │
   ┌──────────▼──────────┐                   ┌──────────▼──────────┐
   │   Optical Encoder    │                   │  SAR Pre-Processing  │
   │   (pixel → latent)   │                   │  NAFBlock (despeckle)│
   └──────────┬──────────┘                   │  DCNv2 (geo-align)   │
              │                               └──────────┬──────────┘
              │                                          │
              └─────────────────────┬────────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │   Vision Mamba Vim   │ ◄── Cross-attention SAR fusion
                          │  Diffusion Bridge    │ ◄── Cloud mask M
                          │  ODE: cloudy→clean   │ ◄── Timestep embedding
                          │   (5–10 OT steps)    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Pretrained Decoder  │
                          │  (latent → pixel)    │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Hard Mask Composite │
                          │  M·output+(1-M)·in   │
                          └────────────────────┘
                                    │
                              Clean Optical
```

## Validation: The "Frozen Judge" Protocol

To prove the generated data is scientifically valid — not just a perceptual hallucination — the research uses a two-tiered evaluation strategy.

### Tier 1: Full-Reference Mathematical Metrics

| Metric | Purpose |
|--------|---------|
| **PSNR** | Measure absolute pixel error |
| **SSIM** | Ensure road networks and building edges align with ground truth |
| **SAM** | Prove multi-spectral chemical reflectance (vegetation vs. water) is mathematically accurate |

### Tier 2: Downstream Task Validation

A pre-trained Land Cover Classification network (Semantic Segmentation) is frozen (`requires_grad = False`). We measure **Mean Intersection over Union (mIoU)** of this classifier on:

- **True clear images** → upper bound
- **AMB-generated images** → our result

The goal: prove that generated pixels retain such high spectral fidelity that a downstream AI can correctly classify forests, urban areas, and water bodies with minimal accuracy drop compared to real sensor data.

## Novel Contributions

| Problem | Prior Work | AMB Solution |
|---------|-----------|-------------|
| Memory bottleneck at high resolution | SADER, SyMDit use O(N²) Transformers | Vision Mamba achieves O(N) — no resolution cap |
| SAR speckle & geometric distortion | Raw SAR injected directly into diffusion | NAF+DCN pre-processing stem translates radar physics before fusion |
| Background hallucination | Standard MSE loss blurs clear pixels | Cloud-Aware Adaptive Loss leaves clear terrain untouched |
| "Pretty picture" fallacy | FID-only evaluation | Downstream mIoU validation on frozen classifier |

## Data

### SEN12MS-CR Dataset

SEN12MS-CR is a large-scale, multi-modal, mono-temporal dataset specifically curated for cloud removal in Earth observation. It is the **first public dataset** for cloud removal to provide global and all-season coverage, containing paired and co-registered space-borne radar measurements (practically unaffected by clouds) alongside cloud-covered and cloud-free multi-spectral optical observations.

**Coverage & Scale**

- **175 globally distributed Regions of Interest** recorded across all four seasons throughout 2018
- **122,218 patch triplets** sliced from full-scene images, each patch of size 256×256
- **~48% average cloud coverage** across all samples — ranging from clear-view images to dense, wide cloud coverage, covering all real-world scenarios

**Each sample is a triplet of:**

| Modality | Satellite | Bands | Notes |
|----------|-----------|-------|-------|
| SAR (radar) | Sentinel-1 GRD | VV, VH (2 bands) | Practically unaffected by clouds |
| Cloudy Optical | Sentinel-2 | 13 multispectral bands | Cloud-contaminated observation |
| Clear Optical | Sentinel-2 | 13 multispectral bands | Ground truth target |

**Why SEN12MS-CR:**
All samples are patch-wise co-registered and fully compatible with the SEN12MS dataset, enabling semantic segmentation and scene classification using available land cover annotations — directly supporting AMB's downstream mIoU validation protocol.

**Cloud mask generation:** Derived from Sentinel-2 coastal blue/cirrus bands (B1/B10).

## Implementation Roadmap

1. Establish the PyTorch DataLoader using the **Sen12MS-CR dataset** to align SAR, Cloudy, and Clear optical triplets.
2. Code the **NAFBlock + DCNv2** SAR pre-processing stem.
3. Import and wrap the **Vision Mamba (Vim)** backbone inside the **Diffusion Bridge ODE** solver.

## References

- Liu et al., _I²SB: Image-to-Image Schrödinger Bridge_ (ICML 2023)
- Rombach et al., _High-Resolution Image Synthesis with Latent Diffusion Models_ (CVPR 2022)
- Gu & Dao, _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_ (2023)
- Zhu et al., _Vision Mamba: Efficient Visual Representation Learning with Bidirectional SSMs_ (2024)
- Ebel et al., _SEN12MS-CR-TS: A Multi-Temporal Cloud Removal Dataset_ (IEEE TGRS 2022)
- Chen et al., _Simple Baselines for Image Restoration (NAFNet)_ (ECCV 2022)
- _Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal_
- _SAR-DeCR: Latent Diffusion for SAR-Fused Thick Cloud Removal_

## License

TBD
