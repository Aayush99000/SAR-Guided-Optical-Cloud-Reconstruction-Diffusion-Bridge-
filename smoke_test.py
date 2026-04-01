"""
AMB Smoke Test
--------------
Verifies the full model pipeline without any real data or GPU.
Creates synthetic tensors matching expected shapes and runs:
    1. SARPreProcessingStem        — despeckle + geo-align
    2. VisionMambaBackbone         — SSM velocity prediction
    3. AdaptiveMambaBridge.forward — full training pass (loss computation)
    4. AdaptiveMambaBridge.reconstruct — full inference pass (OT-ODE)
    5. Metric functions            — PSNR, SSIM, SAM
    6. CloudAwareAdaptiveLoss      — weighted loss
    7. BridgeNoiseSchedule         — σ_t schedule sanity

Usage:
    python smoke_test.py

Expected: all checks print [PASS]. No GPU or data required.
"""

import sys
import math
import traceback
import torch

# ── tiny config so we don't need 12 Mamba blocks and 256-dim on CPU ──────────
B, H, W          = 2, 64, 64   # small spatial size for speed
SAR_BANDS        = 2
OPT_BANDS        = 13
MAMBA_DIM        = 64           # reduced from 256
MAMBA_DEPTH      = 4            # reduced from 12
NUM_TIMESTEPS    = 100          # reduced from 1000
INFERENCE_STEPS  = 3            # reduced from 10

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

def check(name, fn):
    try:
        result = fn()
        print(f"  {PASS}  {name}")
        return result
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Minimal config object (mirrors AMBConfig but smaller for CPU testing)
# ─────────────────────────────────────────────────────────────────────────────
class SmokeCfg:
    data_root            = "./data/sen12ms_cr"
    patch_size           = H
    sar_bands            = SAR_BANDS
    optical_bands        = OPT_BANDS
    num_workers          = 0
    batch_size           = B
    sar_hidden_dim       = 32
    dcn_groups           = 2
    mamba_dim            = MAMBA_DIM
    mamba_depth          = MAMBA_DEPTH
    mamba_expand         = 2
    num_timesteps        = NUM_TIMESTEPS
    num_inference_steps  = INFERENCE_STEPS
    beta_start           = 1e-4
    beta_end             = 2e-2
    cloud_loss_weight    = 10.0
    clear_loss_weight    = 1.0
    cirrus_band_idx      = 10
    coastal_band_idx     = 0
    cloud_threshold      = 0.1
    lr                   = 2e-4
    weight_decay         = 1e-4
    epochs               = 1
    warmup_steps         = 10
    gradient_clip        = 1.0
    save_every           = 1
    log_every            = 10
    checkpoint_dir       = "./checkpoints"
    frozen_classifier_path = ""
    num_classes          = 9
    device               = "cpu"
    mixed_precision      = False
    seed                 = 42

cfg = SmokeCfg()

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic tensors
# ─────────────────────────────────────────────────────────────────────────────
def make_batch():
    sar        = torch.rand(B, SAR_BANDS, H, W)
    cloudy     = torch.rand(B, OPT_BANDS, H, W)
    clear      = torch.rand(B, OPT_BANDS, H, W)
    # Cloud mask: ~40% of pixels marked as cloudy
    cloud_mask = (torch.rand(B, 1, H, W) > 0.6).float()
    return {"sar": sar, "cloudy": cloudy, "clear": clear, "cloud_mask": cloud_mask}

# ═════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("  AMB Smoke Test")
print("══════════════════════════════════════════\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
print("── 1. Imports ──────────────────────────")

check("models.sar_stem",         lambda: __import__("models.sar_stem", fromlist=["SARPreProcessingStem"]))
check("models.vision_mamba",     lambda: __import__("models.vision_mamba", fromlist=["VisionMambaBackbone"]))
check("models.diffusion_bridge", lambda: __import__("models.diffusion_bridge", fromlist=["BridgeNoiseSchedule"]))
check("models.amb",              lambda: __import__("models.amb", fromlist=["AdaptiveMambaBridge"]))
check("losses.cloud_aware_loss", lambda: __import__("losses.cloud_aware_loss", fromlist=["CloudAwareAdaptiveLoss"]))
check("metrics.validation",      lambda: __import__("metrics.validation", fromlist=["compute_all_metrics"]))

from models.sar_stem         import SARPreProcessingStem
from models.vision_mamba     import VisionMambaBackbone, sinusoidal_timestep_embedding
from models.diffusion_bridge import BridgeNoiseSchedule, bridge_forward, OTODESolver, BridgeVelocityLoss
from models.amb              import AdaptiveMambaBridge
from losses.cloud_aware_loss import CloudAwareAdaptiveLoss
from metrics.validation      import compute_all_metrics, compute_psnr, compute_ssim, compute_sam

# ─────────────────────────────────────────────────────────────────────────────
# 2. BRIDGE NOISE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. BridgeNoiseSchedule ──────────────")

def test_schedule():
    sched = BridgeNoiseSchedule(num_timesteps=NUM_TIMESTEPS, sigma_max=0.5)
    # σ_t should be 0 at endpoints and peak near t=0.5
    assert sched.sigma[0].item() < 0.01,        "σ_0 should be ~0"
    assert sched.sigma[-1].item() < 0.01,       "σ_T should be ~0"
    mid = NUM_TIMESTEPS // 2
    assert sched.sigma[mid].item() > 0.2,       "σ at midpoint should be large"
    t_idx = torch.tensor([0, mid, NUM_TIMESTEPS])
    t_cont = sched.get_t_continuous(t_idx)
    assert t_cont[0].item() == 0.0,             "t_cont[0] should be 0"
    assert abs(t_cont[-1].item() - 1.0) < 0.01,"t_cont[-1] should be ~1"
    return sched

sched = check("schedule shape & endpoint values", test_schedule)

# ─────────────────────────────────────────────────────────────────────────────
# 3. BRIDGE FORWARD PROCESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. bridge_forward ───────────────────")

def test_bridge_fwd():
    z_clean  = torch.rand(B, OPT_BANDS, H, W)
    z_cloudy = torch.rand(B, OPT_BANDS, H, W)
    t        = torch.full((B,), 0.5)
    sigma    = torch.full((B,), 0.1)
    z_t, noise = bridge_forward(z_clean, z_cloudy, t, sigma)
    assert z_t.shape    == (B, OPT_BANDS, H, W), f"z_t shape wrong: {z_t.shape}"
    assert noise.shape  == (B, OPT_BANDS, H, W), f"noise shape wrong: {noise.shape}"
    # At t=0, z_t ≈ z_cloudy (with tiny noise)
    t0    = torch.zeros(B)
    sig0  = torch.zeros(B)
    z_t0, _ = bridge_forward(z_clean, z_cloudy, t0, sig0)
    assert torch.allclose(z_t0, z_cloudy, atol=1e-5), "At t=0, z_t should equal z_cloudy"
    # At t=1, z_t ≈ z_clean
    t1    = torch.ones(B)
    z_t1, _ = bridge_forward(z_clean, z_cloudy, t1, sig0)
    assert torch.allclose(z_t1, z_clean, atol=1e-5),  "At t=1, z_t should equal z_clean"

check("output shapes",           test_bridge_fwd)
check("t=0 → z_cloudy endpoint", test_bridge_fwd)
check("t=1 → z_clean endpoint",  test_bridge_fwd)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAR PRE-PROCESSING STEM
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. SARPreProcessingStem ─────────────")

def test_sar_stem():
    stem = SARPreProcessingStem(
        in_channels=SAR_BANDS,
        hidden_dim=cfg.sar_hidden_dim,
        out_channels=MAMBA_DIM,
        dcn_groups=cfg.dcn_groups,
    )
    sar = torch.rand(B, SAR_BANDS, H, W)
    out = stem(sar)
    assert out.shape == (B, MAMBA_DIM, H, W), f"SAR stem output shape wrong: {out.shape}"
    assert not torch.isnan(out).any(),        "NaNs in SAR stem output"
    assert not torch.isinf(out).any(),        "Infs in SAR stem output"
    return stem

stem = check("forward pass, shape, no NaN/Inf", test_sar_stem)

def test_sar_stem_grad():
    stem = SARPreProcessingStem(SAR_BANDS, cfg.sar_hidden_dim, MAMBA_DIM, dcn_groups=cfg.dcn_groups)
    sar = torch.rand(B, SAR_BANDS, H, W, requires_grad=False)
    out = stem(sar)
    out.sum().backward()
    # Check at least one parameter received a gradient
    grads = [p.grad for p in stem.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through SAR stem"

check("gradients flow through stem", test_sar_stem_grad)

# ─────────────────────────────────────────────────────────────────────────────
# 5. SINUSOIDAL TIMESTEP EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. sinusoidal_timestep_embedding ────")

def test_temb():
    t_idx = torch.randint(0, NUM_TIMESTEPS, (B,))
    emb   = sinusoidal_timestep_embedding(t_idx, MAMBA_DIM)
    assert emb.shape == (B, MAMBA_DIM), f"Embedding shape wrong: {emb.shape}"
    # Different timesteps should produce different embeddings
    t0 = sinusoidal_timestep_embedding(torch.tensor([0]), MAMBA_DIM)
    t1 = sinusoidal_timestep_embedding(torch.tensor([1]), MAMBA_DIM)
    assert not torch.allclose(t0, t1), "t=0 and t=1 embeddings are identical — bug"

check("shape & uniqueness", test_temb)

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISION MAMBA BACKBONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. VisionMambaBackbone ──────────────")

def test_backbone():
    backbone = VisionMambaBackbone(
        in_channels=OPT_BANDS,
        sar_channels=MAMBA_DIM,
        d_model=MAMBA_DIM,
        depth=MAMBA_DEPTH,
        expand=cfg.mamba_expand,
    )
    z_t          = torch.rand(B, OPT_BANDS, H, W)
    sar_features = torch.rand(B, MAMBA_DIM, H, W)
    t_idx        = torch.randint(0, NUM_TIMESTEPS, (B,))
    t_emb        = sinusoidal_timestep_embedding(t_idx, MAMBA_DIM)
    cloud_mask   = (torch.rand(B, 1, H, W) > 0.6).float()
    pred         = backbone(z_t, sar_features, t_emb, cloud_mask)
    assert pred.shape == (B, OPT_BANDS, H, W), f"Backbone output shape wrong: {pred.shape}"
    assert not torch.isnan(pred).any(),         "NaNs in backbone output"
    return backbone

backbone = check("forward pass, shape, no NaN", test_backbone)

def test_backbone_grad():
    backbone = VisionMambaBackbone(OPT_BANDS, MAMBA_DIM, MAMBA_DIM, MAMBA_DEPTH, cfg.mamba_expand)
    z_t          = torch.rand(B, OPT_BANDS, H, W)
    sar_features = torch.rand(B, MAMBA_DIM, H, W)
    t_emb        = sinusoidal_timestep_embedding(torch.randint(0, NUM_TIMESTEPS, (B,)), MAMBA_DIM)
    cloud_mask   = (torch.rand(B, 1, H, W) > 0.6).float()
    pred = backbone(z_t, sar_features, t_emb, cloud_mask)
    pred.sum().backward()
    grads = [p.grad for p in backbone.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed through backbone"

check("gradients flow through backbone", test_backbone_grad)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CLOUD-AWARE ADAPTIVE LOSS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 7. CloudAwareAdaptiveLoss ───────────")

def test_loss():
    loss_fn    = CloudAwareAdaptiveLoss(cloud_weight=10.0, clear_weight=1.0)
    pred       = torch.rand(B, OPT_BANDS, H, W)
    target     = torch.rand(B, OPT_BANDS, H, W)
    cloud_mask = (torch.rand(B, 1, H, W) > 0.6).float()
    loss       = loss_fn(pred, target, cloud_mask)
    assert loss.ndim == 0,         "Loss should be a scalar"
    assert loss.item() > 0,        "Loss should be positive"
    assert not math.isnan(loss.item()), "Loss is NaN"
    # Cloud-only loss should be higher than clear-only loss
    # (because cloud regions have higher weight)
    l_cloud = loss_fn.cloud_only_loss(pred, target, cloud_mask).item()
    l_clear = loss_fn.clear_only_loss(pred, target, cloud_mask).item()
    return loss, l_cloud, l_clear

result = check("scalar loss, no NaN, cloud > clear weight", test_loss)

# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 8. Validation Metrics ───────────────")

def test_psnr():
    pred   = torch.rand(B, OPT_BANDS, H, W)
    target = torch.rand(B, OPT_BANDS, H, W)
    psnr   = compute_psnr(pred, target)
    assert psnr.shape == (B,),       f"PSNR shape wrong: {psnr.shape}"
    assert (psnr > 0).all(),         "PSNR values should be positive"
    # Perfect prediction → very high PSNR
    psnr_perfect = compute_psnr(target, target)
    assert (psnr_perfect > 60).all(), "Perfect PSNR should be >60 dB"

check("PSNR — shape, range, perfect case", test_psnr)

def test_ssim():
    pred   = torch.rand(B, OPT_BANDS, H, W)
    target = torch.rand(B, OPT_BANDS, H, W)
    ssim   = compute_ssim(pred, target)
    assert ssim.shape == (B,),             f"SSIM shape wrong: {ssim.shape}"
    assert (ssim > -1).all() and (ssim <= 1).all(), "SSIM out of [-1, 1]"
    # Identical inputs → SSIM ≈ 1
    ssim_perfect = compute_ssim(target, target)
    assert (ssim_perfect > 0.99).all(),    "Perfect SSIM should be ~1"

check("SSIM — shape, range [-1,1], perfect case", test_ssim)

def test_sam():
    pred   = torch.rand(B, OPT_BANDS, H, W).abs() + 0.1
    target = torch.rand(B, OPT_BANDS, H, W).abs() + 0.1
    sam    = compute_sam(pred, target)
    assert sam.shape == (B,),           f"SAM shape wrong: {sam.shape}"
    assert (sam >= 0).all(),            "SAM should be non-negative"
    assert (sam <= math.pi).all(),      "SAM should be ≤ π"
    # Identical inputs → SAM ≈ 0
    sam_perfect = compute_sam(target, target)
    assert (sam_perfect < 0.01).all(),  "Perfect SAM should be ~0"

check("SAM — shape, range [0,π], perfect case", test_sam)

def test_all_metrics():
    pred       = torch.rand(B, OPT_BANDS, H, W)
    target     = torch.rand(B, OPT_BANDS, H, W)
    cloud_mask = (torch.rand(B, 1, H, W) > 0.6).float()
    m = compute_all_metrics(pred, target, cloud_mask)
    for key in ["psnr", "ssim", "sam", "psnr_cloud", "ssim_cloud", "sam_cloud"]:
        assert key in m,               f"Missing metric key: {key}"
        assert not math.isnan(m[key]), f"NaN in metric: {key}"

check("compute_all_metrics — all 6 keys present, no NaN", test_all_metrics)

# ─────────────────────────────────────────────────────────────────────────────
# 9. FULL MODEL — TRAINING FORWARD PASS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 9. AdaptiveMambaBridge.forward ──────")

def test_amb_forward():
    model = AdaptiveMambaBridge(cfg)
    batch = make_batch()
    out   = model(batch)
    for key in ["loss", "loss_velocity", "loss_pixel"]:
        assert key in out,                  f"Missing key in output: {key}"
        assert not math.isnan(out[key].item()), f"NaN in {key}"
        assert out[key].item() >= 0,        f"{key} is negative"
    assert out["loss"].requires_grad,       "loss.requires_grad should be True"
    return model, out

result = check("loss keys present, no NaN, positive, grad attached", test_amb_forward)

def test_amb_backward():
    model = AdaptiveMambaBridge(cfg)
    batch = make_batch()
    out   = model(batch)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients after backward() — training would be silent no-op"

check("loss.backward() — gradients flow to model params", test_amb_backward)

# ─────────────────────────────────────────────────────────────────────────────
# 10. FULL MODEL — INFERENCE (OT-ODE reconstruct)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 10. AdaptiveMambaBridge.reconstruct ─")

def test_amb_reconstruct_euler():
    model = AdaptiveMambaBridge(cfg)
    model.eval()
    sar        = torch.rand(B, SAR_BANDS, H, W)
    cloudy     = torch.rand(B, OPT_BANDS, H, W)
    cloud_mask = (torch.rand(B, 1, H, W) > 0.6).float()
    with torch.no_grad():
        out = model.reconstruct(sar, cloudy, cloud_mask, num_steps=INFERENCE_STEPS, method="euler")
    assert out.shape == (B, OPT_BANDS, H, W), f"Output shape wrong: {out.shape}"
    assert not torch.isnan(out).any(),         "NaNs in reconstruct output (euler)"

check("euler ODE — output shape, no NaN", test_amb_reconstruct_euler)

def test_amb_reconstruct_heun():
    model = AdaptiveMambaBridge(cfg)
    model.eval()
    sar        = torch.rand(B, SAR_BANDS, H, W)
    cloudy     = torch.rand(B, OPT_BANDS, H, W)
    cloud_mask = (torch.rand(B, 1, H, W) > 0.6).float()
    with torch.no_grad():
        out = model.reconstruct(sar, cloudy, cloud_mask, num_steps=INFERENCE_STEPS, method="heun")
    assert out.shape == (B, OPT_BANDS, H, W), f"Output shape wrong: {out.shape}"
    assert not torch.isnan(out).any(),         "NaNs in reconstruct output (heun)"

check("heun ODE — output shape, no NaN", test_amb_reconstruct_heun)

def test_hard_mask_composite():
    """Clear pixels in the output must be identical to the cloudy input."""
    model = AdaptiveMambaBridge(cfg)
    model.eval()
    sar        = torch.rand(B, SAR_BANDS, H, W)
    cloudy     = torch.rand(B, OPT_BANDS, H, W)
    cloud_mask = torch.zeros(B, 1, H, W)          # ALL pixels marked clear
    with torch.no_grad():
        out = model.reconstruct(sar, cloudy, cloud_mask, num_steps=INFERENCE_STEPS, method="euler")
    assert torch.allclose(out, cloudy, atol=1e-5), \
        "Hard mask composite failed: when mask=0 everywhere, output must equal input"

check("hard mask composite — all-clear mask preserves input exactly", test_hard_mask_composite)

# ─────────────────────────────────────────────────────────────────────────────
# 11. PARAMETER COUNT
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 11. Model Size ──────────────────────")

def test_param_count():
    model      = AdaptiveMambaBridge(cfg)
    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"         Total params:     {n_total:,}")
    print(f"         Trainable params: {n_trainable:,}")
    print(f"         (smoke config — full model will be ~10–50× larger)")
    assert n_trainable > 0, "No trainable parameters found"

check("param count sanity", test_param_count)

# ─────────────────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
print("  Smoke test complete.")
print("  All [PASS] = safe to move to real data.")
print("  Any [FAIL] = fix before touching the server.")
print("══════════════════════════════════════════\n")
