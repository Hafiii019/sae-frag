"""
Load TorchXRayVision chest X-ray pretrained weights into the project backbone.

TorchXRayVision ships ResNet-50 models trained on MIMIC-CXR, CheXpert, NIH
ChestX-ray14, PadChest and more — no credentialing required (~100 MB download).

The project uses ResNet-101.  ResNet-50 and ResNet-101 share identical
architecture for layer0-layer3.  Only layer4 differs (bottleneck count).
This script copies the shared layers and saves the result so train_stage1.py
auto-loads it on its first run.

Usage
-----
    pip install torchxrayvision
    python scripts/prepare/init_xrv_weights.py
    python scripts/train/train_stage1.py        # auto-loads the checkpoint

Available weight presets (--weights argument):
    resnet50-res512-all        trained on ALL available CXR datasets (default)
    resnet50-res512-mimic_nb   MIMIC-CXR only
    resnet50-res512-chexpert   CheXpert only
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import torch

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from models import MultiViewBackbone

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# XRV key prefix -> project MultiViewBackbone state-dict key prefix
_XRV_TO_PROJ = {
    "conv1.":  "backbone.layer0.0.",
    "bn1.":    "backbone.layer0.1.",
    "layer1.": "backbone.layer1.",
    "layer2.": "backbone.layer2.",
    "layer3.": "backbone.layer3.",
}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", default="resnet50-res512-all",
        help="TorchXRayVision weight preset.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_ROOT, "checkpoints", "stage1", "mimic_pretrain.pth"),
        help="Output path for the adapted checkpoint.",
    )
    args = parser.parse_args()

    try:
        import torchxrayvision as xrv
    except ImportError:
        log.error("torchxrayvision not installed. Run: pip install torchxrayvision")
        sys.exit(1)

    log.info(f"Loading TorchXRayVision model: {args.weights}  (downloads ~100 MB on first run)")
    xrv_model = xrv.models.ResNet(weights=args.weights)
    xrv_sd    = xrv_model.model.state_dict()

    backbone = MultiViewBackbone()
    proj_sd  = backbone.state_dict()

    copied  = 0
    skipped = 0

    for xrv_key, val in xrv_sd.items():
        proj_key = None
        for xrv_prefix, proj_prefix in _XRV_TO_PROJ.items():
            if xrv_key.startswith(xrv_prefix):
                proj_key = proj_prefix + xrv_key[len(xrv_prefix):]
                break

        if proj_key is None or proj_key not in proj_sd:
            skipped += 1
            continue

        if proj_sd[proj_key].shape != val.shape:
            log.info(f"  Shape mismatch {proj_key}: xrv={tuple(val.shape)} proj={tuple(proj_sd[proj_key].shape)}")
            skipped += 1
            continue

        proj_sd[proj_key] = val.clone()
        copied += 1

    backbone.load_state_dict(proj_sd, strict=False)

    log.info(f"\nCopied {copied} tensors from XRV -> project backbone (skipped {skipped})")
    log.info("CXR-initialised: conv1/bn1 (layer0), layer1, layer2, layer3")
    log.info("ImageNet/random : layer4, FPN, SAFE, view_attn")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"visual_model": backbone.state_dict()}, args.out)

    log.info(f"\nSaved -> {os.path.relpath(args.out, _ROOT)}")
    log.info("train_stage1.py will auto-load this checkpoint on its first run.")


if __name__ == "__main__":
    main()
