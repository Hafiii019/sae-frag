"""
Stage 2b: Train SAEImageClassifier using soft pseudo-labels from ReportClassifier.

The Stage-1 backbone is used as initialisation.  Early backbone layers are frozen
to preserve the contrastive features learned in Stage 1.  Only layer4, FPN, SAFE,
and the classification head are updated.

Prerequisites
-------------
    python scripts/train/train_stage1.py
    python scripts/train/train_report_classifier.py

Usage
-----
    python scripts/train/train_stage2.py
    python scripts/train/train_stage2.py --resume
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/stage2/resume.pt")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP    = DEVICE.type == "cuda"
    NUM_EPOCHS = 25
    BATCH_SIZE = 16
    LR         = 3e-4
    PATIENCE   = 5
    GRAD_CLIP  = 1.0

    CKPT_DIR    = os.path.join(_ROOT, "checkpoints", "stage2")
    BEST_CKPT   = os.path.join(CKPT_DIR, "image_classifier.pth")
    RESUME_FILE = os.path.join(CKPT_DIR, "resume.pt")
    S1_CKPT     = os.path.join(_ROOT, "checkpoints", "stage1", "best.pth")
    os.makedirs(CKPT_DIR, exist_ok=True)

    log.info(f"Device: {DEVICE}  |  AMP: {USE_AMP}")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = IUXrayMultiViewDataset(root_dir=Config.DATA_ROOT, split="train")
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, pin_memory=USE_AMP, num_workers=0,
    )
    log.info(f"Train samples: {len(dataset)}")

    # ── Report classifier (frozen teacher) ────────────────────────────────
    report_classifier = ReportClassifier().to(DEVICE)
    report_classifier.load_state_dict(
        torch.load(
            os.path.join(CKPT_DIR, "report_classifier.pth"),
            map_location=DEVICE, weights_only=False,
        )
    )
    report_classifier.eval()

    # ── Image classifier ──────────────────────────────────────────────────
    image_classifier = SAEImageClassifier().to(DEVICE)

    if os.path.exists(S1_CKPT):
        s1 = torch.load(S1_CKPT, map_location=DEVICE, weights_only=False)
        image_classifier.backbone.load_state_dict(s1["visual_model"], strict=False)
        log.info(f"Loaded Stage-1 backbone from {os.path.relpath(S1_CKPT, _ROOT)}")
    else:
        log.warning(f"Stage-1 checkpoint not found — training from ImageNet init.")

    # Freeze stem + ResNet stages 0-3
    frozen = trainable = 0
    for name, param in image_classifier.backbone.named_parameters():
        if any(f"layer{i}" in name for i in range(4)):
            param.requires_grad = False
            frozen += param.numel()
        else:
            trainable += param.numel()
    log.info(f"Backbone: frozen={frozen/1e6:.1f}M | trainable={trainable/1e6:.1f}M")

    # ── Loss / optimiser / scheduler ──────────────────────────────────────
    pos_weight = torch.ones(14, device=DEVICE) * 5.0
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler     = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, image_classifier.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── Resume ────────────────────────────────────────────────────────────
    best_loss   = float("inf")
    start_epoch = 0
    no_improve  = 0

    if args.resume and os.path.exists(RESUME_FILE):
        state = torch.load(RESUME_FILE, map_location=DEVICE, weights_only=False)
        image_classifier.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        no_improve  = state.get("no_improve", 0)
        best_loss   = state.get("best_loss", float("inf"))
        log.info(f"Resumed from epoch {start_epoch} | no_improve={no_improve}")
    elif args.resume:
        log.warning("--resume requested but no resume.pt found — starting fresh.")

    log.info(f"\nStarting Stage-2 image classifier training\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        image_classifier.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images, reports in loop:
            images = images.to(DEVICE)

            with torch.no_grad():
                labels = torch.sigmoid(report_classifier(reports)).to(DEVICE)

            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP, dtype=torch.bfloat16):
                logits = image_classifier(images)
                loss   = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, image_classifier.parameters()),
                GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        scheduler.step()

        log.info(
            f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
            f"avg_loss={avg_loss:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        torch.save({
            "epoch":      epoch,
            "model":      image_classifier.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_loss":  best_loss,
            "no_improve": no_improve,
        }, RESUME_FILE)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save(image_classifier.state_dict(), BEST_CKPT)
            log.info(f"  Best checkpoint saved (loss={best_loss:.4f})")
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                log.info(f"Early stopping after epoch {epoch + 1}.")
                break

    log.info(f"\nTraining complete. Best checkpoint -> {os.path.relpath(BEST_CKPT, _ROOT)}")


if __name__ == "__main__":
    main()
