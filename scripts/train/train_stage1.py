"""
Stage 1: Visual-Language Alignment via NT-Xent Contrastive Learning.

Trains a dual-encoder with learnable temperature on IU X-Ray multi-view images
paired with radiology report text.  Uses cosine LR schedule with linear warmup
and gradient accumulation to simulate a large effective batch size.

Usage
-----
    python scripts/train/train_stage1.py           # fresh start
    python scripts/train/train_stage1.py --resume  # continue from latest checkpoint
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models import CrossModalAlignment, MultiViewBackbone, ProjectionHead
from utils.losses import NTXentLoss

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/stage1/resume.pt")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    DEVICE       = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS   = Config.NUM_EPOCHS
    BATCH_SIZE   = Config.BATCH_SIZE
    ACCUM_STEPS  = 4
    LR           = Config.LR
    WARMUP_RATIO = 0.05
    GRAD_CLIP    = 1.0
    USE_AMP      = DEVICE.type == "cuda"

    CKPT_DIR    = os.path.join(_ROOT, "checkpoints", "stage1")
    BEST_CKPT   = os.path.join(CKPT_DIR, "best.pth")
    LATEST_CKPT = os.path.join(CKPT_DIR, "latest.pth")
    RESUME_FILE = os.path.join(CKPT_DIR, "resume.pt")
    MIMIC_INIT  = os.path.join(CKPT_DIR, "mimic_pretrain.pth")
    os.makedirs(CKPT_DIR, exist_ok=True)

    log.info(f"Device : {DEVICE}  |  AMP: {USE_AMP}")
    log.info(f"Effective batch: {BATCH_SIZE} x {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS}")

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = IUXrayMultiViewDataset(root_dir=Config.DATA_ROOT, split="train")
    val_dataset   = IUXrayMultiViewDataset(root_dir=Config.DATA_ROOT, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, pin_memory=USE_AMP, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, pin_memory=USE_AMP, num_workers=0,
    )
    log.info(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── Models ────────────────────────────────────────────────────────────
    visual_model = MultiViewBackbone().to(DEVICE)
    alignment    = CrossModalAlignment().to(DEVICE)
    proj_img     = ProjectionHead().to(DEVICE)
    proj_txt     = ProjectionHead().to(DEVICE)
    criterion    = NTXentLoss(init_temperature=0.07).to(DEVICE)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(visual_model.parameters())
        + list(alignment.parameters())
        + list(proj_img.parameters())
        + list(proj_txt.parameters())
        + list(criterion.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )

    # ── LR scheduler ──────────────────────────────────────────────────────
    total_steps  = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    scaler = GradScaler("cuda", enabled=USE_AMP)

    # ── Resume / init state ───────────────────────────────────────────────
    best_val_loss = float("inf")
    start_epoch   = 0
    patience      = 8
    no_improve    = 0

    if not args.resume and os.path.exists(MIMIC_INIT):
        ckpt = torch.load(MIMIC_INIT, map_location=DEVICE, weights_only=False)
        visual_model.load_state_dict(ckpt["visual_model"], strict=False)
        if "alignment" in ckpt:
            alignment.load_state_dict(ckpt["alignment"], strict=False)
        if "proj_img" in ckpt:
            proj_img.load_state_dict(ckpt["proj_img"])
        if "proj_txt" in ckpt:
            proj_txt.load_state_dict(ckpt["proj_txt"])
        if "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])
        log.info(f"Loaded CXR pre-trained weights from {os.path.relpath(MIMIC_INIT, _ROOT)}")

    if args.resume and os.path.exists(RESUME_FILE):
        ckpt = torch.load(RESUME_FILE, map_location=DEVICE, weights_only=False)
        visual_model.load_state_dict(ckpt["visual_model"], strict=False)
        alignment.load_state_dict(ckpt["alignment"], strict=False)
        proj_img.load_state_dict(ckpt["proj_img"])
        proj_txt.load_state_dict(ckpt["proj_txt"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        no_improve    = ckpt.get("no_improve", 0)
        log.info(f"Resumed epoch {start_epoch} | best_val_loss={best_val_loss:.6f}")
    elif args.resume:
        log.warning("--resume requested but no resume.pt found — starting fresh.")

    log.info(f"\nStarting Stage-1 | epochs={NUM_EPOCHS} | total steps={total_steps}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    all_params = (
        list(visual_model.parameters()) + list(alignment.parameters())
        + list(proj_img.parameters()) + list(proj_txt.parameters())
        + list(criterion.parameters())
    )

    for epoch in range(start_epoch, NUM_EPOCHS):

        # ── Train ─────────────────────────────────────────────────────────
        visual_model.train(); alignment.train()
        proj_img.train(); proj_txt.train(); criterion.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [train]")
        for step, (images, reports, *_) in enumerate(loop):
            images = images.to(DEVICE)

            with autocast(device_type="cuda", enabled=USE_AMP, dtype=torch.bfloat16):
                visual_features   = visual_model(images)
                aligned, text_cls, _ = alignment(visual_features, reports)
                img_emb = proj_img(aligned.mean(dim=1))
                txt_emb = proj_txt(text_cls)
                loss    = criterion(img_emb, txt_emb) / ACCUM_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

            loop.set_postfix(
                loss=f"{epoch_loss / (step + 1):.4f}",
                temp=f"{criterion.temperature.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        if (len(train_loader) % ACCUM_STEPS) != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad()

        avg_train = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        visual_model.eval(); alignment.eval()
        proj_img.eval(); proj_txt.eval(); criterion.eval()

        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_reports in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [val]", leave=False
            ):
                val_images = val_images.to(DEVICE)
                with autocast(device_type="cuda", enabled=USE_AMP, dtype=torch.bfloat16):
                    vf = visual_model(val_images)
                    va, vcls, _ = alignment(vf, val_reports)
                    vi = proj_img(va.mean(dim=1))
                    vt = proj_txt(vcls)
                    val_loss += criterion(vi, vt).item()

        avg_val = val_loss / max(len(val_loader), 1)
        torch.cuda.empty_cache()

        log.info(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
            f"train={avg_train:.6f} | val={avg_val:.6f} | "
            f"temp={criterion.temperature.item():.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        weights = {
            "visual_model": visual_model.state_dict(),
            "alignment":    alignment.state_dict(),
            "proj_img":     proj_img.state_dict(),
            "proj_txt":     proj_txt.state_dict(),
        }
        torch.save(weights, LATEST_CKPT)
        torch.save({
            **weights,
            "criterion":     criterion.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "scaler":        scaler.state_dict(),
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            "no_improve":    no_improve,
        }, RESUME_FILE)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save({**weights, "epoch": epoch, "val_loss": avg_val}, BEST_CKPT)
            log.info(f"  Best checkpoint saved (val_loss={best_val_loss:.6f})")
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                log.info(f"Early stopping after epoch {epoch + 1}.")
                break

    log.info(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    log.info(f"Best checkpoint -> {os.path.relpath(BEST_CKPT, _ROOT)}")


if __name__ == "__main__":
    main()
