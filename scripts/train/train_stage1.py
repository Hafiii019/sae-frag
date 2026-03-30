"""Stage 1: Visual-Language Alignment via NT-Xent Contrastive Learning.

Key improvements over baseline
--------------------------------
* AMP (bfloat16) — ~2× faster, no NaN risk on modern GPUs
* NTXentLoss with **learnable temperature** (CLIP-style log-scale param)
* Cosine LR schedule with linear warmup (transformers API)
* Gradient clipping (max_norm=1.0)
* Validation split monitoring — best checkpoint saved by val loss
* Gradient accumulation → effective batch = BATCH_SIZE × ACCUM_STEPS
* LR warmup prevents early divergence when fine-tuning the large backbone

Usage
-----
  python scripts/train/train_stage1.py           # fresh start
  python scripts/train/train_stage1.py --resume  # continue
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead
from utils.losses import NTXentLoss

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="Resume from checkpoints/stage1/resume.pt")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────
DEVICE      = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
NUM_EPOCHS  = Config.NUM_EPOCHS
BATCH_SIZE  = Config.BATCH_SIZE        # per-GPU batch (keep <= 8 for 8GB VRAM)
ACCUM_STEPS = 4                        # effective batch = BATCH_SIZE × ACCUM_STEPS
LR          = Config.LR                # peak LR for new layers; BERT proj uses same
WARMUP_RATIO = 0.05                    # 5% of total steps used for linear warmup
GRAD_CLIP   = 1.0
USE_AMP     = DEVICE.type == "cuda"

CKPT_DIR    = os.path.join(ROOT, "checkpoints", "stage1")
BEST_CKPT   = os.path.join(CKPT_DIR, "best.pth")
LATEST_CKPT = os.path.join(CKPT_DIR, "latest.pth")
RESUME_FILE = os.path.join(CKPT_DIR, "resume.pt")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"Device : {DEVICE}  |  AMP: {USE_AMP}")
print(f"Effective batch size: {BATCH_SIZE} × {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS}")

# ── Datasets / DataLoaders ────────────────────────────────────────────────
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

print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

# ── Models ────────────────────────────────────────────────────────────────
visual_model = MultiViewBackbone().to(DEVICE)
alignment    = CrossModalAlignment().to(DEVICE)
proj_img     = ProjectionHead().to(DEVICE)
proj_txt     = ProjectionHead().to(DEVICE)

# NT-Xent with learnable log-temperature (CLIP-style)
criterion = NTXentLoss(init_temperature=0.07).to(DEVICE)

# ── Optimizer ─────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(visual_model.parameters())
    + list(alignment.parameters())
    + list(proj_img.parameters())
    + list(proj_txt.parameters())
    + list(criterion.parameters()),   # include learnable temperature
    lr=LR,
    weight_decay=1e-4,
)

# ── LR Scheduler (cosine with linear warmup) ─────────────────────────────
total_optim_steps = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
warmup_steps      = int(total_optim_steps * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optim_steps,
)

# ── AMP scaler ────────────────────────────────────────────────────────────
scaler = GradScaler("cuda", enabled=USE_AMP)

# ── Resume state ──────────────────────────────────────────────────────────
best_val_loss = float("inf")
start_epoch   = 0
patience      = 8
no_improve    = 0

if args.resume and os.path.exists(RESUME_FILE):
    ckpt = torch.load(RESUME_FILE, map_location=DEVICE, weights_only=False)
    visual_model.load_state_dict(ckpt["visual_model"])
    alignment.load_state_dict(ckpt["alignment"])
    proj_img.load_state_dict(ckpt["proj_img"])
    proj_txt.load_state_dict(ckpt["proj_txt"])
    criterion.load_state_dict(ckpt["criterion"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    no_improve    = ckpt.get("no_improve", 0)
    print(f"Resumed epoch {ckpt['epoch'] + 1} | best_val_loss={best_val_loss:.6f} | temperature={criterion.temperature.item():.4f}")
elif args.resume:
    print("--resume requested but no resume.pt found — starting fresh.")

print(f"\nStarting Stage-1 Training | epochs={NUM_EPOCHS} | warmup={warmup_steps} steps | total={total_optim_steps} steps")

# ── Training loop ─────────────────────────────────────────────────────────
for epoch in range(start_epoch, NUM_EPOCHS):

    # ── Train ──────────────────────────────────────────────────────────
    visual_model.train()
    alignment.train()
    proj_img.train()
    proj_txt.train()
    criterion.train()

    epoch_loss = 0.0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [train]")

    for step, (images, reports) in enumerate(loop):
        images = images.to(DEVICE)

        with autocast(device_type="cuda", enabled=USE_AMP, dtype=torch.bfloat16):
            visual_features = visual_model(images)
            aligned, text_cls, _ = alignment(visual_features, reports)

            img_global = aligned.mean(dim=1)               # (B, 256)
            img_emb = proj_img(img_global)                 # (B, 256) L2-normalised
            txt_emb = proj_txt(text_cls)                   # (B, 256) L2-normalised

            loss = criterion(img_emb, txt_emb) / ACCUM_STEPS

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(visual_model.parameters())
                + list(alignment.parameters())
                + list(proj_img.parameters())
                + list(proj_txt.parameters())
                + list(criterion.parameters()),
                GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        loop.set_postfix(
            loss=f"{epoch_loss / (step + 1):.4f}",
            temp=f"{criterion.temperature.item():.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    # Flush any leftover accumulated gradients
    if (len(train_loader) % ACCUM_STEPS) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(visual_model.parameters()) + list(alignment.parameters())
            + list(proj_img.parameters()) + list(proj_txt.parameters())
            + list(criterion.parameters()),
            GRAD_CLIP,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = epoch_loss / len(train_loader)

    # ── Validation ─────────────────────────────────────────────────────
    visual_model.eval()
    alignment.eval()
    proj_img.eval()
    proj_txt.eval()
    criterion.eval()

    val_loss_total = 0.0
    with torch.no_grad():
        for val_images, val_reports in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [val]  ", leave=False):
            val_images = val_images.to(DEVICE)
            with autocast(device_type="cuda", enabled=USE_AMP, dtype=torch.bfloat16):
                vf = visual_model(val_images)
                va, vcls, _ = alignment(vf, val_reports)
                vi = proj_img(va.mean(dim=1))
                vt = proj_txt(vcls)
                vl = criterion(vi, vt)
            val_loss_total += vl.item()

    avg_val_loss = val_loss_total / max(len(val_loader), 1)
    torch.cuda.empty_cache()

    print(
        f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
        f"train={avg_train_loss:.6f} | val={avg_val_loss:.6f} | "
        f"temp={criterion.temperature.item():.4f} | "
        f"lr={scheduler.get_last_lr()[0]:.2e}"
    )

    # ── Checkpointing ──────────────────────────────────────────────────
    weights_only = {
        "visual_model": visual_model.state_dict(),
        "alignment":    alignment.state_dict(),
        "proj_img":     proj_img.state_dict(),
        "proj_txt":     proj_txt.state_dict(),
    }

    torch.save(weights_only, LATEST_CKPT)

    torch.save({
        **weights_only,
        "criterion":    criterion.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "scaler":       scaler.state_dict(),
        "epoch":        epoch,
        "best_val_loss": best_val_loss,
        "no_improve":   no_improve,
    }, RESUME_FILE)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve    = 0
        torch.save({**weights_only, "epoch": epoch, "val_loss": avg_val_loss}, BEST_CKPT)
        print(f"  ** Best model saved -> {BEST_CKPT}  (val_loss={best_val_loss:.6f})")
    else:
        no_improve += 1
        print(f"  No improvement {no_improve}/{patience}")
        if no_improve >= patience:
            print(f"Early stopping after epoch {epoch + 1}.")
            break

print(f"\nStage-1 complete | best_val_loss={best_val_loss:.6f}")
print(f"  Best   -> {BEST_CKPT}")
print(f"  Latest -> {LATEST_CKPT}")
print(f"  Resume -> {RESUME_FILE}")
print("\nNext step: python scripts/train/train_report_classifier.py")