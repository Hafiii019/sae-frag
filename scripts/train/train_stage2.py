"""Stage 2b: Train SAEImageClassifier using soft pseudo-labels from ReportClassifier.

Key improvements over baseline
--------------------------------
* Stage-1 backbone weights loaded as initialisation (not random)
* Early backbone layers (stem + stages 0-3) frozen — only layer4, FPN, SAFE,
  and the classification head are updated, preserving contrastive features
* AMP (bfloat16) for ~2x speed-up on CUDA
* BCEWithLogitsLoss with per-class pos_weight to handle IU-Xray label imbalance
* Gradient clipping (max_norm=1.0)

Prerequisites
-------------
  python scripts/train/train_stage1.py
  python scripts/train/train_report_classifier.py
"""

import argparse
import os
import sys

# Bootstrap: make src/ importable
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/stage2/resume.pt")
args = parser.parse_args()

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier


# =========================================
# Device
# =========================================
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
ROOT    = Config.DATA_ROOT

print(f"device={device} | AMP={USE_AMP}")

# =========================================
# TRAIN SPLIT ONLY
# =========================================
dataset = IUXrayMultiViewDataset(
    root_dir=ROOT,
    split="train"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    pin_memory=USE_AMP,
)


# =========================================
# Load trained report classifier
# =========================================
report_model = ReportClassifier().to(device)

report_model.load_state_dict(
    torch.load(
        os.path.join(PROJ_ROOT, "checkpoints", "stage2", "report_classifier.pth"),
        map_location=device
    )
)

report_model.eval()


# =========================================
# Image classifier (SAEnet based)
# =========================================
image_model = SAEImageClassifier().to(device)

# ── Warm-start backbone from Stage-1 contrastive checkpoint ──────────────
# Loading stage-1 weights before Stage-2 training means the backbone starts
# from a vision-language-aligned feature space rather than plain ImageNet.
S1_CKPT = os.path.join(PROJ_ROOT, "checkpoints", "stage1", "best.pth")
if os.path.exists(S1_CKPT):
    s1_weights = torch.load(S1_CKPT, map_location=device, weights_only=False)
    image_model.backbone.load_state_dict(s1_weights["visual_model"])
    print(f"Loaded Stage-1 backbone from {S1_CKPT}")
else:
    print(f"WARNING: Stage-1 checkpoint not found at {S1_CKPT} — training from ImageNet init.")

# ── Freeze stem + ResNet stages 0-3 to preserve Stage-1 features ─────────
# Only layer4 (high-level semantics), FPN, SAFE, and the head are updated.
frozen, trainable = 0, 0
for name, param in image_model.backbone.named_parameters():
    if any(f"backbone.layer{i}" in name for i in range(4)):  # layer0..layer3
        param.requires_grad = False
        frozen += param.numel()
    else:
        param.requires_grad = True
        trainable += param.numel()
print(f"Backbone: frozen={frozen/1e6:.1f}M params | trainable={trainable/1e6:.1f}M params")

CKPT2_DIR   = os.path.join(PROJ_ROOT, "checkpoints", "stage2")
RESUME_FILE = os.path.join(CKPT2_DIR, "resume.pt")
os.makedirs(CKPT2_DIR, exist_ok=True)

# ── BCEWithLogitsLoss with pos_weight for label imbalance ─────────────────
# Chest pathologies are rare; without pos_weight the model predicts all-negative.
# A conservative weight of 5 (tunable) gives ~5× more gradient for positives.
pos_weight = torch.ones(14, device=device) * 5.0
criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AMP scaler
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, image_model.parameters()),
    lr=3e-4,
    weight_decay=1e-4,
)

BEST_CKPT   = os.path.join(CKPT2_DIR, "image_classifier.pth")
NUM_EPOCHS  = 25
start_epoch = 0
patience    = 5
no_improve  = 0
best_loss   = float("inf")

# ── Resume ───────────────────────────────────────────────────────────────
if args.resume and os.path.exists(RESUME_FILE):
    resume = torch.load(RESUME_FILE, map_location=device, weights_only=False)
    image_model.load_state_dict(resume["model"])
    optimizer.load_state_dict(resume["optimizer"])
    start_epoch = resume["epoch"] + 1
    no_improve  = resume.get("no_improve", 0)
    best_loss   = resume.get("best_loss", float("inf"))
    print(f"Resumed from epoch {resume['epoch']+1} | no_improve={no_improve}")
elif args.resume:
    print("--resume requested but no resume.pt found. Starting fresh.")

# Cosine LR scheduler — decays smoothly to near-zero over all epochs
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print("====================================")
print("Starting Image Classifier Training")
print("====================================")

image_model.train()

for epoch in range(start_epoch, NUM_EPOCHS):

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    total_loss = 0

    for images, reports in loop:

        images = images.to(device)

        # Soft pseudo-labels: keep the full [0,1] confidence from the text
        # classifier rather than hard-thresholding. This retains uncertainty
        # information and gives much stronger learning signal.
        with torch.no_grad():
            labels = torch.sigmoid(report_model(reports)).to(device)

        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP, dtype=torch.bfloat16):
            logits = image_model(images)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, image_model.parameters()), 1.0
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    lr_now   = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | avg_loss={avg_loss:.4f} | lr={lr_now:.2e}")

    scheduler.step()

    # Save resume state every epoch
    torch.save({
        "epoch":      epoch,
        "model":      image_model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "best_loss":  best_loss,
        "no_improve": no_improve,
    }, RESUME_FILE)

    if avg_loss < best_loss:
        best_loss  = avg_loss
        no_improve = 0
        torch.save(image_model.state_dict(), BEST_CKPT)
        print(f"  ** Best model saved (loss={best_loss:.4f})")
    else:
        no_improve += 1
        print(f"  No improvement for {no_improve}/{patience} epochs")
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

print("====================================")
print("Image classifier training complete.")
print(f"Best model -> {BEST_CKPT}")
print(f"Resume     -> {RESUME_FILE}")
print("====================================")