import argparse
import os
import sys

# Bootstrap: make src/ importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/stage1/resume.pt")
args = parser.parse_args()

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead
from utils.losses import contrastive_loss


# ==========================
# Setup
# ==========================
device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.join(ROOT, "checkpoints", "stage1"), exist_ok=True)
CKPT_DIR    = os.path.join(ROOT, "checkpoints", "stage1")
BEST_CKPT   = os.path.join(CKPT_DIR, "best.pth")
LATEST_CKPT = os.path.join(CKPT_DIR, "latest.pth")
RESUME_FILE = os.path.join(CKPT_DIR, "resume.pt")

print("Using device:", device)


# ==========================
# Dataset
# ==========================
dataset = IUXrayMultiViewDataset(root_dir=Config.DATA_ROOT)

loader = DataLoader(
    dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print("Dataset size:", len(dataset))


# ==========================
# Models
# ==========================
visual_model = MultiViewBackbone().to(device)
alignment = CrossModalAlignment().to(device)
proj_img = ProjectionHead().to(device)
proj_txt = ProjectionHead().to(device)

optimizer = torch.optim.Adam(
    list(visual_model.parameters()) +
    list(alignment.parameters()) +
    list(proj_img.parameters()) +
    list(proj_txt.parameters()),
    lr=Config.LR
)

best_loss    = float("inf")
start_epoch  = 0
patience     = 5
no_improve   = 0

# ── Resume ───────────────────────────────────────────────────────────────
if args.resume and os.path.exists(RESUME_FILE):
    resume = torch.load(RESUME_FILE, map_location=device, weights_only=False)
    visual_model.load_state_dict(resume["visual_model"])
    alignment.load_state_dict(resume["alignment"])
    proj_img.load_state_dict(resume["proj_img"])
    proj_txt.load_state_dict(resume["proj_txt"])
    optimizer.load_state_dict(resume["optimizer"])
    start_epoch = resume["epoch"] + 1
    best_loss   = resume["best_loss"]
    no_improve  = resume.get("no_improve", 0)
    print(f"Resumed from epoch {resume['epoch']+1} | best_loss={best_loss:.6f} | no_improve={no_improve}")
elif args.resume:
    print("--resume requested but no resume.pt found. Starting fresh.")

print("Starting Stage-1 Training...")


# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(start_epoch, Config.NUM_EPOCHS):

    visual_model.train()
    alignment.train()
    proj_img.train()
    proj_txt.train()

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    epoch_loss = 0.0

    for images, reports in loop:

        images = images.to(device)

        # Forward
        visual_features = visual_model(images)
        aligned, text_cls, attn_weights = alignment(visual_features, reports)

        img_global = aligned.mean(dim=1)

        img_emb = proj_img(img_global)
        txt_emb = proj_txt(text_cls)

        loss = contrastive_loss(img_emb, txt_emb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(loader)

    print(f"Epoch {epoch+1} Finished | Avg Loss: {avg_epoch_loss:.6f}")

    # Save full resume state every epoch
    torch.save({
        "visual_model": visual_model.state_dict(),
        "alignment":    alignment.state_dict(),
        "proj_img":     proj_img.state_dict(),
        "proj_txt":     proj_txt.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "epoch":        epoch,
        "best_loss":    best_loss,
        "no_improve":   no_improve,
    }, RESUME_FILE)

    # Save latest model weights
    torch.save({
        "visual_model": visual_model.state_dict(),
        "alignment":    alignment.state_dict(),
        "proj_img":     proj_img.state_dict(),
        "proj_txt":     proj_txt.state_dict(),
    }, LATEST_CKPT)

    # Save best checkpoint
    if avg_epoch_loss < best_loss:
        best_loss  = avg_epoch_loss
        no_improve = 0
        torch.save({
            "visual_model": visual_model.state_dict(),
            "alignment":    alignment.state_dict(),
            "proj_img":     proj_img.state_dict(),
            "proj_txt":     proj_txt.state_dict(),
            "epoch":        epoch,
            "loss":         avg_epoch_loss,
        }, BEST_CKPT)
        print(f"  ** Best model saved -> {BEST_CKPT}  (loss={best_loss:.6f})")
    else:
        no_improve += 1
        print(f"  No improvement for {no_improve}/{patience} epochs")
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

print(f"\nTraining Complete | best_loss={best_loss:.6f}")
print(f"Best   -> {BEST_CKPT}")
print(f"Latest -> {LATEST_CKPT}")
print(f"Resume -> {RESUME_FILE}")

print("Saved new best model.")


print("Training finished.")

# Save final model
torch.save({
    "visual_model": visual_model.state_dict(),
    "alignment": alignment.state_dict(),
    "proj_img": proj_img.state_dict(),
    "proj_txt": proj_txt.state_dict(),
}, "checkpoints/final_stage1.pth")

print("Final model saved.")


# ==========================
# TOKEN-SPECIFIC HEATMAP
# ==========================
print("Generating token-specific heatmap...")

visual_model.eval()
alignment.eval()

with torch.no_grad():

    images, reports = next(iter(loader))
    images = images.to(device)

    visual_features = visual_model(images)
    aligned, text_cls, attn_weights = alignment(visual_features, reports)

    tokenizer = alignment.tokenizer

    encoding = tokenizer(
        reports,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )

    print("Tokens:", tokens)

    target_word = "pneumothorax"

    token_indices = [
        i for i, tok in enumerate(tokens)
        if target_word in tok.replace("##", "")
    ]

    if len(token_indices) == 0:
        print(f"{target_word} not found, using CLS token.")
        token_indices = [0]

    patch_scores = attn_weights[0][:, token_indices].mean(dim=1)

    heatmap = patch_scores.reshape(7, 7).cpu()

    plt.imshow(heatmap, cmap="hot")
    plt.title(f"Attention Heatmap for '{target_word}'")
    plt.colorbar()
    plt.show()