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

from data.dataset import IUXrayMultiViewDataset
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier


# =========================================
# Device
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "C:/Datasets/IU_Xray"


# =========================================
# TRAIN SPLIT ONLY
# =========================================
dataset = IUXrayMultiViewDataset(
    root_dir=ROOT,
    split="train"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True
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

CKPT2_DIR   = os.path.join(PROJ_ROOT, "checkpoints", "stage2")
RESUME_FILE = os.path.join(CKPT2_DIR, "resume.pt")
os.makedirs(CKPT2_DIR, exist_ok=True)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    image_model.parameters(),
    lr=1e-4
)

NUM_EPOCHS  = 20
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

print("====================================")
print("Starting Image Classifier Training")
print("====================================")

image_model.train()

for epoch in range(start_epoch, NUM_EPOCHS):

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    total_loss = 0

    for images, reports in loop:

        images = images.to(device)

        # Generate pseudo-labels from report classifier
        with torch.no_grad():
            report_logits = report_model(reports)
            labels = (torch.sigmoid(report_logits) > 0.3).float()

        labels = labels.to(device)

        logits = image_model(images)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | avg_loss={avg_loss:.4f}")

    # Save resume state every epoch so training can be continued
    torch.save({
        "epoch":      epoch,
        "model":      image_model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "best_loss":  best_loss,
        "no_improve": no_improve,
    }, RESUME_FILE)

    if avg_loss < best_loss:
        best_loss  = avg_loss
        no_improve = 0
    else:
        no_improve += 1
        print(f"  No improvement for {no_improve}/{patience} epochs")
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

_save_path = os.path.join(CKPT2_DIR, "image_classifier.pth")
torch.save(image_model.state_dict(), _save_path)

print("====================================")
print("Image classifier training complete.")
print(f"Model saved to {_save_path}")
print(f"Resume state -> {RESUME_FILE}")
print("====================================")