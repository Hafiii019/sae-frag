"""
Stage 2a: Train the ReportClassifier (text-side pathology label extractor).

Generates pseudo-labels for 14 CheXpert pathology classes from IU X-Ray report
text via keyword/regex rules, then fine-tunes a Bio_ClinicalBERT classification
head on those labels.  The trained model is used as a soft-label teacher in
Stage 2b and as a clinical metric model in evaluation.

Output
------
    checkpoints/stage2/report_classifier.pth

Usage
-----
    python scripts/train/train_report_classifier.py
    python scripts/train/train_report_classifier.py --epochs 15 --batch 64
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import re
import sys
from typing import List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from classification.report_labeler import ReportClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Class definitions ─────────────────────────────────────────────────────
NUM_CLASSES = 14
CLASS_NAMES = [
    "No Finding", "Cardiomegaly", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Atelectasis", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Nodule", "Mass", "Hernia", "Infiltrate",
]

# Keyword rules for pseudo-labeling — (class_index, [regex_patterns])
_RULES: List[Tuple[int, List[str]]] = [
    (1,  [r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bcardiac enlargement\b",
          r"\bheart is enlarged\b", r"\benlarged cardiac silhouette\b"]),
    (2,  [r"\bpleural effusion\b", r"\beffusion\b", r"\bpleural fluid\b",
          r"\bcostophrenic angle.*blunt", r"\bblunting.*costophrenic\b"]),
    (3,  [r"\bpneumonia\b"]),
    (4,  [r"\bpneumothorax\b"]),
    (5,  [r"\batelectasis\b", r"\batelectatic\b", r"\bbibasilar atelectasis\b",
          r"\bsubsegmental.*collapse\b", r"\bcollapse.*lobe\b"]),
    (6,  [r"\bconsolidation\b", r"\bconsolida\b"]),
    (7,  [r"\bedema\b", r"\bpulmonary edema\b", r"\bpulmonary venous congestion\b",
          r"\bvascular congestion\b", r"\bcongestive heart\b"]),
    (8,  [r"\bemphysema\b", r"\bhyperinflat\b", r"\bhyperexpan\b", r"\bair trap\b"]),
    (9,  [r"\bfibrosis\b", r"\bfibrotic\b", r"\binterstitial fibrosis\b"]),
    (10, [r"\bnodule\b", r"\bnodular\b", r"\bgranuloma\b", r"\bcalcified granuloma\b"]),
    (11, [r"\bmass\b", r"\bmasses\b", r"\btumor\b", r"\bneoplasm\b", r"\blung cancer\b"]),
    (12, [r"\bhernia\b", r"\bhiatal hernia\b", r"\bherniation\b"]),
    (13, [r"\binfiltrate\b", r"\binfiltration\b", r"\bopacity\b", r"\bopacit\b",
          r"\bairspace disease\b", r"\bground.glass\b"]),
]


def _label_report(text: str) -> torch.Tensor:
    """Map a radiology report string to a binary 14-class label tensor."""
    text   = text.lower()
    labels = torch.zeros(NUM_CLASSES)
    for cls_idx, patterns in _RULES:
        if any(re.search(pat, text) for pat in patterns):
            labels[cls_idx] = 1.0
    if labels[1:].sum() == 0:
        labels[0] = 1.0
    return labels


# =============================================================================
# Dataset
# =============================================================================

class ReportLabelDataset(Dataset):
    def __init__(self, split: str) -> None:
        base         = IUXrayMultiViewDataset(Config.DATA_ROOT, split=split)
        self.reports = [s["report"] for s in base.samples]
        self.labels  = [_label_report(r) for r in self.reports]
        positives    = torch.stack(self.labels).sum(0)
        log.info(f"  [{split}] {len(self.reports)} samples | positives: {positives.int().tolist()}")

    def __len__(self) -> int:
        return len(self.reports)

    def __getitem__(self, idx):
        return self.reports[idx], self.labels[idx]


def _collate(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.stack(labels)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=10)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--lr",       type=float, default=2e-5)
    parser.add_argument("--patience", type=int,   default=5)
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP  = DEVICE.type == "cuda"
    CKPT_DIR = os.path.join(_ROOT, "checkpoints", "stage2")
    BEST_CKPT = os.path.join(CKPT_DIR, "report_classifier.pth")
    os.makedirs(CKPT_DIR, exist_ok=True)

    log.info(f"Device: {DEVICE}  |  AMP: {USE_AMP}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building datasets...")
    train_ds = ReportLabelDataset("train")
    val_ds   = ReportLabelDataset("val")

    all_train_labels = torch.stack(train_ds.labels)
    pos_freq   = all_train_labels.mean(0).clamp(min=0.01)
    pos_weight = ((1 - pos_freq) / pos_freq).to(DEVICE)
    log.info(f"Pos-weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=_collate, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, collate_fn=_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ReportClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bert_params = [p for n, p in model.named_parameters() if "encoder" in n]
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    optimizer   = torch.optim.AdamW(
        [{"params": bert_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr * 10}],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
            labels = labels.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                loss = criterion(model(texts), labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        tp = torch.zeros(NUM_CLASSES)
        fp = torch.zeros(NUM_CLASSES)
        fn = torch.zeros(NUM_CLASSES)

        with torch.no_grad():
            for texts, labels in val_loader:
                labels = labels.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                    logits = model(texts)
                    val_loss += criterion(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                gt    = labels.cpu()
                tp   += (preds * gt).sum(0)
                fp   += (preds * (1 - gt)).sum(0)
                fn   += ((1 - preds) * gt).sum(0)

        avg_val = val_loss / max(len(val_loader), 1)
        prec    = tp / (tp + fp + 1e-8)
        rec     = tp / (tp + fn + 1e-8)
        mean_f1 = (2 * prec * rec / (prec + rec + 1e-8)).mean().item()

        log.info(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train={avg_train:.4f} | val={avg_val:.4f} | mean_F1={mean_f1:.4f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save(model.state_dict(), BEST_CKPT)
            log.info(f"  Best checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info("Early stopping.")
                break

    log.info(f"\nTraining complete. Best checkpoint -> {os.path.relpath(BEST_CKPT, _ROOT)}")
    log.info("Next: python scripts/train/train_stage2.py")


if __name__ == "__main__":
    main()
