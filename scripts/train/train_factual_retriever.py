"""
Train the fact-aware multimodal retriever using factually-mined positive pairs.

Implements the Multimodal Dense Retrieval training step from FactMM-RAG (NAACL 2025),
adapted to the SAE-FRAG visual encoder architecture.

The query encoder encodes images only; the document encoder cross-attends image
features with report text via CrossModalAlignment.  Training uses InfoNCE with
in-batch negatives.

Prerequisites
-------------
    python scripts/prepare/mine_factual_pairs.py

Outputs  ->  checkpoints/stage1/
    factual_retriever.pth  -- {visual_model, alignment, proj_img, proj_doc}
    factual_retriever_resume.pt  -- full training state

Usage
-----
    python scripts/train/train_factual_retriever.py
    python scripts/train/train_factual_retriever.py --resume
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import pickle
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models import (
    CrossModalAlignment, DocumentProjectionHead,
    MultiViewBackbone, ProjectionHead,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


_FPN_KEY_REMAP = {
    "fpn.output_c2": "fpn.output_p2",
    "fpn.output_c3": "fpn.output_p3",
    "fpn.output_c4": "fpn.output_p4",
    "fpn.output_c5": "fpn.output_p5",
}


def _remap_fpn_keys(state_dict: dict) -> dict:
    """Translate legacy output_c* keys to output_p* when needed."""
    new_sd = {}
    for k, v in state_dict.items():
        for old, new in _FPN_KEY_REMAP.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        new_sd[k] = v
    return new_sd


# =============================================================================
# Dataset
# =============================================================================

class FactualPairDataset(Dataset):
    """Yields (query_image, doc_image, doc_text) factual training triples."""

    def __init__(
        self,
        base_dataset: IUXrayMultiViewDataset,
        factual_pairs: dict,
    ) -> None:
        self.base  = base_dataset
        self.pairs = [
            (q_idx, d_idx)
            for q_idx, pos_list in factual_pairs.items()
            for d_idx in pos_list
        ]
        if not self.pairs:
            raise ValueError(
                "No factual pairs found. Run mine_factual_pairs.py first."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        q_idx, d_idx = self.pairs[idx]
        q_img, *_    = self.base[q_idx]
        d_img, d_txt = self.base[d_idx][:2]
        return q_img, d_img, d_txt


def _collate(batch):
    q_imgs  = torch.stack([b[0] for b in batch])
    d_imgs  = torch.stack([b[1] for b in batch])
    d_texts = [b[2] for b in batch]
    return q_imgs, d_imgs, d_texts


# =============================================================================
# Helpers
# =============================================================================

def _encode_query(
    visual_encoder: nn.Module,
    proj_img: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Image-only query embedding. Returns L2-normalised (B, 256) tensor."""
    feats       = visual_encoder(images)               # (B, 256, 14, 14)
    global_feat = feats.flatten(2).mean(dim=2)         # (B, 256)
    return proj_img(global_feat)                       # (B, 256)


def _encode_document(
    visual_encoder: nn.Module,
    alignment: nn.Module,
    proj_doc: nn.Module,
    images: torch.Tensor,
    texts: list,
) -> torch.Tensor:
    """Image + text document embedding. Returns L2-normalised (B, 256) tensor."""
    feats             = visual_encoder(images)          # (B, 256, 14, 14)
    aligned, _, _     = alignment(feats, texts)         # (B, 196, 256)
    return proj_doc(aligned.mean(dim=1))                # (B, 256)


def _info_nce_loss(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives. q_emb, d_emb: (B, 256) L2-normalised."""
    scores  = torch.matmul(q_emb, d_emb.t()) / tau     # (B, B)
    targets = torch.arange(q_emb.size(0), device=q_emb.device)
    return F.cross_entropy(scores, targets)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP      = DEVICE.type == "cuda"
    log.info(f"Using device: {DEVICE} | AMP: {USE_AMP}")
    NUM_EPOCHS   = 15
    BATCH_SIZE   = 16
    ACCUM_STEPS  = 2
    LR           = 5e-6
    WEIGHT_DECAY = 0.2
    WARMUP_RATIO = 0.1
    TEMPERATURE  = 0.01
    EARLY_STOP   = 5
    GRAD_CLIP    = 1.0

    STAGE1_DIR  = os.path.join(_ROOT, "checkpoints", "stage1")
    STAGE1_CKPT = os.path.join(STAGE1_DIR, "best.pth")
    OUT_CKPT    = os.path.join(STAGE1_DIR, "factual_retriever.pth")
    RESUME_FILE = os.path.join(STAGE1_DIR, "factual_retriever_resume.pt")
    PAIRS_PATH  = os.path.join(_ROOT, "store", "factual_pairs.pkl")
    os.makedirs(STAGE1_DIR, exist_ok=True)

    # ── Load factual pairs ────────────────────────────────────────────────
    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(
            f"Factual pairs not found: {PAIRS_PATH}\n"
            "Run: python scripts/prepare/mine_factual_pairs.py"
        )
    with open(PAIRS_PATH, "rb") as f:
        factual_pairs: dict = pickle.load(f)

    total_pairs = sum(len(v) for v in factual_pairs.values())
    log.info(f"Loaded {total_pairs} factual training pairs.")

    # ── Dataset ───────────────────────────────────────────────────────────
    train_base = IUXrayMultiViewDataset(Config.DATA_ROOT, split="train")
    train_ds   = FactualPairDataset(train_base, factual_pairs)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=_collate, drop_last=True, pin_memory=USE_AMP,
    )
    log.info(f"Training pairs: {len(train_ds)} | Batch: {BATCH_SIZE}")

    # ── Models ────────────────────────────────────────────────────────────
    ckpt           = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
    visual_encoder = MultiViewBackbone().to(DEVICE)
    visual_encoder.load_state_dict(_remap_fpn_keys(ckpt["visual_model"]))

    alignment = CrossModalAlignment().to(DEVICE)
    alignment.load_state_dict(ckpt["alignment"])

    proj_img = ProjectionHead().to(DEVICE)
    proj_img.load_state_dict(ckpt["proj_img"])

    proj_doc = DocumentProjectionHead().to(DEVICE)

    # Unfreeze visual backbone and alignment projection for fine-tuning
    for param in visual_encoder.parameters():
        param.requires_grad = True
    for param in alignment.text_proj.parameters():
        param.requires_grad = True
    for param in alignment.cross_attention.parameters():
        param.requires_grad = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable = (
        list(visual_encoder.parameters())
        + list(alignment.text_proj.parameters())
        + list(alignment.cross_attention.parameters())
        + list(proj_img.parameters())
        + list(proj_doc.parameters())
    )
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in trainable if p.ndim >= 2], "weight_decay": WEIGHT_DECAY},
         {"params": [p for p in trainable if p.ndim < 2],  "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.98), eps=1e-6,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps  = len(train_loader) // ACCUM_STEPS * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")
    no_improve  = 0

    if args.resume and os.path.exists(RESUME_FILE):
        state = torch.load(RESUME_FILE, map_location=DEVICE, weights_only=False)
        visual_encoder.load_state_dict(_remap_fpn_keys(state["visual_model"]))
        alignment.load_state_dict(state["alignment"])
        proj_img.load_state_dict(state["proj_img"])
        proj_doc.load_state_dict(state["proj_doc"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_loss   = state["best_loss"]
        no_improve  = state.get("no_improve", 0)
        log.info(f"Resumed from epoch {start_epoch}. Best loss: {best_loss:.4f}")
    elif args.resume:
        log.warning("--resume requested but no resume.pt found — starting fresh.")

    log.info(f"\nStarting factual retriever training\n")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        visual_encoder.train(); alignment.train()
        proj_img.train(); proj_doc.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, (q_imgs, d_imgs, d_texts) in enumerate(pbar):
            q_imgs = q_imgs.to(DEVICE)
            d_imgs = d_imgs.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP, dtype=torch.bfloat16):
                q_emb = _encode_query(visual_encoder, proj_img, q_imgs)
                d_emb = _encode_document(visual_encoder, alignment, proj_doc, d_imgs, d_texts)
                loss = _info_nce_loss(q_emb, d_emb, TEMPERATURE) / ACCUM_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        log.info(f"Epoch {epoch + 1}  avg_loss={avg_loss:.4f}")

        # ── Checkpointing ─────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss  = avg_loss
            no_improve = 0
            torch.save({
                "visual_model": visual_encoder.state_dict(),
                "alignment":    alignment.state_dict(),
                "proj_img":     proj_img.state_dict(),
                "proj_doc":     proj_doc.state_dict(),
                "epoch":        epoch,
                "loss":         avg_loss,
            }, OUT_CKPT)
            log.info(f"  Best checkpoint saved -> {os.path.relpath(OUT_CKPT, _ROOT)}")
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{EARLY_STOP}")

        torch.save({
            "visual_model": visual_encoder.state_dict(),
            "alignment":    alignment.state_dict(),
            "proj_img":     proj_img.state_dict(),
            "proj_doc":     proj_doc.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "scaler":       scaler.state_dict(),
            "epoch":        epoch,
            "best_loss":    best_loss,
            "no_improve":   no_improve,
        }, RESUME_FILE)

        if no_improve >= EARLY_STOP:
            log.info("Early stopping triggered.")
            break

    log.info("Factual retriever training complete.")
    log.info(f"Checkpoint -> {os.path.relpath(OUT_CKPT, _ROOT)}")


if __name__ == "__main__":
    main()
