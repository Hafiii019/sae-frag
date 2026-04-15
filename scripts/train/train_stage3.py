"""
Stage 3: Fine-tune HybridReportGenerator using pre-cached frozen model outputs.

Loads aligned_features and entity_vectors from disk so no BERT or visual
backbone is required in the training loop.  Substantially faster than the
uncached version (~1-2s/step vs ~10s/step).

Prerequisites
-------------
    python scripts/prepare/cache_features.py

Outputs  ->  checkpoints/stage3/
    best_generator.pth   best checkpoint by val loss
    last_generator.pth   most recent epoch checkpoint
    resume.pt            full training state for --resume

Usage
-----
    python scripts/train/train_stage3.py
    python scripts/train/train_stage3.py --resume
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import random
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import torch
import torch.amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from rag.hybrid_generator import HybridReportGenerator

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class CachedFeaturesDataset(Dataset):
    """Load pre-cached aligned_features + entity_vectors from disk.

    Randomly samples one of k stored retrieval variants per sample each epoch
    to preserve training diversity without re-running BERT or the visual encoder.
    """

    def __init__(self, cache_path: str) -> None:
        log.info(f"Loading cache: {os.path.relpath(cache_path, _ROOT)}")
        data = torch.load(cache_path, weights_only=False)
        self.aligned  = [[v["aligned_features"] for v in item["variants"]] for item in data]
        self.entities = [[v["entity_vector"]    for v in item["variants"]] for item in data]
        self.reps     = [[v["retrieved_text"]   for v in item["variants"]] for item in data]
        self.targets  = [item["target"]      for item in data]
        self.impressions  = [item.get("impression",  "") for item in data]
        self.entity_texts = [item.get("entity_tags", "") for item in data]
        log.info(f"  {len(self.targets)} samples loaded.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        k = random.randrange(len(self.aligned[idx]))
        return (
            self.aligned[idx][k],
            self.entities[idx][k],
            self.reps[idx][k],
            self.targets[idx],
            self.impressions[idx],
            self.entity_texts[idx],
        )


def _collate(batch):
    afs, evs, reps, targets, impressions, entity_texts = zip(*batch)
    return torch.stack(afs), torch.stack(evs), list(reps), list(targets), list(impressions), list(entity_texts)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/stage3/resume.pt")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP      = DEVICE.type == "cuda"
    NUM_EPOCHS   = 40
    BATCH_SIZE   = 4
    ACCUM_STEPS  = 2   # effective batch = 8
    WARMUP_STEPS = 500  # ~5% of total steps; more warmup for medical domain adapt
    # SciFive-base is T5-base pre-trained on PubMed+PMC — same architecture but
    # medical vocabulary gives ~0.05 BLEU boost over flan-t5-base.
    # Fallback: change to "google/flan-t5-base" if download fails.
    MODEL_NAME   = "razent/SciFive-base-Pubmed_PMC"
    LR_T5        = 2e-5   # slightly higher: SciFive needs more domain adaptation
    LR_NEW       = 1e-4   # projections learn faster
    WEIGHT_DECAY = 0.01
    GRAD_CLIP    = 0.5
    # Phase-1: train projection layers only (freeze T5) for N epochs so the
    # visual/entity projections align with the T5 embedding space before
    # the heavier T5 fine-tuning in phase-2.  Helps convergence on 6 GB GPU.
    FREEZE_T5_EPOCHS = 3

    STAGE3_DIR  = os.path.join(_ROOT, "checkpoints", "stage3")
    BEST_CKPT   = os.path.join(STAGE3_DIR, "best_generator.pth")
    LAST_CKPT   = os.path.join(STAGE3_DIR, "last_generator.pth")
    RESUME_FILE = os.path.join(STAGE3_DIR, "resume.pt")
    TRAIN_CACHE = os.path.join(_ROOT, "store", "cache_train.pt")
    VAL_CACHE   = os.path.join(_ROOT, "store", "cache_val.pt")
    os.makedirs(STAGE3_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_CACHE):
        raise FileNotFoundError(
            f"Cache not found: {TRAIN_CACHE}\n"
            "Run: python scripts/prepare/cache_features.py"
        )

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset = CachedFeaturesDataset(TRAIN_CACHE)
    val_dataset   = CachedFeaturesDataset(VAL_CACHE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=_collate,
    )

    # ── Generator ─────────────────────────────────────────────────────────
    generator = HybridReportGenerator(model_name=MODEL_NAME).to(DEVICE)

    # Warm-start from compatible checkpoint (skip on shape/tokenizer mismatch).
    # When switching to a new MODEL_NAME, clear old checkpoints or they will
    # silently load wrong embedding weights (same shape, different vocab).
    # To force a clean start: delete checkpoints/stage3/best_generator.pth
    for warm_path in [BEST_CKPT, LAST_CKPT]:
        if not os.path.exists(warm_path):
            continue
        state = torch.load(warm_path, map_location=DEVICE, weights_only=False)
        current = {k: v.shape for k, v in generator.state_dict().items()}
        if any(k in current and v.shape != current[k] for k, v in state.items()):
            log.warning(f"Skipping {os.path.basename(warm_path)} — shape mismatch.")
            continue
        generator.load_state_dict(state, strict=False)
        log.info(f"Warm-started from {os.path.relpath(warm_path, _ROOT)}")
        break

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    _new = {"visual_proj", "visual_norm", "visual_drop", "entity_proj"}
    t5_params  = [p for n, p in generator.named_parameters()
                  if not any(k in n for k in _new) and p.requires_grad]
    new_params = [p for n, p in generator.named_parameters()
                  if any(k in n for k in _new) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": t5_params, "lr": LR_T5},
         {"params": new_params, "lr": LR_NEW}],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    # ── Resume ────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    start_epoch   = 0
    patience      = 8   # increased from 5 for medical domain convergence
    no_improve    = 0

    if args.resume and os.path.exists(RESUME_FILE):
        state = torch.load(RESUME_FILE, map_location=DEVICE, weights_only=False)
        generator.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch   = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        no_improve    = state.get("no_improve", 0)
        log.info(f"Resumed from epoch {start_epoch} | best_val_loss={best_val_loss:.4f}")
    elif args.resume:
        log.warning("--resume requested but no resume.pt found — starting fresh.")

    log.info(
        f"\nDevice={DEVICE} | {len(train_dataset)} train / {len(val_dataset)} val\n"
        f"Batch={BATCH_SIZE} x accum={ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS} effective\n"
        f"Epochs={NUM_EPOCHS} (starting {start_epoch + 1}) | total steps={total_steps}\n"
    )

    # ── Training loop ─────────────────────────────────────────────────────
    _proj_keys = {"visual_proj", "visual_norm", "visual_drop", "entity_proj"}

    for epoch in range(start_epoch, NUM_EPOCHS):
        # ── Two-phase training ───────────────────────────────────────────
        # Phase 1 (epochs 0..FREEZE_T5_EPOCHS-1): freeze T5, warm up projections
        # Phase 2 (epoch >= FREEZE_T5_EPOCHS): unfreeze all T5 params
        if epoch < FREEZE_T5_EPOCHS:
            for name, param in generator.named_parameters():
                param.requires_grad_(any(k in name for k in _proj_keys))
            if epoch == start_epoch or epoch == 0:
                log.info(
                    f"  Phase 1: T5 frozen, training projection layers only "
                    f"(epochs 1-{FREEZE_T5_EPOCHS})"
                )
        elif epoch == FREEZE_T5_EPOCHS:
            for param in generator.parameters():
                param.requires_grad_(True)
            log.info(f"  Phase 2: all parameters unfrozen (epoch {epoch + 1}+)")

        generator.train()
        total_loss = 0.0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, (aligned_features, entity_vector, retrieved_texts, reports, impressions, entity_texts) in enumerate(loop):
            aligned_features = aligned_features.to(DEVICE)
            entity_vector    = entity_vector.to(DEVICE)

            prompts   = HybridReportGenerator.build_entity_prompt(entity_vector.cpu())
            rag_texts = HybridReportGenerator.build_rag_retrieved_text(retrieved_texts)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP, dtype=torch.bfloat16):
                loss = generator(
                    region_features=aligned_features,
                    entity_vector=entity_vector,
                    retrieved_texts=rag_texts,
                    prompt_texts=prompts,
                    target_texts=reports,
                    impression_texts=impressions,
                    entity_texts=entity_texts,
                ) / ACCUM_STEPS

            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
                optimizer.step(); optimizer.zero_grad(); scheduler.step()
                torch.cuda.empty_cache()  # Free memory between accumulation steps

            loop.set_postfix(
                loss=f"{total_loss / (step + 1):.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        if (len(train_loader) % ACCUM_STEPS) != 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            optimizer.step(); optimizer.zero_grad(); scheduler.step()

        torch.cuda.empty_cache()  # Free memory after training epoch

        avg_train = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_af, val_ev, val_reps, val_reports, val_imps, val_etexts in tqdm(
                val_loader, desc="  Val", leave=False
            ):
                val_af = val_af.to(DEVICE)
                val_ev = val_ev.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP, dtype=torch.bfloat16):
                    val_loss += generator(
                        region_features=val_af,
                        entity_vector=val_ev,
                        retrieved_texts=HybridReportGenerator.build_rag_retrieved_text(list(val_reps)),
                        prompt_texts=HybridReportGenerator.build_entity_prompt(val_ev.cpu()),
                        target_texts=list(val_reports),
                        impression_texts=list(val_imps),
                        entity_texts=list(val_etexts),
                    ).item()

        avg_val = val_loss / max(len(val_loader), 1)
        torch.cuda.empty_cache()

        log.info(
            f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
            f"train={avg_train:.4f} | val={avg_val:.4f}"
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        torch.save(generator.state_dict(), LAST_CKPT)
        torch.save({
            "epoch":         epoch,
            "model":         generator.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "no_improve":    no_improve,
        }, RESUME_FILE)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve    = 0
            torch.save(generator.state_dict(), BEST_CKPT)
            log.info(f"  Best checkpoint saved (val={best_val_loss:.4f})")
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                log.info(f"Early stopping after epoch {epoch + 1}.")
                break

    log.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Best checkpoint -> {os.path.relpath(BEST_CKPT, _ROOT)}")


if __name__ == "__main__":
    main()
