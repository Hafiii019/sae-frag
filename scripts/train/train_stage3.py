"""Stage 3: Train the HybridReportGenerator using pre-cached frozen model outputs.

Only the generator is loaded onto GPU — no BERT in the training loop.
Per-step time: ~1-2s   (vs ~10s in the uncached version).

Prerequisites
-------------
  python scripts/prepare/cache_features.py

Outputs  →  checkpoints/stage3/
  best_generator.pth   best checkpoint by val loss  (use for eval / inference)
  last_generator.pth   most recent epoch checkpoint
  resume.pt            full training state for --resume

Usage
-----
  python scripts/train/train_stage3.py            # fresh start
  python scripts/train/train_stage3.py --resume   # continue from last checkpoint
"""

import argparse
import os
import random
import sys

# ── Bootstrap: make src/ importable regardless of cwd ──────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.amp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from rag.hybrid_generator import HybridReportGenerator

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="Resume from models/resume.pt")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS   = 100
BATCH_SIZE   = 4
ACCUM_STEPS  = 4           # effective batch = 16
WARMUP_STEPS = 300
LR_T5        = 5e-6
LR_NEW       = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 0.5
TRAIN_CACHE  = os.path.join(ROOT, "store", "cache_train.pt")
VAL_CACHE    = os.path.join(ROOT, "store", "cache_val.pt")

# Output paths
STAGE3_DIR  = os.path.join(ROOT, "checkpoints", "stage3")
BEST_CKPT   = os.path.join(STAGE3_DIR, "best_generator.pth")
LAST_CKPT   = os.path.join(STAGE3_DIR, "last_generator.pth")
RESUME_FILE = os.path.join(STAGE3_DIR, "resume.pt")

os.makedirs(STAGE3_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────
class CachedFeaturesDataset(Dataset):
    """Loads pre-cached aligned_features + entity_vectors.

    The entire cache is held in RAM after the first load.
    Randomly picks one of the k stored variants per sample each epoch
    to preserve training diversity without re-running BERT.
    """

    def __init__(self, cache_path):
        print(f"Loading cache into RAM: {cache_path} ...")
        self.data = torch.load(cache_path, weights_only=False)
        # Pre-extract tensors into flat lists for zero-copy indexing
        self.aligned = [[v["aligned_features"] for v in item["variants"]] for item in self.data]
        self.entities = [[v["entity_vector"]    for v in item["variants"]] for item in self.data]
        self.reps     = [[v["retrieved_text"]   for v in item["variants"]] for item in self.data]
        self.targets  = [item["target"] for item in self.data]
        del self.data  # free the list-of-dicts overhead
        print(f"  Loaded {len(self.targets)} samples.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        k = random.randrange(len(self.aligned[idx]))
        return (
            self.aligned[idx][k],   # (49, 256) — already a tensor
            self.entities[idx][k],  # (14,)
            self.reps[idx][k],      # str
            self.targets[idx],      # str
        )


def collate_fn(batch):
    afs, evs, reps, targets = zip(*batch)
    return torch.stack(afs), torch.stack(evs), list(reps), list(targets)


assert os.path.exists(TRAIN_CACHE), (
    f"Cache not found: {TRAIN_CACHE}\nRun: python scripts/prepare/cache_features.py"
)

train_dataset = CachedFeaturesDataset(TRAIN_CACHE)
val_dataset   = CachedFeaturesDataset(VAL_CACHE)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=0, collate_fn=collate_fn,
)


# ── Generator ─────────────────────────────────────────────────────────────
generator = HybridReportGenerator().to(DEVICE)

# Warm-start priority:
#  1. models/best_generator.pth  (previous best from this script)
#  2. rag/hybrid_generator_best.pth  (legacy path from old script)
#  3. rag/hybrid_generator.pth       (legacy last checkpoint)
for warm_path in [BEST_CKPT,
                  os.path.join(ROOT, "checkpoints", "stage3", "best_generator.pth"),
                  os.path.join(ROOT, "checkpoints", "best_generator.pth")]:
    if os.path.exists(warm_path):
        state = torch.load(warm_path, map_location=DEVICE, weights_only=False)
        generator.load_state_dict(state, strict=False)
        print(f"Warm-started generator from {warm_path}")
        break
else:
    print("No prior checkpoint — training from scratch.")


# ── Optimizer & scheduler ─────────────────────────────────────────────────
_new_layers = {"visual_proj", "visual_norm", "visual_drop", "entity_embed"}
t5_params   = [p for n, p in generator.named_parameters()
               if not any(k in n for k in _new_layers) and p.requires_grad]
new_params  = [p for n, p in generator.named_parameters()
               if any(k in n for k in _new_layers) and p.requires_grad]

optimizer = torch.optim.AdamW(
    [{"params": t5_params, "lr": LR_T5},
     {"params": new_params, "lr": LR_NEW}],
    weight_decay=WEIGHT_DECAY,
)

total_steps = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

# ── Resume ────────────────────────────────────────────────────────────────
start_epoch   = 0
best_val_loss = float("inf")
patience      = 5
no_improve    = 0

if args.resume and os.path.exists(RESUME_FILE):
    resume = torch.load(RESUME_FILE, map_location=DEVICE, weights_only=False)
    generator.load_state_dict(resume["model"], strict=False)
    optimizer.load_state_dict(resume["optimizer"])
    scheduler.load_state_dict(resume["scheduler"])
    start_epoch   = resume["epoch"] + 1
    best_val_loss = resume["best_val_loss"]
    no_improve    = resume.get("no_improve", 0)
    print(f"Resumed from epoch {resume['epoch']+1} | best_val_loss={best_val_loss:.4f} | no_improve={no_improve}")
elif args.resume:
    print("--resume requested but no resume.pt found. Starting fresh.")

print(f"\ndevice={DEVICE} | {len(train_dataset)} train / {len(val_dataset)} val")
print(f"batch={BATCH_SIZE} x accum={ACCUM_STEPS} = effective {BATCH_SIZE * ACCUM_STEPS}")
print(f"epochs={NUM_EPOCHS} (starting from {start_epoch+1}) | total optim steps={total_steps}")
print(f"Outputs -> {STAGE3_DIR}/\n")

# ── Train loop ────────────────────────────────────────────────────────────
for epoch in range(start_epoch, NUM_EPOCHS):

    generator.train()
    total_loss = 0.0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for step, (aligned_features, entity_vector, retrieved_texts, reports) in enumerate(loop):
        aligned_features = aligned_features.to(DEVICE)
        entity_vector    = entity_vector.to(DEVICE)

        prompts = [
            "Generate a detailed radiology report for the chest X-ray."
        ] * len(reports)

        # BFloat16 AMP — same exponent range as FP32, no NaN risk
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = generator(
                region_features=aligned_features,
                entity_vector=entity_vector,
                retrieved_texts=retrieved_texts,
                prompt_texts=prompts,
                target_texts=reports,
            )
            loss = loss / ACCUM_STEPS

        loss.backward()
        total_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        loop.set_postfix(
            loss=f"{total_loss / (step + 1):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    avg_train_loss = total_loss / len(train_loader)

    # ── Validation ────────────────────────────────────────────────────────
    generator.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        for val_af, val_ev, val_reps, val_reports in tqdm(val_loader, desc="  Val", leave=False):
            val_af = val_af.to(DEVICE)
            val_ev = val_ev.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                vl = generator(
                    region_features=val_af,
                    entity_vector=val_ev,
                    retrieved_texts=list(val_reps),
                    prompt_texts=["Generate a detailed radiology report for the chest X-ray."],
                    target_texts=list(val_reports),
                )
            val_loss_total += vl.item()

    avg_val_loss = val_loss_total / max(len(val_loader), 1)
    torch.cuda.empty_cache()

    print(
        f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
        f"train={avg_train_loss:.4f} | "
        f"val={avg_val_loss:.4f}"
    )

    # Always save last checkpoint
    torch.save(generator.state_dict(), LAST_CKPT)

    # Save full training state for resume
    torch.save({
        "epoch":          epoch,
        "model":          generator.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "best_val_loss":  best_val_loss,
        "no_improve":     no_improve,
    }, RESUME_FILE)

    # Save best checkpoint
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve    = 0
        torch.save(generator.state_dict(), BEST_CKPT)
        print(f"  ** Best model saved -> {BEST_CKPT}  (val={best_val_loss:.4f})")
    else:
        no_improve += 1
        print(f"  No improvement for {no_improve}/{patience} epochs")
        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

print("\nTraining Complete")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Best model    : {BEST_CKPT}")
print(f"Last model    : {LAST_CKPT}")
print(f"Resume state  : {RESUME_FILE}")
