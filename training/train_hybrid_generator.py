import os
import random
import torch
import torch.amp
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import pickle
import numpy as np
from transformers import get_cosine_schedule_with_warmup

from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier
from rag.hybrid_generator import HybridReportGenerator
from rag.verifier import ReportVerifier

# =====================================================
# CONFIG
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT            = "C:/Datasets/IU_Xray"
NUM_EPOCHS      = 20
BATCH_SIZE      = 4
ACCUM_STEPS     = 4          # effective batch = 16
RETRIEVAL_K     = 3          # sample from top-K to avoid report memorisation
WARMUP_STEPS    = 150
LR_T5           = 3e-5       # conservative LR for pretrained T5 layers
LR_NEW          = 3e-4       # aggressive LR for randomly-init layers
WEIGHT_DECAY    = 0.01
GRAD_CLIP       = 0.5
CHECKPOINT_OUT  = "rag/hybrid_generator.pth"
BEST_CKPT_OUT   = "rag/hybrid_generator_best.pth"

# =====================================================
# DATA
# =====================================================
train_dataset = IUXrayMultiViewDataset(ROOT, split="train")
train_loader  = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=True,
)

val_dataset  = IUXrayMultiViewDataset(ROOT, split="val")
# Subsample val to 100 random samples to cut ~44 min/epoch from validation
_val_indices = random.sample(range(len(val_dataset)), min(100, len(val_dataset)))
val_subset   = torch.utils.data.Subset(val_dataset, _val_indices)
val_loader   = DataLoader(val_subset, batch_size=1, shuffle=False)

# =====================================================
# LOAD FAISS + METADATA
# =====================================================

index = faiss.read_index("rag/faiss_index.bin")

with open("rag/train_reports.pkl", "rb") as f:
    train_metadata = pickle.load(f)

# =====================================================
# LOAD STAGE-1 BACKBONE + ALIGNMENT + PROJECTION (FROZEN)
# =====================================================

checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=device)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()
for p in visual_encoder.parameters():
    p.requires_grad = False

# CrossModalAlignment: image regions attend to retrieved report tokens
# Produces region-aligned features (B, 49, 256) shaped by report context
alignment = CrossModalAlignment().to(device)
alignment.load_state_dict(checkpoint["alignment"])
alignment.eval()
for p in alignment.parameters():
    p.requires_grad = False

proj_img = ProjectionHead().to(device)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()
for p in proj_img.parameters():
    p.requires_grad = False

# =====================================================
# LOAD IMAGE CLASSIFIER (FROZEN)
# =====================================================

image_classifier = SAEImageClassifier().to(device)
image_classifier.load_state_dict(
    torch.load("classification/image_classifier.pth", map_location=device)
)
image_classifier.eval()
for p in image_classifier.parameters():
    p.requires_grad = False

# =====================================================
# LOAD REPORT CLASSIFIER (FROZEN)
# Used for soft-AND entity verification during training
# =====================================================

report_classifier = ReportClassifier().to(device)
report_classifier.load_state_dict(
    torch.load("classification/report_classifier.pth", map_location=device)
)
report_classifier.eval()
for p in report_classifier.parameters():
    p.requires_grad = False

# =====================================================
# VERIFIER
# Scores retrieved reports by cross-modal attention;
# used here for diagnostic logging only (does not
# affect the training loss).
# =====================================================

verifier = ReportVerifier(alignment=alignment, min_score=0.0)

# =====================================================
# GENERATOR  (warm-start from existing checkpoint)
# =====================================================
generator = HybridReportGenerator().to(device)

if os.path.exists(CHECKPOINT_OUT):
    state = torch.load(CHECKPOINT_OUT, map_location=device, weights_only=False)
    # Strict=False: allows new keys (visual_norm, visual_drop) to init from scratch
    # while all matching keys (T5, visual_proj, entity_embed) get pretrained weights
    missing, unexpected = generator.load_state_dict(state, strict=False)
    print(f"Warm-started from {CHECKPOINT_OUT}")
    if missing:
        print(f"  New params (random init): {missing}")
else:
    print("No checkpoint found — training from scratch.")

# =====================================================
# OPTIMIZER  (differential learning rates)
# Pretrained T5 layers get a conservative LR;
# new projection / norm / entity layers get a higher LR
# so they can catch up faster.
# =====================================================
_new_layers = {"visual_proj", "visual_norm", "visual_drop", "entity_embed"}
t5_params  = [p for n, p in generator.named_parameters()
              if not any(k in n for k in _new_layers) and p.requires_grad]
new_params = [p for n, p in generator.named_parameters()
              if any(k in n for k in _new_layers) and p.requires_grad]

optimizer = torch.optim.AdamW(
    [
        {"params": t5_params,  "lr": LR_T5},
        {"params": new_params, "lr": LR_NEW},
    ],
    weight_decay=WEIGHT_DECAY,
)

# =====================================================
# SCHEDULER  (cosine decay with linear warmup)
# =====================================================
total_optim_steps = (len(train_loader) // ACCUM_STEPS) * NUM_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_optim_steps,
)

print(f"Training on {device} | {len(train_dataset)} train / {len(val_subset)} val (subsampled)")
print(f"Effective batch = {BATCH_SIZE * ACCUM_STEPS} | {NUM_EPOCHS} epochs")
print(f"Total optimiser steps: {total_optim_steps}")

# =====================================================
# TRAIN LOOP
# =====================================================
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    generator.train()
    total_loss  = 0.0
    optimizer.zero_grad()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for step, (images, reports) in enumerate(loop):

        images = images.to(device)

        with torch.no_grad():

            # ------------------------------------------
            # 1. Visual backbone: (B, 256, 7, 7)
            # ------------------------------------------
            visual_features = visual_encoder(images)

            # ------------------------------------------
            # 2. FAISS retrieval — sample from top-K for diversity
            #    Sampling from top-3 forces the model to synthesise
            #    rather than memorise / copy rank-1.
            # ------------------------------------------
            global_feat = visual_features.flatten(2).mean(dim=2)
            img_emb = proj_img(global_feat)
            img_np  = img_emb.cpu().numpy().astype("float32")
            D, I    = index.search(img_np, k=RETRIEVAL_K)

            retrieved_reports = []
            for b in range(images.size(0)):
                pick = random.randint(0, RETRIEVAL_K - 1)
                retrieved_reports.append(train_metadata[I[b][pick]]["report"])

            # ------------------------------------------
            # 3. Region-aligned features  (B, 49, 256)
            # ------------------------------------------
            aligned_features, _, _ = alignment(visual_features, retrieved_reports)

            # ------------------------------------------
            # 4. Soft entity predictions (B, 14) in [0, 1]
            # ------------------------------------------
            img_logits         = image_classifier(images)
            img_entities       = torch.sigmoid(img_logits)

            rep_logits         = report_classifier(retrieved_reports)
            rep_entities_batch = torch.sigmoid(rep_logits)

        # ------------------------------------------
        # 5. Soft-AND entity vector
        # ------------------------------------------
        verified_entities = img_entities * rep_entities_batch  # (B, 14)

        prompts = [
            "Generate a detailed radiology report for the chest X-ray."
            for _ in range(images.size(0))
        ]

        # ------------------------------------------
        # 6. Forward pass — BFloat16 AMP on generator only
        #    (BF16 has same exponent range as FP32, no NaN/overflow risk)
        # ------------------------------------------
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = generator(
                region_features=aligned_features,
                entity_vector=verified_entities,
                retrieved_texts=retrieved_reports,
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
            del loss
            torch.cuda.empty_cache()

        loop.set_postfix(
            loss=f"{total_loss / (step + 1):.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    avg_train_loss = total_loss / len(train_loader)

    # ==================================================
    # VALIDATION
    # ==================================================
    generator.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        for val_images, val_reports in tqdm(val_loader, desc="  Val", leave=False):
            val_images = val_images.to(device)

            vf       = visual_encoder(val_images)
            g_feat   = vf.flatten(2).mean(dim=2)
            v_np     = proj_img(g_feat).cpu().numpy().astype("float32")
            _, vI    = index.search(v_np, k=1)
            v_rep    = [train_metadata[int(vI[0][0])]["report"]]

            val_af, _, _ = alignment(vf, v_rep)

            vi_ent = torch.sigmoid(image_classifier(val_images))
            vr_ent = torch.sigmoid(report_classifier(v_rep))
            v_ent  = vi_ent * vr_ent

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                vl = generator(
                    region_features=val_af,
                    entity_vector=v_ent,
                    retrieved_texts=v_rep,
                    prompt_texts=["Generate a detailed radiology report for the chest X-ray."],
                    target_texts=val_reports,
                )
            val_loss_total += vl.item()

    avg_val_loss = val_loss_total / max(len(val_loader), 1)
    torch.cuda.empty_cache()

    print(
        f"Epoch {epoch+1:2d} | "
        f"train={avg_train_loss:.4f} | "
        f"val={avg_val_loss:.4f}"
    )

    torch.save(generator.state_dict(), CHECKPOINT_OUT)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(generator.state_dict(), BEST_CKPT_OUT)
        print(f"  ** Best model saved (val={best_val_loss:.4f})")

print("\nHybrid Training Complete")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Best model    : {BEST_CKPT_OUT}")