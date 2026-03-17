"""Pre-compute and cache frozen model outputs for all train/val samples.

Run ONCE before using train_cached.py.
Estimated time: ~60-90 minutes (one-time cost).

Outputs
-------
rag/cache_train.pt  — 2695 items each with 3 retrieval variants
rag/cache_val.pt    — 385 items each with 1 retrieval variant
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pickle

import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from data.dataset import IUXrayMultiViewDataset
from models.alignment import CrossModalAlignment
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT        = "C:/Datasets/IU_Xray"
RETRIEVAL_K = 5
BATCH_SIZE  = 8   # larger batch — no grad, pure inference

# ── Load frozen models ────────────────────────────────────────────────────
print("Loading frozen models...")
ckpt = torch.load(os.path.join(_ROOT, "checkpoints", "stage1", "best.pth"), map_location=DEVICE, weights_only=False)

visual_encoder = MultiViewBackbone().to(DEVICE)
visual_encoder.load_state_dict(ckpt["visual_model"])
visual_encoder.eval()

alignment = CrossModalAlignment().to(DEVICE)
alignment.load_state_dict(ckpt["alignment"])
alignment.eval()

proj_img = ProjectionHead().to(DEVICE)
proj_img.load_state_dict(ckpt["proj_img"])
proj_img.eval()

image_classifier = SAEImageClassifier().to(DEVICE)
image_classifier.load_state_dict(
    torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"), map_location=DEVICE, weights_only=False)
)
image_classifier.eval()

report_classifier = ReportClassifier().to(DEVICE)
report_classifier.load_state_dict(
    torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"), map_location=DEVICE, weights_only=False)
)
report_classifier.eval()

index = faiss.read_index(os.path.join(_ROOT, "store", "faiss_index.bin"))
with open(os.path.join(_ROOT, "store", "train_reports.pkl"), "rb") as f:
    train_metadata = pickle.load(f)

print("All models loaded.")


# ── Cache builder ─────────────────────────────────────────────────────────
def build_cache(split, out_path, k_variants):
    dataset = IUXrayMultiViewDataset(ROOT, split=split)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    cache   = []

    with torch.no_grad():
        for images, reports in tqdm(loader, desc=f"Caching {split}"):
            images = images.to(DEVICE)
            B      = images.size(0)

            visual_features = visual_encoder(images)                        # (B,256,7,7)
            global_feat     = visual_features.flatten(2).mean(dim=2)        # (B,256)
            img_emb         = proj_img(global_feat)                         # (B,768)
            img_np          = img_emb.cpu().numpy().astype("float32")

            _, I = index.search(img_np, k=k_variants)                       # (B, k)

            img_entities = torch.sigmoid(image_classifier(images))          # (B,14)

            # Run alignment + report_classifier once per retrieval rank
            # (i.e. batch-level, not per-sample) — much faster
            per_rank = []
            for k in range(k_variants):
                reps_k     = [train_metadata[I[b][k]]["report"] for b in range(B)]
                af_k, _, _ = alignment(visual_features, reps_k)             # (B,49,256)
                rep_ent_k  = torch.sigmoid(report_classifier(reps_k))       # (B,14)
                ev_k       = img_entities * rep_ent_k                       # (B,14) soft-AND
                per_rank.append((af_k.cpu().float(), ev_k.cpu().float(), reps_k))

            for b in range(B):
                variants = []
                for k in range(k_variants):
                    af_k, ev_k, reps_k = per_rank[k]
                    variants.append({
                        "aligned_features": af_k[b],   # (49,256) fp32
                        "entity_vector":    ev_k[b],   # (14,)
                        "retrieved_text":   reps_k[b],
                    })
                cache.append({
                    "variants": variants,
                    "target":   reports[b],
                })

    torch.save(cache, out_path)
    print(f"Saved {len(cache)} samples -> {out_path}")


# ── Build both splits ─────────────────────────────────────────────────────
build_cache("train", os.path.join(_ROOT, "store", "cache_train.pt"), k_variants=RETRIEVAL_K)
build_cache("val",   os.path.join(_ROOT, "store", "cache_val.pt"),   k_variants=1)

print("\nDone. Now run:")
print("  python scripts/train/train_stage3.py")
