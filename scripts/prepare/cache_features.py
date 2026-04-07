"""
Pre-compute and cache frozen model outputs for all train/val samples.

Run once before train_stage3.py.  All BERT, alignment, and classifier
inference is done here so the Stage-3 training loop only loads a generator.

If checkpoints/stage1/factual_retriever.pth exists the cache is built with
the FactMM-RAG-trained encoder for better retrieval quality.

Outputs  ->  store/
    cache_train.pt   2695 samples, each with 5 retrieval variants
    cache_val.pt     385 samples, each with 1 retrieval variant

Usage
-----
    python scripts/prepare/cache_features.py
"""

# ── Standard library ──────────────────────────────────────────────────────
import logging
import os
import pickle
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import faiss
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models import CrossModalAlignment, MultiViewBackbone, ProjectionHead

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _build_cache(
    split: str,
    out_path: str,
    k_variants: int,
    device: torch.device,
    visual_encoder: torch.nn.Module,
    alignment: torch.nn.Module,
    proj_img: torch.nn.Module,
    image_classifier: torch.nn.Module,
    report_classifier: torch.nn.Module,
    index: faiss.Index,
    train_metadata: list,
    batch_size: int = 8,
) -> None:
    dataset = IUXrayMultiViewDataset(Config.DATA_ROOT, split=split)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cache   = []

    with torch.no_grad():
        for images, reports in tqdm(loader, desc=f"Caching {split}"):
            images = images.to(device)
            B      = images.size(0)

            visual_features = visual_encoder(images)                   # (B, 256, 14, 14)
            global_feat     = visual_features.flatten(2).mean(dim=2)   # (B, 256)
            img_emb         = proj_img(global_feat)                    # (B, 256)
            img_np          = img_emb.cpu().numpy().astype("float32")
            faiss.normalize_L2(img_np)

            _, I = index.search(img_np, k=k_variants)                  # (B, k)

            img_entities = torch.sigmoid(image_classifier(images))     # (B, 14)

            per_rank = []
            for k in range(k_variants):
                reps_k     = [train_metadata[I[b][k]]["report"] for b in range(B)]
                af_k, _, _ = alignment(visual_features, reps_k)        # (B, 196, 256)
                rep_ent_k  = torch.sigmoid(report_classifier(reps_k))  # (B, 14)
                ev_k       = img_entities * rep_ent_k                  # (B, 14) soft-AND
                per_rank.append((af_k.cpu().float(), ev_k.cpu().float(), reps_k))

            for b in range(B):
                cache.append({
                    "variants": [
                        {
                            "aligned_features": per_rank[k][0][b],  # (196, 256)
                            "entity_vector":    per_rank[k][1][b],  # (14,)
                            "retrieved_text":   per_rank[k][2][b],
                        }
                        for k in range(k_variants)
                    ],
                    "target": reports[b],
                })

    torch.save(cache, out_path)
    log.info(f"Saved {len(cache)} samples -> {os.path.relpath(out_path, _ROOT)}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FACTUAL_CKPT = os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth")
    STAGE1_CKPT  = os.path.join(_ROOT, "checkpoints", "stage1", "best.pth")
    use_factual  = os.path.exists(FACTUAL_CKPT)

    if use_factual:
        log.info("Using FactMM-RAG factual retriever checkpoint.")
        ckpt = torch.load(FACTUAL_CKPT, map_location=device, weights_only=False)
    else:
        log.info("Using Stage-1 checkpoint (factual retriever not found).")
        ckpt = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)

    # ── Load frozen models ─────────────────────────────────────────────────
    visual_encoder = MultiViewBackbone().to(device)
    visual_encoder.load_state_dict(ckpt["visual_model"])
    visual_encoder.eval()

    alignment = CrossModalAlignment().to(device)
    alignment.load_state_dict(ckpt["alignment"])
    alignment.eval()

    proj_img = ProjectionHead().to(device)
    proj_img.load_state_dict(ckpt["proj_img"])
    proj_img.eval()

    image_classifier = SAEImageClassifier().to(device)
    image_classifier.load_state_dict(
        torch.load(
            os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"),
            map_location=device, weights_only=False,
        ),
        strict=False,
    )
    image_classifier.eval()

    report_classifier = ReportClassifier().to(device)
    report_classifier.load_state_dict(
        torch.load(
            os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"),
            map_location=device, weights_only=False,
        )
    )
    report_classifier.eval()

    index = faiss.read_index(os.path.join(_ROOT, "store", "faiss_index.bin"))
    with open(os.path.join(_ROOT, "store", "train_reports.pkl"), "rb") as f:
        train_metadata = pickle.load(f)

    log.info("All models loaded.\n")

    # ── Build caches ───────────────────────────────────────────────────────
    store = os.path.join(_ROOT, "store")
    kwargs = dict(
        device=device,
        visual_encoder=visual_encoder,
        alignment=alignment,
        proj_img=proj_img,
        image_classifier=image_classifier,
        report_classifier=report_classifier,
        index=index,
        train_metadata=train_metadata,
    )

    _build_cache("train", os.path.join(store, "cache_train.pt"), k_variants=5, **kwargs)
    _build_cache("val",   os.path.join(store, "cache_val.pt"),   k_variants=1, **kwargs)

    log.info("\nDone. Next: python scripts/train/train_stage3.py")


if __name__ == "__main__":
    main()
