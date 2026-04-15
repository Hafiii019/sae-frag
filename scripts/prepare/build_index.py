"""
Build the FAISS retrieval index for Stage 3 report generation.

Encodes all training-split images (and reports, when the factual retriever is
available) and stores them in a flat inner-product FAISS index.  Also
pre-computes RadGraph entity embeddings for the training reports.

If checkpoints/stage1/factual_retriever.pth exists, the index is built using
the FactMM-RAG document encoder (image + text via CrossModalAlignment).
Otherwise falls back to the Stage-1 image-only encoder.

Outputs  ->  store/
    faiss_index.bin                 FAISS flat inner-product index
    train_reports.pkl               per-sample metadata (report, entity_vector)
    radgraph_entity_embeddings.npy  (N, 768) ClinicalBERT embeddings
    radgraph_entity_labels.json     list of entity label lists

Usage
-----
    python scripts/prepare/build_index.py
"""

# ── Standard library ──────────────────────────────────────────────────────
import json
import logging
import os
import pickle
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from classification.report_labeler import ReportClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models import (
    CrossModalAlignment, DocumentProjectionHead,
    MultiViewBackbone, ProjectionHead,
)
from rag.radgraph_extractor import RadGraphExtractor

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FACTUAL_CKPT = os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth")
    STAGE1_CKPT  = os.path.join(_ROOT, "checkpoints", "stage1", "best.pth")
    use_factual  = os.path.exists(FACTUAL_CKPT)

    STORE_DIR = os.path.join(_ROOT, "store")
    os.makedirs(STORE_DIR, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = IUXrayMultiViewDataset(Config.DATA_ROOT, split="train")
    loader  = DataLoader(dataset, batch_size=1, num_workers=0)

    # ── Load encoder checkpoint ───────────────────────────────────────────
    ckpt_path = FACTUAL_CKPT if use_factual else STAGE1_CKPT
    if use_factual:
        log.info("Using FactMM-RAG document encoder (image + text).")
    else:
        log.info("Using Stage-1 image-only encoder (factual retriever not found).")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    visual_encoder = MultiViewBackbone().to(device)
    visual_encoder.load_state_dict(checkpoint["visual_model"])
    visual_encoder.eval()

    proj_img = ProjectionHead().to(device)
    proj_img.load_state_dict(checkpoint["proj_img"])
    proj_img.eval()

    alignment = proj_doc = None
    if use_factual:
        alignment = CrossModalAlignment().to(device)
        alignment.load_state_dict(checkpoint["alignment"])
        alignment.eval()

        proj_doc = DocumentProjectionHead().to(device)
        proj_doc.load_state_dict(checkpoint["proj_doc"])
        proj_doc.eval()

    report_classifier = ReportClassifier().to(device)
    report_classifier.load_state_dict(
        torch.load(
            os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"),
            map_location=device, weights_only=False,
        )
    )
    report_classifier.eval()

    # ── Phase 1: Build FAISS index ────────────────────────────────────────
    log.info(f"\nBuilding FAISS index over {len(dataset)} training samples...")

    embeddings: list = []
    metadata:   list = []

    with torch.no_grad():
        for images, reports, *_ in tqdm(loader, desc="Encoding"):
            images = images.to(device)
            feats  = visual_encoder(images)              # (1, 256, 14, 14)

            if use_factual:
                aligned, _, _ = alignment(feats, list(reports))
                emb = proj_doc(aligned.mean(dim=1))      # (1, 256) normalised
            else:
                global_feat = feats.flatten(2).mean(dim=2)
                emb = proj_img(global_feat)              # (1, 256)

            embeddings.append(emb.cpu().numpy())

            rep_entities = (
                torch.sigmoid(report_classifier(list(reports))) > 0.5
            ).float()
            metadata.append({
                "report":        reports[0],
                "entity_vector": rep_entities.cpu(),
            })

    emb_matrix = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(emb_matrix)

    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)

    faiss.write_index(index, os.path.join(STORE_DIR, "faiss_index.bin"))
    with open(os.path.join(STORE_DIR, "train_reports.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    log.info(f"FAISS index built — {index.ntotal} vectors, dim={emb_matrix.shape[1]}")

    # ── Phase 2: RadGraph entity embeddings ───────────────────────────────
    log.info("\nComputing RadGraph entity embeddings...")

    extractor = RadGraphExtractor(
        device=str(device),
        cache_path=os.path.join(STORE_DIR, "radgraph_cache.json"),
    )
    bert_encoder   = report_classifier.encoder
    bert_tokenizer = report_classifier.tokenizer

    radgraph_embeddings: list = []
    radgraph_labels:     list = []

    for sample in tqdm(metadata, desc="RadGraph"):
        result      = extractor.extract(sample["report"])
        entity_text = extractor.to_entity_text(result)
        emb         = extractor.to_entity_embedding(entity_text, bert_encoder, bert_tokenizer)
        radgraph_embeddings.append(emb.numpy())
        radgraph_labels.append([
            v["label"]
            for v in result.get("entities", {}).values()
            if isinstance(v, dict) and "label" in v
        ])

    extractor.save_cache()

    emb_np = np.vstack(radgraph_embeddings).astype("float32")
    np.save(os.path.join(STORE_DIR, "radgraph_entity_embeddings.npy"), emb_np)
    with open(os.path.join(STORE_DIR, "radgraph_entity_labels.json"), "w", encoding="utf-8") as f:
        json.dump(radgraph_labels, f, indent=2, ensure_ascii=False)

    log.info(f"RadGraph embeddings saved — shape: {emb_np.shape}")
    log.info("\nNext: python scripts/prepare/cache_features.py")


if __name__ == "__main__":
    main()
