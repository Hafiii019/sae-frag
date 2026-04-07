"""
Mine factually-informed report pairs using CheXbert label similarity.

Implements the Factual Report Pairs Mining step from FactMM-RAG (NAACL 2025).
Uses the trained ReportClassifier (Bio_ClinicalBERT -> 14 CheXpert labels) to
produce label vectors for each training report, then computes pairwise Jaccard
similarity over predicted positive labels.

Output
------
    store/factual_pairs.pkl  -- dict {train_idx: [positive_train_idx, ...]}

Usage
-----
    python scripts/prepare/mine_factual_pairs.py
    python scripts/prepare/mine_factual_pairs.py --delta 0.5 --top_k 2
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import pickle
import sys
from typing import Dict, List

# ── Third-party ───────────────────────────────────────────────────────────
import numpy as np
import torch
from tqdm import tqdm

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from classification.report_labeler import ReportClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine factual report pairs for retriever training."
    )
    parser.add_argument("--delta",      type=float, default=0.5,
                        help="Label-similarity threshold for positive pairs.")
    parser.add_argument("--top_k",      type=int,   default=2,
                        help="Max positive pairs per query.")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--out",        type=str,
                        default=os.path.join(_ROOT, "store", "factual_pairs.pkl"))
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load training reports ─────────────────────────────────────────────
    log.info("Loading IU X-Ray training split...")
    dataset       = IUXrayMultiViewDataset(Config.DATA_ROOT, split="train")
    train_reports: List[str] = [s["report"] for s in dataset.samples]
    N = len(train_reports)
    log.info(f"  {N} training reports loaded.")

    # ── Load ReportClassifier ─────────────────────────────────────────────
    CLASSIFIER_PATH = os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth")
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"ReportClassifier not found: {CLASSIFIER_PATH}\n"
            "Run: python scripts/train/train_report_classifier.py"
        )

    report_classifier = ReportClassifier().to(DEVICE)
    report_classifier.load_state_dict(
        torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False)
    )
    report_classifier.eval()

    # ── Batch-predict label vectors ───────────────────────────────────────
    log.info("Predicting CheXpert label vectors for all training reports...")
    label_matrix = np.zeros((N, 14), dtype=np.int8)

    with torch.no_grad():
        for start in tqdm(range(0, N, args.batch_size), desc="Label inference"):
            batch  = train_reports[start : start + args.batch_size]
            logits = report_classifier(batch)
            preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.int8)
            label_matrix[start : start + len(batch)] = preds

    pos_rate = label_matrix.mean(axis=0)
    log.info(f"  Label positive rates: {np.round(pos_rate, 3)}")

    # ── Pairwise Jaccard similarity (vectorised, chunked for memory) ──────
    log.info(f"Mining positive pairs (delta={args.delta}, top_k={args.top_k})...")

    CHUNK = 256
    factual_pairs: Dict[int, List[int]] = {i: [] for i in range(N)}
    lm = label_matrix.astype(np.float32)

    for chunk_start in tqdm(range(0, N, CHUNK), desc="Similarity matrix"):
        chunk_end = min(chunk_start + CHUNK, N)
        chunk     = lm[chunk_start:chunk_end]

        inter      = chunk @ lm.T
        union      = chunk.sum(axis=1, keepdims=True) + lm.sum(axis=1, keepdims=True).T - inter
        union_safe = np.where(union == 0, 1.0, union)
        sim        = inter / union_safe

        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            row         = sim[local_i].copy()
            row[global_i] = -1.0                        # exclude self
            candidates  = np.where(row >= args.delta)[0]
            if len(candidates) == 0:
                continue
            sorted_idx               = candidates[np.argsort(row[candidates])[::-1]]
            factual_pairs[global_i]  = sorted_idx[: args.top_k].tolist()

    # ── Summary ───────────────────────────────────────────────────────────
    queries_with_pos = sum(1 for v in factual_pairs.values() if v)
    total_pairs      = sum(len(v) for v in factual_pairs.values())
    log.info(f"  Queries with >= 1 positive: {queries_with_pos}/{N}")
    log.info(f"  Total positive pairs      : {total_pairs}")
    log.info(f"  Avg pairs per query       : {total_pairs / max(queries_with_pos, 1):.1f}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(factual_pairs, f)
    log.info(f"\nSaved -> {os.path.relpath(args.out, _ROOT)}")
    log.info("Next: python scripts/train/train_factual_retriever.py")


if __name__ == "__main__":
    main()
