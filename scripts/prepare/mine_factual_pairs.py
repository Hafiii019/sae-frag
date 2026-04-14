"""
Mine factually-informed report pairs using RadGraph entity-F1 similarity.

Implements the Factual Report Pairs Mining step from FactMM-RAG (NAACL 2025)
equation 1:

    s(q_txt, d_txt) = 2 * |q_hat ∩ d_hat| / (len(q_hat) + len(d_hat))

where q_hat / d_hat are (token, label) entity tuples extracted by RadGraph.
This is F1RadGraph with reward_level='partial'.

Strategy (tractable on 6 GB GPU)
---------------------------------
1. Pre-filter candidate pairs using fast CheXbert Jaccard (threshold 0.1)
   to avoid O(N²) RadGraph calls.
2. For each candidate pair, compute RadGraph entity-F1 (the real FactMM-RAG
   similarity) from cached entity sets.
3. Keep pairs with entity-F1 ≥ delta and take top_k per query.

Output
------
    store/factual_pairs.pkl  -- dict {train_idx: [positive_train_idx, ...]}

Usage
-----
    python scripts/prepare/mine_factual_pairs.py
    python scripts/prepare/mine_factual_pairs.py --delta 0.3 --top_k 2
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import pickle
import sys
from typing import Dict, List, Set, Tuple

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
from rag.radgraph_extractor import RadGraphExtractor

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
    parser.add_argument("--delta",      type=float, default=0.3,
                        help="F1RadGraph entity-similarity threshold (FactMM-RAG eq.1).")
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

    # ── Step 2: Jaccard pre-filter (loose, threshold 0.10) ────────────────
    # Avoids running RadGraph on O(N²) pairs — only evaluate promising candidates.
    JACCARD_PREFILTER = 0.10
    log.info(
        f"Pre-filtering with Jaccard≥{JACCARD_PREFILTER} before F1RadGraph evaluation..."
    )
    CHUNK = 256
    # candidates_map[i] = list of j indices passing the Jaccard pre-filter
    candidates_map: Dict[int, List[int]] = {i: [] for i in range(N)}
    lm = label_matrix.astype(np.float32)

    for chunk_start in tqdm(range(0, N, CHUNK), desc="Jaccard pre-filter"):
        chunk_end = min(chunk_start + CHUNK, N)
        chunk     = lm[chunk_start:chunk_end]

        inter      = chunk @ lm.T
        union      = chunk.sum(axis=1, keepdims=True) + lm.sum(axis=1, keepdims=True).T - inter
        union_safe = np.where(union == 0, 1.0, union)
        sim        = inter / union_safe

        for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
            row            = sim[local_i].copy()
            row[global_i]  = -1.0
            hit            = np.where(row >= JACCARD_PREFILTER)[0].tolist()
            candidates_map[global_i] = hit

    total_candidates = sum(len(v) for v in candidates_map.values())
    log.info(f"  Total candidate pairs after pre-filter: {total_candidates}")

    # ── Step 3: RadGraph entity-F1 — FactMM-RAG eq.1 ─────────────────────
    #   s(q, d) = 2 * |q_hat ∩ d_hat| / (|q_hat| + |d_hat|)    [Dice F1]
    # Extract entities once per report (O(N) forward passes), then O(N candidate)
    # set operations — no O(N²) model calls.
    ENTITY_CACHE = os.path.join(_ROOT, "store", "radgraph_cache.json")
    log.info("Extracting RadGraph entities for all training reports (cached)...")
    extractor = RadGraphExtractor(cache_path=ENTITY_CACHE)

    # entity_sets[i] = frozenset of (token_lower, label) tuples
    entity_sets: List[frozenset] = []
    for report in tqdm(train_reports, desc="RadGraph extraction"):
        result   = extractor.extract(report)
        entities = result.get("entities", {})
        eset = frozenset(
            (str(v.get("tokens", "")).lower().strip(), str(v.get("label", "")))
            for v in entities.values()
            if isinstance(v, dict) and v.get("tokens")
        )
        entity_sets.append(eset)

    extractor.save_cache()
    using_real = extractor.using_radgraph
    log.info(
        f"  Extraction done. Mode: {'RadGraph (real NER+RE)' if using_real else 'keyword fallback'}"
    )

    # ── Step 4: Score candidates with F1RadGraph and collect top_k ────────
    log.info(f"Scoring with F1RadGraph (delta={args.delta}, top_k={args.top_k})...")
    factual_pairs: Dict[int, List[int]] = {i: [] for i in range(N)}

    for i in tqdm(range(N), desc="F1RadGraph pair scoring"):
        q_set = entity_sets[i]
        q_len = len(q_set)
        cands = candidates_map[i]
        if not cands or q_len == 0:
            continue

        scored = []
        for j in cands:
            d_set = entity_sets[j]
            d_len = len(d_set)
            if d_len == 0:
                continue
            inter = len(q_set & d_set)
            f1    = 2.0 * inter / (q_len + d_len)   # FactMM-RAG eq.1
            if f1 >= args.delta:
                scored.append((f1, j))

        scored.sort(key=lambda x: x[0], reverse=True)
        factual_pairs[i] = [j for _, j in scored[: args.top_k]]

    # ── Summary ───────────────────────────────────────────────────────────
    queries_with_pos = sum(1 for v in factual_pairs.values() if v)
    total_pairs      = sum(len(v) for v in factual_pairs.values())
    log.info(f"  Queries with ≥ 1 positive: {queries_with_pos}/{N}")
    log.info(f"  Total positive pairs      : {total_pairs}")
    log.info(f"  Avg pairs per query       : {total_pairs / max(queries_with_pos, 1):.1f}")
    if not using_real:
        log.warning(
            "  WARNING: RadGraph unavailable — entities from keyword fallback.\n"
            "  Pair quality is reduced.  Run: pip install radgraph>=0.1.18"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(factual_pairs, f)
    log.info(f"\nSaved -> {os.path.relpath(args.out, _ROOT)}")
    log.info("Next: python scripts/train/train_factual_retriever.py")


if __name__ == "__main__":
    main()
