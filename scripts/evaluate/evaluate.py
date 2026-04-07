"""
Evaluation script for SAE-FRAG radiology report generation.

Metrics
-------
NLG     : BLEU-1/2/3/4, METEOR, ROUGE-1/2/L, CIDEr
Clinical: CheXBert Label F1 (micro/macro/per-class), Entity F1

Outputs
-------
results/metrics.json            -- all scalar scores
results/generated_reports.json  -- per-sample reference / generated / uid

Usage
-----
    python scripts/evaluate/evaluate.py
"""

# ── Standard library ──────────────────────────────────────────────────────
import json
import logging
import math
import os
import sys
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────────────
import faiss
import nltk
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
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
from rag.hybrid_generator import HybridReportGenerator
from rag.radgraph_extractor import RadGraphExtractor
from rag.verifier import ReportVerifier
from utils.clinical_metrics import ClinicalMetrics

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# =============================================================================
# Helpers
# =============================================================================

def _load_checkpoint(device: torch.device) -> tuple:
    """Return (state_dict, path) for the best available Stage-1 checkpoint."""
    candidates = [
        os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth"),
        os.path.join(_ROOT, "checkpoints", "stage1", "best.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return torch.load(path, map_location=device, weights_only=False), path
    raise FileNotFoundError(
        "No Stage-1 checkpoint found. Run: python scripts/train/train_stage1.py"
    )


def _load_generator(device: torch.device) -> torch.nn.Module:
    """Load the fine-tuned report generator from the best available checkpoint."""
    candidates = [
        os.path.join(_ROOT, "checkpoints", "stage3", "best_generator.pth"),
        os.path.join(_ROOT, "checkpoints", "stage3", "last_generator.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            model = HybridReportGenerator().to(device)
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=False),
                strict=False,
            )
            model.eval()
            return model
    raise FileNotFoundError(
        "No Stage-3 checkpoint found. Run: python scripts/train/train_stage3.py"
    )


def _compute_cider(references: list, hypotheses: list) -> float:
    """
    TF-IDF weighted n-gram consensus score (CIDEr-D, n=1..4).
    Vedantam et al., CVPR 2015.
    """
    def _ngrams(tokens, n):
        return Counter(zip(*[tokens[i:] for i in range(n)]))

    def _tfidf(counts, df, N):
        total = sum(counts.values()) + 1e-9
        return {
            ng: (cnt / total) * math.log((N + 1.0) / (df.get(ng, 0) + 1.0))
            for ng, cnt in counts.items()
        }

    N = len(references)
    order_scores = []

    for n in range(1, 5):
        df: Counter = Counter()
        for refs in references:
            for ref in refs:
                df.update(set(_ngrams(ref, n).keys()))

        sample_scores = []
        for refs, hyp in zip(references, hypotheses):
            h_vec  = _tfidf(_ngrams(hyp, n), df, N)
            r_vecs = [_tfidf(_ngrams(r, n), df, N) for r in refs]

            all_keys = set(h_vec) | {k for rv in r_vecs for k in rv}
            r_mean   = {k: np.mean([rv.get(k, 0.0) for rv in r_vecs]) for k in all_keys}

            dot    = sum(h_vec.get(k, 0.0) * r_mean.get(k, 0.0) for k in all_keys)
            norm_h = math.sqrt(sum(v ** 2 for v in h_vec.values()) + 1e-9)
            norm_r = math.sqrt(sum(v ** 2 for v in r_mean.values()) + 1e-9)
            sample_scores.append(dot / (norm_h * norm_r))

        order_scores.append(float(np.mean(sample_scores)))

    return round(float(np.mean(order_scores)) * 10, 4)  # standard x10 scale


def _print_results(results: dict) -> None:
    """Print a formatted results summary."""
    W = 55

    def section(title, keys):
        log.info(f"\n  {title}")
        for k in keys:
            if k in results:
                log.info(f"    {k:<32} {results[k]}")

    log.info("\n" + "=" * W)
    log.info("  EVALUATION RESULTS")
    log.info("=" * W)

    section(
        "NLG Metrics",
        ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
         "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "CIDEr"],
    )
    section(
        "Retrieval Quality",
        ["verify_score_mean", "verify_score_median"],
    )
    section(
        "CheXBert Label F1",
        ["chexbert_f1_micro", "chexbert_f1_macro",
         "chexbert_precision_micro", "chexbert_recall_micro"],
    )
    section(
        "Entity F1",
        ["entity_precision", "entity_recall", "entity_f1"],
    )

    log.info("\n  Per-class CheXBert F1:")
    for k, v in results.items():
        if k.startswith("chexbert_f1_") and k not in (
            "chexbert_f1_micro", "chexbert_f1_macro"
        ):
            log.info(f"    {k:<40} {v}")

    if results.get("radgraph_mode") == "keyword_fallback":
        log.info(
            "\n  WARNING: Entity F1 computed via keyword fallback "
            "(RadGraph model unavailable)."
        )

    log.info("=" * W)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # ── Dataset ───────────────────────────────────────────────────────────
    test_dataset = IUXrayMultiViewDataset(Config.DATA_ROOT, split="test")
    test_loader  = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
    )

    # ── FAISS index + training report metadata ────────────────────────────
    faiss_path = os.path.join(_ROOT, "store", "faiss_index.bin")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"FAISS index not found: {faiss_path}\n"
            "Run: python scripts/prepare/build_index.py"
        )
    index          = faiss.read_index(faiss_path)
    train_metadata = torch.load(
        os.path.join(_ROOT, "store", "train_reports.pkl"), weights_only=False
    )

    # ── Models ────────────────────────────────────────────────────────────
    checkpoint, ckpt_path = _load_checkpoint(device)
    log.info(f"Checkpoint : {os.path.relpath(ckpt_path, _ROOT)}")

    visual_encoder = MultiViewBackbone().to(device)
    visual_encoder.load_state_dict(checkpoint["visual_model"], strict=False)
    visual_encoder.eval()

    alignment = CrossModalAlignment().to(device)
    alignment.load_state_dict(checkpoint["alignment"], strict=False)
    alignment.eval()

    proj_img = ProjectionHead().to(device)
    proj_img.load_state_dict(checkpoint["proj_img"])
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

    generator = _load_generator(device)

    verifier = ReportVerifier(alignment=alignment, min_score=0.0)

    # ── Metric objects ────────────────────────────────────────────────────
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    radgraph_extractor = RadGraphExtractor(
        cache_path=os.path.join(_ROOT, "store", "radgraph_cache.json")
    )
    clinical_metrics = ClinicalMetrics(
        report_classifier=report_classifier,
        radgraph_extractor=radgraph_extractor,
        threshold=0.5,
        device=device,
    )

    # ── Accumulators ─────────────────────────────────────────────────────
    all_references: list = []
    all_hypotheses: list = []
    meteor_scores:  list = []
    rouge1_scores:  list = []
    rouge2_scores:  list = []
    rougeL_scores:  list = []
    verify_scores:  list = []
    generated_log:  list = []

    # ── Evaluation loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for sample_idx, (images, reports) in enumerate(
            tqdm(test_loader, desc="Evaluating", unit="sample")
        ):
            images    = images.to(device)
            reference = reports[0]

            # Visual features: (1, 256, 14, 14) — 196 spatial tokens at P4
            visual_features = visual_encoder(images)

            # FAISS retrieval — top-5 candidates
            global_feat = visual_features.flatten(2).mean(dim=2)
            img_emb     = proj_img(global_feat)
            img_np      = img_emb.cpu().float().numpy()
            faiss.normalize_L2(img_np)
            _, I = index.search(img_np, k=5)

            candidates = [train_metadata[int(i)]["report"] for i in I[0]]

            # Verifier — pick best candidate via cross-modal cosine score
            retrieved_report, verify_score = verifier.verify(visual_features, candidates)
            verify_scores.append(verify_score)

            # Multi-retrieved context: verified best first + FAISS top-3
            top3    = [train_metadata[int(i)]["report"] for i in I[0][:3]]
            others  = [r for r in top3 if r != retrieved_report][:2]
            context = " [SEP] ".join([retrieved_report] + others)
            rag_context = HybridReportGenerator.build_rag_retrieved_text([context])[0]

            # Region-aligned features: image regions attend to retrieved report
            # Output: (1, 196, 256)
            aligned_features, _, _ = alignment(visual_features, [retrieved_report])

            # Soft-AND entity verification (image x text classifiers)
            img_entities  = torch.sigmoid(image_classifier(images))
            rep_entities  = torch.sigmoid(report_classifier([retrieved_report]))
            verified_ents = img_entities * rep_entities

            # Generate report
            prompt    = HybridReportGenerator.build_entity_prompt(verified_ents.cpu())
            generated = generator(
                region_features=aligned_features,
                entity_vector=verified_ents,
                retrieved_texts=[rag_context],
                prompt_texts=prompt,
                target_texts=None,
            )[0]

            # Accumulate NLG metrics
            ref_tokens = reference.split()
            gen_tokens = generated.split()

            all_references.append([ref_tokens])
            all_hypotheses.append(gen_tokens)
            meteor_scores.append(meteor_score([ref_tokens], gen_tokens))

            rouge_out = rouge_scorer_obj.score(reference, generated)
            rouge1_scores.append(rouge_out["rouge1"].fmeasure)
            rouge2_scores.append(rouge_out["rouge2"].fmeasure)
            rougeL_scores.append(rouge_out["rougeL"].fmeasure)

            clinical_metrics.update(reference, generated)

            generated_log.append({
                "sample_idx":   sample_idx,
                "uid":          str(test_dataset.samples[sample_idx]["uid"]),
                "reference":    reference,
                "generated":    generated,
                "retrieved":    retrieved_report,
                "verify_score": round(float(verify_score), 4),
            })

    # ── Compute aggregate scores ──────────────────────────────────────────
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(1/3, 1/3, 1/3, 0))
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    cider = _compute_cider(all_references, all_hypotheses)

    clinical_scores = clinical_metrics.compute()
    radgraph_extractor.save_cache()

    results = {
        "BLEU-1":  round(bleu1, 4),
        "BLEU-2":  round(bleu2, 4),
        "BLEU-3":  round(bleu3, 4),
        "BLEU-4":  round(bleu4, 4),
        "METEOR":  round(float(np.mean(meteor_scores)), 4),
        "ROUGE-1": round(float(np.mean(rouge1_scores)), 4),
        "ROUGE-2": round(float(np.mean(rouge2_scores)), 4),
        "ROUGE-L": round(float(np.mean(rougeL_scores)), 4),
        "CIDEr":   cider,
        "verify_score_mean":   round(float(np.mean(verify_scores)),   4),
        "verify_score_median": round(float(np.median(verify_scores)), 4),
        **clinical_scores,
        "num_test_samples": len(all_hypotheses),
        "radgraph_mode":    "radgraph" if radgraph_extractor.using_radgraph else "keyword_fallback",
    }

    _print_results(results)

    # ── Save outputs ──────────────────────────────────────────────────────
    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    reports_path = os.path.join(out_dir, "generated_reports.json")
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(generated_log, f, indent=2, ensure_ascii=False)

    log.info(f"\nSaved  metrics -> {os.path.relpath(metrics_path, _ROOT)}")
    log.info(f"Saved  reports -> {os.path.relpath(reports_path, _ROOT)}")


if __name__ == "__main__":
    main()
"""Evaluation script for SAE-FRAG radiology report generation.

Metrics reported
----------------
NLG (standard for report generation papers):
  BLEU-1 / 2 / 3 / 4   corpus-level (Papineni et al., 2002)
  METEOR                sentence-level, corpus-averaged
  ROUGE-1 / 2 / L       F-measure, corpus-averaged
  CIDEr                 consensus-based image description evaluation

Clinical (standard for chest X-ray report generation):
  CheXBert Label F1     micro and macro F1 over 14 CheXpert pathology labels;
                        matches the CheXBert metric in R2Gen, RGRG, FactMM-RAG
  Entity F1             precision / recall / F1 over RadGraph entity sets;
                        falls back to keyword matching if RadGraph unavailable

Outputs
-------
  results/metrics.json            all scalar metrics
  results/generated_reports.json  reference / generated / scores per sample

Usage
-----
  python scripts/evaluate/evaluate.py
"""

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import faiss
import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models import CrossModalAlignment, MultiViewBackbone, ProjectionHead
from rag.hybrid_generator import HybridReportGenerator
from rag.radgraph_extractor import RadGraphExtractor
from rag.verifier import ReportVerifier
from utils.clinical_metrics import ClinicalMetrics

try:
    from nltk.translate.bleu_score import corpus_bleu as _
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4",  quiet=True)
except Exception:
    pass

# ── Device ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}")

# ── Test dataset ──────────────────────────────────────────────────────────
test_dataset = IUXrayMultiViewDataset(Config.DATA_ROOT, split="test")
test_loader  = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=USE_AMP,
)
print(f"Test set: {len(test_dataset)} samples")

# ── FAISS index + report database ─────────────────────────────────────────
_faiss_path   = os.path.join(_ROOT, "store", "faiss_index.bin")
_reports_path = os.path.join(_ROOT, "store", "train_reports.pkl")

if not os.path.exists(_faiss_path):
    raise FileNotFoundError(
        f"FAISS index not found: {_faiss_path}\n"
        "Run: python scripts/prepare/build_index.py"
    )
index          = faiss.read_index(_faiss_path)
train_metadata = torch.load(_reports_path, weights_only=False)

# ── Stage-1 checkpoint: factual retriever > stage-1 best ──────────────────
_factual_ckpt = os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth")
_stage1_ckpt  = os.path.join(_ROOT, "checkpoints", "stage1", "best.pth")
_ckpt_path    = _factual_ckpt if os.path.exists(_factual_ckpt) else _stage1_ckpt
if not os.path.exists(_ckpt_path):
    raise FileNotFoundError(
        f"No Stage-1 checkpoint found.\n  Factual: {_factual_ckpt}\n  Stage-1: {_stage1_ckpt}"
    )
print(f"Checkpoint: {os.path.basename(_ckpt_path)}")
checkpoint = torch.load(_ckpt_path, map_location=device, weights_only=False)

# ── Visual encoder ────────────────────────────────────────────────────────
# Output: (B, 256, 14, 14) — 196 spatial tokens at P4 resolution
visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"], strict=False)
visual_encoder.eval()

# ── Cross-modal alignment (ClinicalBERT cross-attention) ──────────────────
# image_features (B, 256, 14, 14) × report tokens → aligned (B, 196, 256)
alignment = CrossModalAlignment().to(device)
alignment.load_state_dict(checkpoint["alignment"], strict=False)
alignment.eval()

# ── Projection head (query encoder for FAISS search) ─────────────────────
proj_img = ProjectionHead().to(device)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()

# ── Image classifier (14 CheXpert labels) ────────────────────────────────
image_classifier = SAEImageClassifier().to(device)
image_classifier.load_state_dict(
    torch.load(
        os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"),
        map_location=device, weights_only=False,
    ),
    strict=False,
)
image_classifier.eval()

# ── Report classifier (14 CheXpert labels from text) ─────────────────────
report_classifier = ReportClassifier().to(device)
report_classifier.load_state_dict(
    torch.load(
        os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"),
        map_location=device, weights_only=False,
    )
)
report_classifier.eval()

# ── Report generator (flan-t5-large) ─────────────────────────────────────
generator = HybridReportGenerator().to(device)
_gen_path = os.path.join(_ROOT, "checkpoints", "stage3", "best_generator.pth")
if not os.path.exists(_gen_path):
    _gen_path = os.path.join(_ROOT, "checkpoints", "stage3", "last_generator.pth")
if not os.path.exists(_gen_path):
    raise FileNotFoundError(
        "No Stage-3 checkpoint found. Run: python scripts/train/train_stage3.py"
    )
print(f"Generator: {os.path.basename(_gen_path)}")
generator.load_state_dict(
    torch.load(_gen_path, map_location=device, weights_only=False),
    strict=False,
)
generator.eval()

# ── Verifier (re-rank FAISS candidates by cross-modal cosine score) ───────
verifier = ReportVerifier(alignment=alignment, min_score=0.0)

# ── Clinical metrics ──────────────────────────────────────────────────────
radgraph_extractor = RadGraphExtractor(
    cache_path=os.path.join(_ROOT, "store", "radgraph_cache.json")
)
clinical_metrics = ClinicalMetrics(
    report_classifier=report_classifier,
    radgraph_extractor=radgraph_extractor,
    threshold=0.5,
    device=device,
)

# ── NLG metric accumulators ───────────────────────────────────────────────
rouge   = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Lists for corpus-level aggregation
all_references: list = []   # [[ref_tokens], ...]
all_hypotheses: list = []   # [hyp_tokens, ...]
meteor_scores:  list = []
rouge1_scores:  list = []
rouge2_scores:  list = []
rougeL_scores:  list = []
verify_scores:  list = []

# Detailed per-sample log for thesis error analysis
generated_log: list = []

print(f"\nEvaluating on {len(test_dataset)} test samples...")

# ── Evaluation loop ───────────────────────────────────────────────────────
with torch.no_grad():
    for sample_idx, (images, reports) in enumerate(tqdm(test_loader)):
        images    = images.to(device)
        reference = reports[0]   # str (batch_size=1)

        # ── 1. Visual features: (1, 256, 14, 14) — 196 spatial tokens ─
        visual_features = visual_encoder(images)

        # ── 2. FAISS retrieval — top-5 candidates for re-ranking ──────
        global_feat = visual_features.flatten(2).mean(dim=2)      # (1, 256)
        img_emb     = proj_img(global_feat)                        # (1, 256) L2-normed
        img_np      = img_emb.cpu().float().numpy()
        faiss.normalize_L2(img_np)
        _D, I = index.search(img_np, k=5)                         # (1, 5)

        candidate_reports = [train_metadata[int(i)]["report"] for i in I[0]]

        # ── 3. Verifier: re-rank by cross-modal cosine similarity ─────
        #    Batches all 5 candidates in one forward pass (score_batch)
        retrieved_report, verify_score = verifier.verify(
            visual_features, candidate_reports
        )
        verify_scores.append(verify_score)

        # ── 4. Multi-retrieved context (top-3, verified best first) ───
        #    Following FactMM-RAG Appendix A: concatenate with [SEP]
        top3_reports  = [train_metadata[int(i)]["report"] for i in I[0][:3]]
        # Deduplicate: put verified best first, then remaining FAISS top-2
        other_reports = [r for r in top3_reports if r != retrieved_report][:2]
        multi_retrieved = " [SEP] ".join([retrieved_report] + other_reports)

        # Wrap in FactMM-RAG prompt template
        rag_context = HybridReportGenerator.build_rag_retrieved_text([multi_retrieved])[0]

        # ── 5. Region-aligned features: image attends to retrieved text ─
        #    (1, 196, 256) — image regions guided by retrieved report
        aligned_features, _, _ = alignment(visual_features, [retrieved_report])

        # ── 6. Soft-AND entity predictions ────────────────────────────
        #    Image classifier + report classifier both predict 14 labels.
        #    Element-wise product keeps only findings confirmed by BOTH.
        img_entities = torch.sigmoid(image_classifier(images))         # (1, 14)
        rep_entities = torch.sigmoid(report_classifier([retrieved_report]))  # (1, 14)
        verified_entities = img_entities * rep_entities                # (1, 14)

        # ── 7. Entity-informed prompt ──────────────────────────────────
        prompt = HybridReportGenerator.build_entity_prompt(verified_entities.cpu())

        # ── 8. Generate report ─────────────────────────────────────────
        generated = generator(
            region_features=aligned_features,    # (1, 196, 256)
            entity_vector=verified_entities,     # (1, 14)
            retrieved_texts=[rag_context],        # list[str]
            prompt_texts=prompt,                  # list[str]
            target_texts=None,
        )[0]

        # ── Accumulate NLG metrics ─────────────────────────────────────
        ref_tokens = reference.split()
        gen_tokens = generated.split()

        all_references.append([ref_tokens])   # corpus_bleu expects list-of-list
        all_hypotheses.append(gen_tokens)

        meteor_scores.append(meteor_score([ref_tokens], gen_tokens))

        rouge_out = rouge.score(reference, generated)
        rouge1_scores.append(rouge_out["rouge1"].fmeasure)
        rouge2_scores.append(rouge_out["rouge2"].fmeasure)
        rougeL_scores.append(rouge_out["rougeL"].fmeasure)

        # ── Accumulate clinical metrics ────────────────────────────────
        clinical_metrics.update(reference, generated)

        # ── Log for qualitative analysis ──────────────────────────────
        uid = test_dataset.samples[sample_idx]["uid"]
        generated_log.append({
            "sample_idx":    sample_idx,
            "uid":           str(uid),
            "reference":     reference,
            "generated":     generated,
            "retrieved":     retrieved_report,
            "verify_score":  round(float(verify_score), 4),
        })

# ── Corpus-level BLEU ─────────────────────────────────────────────────────
# Corpus BLEU is the correct method for NLG evaluation:
# averaging sentence BLEU introduces length bias (inflated scores).
bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1,    0,    0,    0))
bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5,  0.5,  0,    0))
bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(1/3,  1/3,  1/3,  0))
bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

# ── CIDEr ─────────────────────────────────────────────────────────────────
# Consensus-based image description evaluation.
# Standard in captioning and report generation (Vedantam et al., 2015).
def _compute_cider(references: list, hypotheses: list) -> float:
    """TF-IDF weighted n-gram consensus score (CIDEr-D variant, n=1..4)."""
    from collections import Counter
    import math

    def _ngrams(tokens, n):
        return Counter(zip(*[tokens[i:] for i in range(n)]))

    def _tfidf(ngram_counts, df, N, n):
        """Compute TF-IDF weighted CIDEr vectors."""
        vec = {}
        total = sum(ngram_counts.values()) + 1e-9
        for ng, cnt in ngram_counts.items():
            tf  = cnt / total
            idf = math.log((N + 1.0) / (df.get(ng, 0) + 1.0))
            vec[ng] = tf * idf
        return vec

    N = len(references)
    scores = []
    for n in range(1, 5):
        # Build document frequency from references
        df: dict = Counter()
        for refs in references:
            for ref in refs:
                df.update(set(_ngrams(ref, n).keys()))

        n_scores = []
        for refs, hyp in zip(references, hypotheses):
            h_vec  = _tfidf(_ngrams(hyp, n), df, N, n)
            r_vecs = [_tfidf(_ngrams(r, n), df, N, n) for r in refs]

            # Mean over reference TF-IDF vectors
            all_keys = set(h_vec) | {k for rv in r_vecs for k in rv}
            r_mean = {k: np.mean([rv.get(k, 0) for rv in r_vecs]) for k in all_keys}

            dot  = sum(h_vec.get(k, 0) * r_mean.get(k, 0) for k in all_keys)
            norm_h = math.sqrt(sum(v**2 for v in h_vec.values()) + 1e-9)
            norm_r = math.sqrt(sum(v**2 for v in r_mean.values()) + 1e-9)
            n_scores.append(dot / (norm_h * norm_r))

        scores.append(float(np.mean(n_scores)))

    return round(float(np.mean(scores)) * 10, 4)   # CIDEr scale factor = 10

cider = _compute_cider(all_references, all_hypotheses)

# ── Compute clinical metrics ───────────────────────────────────────────────
clinical_scores = clinical_metrics.compute()
radgraph_extractor.save_cache()

# ── Assemble results ──────────────────────────────────────────────────────
results = {
    # NLG
    "BLEU-1":  round(bleu1, 4),
    "BLEU-2":  round(bleu2, 4),
    "BLEU-3":  round(bleu3, 4),
    "BLEU-4":  round(bleu4, 4),
    "METEOR":  round(float(np.mean(meteor_scores)), 4),
    "ROUGE-1": round(float(np.mean(rouge1_scores)), 4),
    "ROUGE-2": round(float(np.mean(rouge2_scores)), 4),
    "ROUGE-L": round(float(np.mean(rougeL_scores)), 4),
    "CIDEr":   cider,
    # Retrieval quality
    "verify_score_mean":   round(float(np.mean(verify_scores)),   4),
    "verify_score_median": round(float(np.median(verify_scores)), 4),
    # Clinical
    **clinical_scores,
    # Meta
    "num_test_samples":  len(all_hypotheses),
    "radgraph_mode":     "radgraph" if radgraph_extractor.using_radgraph else "keyword_fallback",
}

# ── Print ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)

def _section(title, keys):
    print(f"\n  {title}")
    for k in keys:
        if k in results:
            print(f"    {k:<32} {results[k]}")

_section(
    "NLG Metrics",
    ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "CIDEr"],
)
_section(
    "Retrieval Quality",
    ["verify_score_mean", "verify_score_median"],
)
_section(
    "CheXBert Label F1",
    ["chexbert_f1_micro", "chexbert_f1_macro",
     "chexbert_precision_micro", "chexbert_recall_micro"],
)
_section(
    "Entity F1",
    ["entity_precision", "entity_recall", "entity_f1"],
)
print(f"\n  Per-class CheXBert F1:")
for k, v in results.items():
    if k.startswith("chexbert_f1_") and k not in (
        "chexbert_f1_micro", "chexbert_f1_macro"
    ):
        print(f"    {k:<40} {v}")

if results["radgraph_mode"] == "keyword_fallback":
    print(
        "\n  NOTE: Entity F1 uses keyword fallback (RadGraph unavailable).\n"
        "        Install radgraph with transformers<4.40 for exact Entity F1."
    )

print("=" * 55)

# ── Save outputs ──────────────────────────────────────────────────────────
out_dir = os.path.join(_ROOT, "results")
os.makedirs(out_dir, exist_ok=True)

_metrics_path = os.path.join(out_dir, "metrics.json")
with open(_metrics_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrics  → {_metrics_path}")

_reports_path = os.path.join(out_dir, "generated_reports.json")
with open(_reports_path, "w", encoding="utf-8") as f:
    json.dump(generated_log, f, indent=2, ensure_ascii=False)
print(f"Reports  → {_reports_path}")


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4",  quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Config.DATA_ROOT

# ------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------
test_dataset = IUXrayMultiViewDataset(ROOT, split="test")
test_loader = DataLoader(test_dataset, batch_size=1)

# ------------------------------------------------
# LOAD FAISS + TRAIN METADATA
# ------------------------------------------------
index = faiss.read_index(os.path.join(_ROOT, "store", "faiss_index.bin"))

with open(os.path.join(_ROOT, "store", "train_reports.pkl"), "rb") as f:
    train_metadata = pickle.load(f)

# ------------------------------------------------
# LOAD STAGE-1 MODELS (FROZEN)
# Use factual retriever if available (FactMM-RAG).
# ------------------------------------------------
_FACTUAL_CKPT = os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth")
_STAGE1_CKPT  = os.path.join(_ROOT, "checkpoints", "stage1", "best.pth")
_use_factual  = os.path.exists(_FACTUAL_CKPT)
_ckpt_path    = _FACTUAL_CKPT if _use_factual else _STAGE1_CKPT
if _use_factual:
    print("[evaluate] Using FactMM-RAG factual retriever checkpoint.")
checkpoint = torch.load(_ckpt_path, map_location=device, weights_only=False)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"], strict=False)
visual_encoder.eval()

# CrossModalAlignment: image regions (Q) attend to retrieved report tokens (K,V)
# Produces region-aligned features (B, 49, 256) shaped by retrieved report context
alignment = CrossModalAlignment().to(device)
alignment.load_state_dict(checkpoint["alignment"])
alignment.eval()

proj_img = ProjectionHead().to(device)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()

image_classifier = SAEImageClassifier().to(device)
image_classifier.load_state_dict(
    torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"), map_location=device),
    strict=False,   # view_attn keys are new; checkpoint predates that addition
)
image_classifier.eval()

report_classifier = ReportClassifier().to(device)
report_classifier.load_state_dict(
    torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"), map_location=device)
)
report_classifier.eval()

generator = HybridReportGenerator().to(device)
MODEL_PATH = os.path.join(_ROOT, "checkpoints", "stage3", "best_generator.pth")
if not os.path.exists(MODEL_PATH):
    for fallback in [os.path.join(_ROOT, "checkpoints", "stage3", "last_generator.pth")]:
        if os.path.exists(fallback):
            MODEL_PATH = fallback
            break
print(f"Loading generator from: {MODEL_PATH}")
generator.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=False),
    strict=False,
)
generator.eval()

# ------------------------------------------------
# VERIFIER
# Re-ranks top-5 FAISS candidates by cross-modal
# attention score before passing to the decoder.
# ------------------------------------------------
verifier = ReportVerifier(alignment=alignment, min_score=0.0)

# ------------------------------------------------
# CLINICAL METRICS
# CheXBert Label F1: reuses ReportClassifier
# (ClinicalBERT → 14 CheXpert labels) to label
# both generated and reference reports, then
# computes micro / macro F1 — the standard clinical
# metric in MIMIC-CXR generation papers.
#
# Entity F1: RadGraphExtractor with keyword fallback
# for entity-level set-overlap precision/recall/F1.
# ------------------------------------------------
radgraph_cache = os.path.join(_ROOT, "store", "radgraph_cache.json")
radgraph_extractor = RadGraphExtractor(cache_path=radgraph_cache)
clinical_metrics = ClinicalMetrics(
    report_classifier=report_classifier,
    radgraph_extractor=radgraph_extractor,
    threshold=0.5,
    device=device,
)

# ------------------------------------------------
# METRICS
# ------------------------------------------------
smooth = SmoothingFunction().method1
rouge  = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Corpus-level accumulation (correct way to report BLEU for NLG papers)
all_references  = []   # list of list[list[str]]  (per sample: [[ref_tokens]])
all_hypotheses  = []   # list of list[str]          (per sample: hyp_tokens)

meteor_scores  = []
rouge1_scores  = []
rouge2_scores  = []
rouge_l_scores = []
verify_scores  = []

# Keep generated texts for qualitative analysis
generated_reports_log = []

print("Evaluating Aligned Hybrid Pipeline...")

# ------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------
with torch.no_grad():

    for images, reports in tqdm(test_loader):

        images = images.to(device)
        reference = reports[0]

        # 1. Visual backbone features: (B, 256, 7, 7)
        visual_features = visual_encoder(images)

        # 2. FAISS retrieval — top-5 candidates for re-ranking
        global_feat = visual_features.flatten(2).mean(dim=2)
        img_emb = proj_img(global_feat)
        img_np = img_emb.cpu().numpy().astype("float32")
        faiss.normalize_L2(img_np)          # match IndexFlatIP index
        D, I = index.search(img_np, k=5)

        candidate_reports = [train_metadata[int(i)]["report"] for i in I[0]]

        # 3. Verification — re-rank candidates by cross-modal attention
        retrieved_report, verify_score = verifier.verify(
            visual_features, candidate_reports
        )
        verify_scores.append(verify_score)

        # Concatenate top-3 candidates (verified best first) as richer context.
        # The generator was trained with retrieval variants, so it can use
        # longer retrieval context without seeing this exact combination.
        top3_idx = [int(i) for i in I[0][:3]]
        top3_reports = [train_metadata[j]["report"] for j in top3_idx]
        # Put the verified best report first; de-duplicate the rest
        top3_ordered = [retrieved_report] + [r for r in top3_reports if r != retrieved_report][:2]
        multi_retrieved = " [SEP] ".join(top3_ordered)

        # FactMM-RAG prompt: wrap retrieved text in context template
        rag_retrieved = HybridReportGenerator.build_rag_retrieved_text([multi_retrieved])[0]

        # 4. Region-aligned features — use best verified report for alignment
        #    image regions attend to retrieved report tokens via ClinicalBERT
        #    → regions are guided by what the retrieved report discusses
        aligned_features, _, _ = alignment(visual_features, [retrieved_report])

        # 5. Soft entity predictions
        #    Both classifiers produce [0, 1] probabilities.
        #    Multiplying keeps only findings confirmed by BOTH sources.
        img_logits = image_classifier(images)
        img_entities = torch.sigmoid(img_logits)                     # (1, 14) soft

        rep_logits   = report_classifier([retrieved_report])
        rep_entities = torch.sigmoid(rep_logits)                     # (1, 14) soft

        # 6. Soft-AND fact verification
        verified_entities = img_entities * rep_entities

        # Entity-informed prompt matches train_stage3.py exactly.
        prompt = HybridReportGenerator.build_entity_prompt(verified_entities.cpu())

        # 7. Generate report
        #    aligned_features  → region-aligned visual tokens
        #    verified_entities → fact-grounded entity tokens
        #    rag_retrieved     → FactMM-RAG wrapped top-3 retrieved context
        #    prompt            → task instruction tokens
        generated = generator(
            region_features=aligned_features,
            entity_vector=verified_entities,
            retrieved_texts=[rag_retrieved],
            prompt_texts=prompt,
            target_texts=None
        )[0]

        # ---------------- Metrics ----------------
        ref_tokens = reference.split()
        gen_tokens = generated.split()

        # Accumulate for corpus-level BLEU (standard in report generation papers)
        all_references.append([ref_tokens])
        all_hypotheses.append(gen_tokens)

        # Sentence-level METEOR (standard: word-level tokens)
        meteor_scores.append(meteor_score([ref_tokens], gen_tokens))

        rouge_out = rouge.score(reference, generated)
        rouge1_scores.append(rouge_out['rouge1'].fmeasure)
        rouge2_scores.append(rouge_out['rouge2'].fmeasure)
        rouge_l_scores.append(rouge_out['rougeL'].fmeasure)

        # Clinical metrics — update per sample
        clinical_metrics.update(reference, generated)

        generated_reports_log.append({
            "reference": reference,
            "generated": generated,
            "verify_score": float(verify_score),
        })

# ------------------------------------------------
# FINAL RESULTS
# ------------------------------------------------
# Corpus-level BLEU: correct for NLG evaluation
# (averaging sentence BLEU per sample inflates scores)
bleu1_c = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0))
bleu2_c = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
bleu3_c = corpus_bleu(all_references, all_hypotheses, weights=(1/3, 1/3, 1/3, 0))
bleu4_c = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

# Compute clinical metrics after the loop
clinical_scores = clinical_metrics.compute()
# Flush RadGraph keyword cache to disk (no-op if already cached)
radgraph_extractor.save_cache()

results = {
    "BLEU-1":  round(bleu1_c, 4),
    "BLEU-2":  round(bleu2_c, 4),
    "BLEU-3":  round(bleu3_c, 4),
    "BLEU-4":  round(bleu4_c, 4),
    "METEOR":  round(float(np.mean(meteor_scores)), 4),
    "ROUGE-1": round(float(np.mean(rouge1_scores)), 4),
    "ROUGE-2": round(float(np.mean(rouge2_scores)), 4),
    "ROUGE-L": round(float(np.mean(rouge_l_scores)), 4),
    "verify_score_mean":   round(float(np.mean(verify_scores)),   4),
    "verify_score_median": round(float(np.median(verify_scores)), 4),
    **clinical_scores,
    "num_samples": len(all_hypotheses),
}

print("\n==== FINAL RESULTS (corpus-level) ====")
_SECTION_KEYS = {
    "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
    "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L",
}
print("  -- NLG Metrics --")
for k, v in results.items():
    if k in _SECTION_KEYS:
        print(f"  {k:<24}: {v}")
print("  -- Clinical Metrics (CheXBert Label F1) --")
for k, v in results.items():
    if k.startswith("chexbert") and "per_class" not in k and "f1_no_finding" not in k:
        print(f"  {k:<24}: {v}")
print("  -- Entity F1 --")
for k, v in results.items():
    if k.startswith("entity"):
        print(f"  {k:<24}: {v}")
print("  -- Per-Class CheXBert F1 --")
for k, v in results.items():
    if k.startswith("chexbert_f1_") and k not in ("chexbert_f1_micro", "chexbert_f1_macro"):
        print(f"  {k:<40}: {v}")

# Save results and generated reports to disk
out_dir   = os.path.join(_ROOT, "results")
os.makedirs(out_dir, exist_ok=True)

results_path = os.path.join(out_dir, "metrics.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrics saved -> {results_path}")

reports_path = os.path.join(out_dir, "generated_reports.json")
with open(reports_path, "w", encoding="utf-8") as f:
    json.dump(generated_reports_log, f, indent=2, ensure_ascii=False)
print(f"Generated reports saved -> {reports_path}")