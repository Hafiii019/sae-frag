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
import pickle
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


# Must match MODEL_NAME in scripts/train/train_stage3.py
_GENERATOR_MODEL = "razent/SciFive-base-Pubmed_PMC"


def _load_generator(device: torch.device) -> torch.nn.Module:
    """Load the fine-tuned report generator from the best available checkpoint."""
    candidates = [
        os.path.join(_ROOT, "checkpoints", "stage3", "best_generator.pth"),
        os.path.join(_ROOT, "checkpoints", "stage3", "last_generator.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            model = HybridReportGenerator(model_name=_GENERATOR_MODEL).to(device)
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
    train_metadata = pickle.load(
        open(os.path.join(_ROOT, "store", "train_reports.pkl"), "rb")
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

