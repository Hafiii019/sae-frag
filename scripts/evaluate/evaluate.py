"""
Evaluation script for SAE-FRAG radiology report generation.

Metrics
-------
NLG     : BLEU-1/2/3/4, METEOR, ROUGE-1/2/L, CIDEr
Clinical: CheXBert Label F1 (micro/macro/per-class), Entity F1
Factual : Fact Verification Score (entity precision/recall/F1 vs reference)

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
import re
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

# ── Factual-verification helpers (from factual_verify.py) ─────────────────
# Inline the clinical-entity vocabulary and scorers so evaluate.py stays
# self-contained and runnable without a separate import step.
import re as _re

_CLINICAL_TERMS = sorted([
    "pleural effusion", "pulmonary edema", "pulmonary fibrosis",
    "pulmonary hypertension", "aortic dissection", "hilar adenopathy",
    "mediastinal widening", "pneumoperitoneum", "pneumothorax",
    "cardiomegaly", "atelectasis", "consolidation", "emphysema",
    "infiltrate", "opacities", "opacity", "effusion", "granuloma",
    "nodule", "mass", "hernia", "fibrosis", "scoliosis", "kyphosis",
    "osteophyte", "fracture", "adenopathy", "calcification", "edema",
    "pneumonia", "pacemaker", "defibrillator", "picc line", "catheter",
    "spinal rods", "hyperexpanded", "hyperinflated",
    "flattened diaphragm", "tortuous aorta", "atherosclerotic",
    "normal heart size", "clear lungs", "no pneumothorax",
    "no effusion", "no consolidation", "no pleural effusion",
    "no focal consolidation", "bibasilar", "left lower lobe",
    "right lower lobe", "left upper lobe", "right upper lobe",
    "right middle lobe", "perihilar", "basilar",
], key=lambda t: -len(t))


def _fv_extract(text: str) -> set:
    """Return clinical terms found in *text* (case-insensitive)."""
    txt = text.lower()
    return {
        term for term in _CLINICAL_TERMS
        if _re.search(r"\b" + _re.escape(term) + r"\b", txt)
    }


def _fv_scores(ref: str, gen: str):
    """Return (precision, recall, f1) for entity overlap."""
    ref_e = _fv_extract(ref)
    gen_e = _fv_extract(gen)
    if not ref_e and not gen_e:
        return None, None, None          # skip trivially empty pairs
    if not gen_e or not ref_e:
        return 0.0, 0.0, 0.0
    tp = len(gen_e & ref_e)
    p  = tp / len(gen_e)
    r  = tp / len(ref_e)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

nltk.download("wordnet",    quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


def _normalize(text: str) -> str:
    """Lowercase + strip punctuation attached to tokens, matching _clean_report().

    Applied to the *generated* text before metric computation so it is
    tokenized identically to the reference (which goes through _clean_report
    at dataset load time: lowercase, xxxx removed, junk chars stripped).
    """
    text = text.lower().strip()
    text = re.sub(r'\bxxxx\b', '', text)
    text = re.sub(r'[^\w\s.,;:\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _tokenize(text: str) -> list:
    """Tokenize text using NLTK word_tokenize (handles punctuation correctly).

    e.g. 'normal.' → ['normal', '.'] not ['normal.']
         'well-defined' → ['well-defined'] (hyphen kept)
    """
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)


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
    W = 62

    log.info("\n" + "=" * W)
    log.info("  SAE-FRAG  ·  EVALUATION RESULTS")
    log.info("=" * W)

    # ── Core thesis metrics table ─────────────────────────────────────────
    log.info("\n  ┌─────────────────────────────────┬──────────┐")
    log.info(  "  │ Metric                          │  Score   │")
    log.info(  "  ├─────────────────────────────────┼──────────┤")

    def row(label, key):
        val = results.get(key, "N/A")
        log.info(f"  │ {label:<31} │ {str(val):>8} │")

    row("BLEU-1",                        "BLEU-1")
    row("BLEU-2",                        "BLEU-2")
    row("BLEU-3",                        "BLEU-3")
    row("BLEU-4",                        "BLEU-4")
    log.info("  ├─────────────────────────────────┼──────────┤")
    row("ROUGE-L",                       "ROUGE-L")
    row("METEOR",                        "METEOR")
    row("CIDEr",                         "CIDEr")
    log.info("  ├─────────────────────────────────┼──────────┤")
    row("RadGraph Entity F1",            "entity_f1")
    row("RadGraph Entity Precision",     "entity_precision")
    row("RadGraph Entity Recall",        "entity_recall")
    log.info("  ├─────────────────────────────────┼──────────┤")
    row("CheXBert F1 (micro)",           "chexbert_f1_micro")
    row("CheXBert F1 (macro)",           "chexbert_f1_macro")
    row("CheXBert Precision (micro)",    "chexbert_precision_micro")
    row("CheXBert Recall (micro)",       "chexbert_recall_micro")
    log.info("  ├─────────────────────────────────┼──────────┤")
    row("Fact Verification F1",          "fact_verify_f1")
    row("Fact Verification Precision",   "fact_verify_precision")
    row("Fact Verification Recall",      "fact_verify_recall")
    log.info("  ├─────────────────────────────────┼──────────┤")
    row("CrossModal Verify Score (mean)","verify_score_mean")
    log.info("  └─────────────────────────────────┴──────────┘")

    # ── Per-class CheXBert ────────────────────────────────────────────────
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
    all_references:   list = []
    all_hypotheses:   list = []
    meteor_scores:    list = []
    rouge1_scores:    list = []
    rouge2_scores:    list = []
    rougeL_scores:    list = []
    verify_scores:    list = []
    generated_log:    list = []
    fv_p_list:        list = []   # fact-verify precision per sample
    fv_r_list:        list = []   # fact-verify recall per sample
    fv_f1_list:       list = []   # fact-verify F1 per sample

    # ── Evaluation loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for sample_idx, (images, reports, impressions, entity_texts) in enumerate(
            tqdm(test_loader, desc="Evaluating", unit="sample")
        ):
            images     = images.to(device)
            reference  = reports[0]
            impression = impressions[0]
            entity_txt = entity_texts[0]

            # Visual features
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

            # Region-aligned features
            aligned_features, _, _ = alignment(visual_features, [retrieved_report])

            # Soft-AND entity verification
            img_entities  = torch.sigmoid(image_classifier(images))
            rep_entities  = torch.sigmoid(report_classifier([retrieved_report]))
            verified_ents = img_entities * rep_entities

            # Generate report — pass impression + Stanza entity text if available
            prompt = HybridReportGenerator.build_entity_prompt(verified_ents.cpu())
            generated = generator(
                region_features=aligned_features,
                entity_vector=verified_ents,
                retrieved_texts=[rag_context],
                prompt_texts=prompt,
                target_texts=None,
                impression_texts=[impression] if impression.strip() else None,
                entity_texts=[entity_txt] if entity_txt.strip() else None,
            )[0]

            # ── Normalize + tokenize for metric computation ────────────────────
            # CRITICAL: reference is already lowercased by _clean_report().
            # Generated text from SciFive is mixed-case — must normalize
            # before metric computation or case mismatches kill BLEU scores.
            generated_norm = _normalize(generated)
            ref_tokens = _tokenize(reference)
            gen_tokens = _tokenize(generated_norm)

            all_references.append([ref_tokens])
            all_hypotheses.append(gen_tokens)
            meteor_scores.append(meteor_score([ref_tokens], gen_tokens))

            rouge_out = rouge_scorer_obj.score(reference, generated_norm)
            rouge1_scores.append(rouge_out["rouge1"].fmeasure)
            rouge2_scores.append(rouge_out["rouge2"].fmeasure)
            rougeL_scores.append(rouge_out["rougeL"].fmeasure)

            clinical_metrics.update(reference, generated_norm)

            # ── Fact verification score ────────────────────────────────────
            fv_p, fv_r, fv_f1 = _fv_scores(reference, generated_norm)
            if fv_f1 is not None:           # skip trivially empty pairs
                fv_p_list.append(fv_p)
                fv_r_list.append(fv_r)
                fv_f1_list.append(fv_f1)

            generated_log.append({
                "sample_idx":    sample_idx,
                "uid":           str(test_dataset.samples[sample_idx]["uid"]),
                "reference":     reference,
                "generated_raw": generated,
                "generated":     generated_norm,
                "retrieved":  retrieved_report,
                "verify_score": round(float(verify_score), 4),
            })

    # ── Compute aggregate scores ──────────────────────────────────────────
    # Corpus-level BLEU with add-1 smoothing (method1).
    # Standard papers use no smoothing at corpus level, but many RRG papers
    # (including R2Gen, PKARG) apply smoothing to handle zero n-gram counts
    # in short medical reports.  Smoothing only affects BLEU-3/4 noticeably.
    from nltk.translate.bleu_score import SmoothingFunction
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0),       smoothing_function=smooth)
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0),   smoothing_function=smooth)
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25,)*4,           smoothing_function=smooth)
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
        "fact_verify_precision": round(float(np.mean(fv_p_list)),  4) if fv_p_list else None,
        "fact_verify_recall":    round(float(np.mean(fv_r_list)),  4) if fv_r_list else None,
        "fact_verify_f1":        round(float(np.mean(fv_f1_list)), 4) if fv_f1_list else None,
        "fact_verify_n_scored":  len(fv_f1_list),
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

