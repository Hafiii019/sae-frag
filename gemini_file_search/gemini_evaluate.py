"""Evaluation script for the Gemini-enhanced SAE-FRAG pipeline.

Computes BLEU-1/2/3/4, METEOR, and ROUGE-L scores for:
  - T5 draft reports     (Steps 1-7 only)
  - Gemini refined reports  (Steps 1-8 with Gemini Files API)

Results are printed to stdout and saved to gemini_eval_output.txt.

Usage
-----
    cd D:\\Thesis\\sae-frag
    $env:GEMINI_API_KEY="your-key"
    python -m gemini_file_search.gemini_evaluate

Optional flags (edit the CONFIG section below):
    MAX_SAMPLES    — limit evaluation to N samples (None = full test set)
    GEMINI_MODEL   — which Gemini model to use
    ENTITY_THRESHOLD — soft-AND confidence threshold
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
from typing import List

import faiss
import nltk
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from data.dataset import IUXrayMultiViewDataset
from models.alignment import CrossModalAlignment
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead
from rag.hybrid_generator import HybridReportGenerator
from rag.verifier import ReportVerifier
from gemini_file_search.gemini_refiner import GeminiReportRefiner

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =====================================================
# CONFIG
# =====================================================
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT             = "C:/Datasets/IU_Xray"
TOP_K            = 5
GEMINI_MODEL     = "gemini-2.0-flash"
ENTITY_THRESHOLD = 0.3
MAX_SAMPLES      = None   # set to an int (e.g. 50) to limit evaluation length
OUTPUT_FILE      = "gemini_eval_output.txt"

PROMPT = (
    "Generate a detailed radiology report based on the chest X-ray regions, "
    "verified clinical findings, and retrieved context."
)

smooth = SmoothingFunction().method1
rouge  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# =====================================================
# DATASET WITH PATHS
# =====================================================

class _DatasetWithPaths(IUXrayMultiViewDataset):
    def __getitem__(self, idx):
        images, report = super().__getitem__(idx)
        sample = self.samples[idx]
        paths = [
            os.path.join(self.image_dir, fname)
            for fname in sample["images"][:2]
        ]
        return images, report, paths


# =====================================================
# METRIC HELPERS
# =====================================================

def _bleu(reference: str, hypothesis: str) -> dict:
    ref_tokens  = reference.split()
    hyp_tokens  = hypothesis.split()
    return {
        "bleu1": sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu2": sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "bleu3": sentence_bleu([ref_tokens], hyp_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth),
        "bleu4": sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }

def _meteor(reference: str, hypothesis: str) -> float:
    return meteor_score([reference.split()], hypothesis.split())

def _rougeL(reference: str, hypothesis: str) -> float:
    return rouge.score(reference, hypothesis)["rougeL"].fmeasure

def _avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0

def _print_and_log(line: str, fh) -> None:
    print(line)
    fh.write(line + "\n")


# =====================================================
# LOAD MODELS
# =====================================================
logger.info("Loading models…")

test_dataset = _DatasetWithPaths(ROOT, split="test")
n_samples    = len(test_dataset) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_dataset))
# Use a Subset if capped
if MAX_SAMPLES is not None and MAX_SAMPLES < len(test_dataset):
    from torch.utils.data import Subset
    test_dataset = Subset(test_dataset, list(range(MAX_SAMPLES)))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
logger.info("Evaluating on %d samples", n_samples)

index = faiss.read_index("rag/faiss_index.bin")

with open("rag/train_reports.pkl", "rb") as fh:
    train_metadata = pickle.load(fh)

checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=DEVICE)

visual_encoder = MultiViewBackbone().to(DEVICE)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()

alignment = CrossModalAlignment().to(DEVICE)
alignment.load_state_dict(checkpoint["alignment"])
alignment.eval()

proj_img = ProjectionHead().to(DEVICE)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()

image_classifier = SAEImageClassifier().to(DEVICE)
image_classifier.load_state_dict(
    torch.load("classification/image_classifier.pth", map_location=DEVICE)
)
image_classifier.eval()

report_classifier = ReportClassifier().to(DEVICE)
report_classifier.load_state_dict(
    torch.load("classification/report_classifier.pth", map_location=DEVICE)
)
report_classifier.eval()

generator = HybridReportGenerator().to(DEVICE)
generator.load_state_dict(torch.load("rag/hybrid_generator.pth", map_location=DEVICE))
generator.eval()

verifier = ReportVerifier(alignment=alignment, min_score=0.0)

gemini_refiner = GeminiReportRefiner(
    model=GEMINI_MODEL,
    entity_threshold=ENTITY_THRESHOLD,
    cache_uploads=True,
)

# =====================================================
# EVALUATION LOOP
# =====================================================
# Per-sample score accumulators
t5_scores      = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rougeL": []}
gemini_scores  = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rougeL": []}

try:
    with torch.no_grad(), open(OUTPUT_FILE, "w", encoding="utf-8") as log_fh:
        for images, true_reports, image_paths in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)

            # Steps 1-2: visual features + FAISS query
            image_features = visual_encoder(images)
            global_feat    = image_features.flatten(2).mean(dim=2)
            query_emb      = proj_img(global_feat)
            query_np       = F.normalize(query_emb, dim=1).cpu().numpy().astype("float32")

            # Step 3: FAISS retrieval
            _, indices = index.search(query_np, TOP_K)
            candidates = [train_metadata[int(i)]["report"] for i in indices[0]]

            # Step 4: Verification
            verified_report, verify_score = verifier.verify(image_features, candidates)

            # Step 5: Aligned region features
            aligned_region_features, _, _ = alignment(image_features, [verified_report])

            # Step 6: Soft entity predictions
            img_entities = torch.sigmoid(image_classifier(images))
            rep_entities = torch.sigmoid(report_classifier([verified_report]))
            entity_vector = img_entities * rep_entities

            # Step 7: T5 draft
            t5_draft = generator(
                region_features=aligned_region_features,
                entity_vector=entity_vector,
                retrieved_texts=[verified_report],
                prompt_texts=[PROMPT],
                target_texts=None,
            )[0]

            # Step 8: Gemini refinement
            raw_paths = [p[0] if isinstance(p, (list, tuple)) else p for p in image_paths]
            gemini_report = gemini_refiner.refine(
                image_paths=raw_paths,
                verified_report=verified_report,
                entity_vector=entity_vector[0],
                draft_report=t5_draft,
                verify_score=verify_score,
            )

            reference = true_reports[0]

            # Accumulate T5 metrics
            for k, v in _bleu(reference, t5_draft).items():
                t5_scores[k].append(v)
            t5_scores["meteor"].append(_meteor(reference, t5_draft))
            t5_scores["rougeL"].append(_rougeL(reference, t5_draft))

            # Accumulate Gemini metrics
            for k, v in _bleu(reference, gemini_report).items():
                gemini_scores[k].append(v)
            gemini_scores["meteor"].append(_meteor(reference, gemini_report))
            gemini_scores["rougeL"].append(_rougeL(reference, gemini_report))

        # --------------------------------------------------
        # Final summary
        # --------------------------------------------------
        header = "\n" + "=" * 70 + "\nEVALUATION RESULTS\n" + "=" * 70
        _print_and_log(header, log_fh)

        metrics = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL"]
        row_fmt = "{:<20}  {:>10}  {:>14}  {:>10}"
        _print_and_log(row_fmt.format("Metric", "T5 Draft", "Gemini Refined", "Delta"), log_fh)
        _print_and_log("-" * 60, log_fh)

        for m in metrics:
            t5_val     = _avg(t5_scores[m])
            gemini_val = _avg(gemini_scores[m])
            delta      = gemini_val - t5_val
            sign       = "+" if delta >= 0 else ""
            _print_and_log(
                row_fmt.format(m.upper(), f"{t5_val:.4f}", f"{gemini_val:.4f}", f"{sign}{delta:.4f}"),
                log_fh,
            )

        _print_and_log(f"\nResults also saved to {OUTPUT_FILE}", log_fh)

finally:
    gemini_refiner.cleanup_uploads()
    logger.info("Evaluation complete.  Uploaded files cleaned up.")
