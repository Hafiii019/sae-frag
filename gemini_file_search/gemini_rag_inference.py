"""End-to-end SAE-FRAG + Gemini inference pipeline.

Extended Pipeline (Steps 1-8)
------------------------------
1.  Image  →  SAE Encoder (MultiViewBackbone)  →  image_features (1, 256, 7, 7)
2.  GAP  →  ProjectionHead  →  L2-normalise  →  query_emb (1, 768)
3.  FAISS search  (top-K)  →  candidate_reports
4.  ReportVerifier  re-rank  →  verified_report  +  verify_score
5.  CrossModalAlignment(image_features, verified_report)
        →  aligned_region_features (1, 49, 256)
6.  SAEImageClassifier  →  img_entities  (1, 14)  [soft sigmoid]
    ReportClassifier    →  rep_entities  (1, 14)  [soft sigmoid]
    verified_entity_vector = img_entities * rep_entities  (soft AND)
7.  HybridReportGenerator  →  t5_draft_report
8.  GeminiReportRefiner (NEW)
        image files  →  Gemini Files API upload
        Gemini  ←  images + verified_report + entities + t5_draft
        →  gemini_refined_report

Requirements
------------
    pip install google-genai

Set your API key:
    export GEMINI_API_KEY="your-key-here"   # Linux / macOS
    $env:GEMINI_API_KEY="your-key-here"     # PowerShell
"""

from __future__ import annotations

import os
import sys
import pickle
import logging

import faiss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make sure the project root is on the path when running this script directly
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = "C:/Datasets/IU_Xray"
TOP_K  = 5

PROMPT = (
    "Generate a detailed radiology report based on the chest X-ray regions, "
    "verified clinical findings, and retrieved context."
)

# Gemini model to use for refinement
GEMINI_MODEL = "gemini-2.5-flash"

# Soft-AND threshold: findings must exceed this score before being sent to Gemini
ENTITY_THRESHOLD = 0.3

# =====================================================
# DATASET  (with image-path access)
# =====================================================

class _DatasetWithPaths(IUXrayMultiViewDataset):
    """Thin subclass that also exposes the raw image file paths."""

    def __getitem__(self, idx):
        images, report = super().__getitem__(idx)
        sample = self.samples[idx]
        # Return up to two frontal/lateral image paths
        paths = [
            os.path.join(self.image_dir, fname)
            for fname in sample["images"][:2]
        ]
        return images, report, paths


test_dataset = _DatasetWithPaths(ROOT, split="test")
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
logger.info("Test split: %d samples", len(test_dataset))

# =====================================================
# LOAD FAISS INDEX + TRAIN METADATA
# =====================================================
index = faiss.read_index("rag/faiss_index.bin")
logger.info("FAISS index size: %d", index.ntotal)

with open("rag/train_reports.pkl", "rb") as fh:
    train_metadata = pickle.load(fh)
logger.info("Train metadata entries: %d", len(train_metadata))

# =====================================================
# LOAD MODELS
# =====================================================
checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=DEVICE)

# --- SAE Encoder ---
visual_encoder = MultiViewBackbone().to(DEVICE)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()

# --- Cross-modal alignment ---
alignment = CrossModalAlignment().to(DEVICE)
alignment.load_state_dict(checkpoint["alignment"])
alignment.eval()

# --- Projection head ---
proj_img = ProjectionHead().to(DEVICE)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()

# --- Pathology classifiers ---
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

# --- T5 Decoder ---
generator = HybridReportGenerator().to(DEVICE)
generator.load_state_dict(
    torch.load("rag/hybrid_generator.pth", map_location=DEVICE)
)
generator.eval()

# --- Verifier ---
verifier = ReportVerifier(alignment=alignment, min_score=0.0)

# --- Gemini Refiner (Step 8) ---
gemini_refiner = GeminiReportRefiner(
    model=GEMINI_MODEL,
    entity_threshold=ENTITY_THRESHOLD,
    cache_uploads=True,
)

# =====================================================
# INFERENCE LOOP
# =====================================================
logger.info("Starting Gemini-enhanced inference on %d samples…", len(test_dataset))

try:
    with torch.no_grad():
        for batch_idx, (images, true_reports, image_paths) in enumerate(test_loader):
            images = images.to(DEVICE)

            # --------------------------------------------------
            # Step 1: SAE image features  →  (1, 256, 7, 7)
            # --------------------------------------------------
            image_features = visual_encoder(images)

            # --------------------------------------------------
            # Step 2: Global embedding  →  L2-normalised FAISS query
            # --------------------------------------------------
            global_feat = image_features.flatten(2).mean(dim=2)        # (1, 256)
            query_emb   = proj_img(global_feat)                         # (1, 768)
            query_np    = F.normalize(query_emb, dim=1).cpu().numpy().astype("float32")

            # --------------------------------------------------
            # Step 3: FAISS retrieval  →  top-K candidates
            # --------------------------------------------------
            _, indices = index.search(query_np, TOP_K)
            candidate_reports = [train_metadata[int(i)]["report"] for i in indices[0]]

            # --------------------------------------------------
            # Step 4: Verification  →  re-rank by cross-modal attention
            # --------------------------------------------------
            verified_report, verify_score = verifier.verify(
                image_features, candidate_reports
            )

            # --------------------------------------------------
            # Step 5: Region-aligned features
            # --------------------------------------------------
            aligned_region_features, _, _ = alignment(
                image_features, [verified_report]
            )

            # --------------------------------------------------
            # Step 6: Soft entity predictions + soft-AND fusion
            # --------------------------------------------------
            img_entities = torch.sigmoid(image_classifier(images))     # (1, 14)
            rep_entities = torch.sigmoid(report_classifier([verified_report]))
            verified_entity_vector = img_entities * rep_entities       # soft AND

            # --------------------------------------------------
            # Step 7: T5 draft report
            # --------------------------------------------------
            t5_draft = generator(
                region_features=aligned_region_features,
                entity_vector=verified_entity_vector,
                retrieved_texts=[verified_report],
                prompt_texts=[PROMPT],
                target_texts=None,
            )[0]

            # --------------------------------------------------
            # Step 8: Gemini refinement via Files API
            # --------------------------------------------------
            # image_paths is a list-of-lists from the DataLoader;
            # flatten to a plain list of path strings for this sample.
            raw_paths = [p[0] if isinstance(p, (list, tuple)) else p
                         for p in image_paths]

            gemini_report = gemini_refiner.refine(
                image_paths=raw_paths,
                verified_report=verified_report,
                entity_vector=verified_entity_vector[0],   # (14,)
                draft_report=t5_draft,
                verify_score=verify_score,
            )

            # --------------------------------------------------
            # Output
            # --------------------------------------------------
            print("\n" + "=" * 70)
            print(f"SAMPLE {batch_idx + 1}")
            print("-" * 70)
            print(f"TRUE REPORT:\n{true_reports[0]}")
            print(f"\nVERIFIED RETRIEVED  (score={verify_score:.4f}):\n{verified_report}")
            print(f"\nT5 DRAFT REPORT:\n{t5_draft}")
            print(f"\nGEMINI REFINED REPORT:\n{gemini_report}")

finally:
    # Clean up uploaded files from the Gemini Files API
    gemini_refiner.cleanup_uploads()
    logger.info("Inference complete.  Uploaded files cleaned up.")
