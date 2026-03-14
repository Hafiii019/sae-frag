"""RAG inference pipeline for X-ray radiology report generation.

Pipeline
--------
1.  Image  →  SAE Encoder (MultiViewBackbone)  →  image_features (1, 256, 7, 7)
2.  GAP  →  ProjectionHead  →  L2-normalise  →  query_emb (1, 768)
3.  FAISS search  (top-K)  →  candidate_reports
4.  ReportVerifier  re-rank  →  verified_report  +  verify_score
5.  CrossModalAlignment(image_features, verified_report)
        →  aligned_region_features (1, 49, 256)
6.  SAEImageClassifier  →  img_entities  (1, 14)  [soft sigmoid]
    ReportClassifier    →  rep_entities  (1, 14)  [soft sigmoid]
    verified_entity_vector = img_entities * rep_entities  (soft AND)
7.  HybridReportGenerator(
        region_features=aligned_region_features,
        entity_vector=verified_entity_vector,
        retrieved_texts=[verified_report],
        prompt_texts=[PROMPT],
    )  →  generated_report
"""

from __future__ import annotations

import pickle

import faiss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from data.dataset import IUXrayMultiViewDataset
from models.alignment import CrossModalAlignment
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead
from rag.hybrid_generator import HybridReportGenerator
from rag.verifier import ReportVerifier


# =====================================================
# CONFIGURATION
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = "C:/Datasets/IU_Xray"
TOP_K  = 5  # number of FAISS candidates to retrieve before re-ranking

PROMPT = (
    "Generate a detailed radiology report based on the chest X-ray regions, "
    "verified clinical findings, and retrieved context."
)

# =====================================================
# LOAD TEST DATA
# =====================================================
test_dataset = IUXrayMultiViewDataset(ROOT, split="test")
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"Test split: {len(test_dataset)} samples")

# =====================================================
# LOAD FAISS INDEX + TRAIN METADATA
# =====================================================
index = faiss.read_index("store/faiss_index.bin")
print(f"FAISS index size: {index.ntotal}")

with open("store/train_reports.pkl", "rb") as fh:
    train_metadata = pickle.load(fh)
print(f"Train metadata entries: {len(train_metadata)}")

# =====================================================
# LOAD MODELS
# =====================================================
checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=DEVICE)

# --- SAE Encoder (backbone + FPN + SAFE) ---
visual_encoder = MultiViewBackbone().to(DEVICE)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()

# --- Cross-modal alignment (image patches attend to report tokens) ---
alignment = CrossModalAlignment().to(DEVICE)
alignment.load_state_dict(checkpoint["alignment"])
alignment.eval()

# --- Projection head: global image embedding → 768-dim FAISS query ---
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

# --- Decoder ---
generator = HybridReportGenerator().to(DEVICE)
generator.load_state_dict(
    torch.load("checkpoints/best_generator.pth", map_location=DEVICE, weights_only=False),
    strict=False,
)
generator.eval()

# --- Verifier (re-ranks FAISS candidates via cross-modal attention) ---
verifier = ReportVerifier(alignment=alignment, min_score=0.0)

# =====================================================
# INFERENCE LOOP
# =====================================================
with torch.no_grad():
    for images, true_reports in test_loader:
        images = images.to(DEVICE)

        # --------------------------------------------------
        # Step 1: SAE Encoder — (1, 256, 7, 7)
        # --------------------------------------------------
        image_features = visual_encoder(images)

        # --------------------------------------------------
        # Step 2: Global embedding → L2-normalised FAISS query
        # --------------------------------------------------
        global_feat = image_features.flatten(2).mean(dim=2)            # (1, 256)
        query_emb   = proj_img(global_feat)                             # (1, 768)
        query_np    = F.normalize(query_emb, dim=1).cpu().numpy().astype("float32")

        # --------------------------------------------------
        # Step 3: FAISS retrieval — top-K candidates
        # --------------------------------------------------
        _, indices = index.search(query_np, TOP_K)
        candidate_reports = [train_metadata[int(i)]["report"] for i in indices[0]]

        # --------------------------------------------------
        # Step 4: Verification — re-rank by cross-modal attention
        # --------------------------------------------------
        verified_report, verify_score = verifier.verify(
            image_features, candidate_reports
        )

        # --------------------------------------------------
        # Step 5: Region-aligned features
        #         Image patches (Q) attend to verified report tokens (K, V)
        #         → each spatial region is conditioned on the retrieved report
        # --------------------------------------------------
        aligned_region_features, _, _ = alignment(
            image_features, [verified_report]
        )  # (1, 49, 256)

        # --------------------------------------------------
        # Step 6: Soft entity predictions + soft-AND verification
        #         Both classifiers produce [0, 1] probabilities.
        #         Multiplying keeps only findings confirmed by
        #         BOTH the image signal AND the retrieved report text.
        # --------------------------------------------------
        img_logits = image_classifier(images)
        img_entities = torch.sigmoid(img_logits)                        # (1, 14)

        rep_logits   = report_classifier([verified_report])
        rep_entities = torch.sigmoid(rep_logits)                        # (1, 14)

        verified_entity_vector = img_entities * rep_entities            # soft AND

        # --------------------------------------------------
        # Step 7: Report generation
        # --------------------------------------------------
        generated_report = generator(
            region_features=aligned_region_features,
            entity_vector=verified_entity_vector,
            retrieved_texts=[verified_report],
            prompt_texts=[PROMPT],
            target_texts=None,
        )[0]

        # --------------------------------------------------
        # Output
        # --------------------------------------------------
        print("\n" + "=" * 60)
        print(f"TRUE REPORT:\n{true_reports[0]}")
        print(f"\nVERIFIED REPORT  (score={verify_score:.4f}):\n{verified_report}")
        print(f"\nGENERATED REPORT:\n{generated_report}")