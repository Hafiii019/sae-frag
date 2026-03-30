import os
import sys
import json

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

from torch.utils.data import DataLoader
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier
from rag.hybrid_generator import HybridReportGenerator
from rag.verifier import ReportVerifier

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
# ------------------------------------------------
checkpoint = torch.load(os.path.join(_ROOT, "checkpoints", "stage1", "best.pth"), map_location=device)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
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
    torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"), map_location=device)
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

        # 4. Region-aligned features
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

        prompt = [
            "Generate a detailed radiology report based on the chest X-ray regions, "
            "verified clinical findings, and retrieved context."
        ]

        # 7. Generate report
        #    aligned_features  → region-aligned visual tokens
        #    verified_entities → fact-grounded entity tokens
        #    retrieved_report  → retrieved context text tokens
        #    prompt            → task instruction tokens
        generated = generator(
            region_features=aligned_features,
            entity_vector=verified_entities,
            retrieved_texts=[retrieved_report],
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
    "num_samples": len(all_hypotheses),
}

print("\n==== FINAL RESULTS (corpus-level) ====")
for k, v in results.items():
    print(f"  {k:<24}: {v}")

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