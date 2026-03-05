import torch
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

from torch.utils.data import DataLoader
from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier
from rag.hybrid_generator import HybridReportGenerator

nltk.download("wordnet")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "C:/Datasets/IU_Xray"

# ------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------
test_dataset = IUXrayMultiViewDataset(ROOT, split="test")
test_loader = DataLoader(test_dataset, batch_size=1)

# ------------------------------------------------
# LOAD FAISS + TRAIN METADATA
# ------------------------------------------------
index = faiss.read_index("rag/faiss_index.bin")

with open("rag/train_reports.pkl", "rb") as f:
    train_metadata = pickle.load(f)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
visual_encoder = MultiViewBackbone().to(device)
checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()

image_classifier = SAEImageClassifier().to(device)
image_classifier.load_state_dict(
    torch.load("classification/image_classifier.pth", map_location=device)
)
image_classifier.eval()

report_classifier = ReportClassifier().to(device)
report_classifier.load_state_dict(
    torch.load("classification/report_classifier.pth", map_location=device)
)
report_classifier.eval()

generator = HybridReportGenerator().to(device)
generator.load_state_dict(
    torch.load("rag/hybrid_generator.pth", map_location=device)
)
generator.eval()

# ------------------------------------------------
# METRICS
# ------------------------------------------------
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

bleu1, bleu2, bleu3, bleu4 = [], [], [], []
meteor_scores = []
rouge_l_scores = []

print("Evaluating Correct Hybrid Pipeline...")

# ------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------
with torch.no_grad():

    for images, reports in tqdm(test_loader):

        images = images.to(device)
        reference = reports[0]

        # 1️⃣ Region features
        feats = visual_encoder(images)
        region_features = feats.flatten(2).transpose(1, 2)

        # 2️⃣ Image entities
        img_logits = image_classifier(images)
        img_entities = (torch.sigmoid(img_logits) > 0.5).float()

        # 3️⃣ Retrieve top-1 report
        global_feat = feats.flatten(2).mean(dim=2)
        global_np = global_feat.cpu().numpy().astype("float32")
        D, I = index.search(global_np, k=1)

        retrieved_report = train_metadata[I[0][0]]["report"]

        # 4️⃣ Retrieved entities
        rep_logits = report_classifier([retrieved_report])
        rep_entities = (torch.sigmoid(rep_logits) > 0.5).float()

        # 5️⃣ Verified intersection
        verified_entities = img_entities * rep_entities

        prompt = ["Generate a detailed radiology report based on the image, verified findings, and retrieved context."]

        # 6️⃣ Generate
        generated = generator(
            region_features=region_features,
            entity_vector=verified_entities,
            retrieved_texts=[retrieved_report],
            prompt_texts=prompt,
            target_texts=None
        )[0]

        # ---------------- Metrics ----------------
        ref_tokens = reference.split()
        gen_tokens = generated.split()

        bleu1.append(sentence_bleu([ref_tokens], gen_tokens,
                                   weights=(1,0,0,0),
                                   smoothing_function=smooth))

        bleu2.append(sentence_bleu([ref_tokens], gen_tokens,
                                   weights=(0.5,0.5,0,0),
                                   smoothing_function=smooth))

        bleu3.append(sentence_bleu([ref_tokens], gen_tokens,
                                   weights=(0.33,0.33,0.33,0),
                                   smoothing_function=smooth))

        bleu4.append(sentence_bleu([ref_tokens], gen_tokens,
                                   weights=(0.25,0.25,0.25,0.25),
                                   smoothing_function=smooth))

        meteor_scores.append(meteor_score([ref_tokens], gen_tokens))

        rouge_l_scores.append(
            rouge.score(reference, generated)['rougeL'].fmeasure
        )

# ------------------------------------------------
# FINAL RESULTS
# ------------------------------------------------
print("\n==== FINAL RESULTS ====")
print("BLEU-1:", np.mean(bleu1))
print("BLEU-2:", np.mean(bleu2))
print("BLEU-3:", np.mean(bleu3))
print("BLEU-4:", np.mean(bleu4))
print("METEOR:", np.mean(meteor_scores))
print("ROUGE-L:", np.mean(rouge_l_scores))