import json
import os
import sys

# Bootstrap: make src/ importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead
from classification.report_labeler import ReportClassifier
from rag.radgraph_extractor import RadGraphExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "C:/Datasets/IU_Xray"

dataset = IUXrayMultiViewDataset(ROOT, split="train")
loader = DataLoader(dataset, batch_size=1)

# Load Stage-1
checkpoint = torch.load(os.path.join(ROOT, "checkpoints", "stage1", "best.pth"), map_location=device)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()

proj_img = ProjectionHead().to(device)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()

# Load report classifier ONCE
report_classifier = ReportClassifier().to(device)
report_classifier.load_state_dict(
    torch.load(os.path.join(ROOT, "checkpoints", "stage2", "report_classifier.pth"), map_location=device)
)
report_classifier.eval()

embeddings = []
metadata = []

print("Building FAISS with precomputed entity vectors...")

with torch.no_grad():

    for images, reports in tqdm(loader):

        images = images.to(device)

        feats = visual_encoder(images)
        global_feat = feats.flatten(2).mean(dim=2)
        img_emb = proj_img(global_feat)

        embeddings.append(img_emb.cpu().numpy())

        # Precompute report entities ONCE
        rep_logits = report_classifier(reports)
        rep_entities = (torch.sigmoid(rep_logits) > 0.5).float()

        metadata.append({
            "report": reports[0],
            "entity_vector": rep_entities.cpu()
        })

embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(ROOT, "store", "faiss_index.bin"))

with open(os.path.join(ROOT, "store", "train_reports.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAST FAISS built.")

# =====================================================
# PHASE 2: RadGraph entity embeddings (offline metadata)
# =====================================================
# Reuse the ClinicalBERT encoder already inside ReportClassifier
# so we avoid loading a second copy of the BERT weights.
extractor = RadGraphExtractor(
    device=str(device),
    cache_path=os.path.join(ROOT, "store", "radgraph_cache.json"),
)

bert_encoder  = report_classifier.encoder    # Bio_ClinicalBERT AutoModel
bert_tokenizer = report_classifier.tokenizer  # matching AutoTokenizer

radgraph_embeddings: list = []
radgraph_labels: list = []

print("Computing RadGraph entity embeddings for training reports...")

for sample in tqdm(metadata, desc="RadGraph"):
    report_text = sample["report"]
    result      = extractor.extract(report_text)
    entity_text = extractor.to_entity_text(result)

    emb = extractor.to_entity_embedding(entity_text, bert_encoder, bert_tokenizer)
    radgraph_embeddings.append(emb.numpy())

    label_list = [
        v["label"]
        for v in result.get("entities", {}).values()
        if isinstance(v, dict) and "label" in v
    ]
    radgraph_labels.append(label_list)

# Persist cache to disk to avoid re-running RadGraph on next build
extractor.save_cache()

radgraph_embeddings_np = np.vstack(radgraph_embeddings).astype("float32")  # (N, 768)
np.save(os.path.join(ROOT, "store", "radgraph_entity_embeddings.npy"), radgraph_embeddings_np)

with open("rag/radgraph_entity_labels.json", "w", encoding="utf-8") as fh:
    json.dump(radgraph_labels, fh, indent=2, ensure_ascii=False)

print(f"✅ RadGraph metadata saved — shape: {radgraph_embeddings_np.shape}")