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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "C:/Datasets/IU_Xray"

dataset = IUXrayMultiViewDataset(ROOT, split="train")
loader = DataLoader(dataset, batch_size=1)

# Load Stage-1
checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=device)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()

proj_img = ProjectionHead().to(device)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()

# Load report classifier ONCE
report_classifier = ReportClassifier().to(device)
report_classifier.load_state_dict(
    torch.load("classification/report_classifier.pth", map_location=device)
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

faiss.write_index(index, "rag/faiss_index.bin")

with open("rag/train_reports.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAST FAISS built.")