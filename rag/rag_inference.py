import torch
import torch.nn.functional as F
import faiss
import numpy as np
import pickle
from torch.utils.data import DataLoader

from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead

from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier


# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "C:/Datasets/IU_Xray"

# =====================================================
# LOAD TEST DATASET
# =====================================================
test_dataset = IUXrayMultiViewDataset(ROOT, split="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("TEST split loaded with", len(test_dataset), "samples.")

# =====================================================
# LOAD FAISS INDEX
# =====================================================
index = faiss.read_index("rag/faiss_index.bin")
print("FAISS size:", index.ntotal)

# =====================================================
# LOAD TRAIN METADATA
# =====================================================
with open("rag/train_reports.pkl", "rb") as f:
    train_samples = pickle.load(f)

print("Train metadata size:", len(train_samples))

# =====================================================
# LOAD MODELS
# =====================================================
visual_model = MultiViewBackbone().to(device)
proj_img = ProjectionHead().to(device)

visual_model.eval()
proj_img.eval()

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

# =====================================================
# RAG INFERENCE
# =====================================================
with torch.no_grad():

    for images, true_report in test_loader:

        images = images.to(device)

        # ---------------------------------------------
        # 1️⃣ Encode Query Image
        # ---------------------------------------------
        visual_features = visual_model(images)  # (B,256,7,7)

        img_global = visual_features.flatten(2).mean(dim=2)  # (B,256)

        img_emb = proj_img(img_global)

        # Normalize for cosine similarity
        img_emb = F.normalize(img_emb, dim=1)

        img_emb_np = img_emb.cpu().numpy().astype("float32")

        # ---------------------------------------------
        # 2️⃣ Retrieve Top-K Reports
        # ---------------------------------------------
        D, I = index.search(img_emb_np, k=3)

        print("\n================================================")
        print("TRUE REPORT:\n")
        print(true_report[0])

        # ---------------------------------------------
        # 3️⃣ Predict Image Entities
        # ---------------------------------------------
        img_logits = image_classifier(images)
        img_labels = (torch.sigmoid(img_logits) > 0.3).int().cpu().numpy()[0]

        print("\nImage Predicted Label Vector:")
        print(img_labels)

        print("\nRETRIEVED TRAIN REPORTS + VERIFIED OVERLAP:")

        # ---------------------------------------------
        # 4️⃣ Verify Each Retrieved Report
        # ---------------------------------------------
        for rank, idx in enumerate(I[0]):

            idx = int(idx)  # ensure python int

            retrieved_report = train_samples[idx]["report"]

            print(f"\n--- Rank {rank+1} ---")
            print(retrieved_report)

            # Predict retrieved report entities
            rep_logits = report_classifier([retrieved_report])
            rep_labels = (torch.sigmoid(rep_logits) > 0.3).int().cpu().numpy()[0]

            # Intersection
            intersection = img_labels & rep_labels

            print("Retrieved Label Vector:", rep_labels)
            print("Verified Overlap:", intersection)

        break  # show only 1 example