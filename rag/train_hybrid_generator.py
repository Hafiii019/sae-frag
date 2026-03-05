import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import pickle
import numpy as np

from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead
from classification.sae_image_classifier import SAEImageClassifier
from rag.hybrid_generator import HybridReportGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "C:/Datasets/IU_Xray"

# =====================================================
# DATA
# =====================================================

train_dataset = IUXrayMultiViewDataset(ROOT, split="train")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# =====================================================
# LOAD FAISS + METADATA
# =====================================================

index = faiss.read_index("rag/faiss_index.bin")

with open("rag/train_reports.pkl", "rb") as f:
    train_metadata = pickle.load(f)

# =====================================================
# LOAD STAGE-1 BACKBONE + PROJECTION (FROZEN)
# =====================================================

checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=device)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()
for p in visual_encoder.parameters():
    p.requires_grad = False

proj_img = ProjectionHead().to(device)
proj_img.load_state_dict(checkpoint["proj_img"])
proj_img.eval()
for p in proj_img.parameters():
    p.requires_grad = False

# =====================================================
# LOAD IMAGE CLASSIFIER (FROZEN)
# =====================================================

image_classifier = SAEImageClassifier().to(device)
image_classifier.load_state_dict(
    torch.load("classification/image_classifier.pth", map_location=device)
)
image_classifier.eval()
for p in image_classifier.parameters():
    p.requires_grad = False

# =====================================================
# HYBRID GENERATOR
# =====================================================

generator = HybridReportGenerator().to(device)

# LOWER LR FOR STABILITY
optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-5)

print("🚀 Stable Hybrid Training Started")

# =====================================================
# TRAIN LOOP
# =====================================================

for epoch in range(8):

    generator.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, reports in loop:

        images = images.to(device)

        # ---------------------------
        # 1️⃣ Region Features
        # ---------------------------
        with torch.no_grad():
            feats = visual_encoder(images)
            region_features = feats.flatten(2).transpose(1, 2)

        # ---------------------------
        # 2️⃣ Image Entities
        # ---------------------------
        with torch.no_grad():
            img_logits = image_classifier(images)
            img_entities = (torch.sigmoid(img_logits) > 0.5).float()

        # ---------------------------
        # 3️⃣ FAISS Retrieval
        # ---------------------------
        with torch.no_grad():
            global_feat = feats.flatten(2).mean(dim=2)
            img_emb = proj_img(global_feat)

            img_np = img_emb.cpu().numpy().astype("float32")
            D, I = index.search(img_np, k=1)

        retrieved_reports = []
        rep_entities_batch = []

        for b in range(images.size(0)):
            idx = I[b][0]
            retrieved_reports.append(train_metadata[idx]["report"])
            rep_entities_batch.append(train_metadata[idx]["entity_vector"])

        rep_entities_batch = torch.cat(rep_entities_batch).to(device)
        verified_entities = img_entities * rep_entities_batch

        prompts = [
            "Generate a detailed radiology report using region features and verified clinical findings."
            for _ in range(images.size(0))
        ]

        # ---------------------------
        # 4️⃣ Forward Pass
        # ---------------------------
        loss = generator(
            region_features=region_features,
            entity_vector=verified_entities,
            retrieved_texts=retrieved_reports,
            prompt_texts=prompts,
            target_texts=reports
        )

        # ---------------------------
        # 5️⃣ Backprop (Stable)
        # ---------------------------
        optimizer.zero_grad()
        loss.backward()

        # VERY IMPORTANT
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    torch.save(generator.state_dict(), "rag/hybrid_generator.pth")

print("✅ Hybrid Training Complete")