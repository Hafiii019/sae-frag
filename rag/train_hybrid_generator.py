import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import pickle
import numpy as np

from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
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
# LOAD STAGE-1 BACKBONE + ALIGNMENT + PROJECTION (FROZEN)
# =====================================================

checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=device)

visual_encoder = MultiViewBackbone().to(device)
visual_encoder.load_state_dict(checkpoint["visual_model"])
visual_encoder.eval()
for p in visual_encoder.parameters():
    p.requires_grad = False

# CrossModalAlignment: image regions attend to retrieved report tokens
# Produces region-aligned features (B, 49, 256) shaped by report context
alignment = CrossModalAlignment().to(device)
alignment.load_state_dict(checkpoint["alignment"])
alignment.eval()
for p in alignment.parameters():
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

optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-5)

print("Starting Aligned Hybrid Training...")

# =====================================================
# TRAIN LOOP
# =====================================================

for epoch in range(8):

    generator.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for images, reports in loop:

        images = images.to(device)

        with torch.no_grad():

            # ------------------------------------------
            # 1. Visual backbone: (B, 256, 7, 7)
            # ------------------------------------------
            visual_features = visual_encoder(images)

            # ------------------------------------------
            # 2. FAISS retrieval using projected embedding
            # ------------------------------------------
            global_feat = visual_features.flatten(2).mean(dim=2)
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

            # ------------------------------------------
            # 3. Region-Aligned Features
            #    Cross-attention: image regions (Q) attend
            #    to retrieved report tokens (K, V) via
            #    ClinicalBERT embeddings → (B, 49, 256)
            # ------------------------------------------
            aligned_features, _, _ = alignment(visual_features, retrieved_reports)

            # ------------------------------------------
            # 4. Image entity predictions (14 pathologies)
            # ------------------------------------------
            img_logits = image_classifier(images)
            img_entities = (torch.sigmoid(img_logits) > 0.5).float()

        # ------------------------------------------
        # 5. Fact verification: keep only entities
        #    confirmed by BOTH image AND retrieved report
        # ------------------------------------------
        verified_entities = img_entities * rep_entities_batch

        prompts = [
            "Generate a detailed radiology report based on the chest X-ray regions, verified clinical findings, and retrieved context."
            for _ in range(images.size(0))
        ]

        # ------------------------------------------
        # 6. Generator forward pass
        #    aligned_features  → visual tokens (region-aligned)
        #    verified_entities → fact-grounded entity tokens
        #    retrieved_reports → retrieved context embeddings
        #    prompts           → task instruction tokens
        #    reports           → target for teacher-forcing
        # ------------------------------------------
        loss = generator(
            region_features=aligned_features,
            entity_vector=verified_entities,
            retrieved_texts=retrieved_reports,
            prompt_texts=prompts,
            target_texts=reports
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    torch.save(generator.state_dict(), "rag/hybrid_generator.pth")

print("Hybrid Training Complete")