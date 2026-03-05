import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import IUXrayDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead


def main():

    # ------------------------------
    # Create checkpoint folder
    # ------------------------------
    os.makedirs("checkpoint", exist_ok=True)

    # ------------------------------
    # Device
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # Dataset
    # ------------------------------
    DATA_ROOT = "C:/Datasets/IU_Xray"

    train_dataset = IUXrayDataset(DATA_ROOT, split="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # Windows safe
        pin_memory=True
    )

    # ------------------------------
    # Models
    # ------------------------------
    backbone = MultiViewBackbone().to(device)
    alignment = CrossModalAlignment().to(device)

    proj_img = ProjectionHead().to(device)
    proj_txt = ProjectionHead().to(device)

    # ------------------------------
    # Optimizer
    # ------------------------------
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) +
        list(alignment.parameters()) +
        list(proj_img.parameters()) +
        list(proj_txt.parameters()),
        lr=1e-4
    )

    # ------------------------------
    # Mixed precision
    # ------------------------------
    scaler = torch.cuda.amp.GradScaler()

    # ------------------------------
    # Contrastive loss
    # ------------------------------
    def contrastive_loss(img, txt, temperature=0.07):

        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

        similarity = torch.matmul(img, txt.T) / temperature
        labels = torch.arange(img.size(0)).to(img.device)

        loss_i = F.cross_entropy(similarity, labels)
        loss_t = F.cross_entropy(similarity.T, labels)

        return (loss_i + loss_t) / 2

    # ------------------------------
    # Training settings
    # ------------------------------
    EPOCHS = 50

    # ------------------------------
    # Training loop
    # ------------------------------
    for epoch in range(EPOCHS):

        backbone.train()
        alignment.train()

        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, (images, reports) in enumerate(progress_bar):

            images = images.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                # Encode images
                image_features = backbone(images)

                # Cross modal alignment
                aligned_features, text_cls, _ = alignment(
                    image_features,
                    reports
                )

                # Global embedding
                img_embed = aligned_features.mean(dim=1)

                img_embed = proj_img(img_embed)
                txt_embed = proj_txt(text_cls)

                # Loss
                loss = contrastive_loss(img_embed, txt_embed)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}\n")

        # ------------------------------
        # Save checkpoint
        # ------------------------------
        torch.save({
            "epoch": epoch + 1,
            "backbone": backbone.state_dict(),
            "alignment": alignment.state_dict(),
            "proj_img": proj_img.state_dict(),
            "proj_txt": proj_txt.state_dict(),
            "optimizer": optimizer.state_dict()
        }, f"checkpoint/sae_epoch_{epoch+1}.pth")


# ------------------------------
# Windows entry point
# ------------------------------
if __name__ == "__main__":
    main()