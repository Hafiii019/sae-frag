import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead
from utils.losses import contrastive_loss


# ==========================
# Setup
# ==========================
device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)

print("Using device:", device)


# ==========================
# Dataset
# ==========================
dataset = IUXrayMultiViewDataset(root_dir=Config.DATA_ROOT)

loader = DataLoader(
    dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print("Dataset size:", len(dataset))


# ==========================
# Models
# ==========================
visual_model = MultiViewBackbone().to(device)
alignment = CrossModalAlignment().to(device)
proj_img = ProjectionHead().to(device)
proj_txt = ProjectionHead().to(device)

optimizer = torch.optim.Adam(
    list(visual_model.parameters()) +
    list(alignment.parameters()) +
    list(proj_img.parameters()) +
    list(proj_txt.parameters()),
    lr=Config.LR
)

best_loss = float("inf")

print("Starting Stage-1 Training...")


# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(Config.NUM_EPOCHS):

    visual_model.train()
    alignment.train()
    proj_img.train()
    proj_txt.train()

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    epoch_loss = 0.0

    for images, reports in loop:

        images = images.to(device)

        # Forward
        visual_features = visual_model(images)
        aligned, text_cls, attn_weights = alignment(visual_features, reports)

        img_global = aligned.mean(dim=1)

        img_emb = proj_img(img_global)
        txt_emb = proj_txt(text_cls)

        loss = contrastive_loss(img_emb, txt_emb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(loader)

    print(f"Epoch {epoch+1} Finished | Avg Loss: {avg_epoch_loss:.6f}")

    # Save latest checkpoint
    torch.save({
        "visual_model": visual_model.state_dict(),
        "alignment": alignment.state_dict(),
        "proj_img": proj_img.state_dict(),
        "proj_txt": proj_txt.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": avg_epoch_loss
    }, "checkpoints/latest_stage1.pth")

    # Save best checkpoint
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss

        torch.save({
            "visual_model": visual_model.state_dict(),
            "alignment": alignment.state_dict(),
            "proj_img": proj_img.state_dict(),
            "proj_txt": proj_txt.state_dict(),
            "epoch": epoch,
            "loss": avg_epoch_loss
        }, "checkpoints/best_stage1.pth")

        print("Saved new best model.")


print("Training finished.")

# Save final model
torch.save({
    "visual_model": visual_model.state_dict(),
    "alignment": alignment.state_dict(),
    "proj_img": proj_img.state_dict(),
    "proj_txt": proj_txt.state_dict(),
}, "checkpoints/final_stage1.pth")

print("Final model saved.")


# ==========================
# TOKEN-SPECIFIC HEATMAP
# ==========================
print("Generating token-specific heatmap...")

visual_model.eval()
alignment.eval()

with torch.no_grad():

    images, reports = next(iter(loader))
    images = images.to(device)

    visual_features = visual_model(images)
    aligned, text_cls, attn_weights = alignment(visual_features, reports)

    tokenizer = alignment.tokenizer

    encoding = tokenizer(
        reports,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    tokens = tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    )

    print("Tokens:", tokens)

    target_word = "pneumothorax"

    token_indices = [
        i for i, tok in enumerate(tokens)
        if target_word in tok.replace("##", "")
    ]

    if len(token_indices) == 0:
        print(f"{target_word} not found, using CLS token.")
        token_indices = [0]

    patch_scores = attn_weights[0][:, token_indices].mean(dim=1)

    heatmap = patch_scores.reshape(7, 7).cpu()

    plt.imshow(heatmap, cmap="hot")
    plt.title(f"Attention Heatmap for '{target_word}'")
    plt.colorbar()
    plt.show()