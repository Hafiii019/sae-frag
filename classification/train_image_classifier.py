import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import IUXrayMultiViewDataset
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler import ReportClassifier


# =========================================
# Device
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "C:/Datasets/IU_Xray"


# =========================================
# TRAIN SPLIT ONLY
# =========================================
dataset = IUXrayMultiViewDataset(
    root_dir=ROOT,
    split="train"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True
)


# =========================================
# Load trained report classifier
# =========================================
report_model = ReportClassifier().to(device)

report_model.load_state_dict(
    torch.load(
        "classification/report_classifier.pth",
        map_location=device
    )
)

report_model.eval()


# =========================================
# Image classifier (SAEnet based)
# =========================================
image_model = SAEImageClassifier().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    image_model.parameters(),
    lr=1e-4
)

print("====================================")
print("Starting Image Classifier Training")
print("====================================")

image_model.train()

NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):

    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    total_loss = 0

    for images, reports in loop:

        images = images.to(device)

        # Generate pseudo-labels from report classifier
        with torch.no_grad():
            report_logits = report_model(reports)
            labels = (torch.sigmoid(report_logits) > 0.3).float()

        labels = labels.to(device)

        logits = image_model(images)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")


# =========================================
# Save model
# =========================================
torch.save(
    image_model.state_dict(),
    "classification/image_classifier.pth"
)

print("====================================")
print("Image classifier training complete.")
print("Model saved to classification/image_classifier.pth")
print("====================================")