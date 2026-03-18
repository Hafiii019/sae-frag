import os
import re
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def _clean_report(text: str) -> str:
    """Normalise raw IU X-Ray report text."""
    text = text.lower().strip()
    text = re.sub(r'\bxxxx\b', '', text)       # remove placeholders
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace
    text = re.sub(r'[^\w\s.,;:\-]', '', text)  # strip junk chars
    return text.strip()


class IUXrayMultiViewDataset(Dataset):

    def __init__(self, root_dir, split="train"):

        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images", "images_normalized")

        reports_path = os.path.join(root_dir, "indiana_reports.csv")
        projections_path = os.path.join(root_dir, "indiana_projections.csv")
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        split_path = os.path.join(_repo_root, "data", "splits", "iu_split.csv")

        # ------------------------
        # Load CSV files
        # ------------------------

        reports = pd.read_csv(reports_path)
        projections = pd.read_csv(projections_path)
        split_df = pd.read_csv(split_path)

        # ------------------------
        # Select split
        # ------------------------

        split_uids = split_df[split_df["split"] == split]["uid"]

        reports = reports[reports["uid"].isin(split_uids)]
        projections = projections[projections["uid"].isin(split_uids)]

        # ------------------------
        # Merge tables
        # ------------------------

        merged = reports.merge(projections, on="uid")

        # Combine findings + impression, then normalise
        merged["report"] = (
            merged["findings"].fillna("") + " " +
            merged["impression"].fillna("")
        ).apply(_clean_report)

        # ------------------------
        # Group by patient
        # ------------------------

        grouped = merged.groupby("uid")

        self.samples = []

        for uid, group in grouped:

            image_files = list(group["filename"])
            report_text = group["report"].iloc[0]

            if len(image_files) >= 1:

                self.samples.append({
                    "uid": uid,
                    "images": image_files,
                    "report": report_text
                })

        print(f"{split} samples:", len(self.samples))

        # ------------------------
        # Image transforms
        # ------------------------

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    
    def clean_report(self, text):

        text = text.lower()

        # remove IU-Xray placeholders
        text = text.replace("xxxx", "")
        text = text.replace("x-xxxx", "")
        text = text.replace("xxxx.", "")

        # remove leftover artifacts
        text = text.replace("x-", "")

        # collapse spaces
        text = " ".join(text.split())

        return text


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        images = []

        # ------------------------
        # Load up to 2 views
        # ------------------------

        for img_name in sample["images"][:2]:

            img_path = os.path.join(self.image_dir, img_name)

            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

            images.append(image)

        # If only one image exists → duplicate it
        if len(images) == 1:
            images.append(images[0])

        images = torch.stack(images)

        # ------------------------
        # Clean report
        # ------------------------

        report = self.clean_report(sample["report"])

        return images, report