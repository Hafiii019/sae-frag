import os
import re
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def _clean_report(text: str) -> str:
    """Normalise raw IU X-Ray report text.

    Caps at 60 words as specified in SAENet (Cao et al. 2025) Section 4.1:
    "The text length of each report is limited to 60 words."
    This aligns train targets with the paper and prevents the generator from
    learning to produce overly long outputs that T5 cannot decode efficiently.
    """
    text = text.lower().strip()
    text = re.sub(r'\bxxxx\b', '', text)       # remove placeholders
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace
    text = re.sub(r'[^\w\s.,;:\-]', '', text)  # strip junk chars
    text = text.strip()
    # Limit to 60 words — matches SAENet paper dataset setting
    words = text.split()
    if len(words) > 60:
        text = " ".join(words[:60])
    return text


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
        # Train uses augmentation; val/test use deterministic resize only.
        #
        # CXR-safe augmentation rules:
        #  ✗ No horizontal flip  — left/right heart position is diagnostic
        #  ✗ No vertical flip    — lung anatomy is orientation-dependent
        #  ✗ No large rotation   — > 15° not realistic for real acquisitions
        #  ✓ RandomResizedCrop   — patient distance / zoom variation
        #  ✓ Small rotation      — patient positioning on the table
        #  ✓ Brightness/contrast — scanner exposure variation
        #  ✓ GaussianBlur        — different scanner sharpness / motion blur
        #  ✓ RandomAdjustSharpness — detail emphasis variation
        #  ✓ RandomEqualize      — global histogram / CLAHE-like normalisation
        #  ✓ RandomErasing       — occlusion robustness (foreign objects, wires)
        #
        # Each view (frontal, lateral) gets the transform applied independently
        # with different random seeds, so the two views in a pair always receive
        # a different random augmentation — giving SimCLR-style contrastive pairs
        # at zero extra cost.
        # ------------------------
        _norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if split == "train":
            self.transform = transforms.Compose([
                # Geometry — simulate patient positioning and zoom
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),

                # Intensity — simulate scanner and exposure variation
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                transforms.RandomEqualize(p=0.2),  # CLAHE-like global contrast
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                    p=0.3,
                ),

                # Convert to tensor before pixel-level augmentations
                transforms.ToTensor(),
                _norm,

                # Occlusion — simulate wires, tubes, foreign objects
                transforms.RandomErasing(
                    p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.0), value=0
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                _norm,
            ])

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

        # Report is already cleaned by _clean_report() during __init__
        report = sample["report"]

        return images, report