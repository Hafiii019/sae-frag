import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ReportClassifier(nn.Module):

    def __init__(self, num_classes=14):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

        self.encoder = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            torch_dtype="auto"
        )

        hidden_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, reports):

        encoding = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(next(self.parameters()).device)

        outputs = self.encoder(**encoding)

        cls_token = outputs.last_hidden_state[:, 0]  # CLS

        logits = self.classifier(cls_token)

        return logits