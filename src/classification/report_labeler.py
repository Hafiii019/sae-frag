import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Use local safetensors copy when available (avoids transformers 5.x torch.load
# safety check for old .bin models — see scripts/prepare/convert_models.py).
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from configs.config import BIO_CLINICAL_BERT


class ReportClassifier(nn.Module):

    def __init__(self, num_classes=14):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(BIO_CLINICAL_BERT)

        self.encoder = AutoModel.from_pretrained(
            BIO_CLINICAL_BERT,
            torch_dtype="auto",
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