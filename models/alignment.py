import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossModalAlignment(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )

        self.text_encoder = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            use_safetensors=True
        )

        # Freeze BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_proj = nn.Linear(768, embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, image_features, reports):
        """
        image_features: (B, 256, 7, 7)
        reports: list[str]
        """

        B, C, H, W = image_features.shape

        # Flatten patches → (B, 49, 256)
        img_tokens = image_features.flatten(2).transpose(1, 2)

        # Tokenize reports
        encoding = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(image_features.device)

        text_outputs = self.text_encoder(**encoding)
        text_tokens = text_outputs.last_hidden_state  # (B, L, 768)

        # Project text to 256
        text_tokens = self.text_proj(text_tokens)  # (B, L, 256)

        # CLS token for global embedding
        cls_token = text_tokens[:, 0]  # (B, 256)

        # Cross attention
        aligned_features, attn_weights = self.cross_attention(
            img_tokens,
            text_tokens,
            text_tokens
        )
        # attn_weights: (B, num_heads, 49, L)

        return aligned_features, cls_token, attn_weights