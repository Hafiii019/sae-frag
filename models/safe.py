import torch
import torch.nn as nn


class SAFE(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()

        self.query_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, c5, p5):
        """
        c5: (B, 2048, 7, 7)
        p5: (B, 256, 7, 7)
        """

        B, _, H, W = c5.shape

        # Project C5 to embed_dim
        q = self.query_proj(c5)  # (B, 256, 7, 7)

        # Flatten spatial dimensions
        q = q.flatten(2).transpose(1, 2)   # (B, 49, 256)
        kv = p5.flatten(2).transpose(1, 2) # (B, 49, 256)

        # Multi-head attention
        enhanced, _ = self.mha(q, kv, kv)

        # Reshape back
        enhanced = enhanced.transpose(1, 2).reshape(B, 256, H, W)

        return enhanced