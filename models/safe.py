import torch
import torch.nn as nn


class SAFE(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super(SAFE, self).__init__()

        self.query_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

    def forward(self, c5, p5):
        """
        c5: (B, 2048, 7, 7)
        p5: (B, 256, 7, 7)
        """

        B, _, H, W = c5.shape

        # Project C5
        q = self.query_proj(c5)

        # Flatten
        q_flat = q.flatten(2).transpose(1, 2)     # (B, 49, 256)
        kv_flat = p5.flatten(2).transpose(1, 2)   # (B, 49, 256)

        # Attention
        enhanced, _ = self.mha(q_flat, kv_flat, kv_flat)

        # Restore shape
        enhanced = enhanced.transpose(1, 2).reshape(B, 256, H, W)

        # Residual connection
        return enhanced + q