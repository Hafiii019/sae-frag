import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(image_emb, text_emb, temperature=0.07):
    """Symmetric NT-Xent loss. Kept for backward compatibility."""
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = torch.matmul(image_emb, text_emb.T) / temperature

    labels = torch.arange(image_emb.size(0)).to(image_emb.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE) loss with a learnable log-temperature (CLIP-style).

    The temperature is parameterised as ``exp(log_temp)`` and clamped to
    ``[0.01, 0.5]`` for numerical stability.  Starting at 0.07 (CLIP default)
    has been shown to outperform fixed-temperature contrastive objectives on
    multi-modal alignment tasks.
    """

    def __init__(self, init_temperature: float = 0.07) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=0.5)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        logits = torch.matmul(img_emb, txt_emb.T) / self.temperature

        labels = torch.arange(img_emb.size(0), device=img_emb.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2