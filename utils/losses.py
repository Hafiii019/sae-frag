import torch
import torch.nn.functional as F


def contrastive_loss(image_emb, text_emb, temperature=0.2):
    logits = torch.matmul(image_emb, text_emb.T) / temperature

    labels = torch.arange(image_emb.size(0)).to(image_emb.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2