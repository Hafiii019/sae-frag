import torch
import torch.nn.functional as F


def contrastive_loss(image_emb, text_emb, temperature=0.2):

    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    logits = torch.matmul(image_emb, text_emb.T) / temperature

    labels = torch.arange(image_emb.size(0)).to(image_emb.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    loss = (loss_i2t + loss_t2i) / 2

    return loss