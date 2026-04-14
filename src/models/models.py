"""SAE-FRAG model definitions — all classes in one place.

Import via the package (preferred):

    from models import MultiViewBackbone, CrossModalAlignment, ProjectionHead

Class hierarchy
---------------
ResNet101Backbone       ResNet-101 split into C2-C5 stages
FPN                     Feature Pyramid Network (P2-P5, 256-ch)
SAFE                    Spatial Attention Feature Enhancement (C5 queries P4)
MultiViewBackbone       ResNet-101 + FPN + SAFE; fuses frontal + lateral views
CrossModalAlignment     Bio_ClinicalBERT cross-modal cross-attention
ProjectionHead          MLP + L2-norm projection head (query/image-text encoder)
DocumentProjectionHead  Same architecture, independent weights (doc encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────
# 1. ResNet-101 Backbone
# ──────────────────────────────────────────────────────────────────────────

class ResNet101Backbone(nn.Module):
    """ResNet-101 split into four feature stages.

    Returns (C2, C3, C4, C5) — channels (256, 512, 1024, 2048).
    Spatial sizes at 224×224 input: 56, 28, 14, 7.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        resnet  = tv_models.resnet101(weights=weights)

        # layer0: conv1 → bn1 → relu → maxpool  (224 → 56)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1   # C2 — 56×56, 256-ch
        self.layer2 = resnet.layer2   # C3 — 28×28, 512-ch
        self.layer3 = resnet.layer3   # C4 — 14×14, 1024-ch
        self.layer4 = resnet.layer4   # C5 —  7×7,  2048-ch

    def forward(self, x):
        x  = self.layer0(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


# ──────────────────────────────────────────────────────────────────────────
# 2. Feature Pyramid Network
# ──────────────────────────────────────────────────────────────────────────

class FPN(nn.Module):
    """Standard top-down FPN with lateral connections.

    Input : (C2, C3, C4, C5) from ResNet — channels (256, 512, 1024, 2048)
    Output: (P2, P3, P4, P5) — all 256-ch, spatial sizes match input stages
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.lateral_c2 = nn.Conv2d(256,  out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(512,  out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.output_p2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p3  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p4  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p5  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        return (
            self.output_p2(p2),
            self.output_p3(p3),
            self.output_p4(p4),
            self.output_p5(p5),
        )


# ──────────────────────────────────────────────────────────────────────────
# 3. SAFE — Spatial Attention Feature Enhancement
# ──────────────────────────────────────────────────────────────────────────

class SAFE(nn.Module):
    """Spatial Attention Feature Enhancement.

    C5 semantic features (7×7) query P3 detail features (28×28) via
    multi-head cross-attention.  P3 retains 16× more spatial tokens than P5,
    preserving small nodules, localised effusions, and subtle opacities.
    Using P3 (28×28=784 tokens) rather than P2 (56×56=3136) keeps VRAM
    within the 6 GB budget while giving 4× better detail than P4 (14×14).

    Matches SAENet eq.5: V' = MHA(Vf, V'f, V'f) + MHA(Vl, V'l, V'l)
    where Vf = C5 global features and V'f = fused P3 local features.

    Input  — c5: (B, 2048, 7, 7)   p3: (B, 256, 28, 28)
    Output — (B, 256, 28, 28)
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.query_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, c5, p3):
        B, _, H3, W3 = p3.shape
        q = self.query_proj(c5)                                      # (B, 256, 7, 7)
        q = F.interpolate(q, size=(H3, W3), mode="bilinear",
                          align_corners=False)                        # (B, 256, 28, 28)
        q_flat  = q.flatten(2).transpose(1, 2)                       # (B, 784, 256)
        kv_flat = p3.flatten(2).transpose(1, 2)                      # (B, 784, 256)
        enhanced, _ = self.mha(q_flat, kv_flat, kv_flat)            # (B, 784, 256)
        enhanced = enhanced.transpose(1, 2).reshape(B, 256, H3, W3)
        return enhanced + q


# ──────────────────────────────────────────────────────────────────────────
# 4. Multi-View Backbone
# ──────────────────────────────────────────────────────────────────────────

class MultiViewBackbone(nn.Module):
    """Dual-view visual encoder: ResNet-101 → FPN → SAFE.

    Input  : (B, 2, 3, 224, 224)
    Output : (B, 256, 28, 28)  — 784 spatial tokens at P3 resolution

    View fusion matches SAENet eq.5:
        V' = MHA(Vf, V'f, V'f) + MHA(Vl, V'l, V'l)
    i.e. each view is independently enhanced by SAFE then summed.
    NOTE: requires full pipeline rebuild (stage1 → build_index → cache_features)
    """

    def __init__(self):
        super().__init__()
        self.backbone = ResNet101Backbone(pretrained=True)
        self.fpn      = FPN(out_channels=256)
        self.safe     = SAFE(embed_dim=256, num_heads=8)

    def forward(self, x):
        B, V, C, H, W = x.shape

        def _encode(view):
            c2, c3, c4, c5 = self.backbone(view)
            _, p3, _, _    = self.fpn(c2, c3, c4, c5)   # P3: 28×28, 256-ch
            return self.safe(c5, p3)

        feat1 = _encode(x[:, 0])   # (B, 256, 28, 28) frontal
        feat2 = _encode(x[:, 1])   # (B, 256, 28, 28) lateral
        # Additive dual-view fusion: SAENet eq.5 V' = MHA(Vf,...) + MHA(Vl,...)
        return feat1 + feat2


# ──────────────────────────────────────────────────────────────────────────
# 5. Cross-Modal Alignment
# ──────────────────────────────────────────────────────────────────────────

class CrossModalAlignment(nn.Module):
    """Bio_ClinicalBERT text encoder + image-text cross-attention.

    The BERT encoder is frozen; only the linear projection and cross-attention
    head are trained.

    Input
    -----
    image_features : (B, 256, H, W)  — any spatial size; P4 gives (B,256,14,14)
    reports        : list[str]

    Output
    ------
    aligned_features : (B, H*W, 256) — image tokens after cross-attending text
    cls_token        : (B, 256)      — global text embedding
    attn_weights     : (B, H*W, L)   — attention weight map
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.tokenizer    = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_encoder = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", torch_dtype="auto"
        )
        # Freeze BERT — only the projection and attention head are trained
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_proj     = nn.Linear(768, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, image_features, reports):
        B, C, H, W = image_features.shape
        img_tokens  = image_features.flatten(2).transpose(1, 2)
        encoding = self.tokenizer(
            reports, padding=True, truncation=True, return_tensors="pt"
        ).to(image_features.device)
        text_out    = self.text_encoder(**encoding)
        text_tokens = self.text_proj(text_out.last_hidden_state)
        cls_token   = text_tokens[:, 0]
        aligned, attn = self.cross_attention(img_tokens, text_tokens, text_tokens)
        return aligned, cls_token, attn


# ──────────────────────────────────────────────────────────────────────────
# 6. Projection Heads
# ──────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP + L2-norm projection head (query / image-text contrastive encoder).

    Input: (B, input_dim)  →  Output: (B, output_dim) L2-normalised
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class DocumentProjectionHead(nn.Module):
    """Same as ProjectionHead with independent weights for the FactMM-RAG doc encoder."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────
# 1. ResNet-101 Backbone
# ──────────────────────────────────────────────────────────────────────────

class ResNet101Backbone(nn.Module):
    """ResNet-101 split into four feature stages.

    Returns (C2, C3, C4, C5) — channels (256, 512, 1024, 2048).
    Spatial sizes at 224×224 input: 56, 28, 14, 7.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        resnet  = tv_models.resnet101(weights=weights)

        # layer0: conv1 → bn1 → relu → maxpool  (224 → 56)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1   # C2 — 56×56, 256-ch
        self.layer2 = resnet.layer2   # C3 — 28×28, 512-ch
        self.layer3 = resnet.layer3   # C4 — 14×14, 1024-ch
        self.layer4 = resnet.layer4   # C5 —  7×7,  2048-ch

    def forward(self, x):
        x  = self.layer0(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


# ──────────────────────────────────────────────────────────────────────────
# 2. Feature Pyramid Network
# ──────────────────────────────────────────────────────────────────────────

class FPN(nn.Module):
    """Standard top-down FPN with lateral connections.

    Input : (C2, C3, C4, C5) from ResNet — channels (256, 512, 1024, 2048)
    Output: (P2, P3, P4, P5) — all 256-ch, spatial sizes match input stages
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        # Lateral 1×1 projections
        self.lateral_c2 = nn.Conv2d(256,  out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(512,  out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(2048, out_channels, kernel_size=1)
        # Output smoothing 3×3
        self.output_p2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p3  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p4  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.output_p5  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        return (
            self.output_p2(p2),
            self.output_p3(p3),
            self.output_p4(p4),
            self.output_p5(p5),
        )


# ──────────────────────────────────────────────────────────────────────────
# 3. SAFE — Spatial Attention Feature Enhancement
# ──────────────────────────────────────────────────────────────────────────

class SAFE(nn.Module):
    """Spatial Attention Feature Enhancement.

    C5 semantic features (7×7) query P4 detail features (14×14) via
    multi-head cross-attention.  P4 retains 4× more spatial tokens than P5,
    preserving small nodules, localised effusions, and subtle opacities.

    Input
    -----
    c5 : (B, 2048, 7,  7)  — high-level backbone features
    p4 : (B, 256,  14, 14) — FPN P4 mid-level features

    Output
    ------
    (B, 256, 14, 14)       — attention-enhanced feature map
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.query_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, c5, p4):
        B, _, H4, W4 = p4.shape

        # Project + upsample C5 to match P4 resolution
        q = self.query_proj(c5)                                     # (B, 256, 7, 7)
        q = F.interpolate(q, size=(H4, W4), mode="bilinear",
                          align_corners=False)                       # (B, 256, 14, 14)

        q_flat  = q.flatten(2).transpose(1, 2)                      # (B, 196, 256)
        kv_flat = p4.flatten(2).transpose(1, 2)                     # (B, 196, 256)

        enhanced, _ = self.mha(q_flat, kv_flat, kv_flat)           # (B, 196, 256)
        enhanced = enhanced.transpose(1, 2).reshape(B, 256, H4, W4)
        return enhanced + q                                          # residual


# ──────────────────────────────────────────────────────────────────────────
# 4. Multi-View Backbone
# ──────────────────────────────────────────────────────────────────────────

class MultiViewBackbone(nn.Module):
    """Dual-view visual encoder: ResNet-101 → FPN → SAFE.

    Processes frontal and lateral CXR views independently, then fuses them
    with a learned attention weight (rather than a naive average).

    Input  : (B, 2, 3, 224, 224)
    Output : (B, 256, 14, 14)  — 196 spatial tokens at P4 resolution
    """

    def __init__(self):
        super().__init__()
        self.backbone  = ResNet101Backbone(pretrained=True)
        self.fpn       = FPN(out_channels=256)
        self.safe      = SAFE(embed_dim=256, num_heads=8)
        # Learnable per-view fusion weights (softmax-normalised)
        self.view_attn = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        B, V, C, H, W = x.shape   # V == 2 always (frontal, lateral)

        def _encode(view):
            c2, c3, c4, c5 = self.backbone(view)
            _, _, p4, _    = self.fpn(c2, c3, c4, c5)
            return self.safe(c5, p4)                   # (B, 256, 14, 14)

        feat1 = _encode(x[:, 0])
        feat2 = _encode(x[:, 1])

        w = torch.softmax(self.view_attn, dim=0)       # (2,)
        return w[0] * feat1 + w[1] * feat2             # (B, 256, 14, 14)


# ──────────────────────────────────────────────────────────────────────────
# 5. Cross-Modal Alignment
# ──────────────────────────────────────────────────────────────────────────

class CrossModalAlignment(nn.Module):
    """Bio_ClinicalBERT text encoder + image-text cross-attention.

    The BERT encoder is frozen; only the linear projection and cross-attention
    head are trained.

    Input
    -----
    image_features : (B, 256, H, W)  — any spatial size; P4 gives (B,256,14,14)
    reports        : list[str]

    Output
    ------
    aligned_features : (B, H*W, 256) — image tokens after cross-attending text
    cls_token        : (B, 256)      — global text embedding
    attn_weights     : (B, H*W, L)   — attention weight map
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.tokenizer    = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_encoder = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", torch_dtype="auto"
        )
        # Freeze BERT — only the projection and attention head are trained
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_proj     = nn.Linear(768, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, image_features, reports):
        B, C, H, W = image_features.shape
        img_tokens  = image_features.flatten(2).transpose(1, 2)    # (B, H*W, 256)

        encoding = self.tokenizer(
            reports, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(image_features.device)

        text_out   = self.text_encoder(**encoding)
        text_tokens = self.text_proj(text_out.last_hidden_state)    # (B, L, 256)
        cls_token   = text_tokens[:, 0]                             # (B, 256)

        aligned, attn = self.cross_attention(img_tokens, text_tokens, text_tokens)
        return aligned, cls_token, attn


# ──────────────────────────────────────────────────────────────────────────
# 6. Projection Heads
# ──────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP projection head with L2 normalisation (query / image-text encoder).

    Used in Stage-1 contrastive training and the FactMM-RAG query encoder.

    Input  : (B, input_dim)
    Output : (B, output_dim)  — L2-normalised
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class DocumentProjectionHead(nn.Module):
    """Projection head for the FactMM-RAG document encoder (image + text).

    Identical architecture to ProjectionHead but with independent weights,
    so query (image-only) and document (image+text) spaces are trained
    separately and aligned via InfoNCE loss.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
