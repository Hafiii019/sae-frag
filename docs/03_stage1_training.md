# 03 — Stage 1: Visual-Language Alignment

## Goal

Train a visual backbone that produces **region-level features aligned to clinical language**. At the end of Stage 1, the model can embed an X-ray image into the same semantic space as radiology report text.

This is the foundation that all subsequent stages depend on.

---

## Script

```bash
python train.py
```

Config is in `configs/config.py`:

| Parameter | Value |
|-----------|-------|
| `BATCH_SIZE` | 4 |
| `NUM_EPOCHS` | 50 |
| `LR` | 3e-5 |
| `IMAGE_SIZE` | 224 |
| `DEVICE` | cuda |

---

## Models Trained

### 1. MultiViewBackbone (`models/multiview_backbone.py`)

Processes both X-ray views simultaneously:

```
View 1: (B, 3, 224, 224)
         │
    ResNet101Backbone
    layer0 → layer1 (C2: 256ch) → layer2 (C3: 512ch)
           → layer3 (C4: 1024ch) → layer4 (C5: 2048ch)
         │
    FPN (Feature Pyramid Network)
    lateral 1x1 convs → top-down fusion with upsampling
    P2(256), P3(256), P4(256), P5(256)
         │
    SAFE (Spatial Attention Feature Enhancement)
    MHA(Q=C5_proj, K=P5, V=P5) + residual
    → enhanced_1: (B, 256, 7, 7)

View 2: same path → enhanced_2: (B, 256, 7, 7)

fused = (enhanced_1 + enhanced_2) / 2   → (B, 256, 7, 7)
```

**ResNet101** provides multi-scale hierarchical features.
**FPN** aggregates them top-down so P5 has both semantic depth from C5 and spatial precision from lower layers.
**SAFE** uses multi-head attention to let the deep semantic features (C5) selectively attend to the pyramid features (P5), enhancing spatially important regions.
**View averaging** fuses the frontal and lateral views into one representation.

---

### 2. CrossModalAlignment (`models/alignment.py`)

Aligns image region tokens to radiology report tokens:

```
image_features: (B, 256, 7, 7)
reports: list[str]

→ img_tokens = flatten + transpose → (B, 49, 256)
                                      49 = 7×7 spatial patches

→ ClinicalBERT tokenize + encode reports
  → text_tokens: (B, L, 768)
  → text_proj: (B, L, 256)   (project 768→256 to match image dim)

→ cls_token = text_tokens[:, 0]  (global report embedding)

→ MultiheadAttention (8 heads, embed=256, batch_first=True):
    Q = img_tokens   (image regions ask questions)
    K = text_tokens  (text tokens provide keys)
    V = text_tokens  (text tokens provide values)

    Each image patch attends to ALL report tokens
    → learns which text concepts correspond to which image regions

→ Output: aligned_features (B, 49, 256)
          attn_weights     (B, 49, L)
```

**Why ClinicalBERT?** The model `emilyalsentzer/Bio_ClinicalBERT` is fine-tuned on clinical notes, giving it domain-specific understanding of radiology terminology. Its weights are frozen during training — only `text_proj` and the cross-attention parameters are learned.

---

### 3. ProjectionHead (`models/projection.py`)

Maps both image and text global embeddings into the shared contrastive space:

```
Input: (B, 256)
→ Linear(256, 512) → ReLU → Linear(512, 256) → L2 normalize
Output: (B, 256) — unit sphere
```

Two separate heads:
- `proj_img`: projects `aligned_features.mean(dim=1)` (mean-pooled image regions)
- `proj_txt`: projects `cls_token` (ClinicalBERT CLS)

---

## Loss Function: Symmetric Contrastive Loss

**File:** `utils/losses.py`

```python
def contrastive_loss(image_emb, text_emb, temperature=0.2):
    logits = (image_emb @ text_emb.T) / temperature
    labels = torch.arange(B)          # diagonal = matched pairs
    loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
```

This is the **InfoNCE / CLIP-style** loss. For a batch of B samples:
- The diagonal of the `B×B` similarity matrix = correct image-report pairs
- Every other cell = negative pairs (other patients' reports)
- Temperature = 0.2 (sharper than CLIP's 0.07) tuned for medical domain

---

## Checkpoints Saved

| File | Saved When |
|------|-----------|
| `checkpoints/latest_stage1.pth` | Every epoch |
| `checkpoints/best_stage1.pth` | When loss improves |
| `checkpoints/final_stage1.pth` | After all epochs |

**Checkpoint contents:**
```python
{
    "visual_model": visual_model.state_dict(),
    "alignment":    alignment.state_dict(),
    "proj_img":     proj_img.state_dict(),
    "proj_txt":     proj_txt.state_dict(),
    "optimizer":    optimizer.state_dict(),
    "epoch":        epoch,
    "loss":         avg_epoch_loss
}
```

**Important:** Later stages load with the key names above:
```python
checkpoint = torch.load("checkpoints/best_stage1.pth")
visual_encoder.load_state_dict(checkpoint["visual_model"])
alignment.load_state_dict(checkpoint["alignment"])
proj_img.load_state_dict(checkpoint["proj_img"])
```

---

## Attention Heatmap (Post-Training)

After training, `train.py` generates a token-specific attention heatmap:

1. Takes one batch from the loader
2. Runs alignment to get `attn_weights (B, 49, L)`
3. Finds token indices in the report matching a target word (e.g. "pneumothorax")
4. Averages attention weights across those tokens → `(49,)` patch scores
5. Reshapes to `(7, 7)` and plots as a heatmap

This visualizes which 7×7 image regions the model associates with that clinical concept.

---

## What is Learned

After Stage 1:
- The backbone produces features where **semantically similar X-rays are nearby** in the embedding space
- Each of the 49 patch features carries information about **which clinical concepts are relevant** to that spatial region (via cross-attention from the report)
- The alignment between image regions and text tokens is used as the visual input to the Stage 4 generator, ensuring the decoder receives **text-conditioned regional features** rather than raw backbone outputs
