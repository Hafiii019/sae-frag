# 04 — Stage 2: Pathology Entity Classifiers

## Purpose

Classify both X-ray images and radiology report text into **14 binary pathology labels** (one per condition). These labels are called "entity vectors" and serve as fact-grounded anchors during report generation.

The two classifiers complement each other:
- **ReportClassifier** — text side: what does the reference report say is present?
- **SAEImageClassifier** — image side: what does the image visually suggest?

Their intersection forms **verified entities** — findings confirmed by both modalities.

---

## Stage 2a — ReportClassifier (Text Side)

**File:** `classification/report_labeler.py`

### Architecture

```
Input: list[str]  (batch of report texts)
         │
    ClinicalBERT tokenizer + encoder
    → last_hidden_state[:, 0]  (CLS token)   (B, 768)
         │
    Linear(768, 512) → ReLU → Dropout(0.3) → Linear(512, 14)
         │
    Output: logits (B, 14)    — raw, before sigmoid
```

### Usage

```python
logits = report_classifier(["heart size is enlarged..."])
probs  = torch.sigmoid(logits)            # (B, 14) in [0,1]
labels = (probs > 0.5).float()             # binary entity vector
```

### Training

This classifier is assumed to be pre-trained (weights already in `classification/report_classifier.pth`). It serves as a **text oracle** — given a ground-truth report, it outputs which of the 14 conditions are mentioned.

During FAISS index building, it processes every training report once and stores the entity vector alongside the report text in the metadata pickle.

---

## Stage 2b — SAEImageClassifier (Image Side)

**File:** `classification/sae_image_classifier.py`

**Training Script:** `classification/train_image_classifier.py`

### Architecture

```
Input: (B, 2, 3, 224, 224)
         │
    MultiViewBackbone (same SAEnet from Stage 1)
    → visual_features: (B, 256, 7, 7)
         │
    AdaptiveAvgPool2d(1)  → avg_feat: (B, 256)
    AdaptiveMaxPool2d(1)  → max_feat: (B, 256)
         │
    concat → (B, 512)
         │
    Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 14)
         │
    Output: logits (B, 14)
```

Combining average pooling and max pooling:
- **Average pooling** captures diffuse patterns (e.g. bilateral opacity)
- **Max pooling** captures focal findings (e.g. a nodule or focal consolidation)

### Training Script

```bash
python -m classification.train_image_classifier
```

**How pseudo-labels are generated:**

The image classifier has **no ground-truth image labels** in the IU X-Ray dataset. Instead:

1. `ReportClassifier` is loaded (frozen) as a teacher
2. For each training batch, the teacher reads the paired ground-truth report
3. At threshold 0.3 (lower than inference 0.5 — more permissive for label coverage):
   ```python
   labels = (torch.sigmoid(report_logits) > 0.3).float()
   ```
4. These pseudo-labels supervise the image classifier

This is a form of **knowledge distillation from text to vision** — the image model learns to predict the same pathologies that the text model sees in the paired report.

**Training config:**

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| LR | 1e-4 |
| Loss | `BCEWithLogitsLoss` (multi-label) |
| Batch size | 4 |

**Output:** `classification/image_classifier.pth`

---

## Entity Vectors

Both classifiers output a `(B, 14)` binary vector at inference (with sigmoid > 0.5 threshold):

```
entity_vector[b] = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    ↑             ↑              ↑
              No Atelectasis  Effusion        Nodule
```

---

## Fact Verification (Intersection)

The key operation in the RAG pipeline:

```python
verified_entities = img_entities * rep_entities
```

Element-wise multiplication of two binary vectors = logical AND:

```
img_entities:    [0, 1, 0, 1, 1, 0, ...]   ← image says these are present
rep_entities:    [1, 1, 0, 0, 1, 0, ...]   ← retrieved report says these
verified:        [0, 1, 0, 0, 1, 0, ...]   ← only findings confirmed by BOTH
```

**Why this matters:** The image classifier may have false positives; the retrieved report may describe a different patient. The intersection keeps only pathologies supported by both evidence sources, reducing hallucination in the final generated report.

---

## Thresholds Summary

| Context | Threshold | Reason |
|---------|-----------|--------|
| Pseudo-label generation (train image classifier) | 0.3 | Recall-biased — cover more labels so the image model sees sufficient positive examples |
| Image entity prediction (pipeline) | 0.5 | Precision-biased — only confident positives enter the generator |
| Report entity prediction (pipeline) | 0.5 | Same |
| RAG inference debug script | 0.3 | Exploratory — see more candidate findings |

---

## Dependency on Stage 1

`SAEImageClassifier` **contains its own `MultiViewBackbone` instance**. This means:
- Its backbone is trained from scratch during Stage 2b (not loaded from Stage 1)
- The backbone in the image classifier is optimized purely for classification
- Stage 1's backbone is separately loaded and frozen in Stages 3 and 4 for feature extraction and FAISS retrieval
