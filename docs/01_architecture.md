# 01 — System Architecture

## High-Level Architecture

```
                        IU X-RAY DATASET
                      (frontal + lateral views)
                              │
                              ▼
              ┌───────────────────────────────┐
              │       DATA PIPELINE           │
              │  IUXrayMultiViewDataset        │
              │  → (B, 2, 3, 224, 224)         │
              │  → cleaned report string       │
              └──────────────┬────────────────┘
                             │
              ┌──────────────▼────────────────┐
              │  STAGE 1: VISUAL-LANGUAGE     │
              │  ALIGNMENT (train.py)          │
              │                               │
              │  MultiViewBackbone             │
              │   ResNet101 → FPN → SAFE       │
              │    fused (B, 256, 7, 7)        │
              │         │                     │
              │  CrossModalAlignment           │
              │   image regions ←→ ClinBERT   │
              │    aligned (B, 49, 256)        │
              │         │                     │
              │  ProjectionHead (img + txt)    │
              │   contrastive loss             │
              │         │                     │
              │  Save: best_stage1.pth         │
              └──────────────┬────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────┐   ┌────────────────────┐
│  STAGE 2a       │ │  STAGE 2b   │   │  STAGE 3           │
│  ReportClassif. │ │  ImageClass.│   │  FAISS INDEX       │
│ (ClinicalBERT)  │ │ (SAEnet)    │   │  build_faiss_fast  │
│  14 pathology   │ │  14 patho.  │   │  - embed train set │
│  pseudo-labels  │ │  labels     │   │  - store metadata  │
│  report_classif.│ │  image_clas.│   │  faiss_index.bin   │
│  .pth           │ │  sifier.pth │   │  train_reports.pkl │
└────────┬────────┘ └──────┬──────┘   └──────────┬─────────┘
         │                 │                     │
         └─────────────────┴──────────┬──────────┘
                                      │
              ┌───────────────────────▼───────────────────┐
              │  STAGE 4: HYBRID GENERATOR TRAINING        │
              │  train_hybrid_generator.py                  │
              │                                            │
              │  For each training sample:                  │
              │                                            │
              │  1. visual_features = Backbone(image)       │
              │  2. retrieved_report = FAISS.search(img)    │
              │  3. aligned_features =                      │
              │        Alignment(visual, retrieved_report)  │
              │  4. img_entities = ImageClassifier(image)   │
              │  5. rep_entities = metadata[idx].entities   │
              │  6. verified = img_entities AND rep_entities│
              │  7. loss = Generator(                       │
              │        aligned_features,   ← visual tokens  │
              │        verified,           ← entity tokens  │
              │        retrieved_report,   ← context tokens │
              │        prompt,             ← prompt tokens  │
              │        target=report       ← teacher-force  │
              │     )                                       │
              │                                            │
              │  Save: hybrid_generator.pth                 │
              └───────────────────────┬───────────────────┘
                                      │
              ┌───────────────────────▼───────────────────┐
              │  INFERENCE / EVALUATION                     │
              │  evaluate_hybrid.py                         │
              │                                            │
              │  Same pipeline, target_texts=None           │
              │  → Flan-T5.generate() with beam search      │
              │  → BLEU-1/2/3/4, METEOR, ROUGE-L            │
              └────────────────────────────────────────────┘
```

---

## Component Breakdown

### Visual Backbone (SAEnet)

```
Input: (B, 2, 3, 224, 224)  — two views per patient
         │
         ├── View 1 → ResNet101 → C2, C3, C4, C5
         │                             │
         │                            FPN
         │                        P2, P3, P4, P5
         │                             │
         │                           SAFE
         │                 C5 (query) ←→ P5 (key/value)
         │                        enhanced_1
         │
         └── View 2 → (same path) → enhanced_2
                                         │
                              fused = (enhanced_1 + enhanced_2) / 2
                              Output: (B, 256, 7, 7)
```

### CrossModalAlignment

```
Input:
  image_features: (B, 256, 7, 7)
  reports: list[str]

Steps:
  img_tokens  = flatten+transpose → (B, 49, 256)
  text_tokens = ClinicalBERT(reports) → (B, L, 768)
              → linear projection   → (B, L, 256)
  cls_token   = text_tokens[:, 0]   → (B, 256)

  aligned_features, attn_weights = MultiheadAttention(
      Q=img_tokens,      ← image region queries
      K=text_tokens,     ← clinical text keys
      V=text_tokens      ← clinical text values
  )

Output:
  aligned_features: (B, 49, 256)   ← regions shaped by text
  cls_token:        (B, 256)        ← global text summary
  attn_weights:     (B, 49, L)      ← token-to-region map
```

### HybridReportGenerator (Flan-T5)

```
Inputs → Encoder Token Sequence:
  ┌─────────────────────────────────────────────────────┐
  │ [visual_tokens] [entity_tokens] [ret_tokens] [prompt]│
  │  (B,49,768)      (B,K,768)      (B,128,768)  (B,T,768)│
  │                                                     │
  │ visual_proj projects aligned_features: 256→768      │
  │ entity_embed selects only active (positive) entities │
  │ T5.embed_tokens on retrieved text + prompt text     │
  └─────────────────────────────────────────────────────┘
  Concatenated → single sequence → T5 encoder
                                         │
                                   T5 decoder
                                         │
                          Train: cross-entropy vs target
                        Infer:  beam search (num_beams=4)
```

---

## Tensor Shape Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `images` | `(B, 2, 3, 224, 224)` | Dual-view chest X-rays |
| `visual_features` | `(B, 256, 7, 7)` | SAEnet backbone output |
| `aligned_features` | `(B, 49, 256)` | Region-aligned via cross-attention |
| `global_feat` | `(B, 256)` | Global pooled feature for FAISS |
| `img_emb` | `(B, 256)` | L2-normalized projection for retrieval |
| `img_entities` | `(B, 14)` | Binary pathology predictions from image |
| `rep_entities` | `(B, 14)` | Binary pathology predictions from report |
| `verified_entities` | `(B, 14)` | Element-wise AND of above two |
| `visual_tokens` (T5) | `(B, 49, 768)` | Projected visual input to T5 encoder |
| `entity_tokens` (T5) | `(B, K, 768)` | Embedded active entities (K ≤ 14) |
| `attn_weights` | `(B, 49, L)` | Image-patch-to-text-token attention map |

---

## Information Flow at Inference

```
Test Image (frontal + lateral)
       │
       ▼
MultiViewBackbone → visual_features (B, 256, 7, 7)
       │
       ├──── flatten + pool ──────────────────────────▶ FAISS query
       │       global_feat → proj_img → img_emb        │
       │                                               ▼
       │                                    top-1 retrieved_report
       │                                    + rep_entities (precomputed)
       │
       ├──── CrossModalAlignment(visual, retrieved_report)
       │       Q=image regions, K/V=ClinicalBERT(retrieved)
       │       → aligned_features (B, 49, 256)
       │
       ├──── SAEImageClassifier(images) → img_entities
       │
       ├──── verified_entities = img_entities * rep_entities
       │       (intersection: both image AND report agree)
       │
       └──── HybridGenerator(
               aligned_features,      ← what the image shows, guided
               verified_entities,     ← confirmed clinical findings
               retrieved_report,      ← similar reference case
               prompt                 ← "Generate a detailed report..."
             )
               → generated report string
```
