# SAE-FRAG: Spatially-Aware Enhanced Feature Retrieval-Augmented Generation

## Project Summary

SAE-FRAG is a multi-stage deep learning system for **automatic radiology report generation** from chest X-ray images. It combines:

- **SAEnet** (Spatially-Aware Enhanced Network) — a multi-view visual backbone built on ResNet101 + FPN + SAFE attention
- **Cross-Modal Alignment** — region-level image-text alignment using ClinicalBERT
- **FAISS-based RAG** — retrieval-augmented generation using a dense knowledge base of training reports
- **Fact Verification** — entity-level intersection between image predictions and retrieved report findings
- **Hybrid Generator** — Flan-T5 decoder that receives region-aligned features, verified entity tokens, retrieved context, and a task prompt

---

## Dataset

**IU X-Ray** (Indiana University Chest X-Ray dataset)

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | Model training + FAISS index |
| Val   | 10% | Monitoring |
| Test  | 20% | Final evaluation |

Each sample contains:
- Two X-ray views (frontal + lateral) as `(2, 3, 224, 224)` tensors
- Combined `findings` + `impression` text as a single cleaned report string

---

## Four-Stage Training Pipeline

```
Stage 1 → Stage 2 → Stage 3 → Stage 4
 Align     Classify  Index     Generate
```

| Stage | Script | Output |
|-------|--------|--------|
| 1. Visual-Language Alignment | `train.py` | `checkpoints/best_stage1.pth` |
| 2a. Report Entity Classifier | *(pre-trained)* | `classification/report_classifier.pth` |
| 2b. Image Entity Classifier | `classification/train_image_classifier.py` | `classification/image_classifier.pth` |
| 3. Build FAISS Knowledge Base | `rag/build_faiss_fast.py` | `rag/faiss_index.bin` + `rag/train_reports.pkl` |
| 4. Train Hybrid Generator | `rag/train_hybrid_generator.py` | `rag/hybrid_generator.pth` |

---

## Evaluation

```bash
python -m rag.evaluate_hybrid
```

Metrics reported: BLEU-1/2/3/4, METEOR, ROUGE-L

---

## Quick-Start Execution Order

```bash
# 0. Create data splits
python tools/create_split.py

# 1. Stage 1 — Visual-Language alignment
python train.py

# 2. Stage 2 — Image classifier (uses report_classifier as pseudo-labeler)
python -m classification.train_image_classifier

# 3. Stage 3 — Build FAISS index
python -m rag.build_faiss_fast

# 4. Stage 4 — Train hybrid generator
python -m rag.train_hybrid_generator

# 5. Evaluate
python -m rag.evaluate_hybrid
```

---

## Documentation Index

| File | Contents |
|------|----------|
| [01_architecture.md](01_architecture.md) | Full system architecture and data-flow diagrams |
| [02_data_pipeline.md](02_data_pipeline.md) | Dataset loading, preprocessing, cleaning |
| [03_stage1_training.md](03_stage1_training.md) | Stage 1: Visual backbone and cross-modal alignment |
| [04_stage2_classifiers.md](04_stage2_classifiers.md) | Stage 2: Pathology entity classifiers |
| [05_stage3_faiss.md](05_stage3_faiss.md) | Stage 3: FAISS knowledge base construction |
| [06_stage4_generation.md](06_stage4_generation.md) | Stage 4: Hybrid report generation |
| [07_evaluation.md](07_evaluation.md) | Evaluation metrics and scripts |
| [08_file_reference.md](08_file_reference.md) | Complete file-by-file reference |
