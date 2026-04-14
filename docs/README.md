# SAE-FRAG: Spatially-Aware Enhanced Feature Retrieval-Augmented Generation

## Project Summary

SAE-FRAG is a multi-stage deep learning system for **automatic radiology report generation** from chest X-ray images. It combines:

- **SAENet** (Spatially-Aware Enhanced Network) — ResNet-101 + FPN + SAFE attention at **P3 resolution (28×28)**
- **Cross-Modal Alignment** — region-level image-text alignment using Bio-ClinicalBERT
- **FactMM-RAG Retrieval** — fact-aware FAISS retrieval trained with RadGraph entity-F1 pair supervision
- **Fact Verification** — soft-AND combining image and retrieved-report entity predictions
- **Hybrid Generator** — SciFive-base decoder (medically pre-trained T5-base) receiving 49 visual + entity + retrieved context tokens

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
- Combined `findings` + `impression` text, cleaned and **capped at 60 words** (SAENet §4.1)

---

## Multi-Stage Training Pipeline

```
Stage 1 → Stage 2 → mine_pairs → retriever → build_index → cache → Stage 3
 Align     Classify   RadGraph     InfoNCE      FAISS         49-tok   Generate
```

| Stage | Script | Output |
|-------|--------|--------|
| 1. Visual-Language Alignment | `train_stage1.py` | `checkpoints/stage1/best.pth` |
| 2a. Report Entity Classifier | `train_report_classifier.py` | `checkpoints/stage2/report_classifier.pth` |
| 2b. Image Entity Classifier | `train_stage2.py` | `checkpoints/stage2/image_classifier.pth` |
| 3a. Factual Pair Mining | `mine_factual_pairs.py` | `store/factual_pairs.pkl` |
| 3b. Fact-Aware Retriever | `train_factual_retriever.py` | `checkpoints/stage1/factual_retriever.pth` |
| 3c. FAISS Index | `build_index.py` | `store/faiss_index.bin` |
| 3d. Feature Cache (49-token) | `cache_features.py` | `store/cache_train.pt` |
| 3e. Report Generator | `train_stage3.py` | `checkpoints/stage3/best_generator.pth` |

---

## Evaluation

```bash
python scripts/evaluate/evaluate.py
```

Metrics reported: BLEU-1/2/3/4, METEOR, ROUGE-1/2/L, CIDEr, CheXBert F1 (micro/macro/per-class), Entity F1

---

## Quick-Start Execution Order

```bash
conda activate ergonomics

# Option A: Full automated pipeline
python run_pipeline.py

# Option B: Manual step-by-step
python scripts/train/train_stage1.py
python scripts/train/train_report_classifier.py
python scripts/train/train_stage2.py
python scripts/prepare/mine_factual_pairs.py --delta 0.3 --top_k 2
python scripts/train/train_factual_retriever.py
python scripts/prepare/build_index.py
python scripts/prepare/cache_features.py
python scripts/train/train_stage3.py
python scripts/evaluate/evaluate.py
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
