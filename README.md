# SAE-FRAG: Sparse Autoencoder Feature-Guided Radiology Report Generation

A multi-stage pipeline for automated chest X-ray report generation using sparse autoencoder visual features, cross-modal alignment, FAISS-based retrieval, and a hybrid T5 decoder.

## Architecture

```
Stage 1 – Visual Encoder
  MultiViewBackbone (ResNet-101 → FPN → SAFE)
  + CrossModalAlignment (Bio-ClinicalBERT cross-attention)
  + ProjectionHead

Stage 2 – Entity Classifiers
  SAEImageClassifier   – predicts 14 CheXpert findings from image features
  ReportClassifier     – predicts 14 CheXpert findings from report text

Stage 3 (FactMM-RAG) – Fact-Aware Retrieval + Hybrid Report Generator
  Factual Pair Mining  → F1RadGraph similarity (Jain et al., 2021)
  Factual Retriever    → InfoNCE training on (image query, image+text document) pairs
  FAISS retrieval      → top-k factually-similar candidate reports
  ReportVerifier       → cross-modal attention re-ranking
  HybridReportGenerator (flan-t5-large) → final report
```

## Results (IU X-Ray test set)

| Metric                    | Score  |
|---------------------------|--------|
| BLEU-1                    | 0.294  |
| BLEU-4                    | 0.085  |
| METEOR                    | 0.273  |
| ROUGE-L                   | 0.319  |
| **CheXBert F1 (micro)**   | —      |
| **CheXBert F1 (macro)**   | —      |
| **Entity F1**             | —      |

## Project Structure

```
sae-frag/
├── src/                        # Importable packages (add to PYTHONPATH)
│   ├── models/                 # backbone, fpn, safe, alignment, projection
│   ├── data/                   # IUXrayMultiViewDataset
│   ├── rag/                    # hybrid_generator, verifier, retriever, radgraph_extractor
│   ├── classification/         # SAEImageClassifier, ReportClassifier
│   ├── utils/                  # losses
│   └── configs/                # config.py
├── scripts/
│   ├── prepare/
│   │   ├── build_index.py      # Build FAISS index from Stage-1 embeddings
│   │   └── cache_features.py   # Pre-compute frozen model outputs (run once)
│   ├── train/
│   │   ├── train_stage1.py     # Train visual encoder + alignment
│   │   ├── train_stage2.py     # Train image & report classifiers
│   │   └── train_stage3.py     # Train hybrid report generator (cached)
│   └── evaluate/
│       ├── evaluate.py         # Full evaluation on test split (BLEU/METEOR/ROUGE)
│       └── infer.py            # Single-sample inference
├── data/
│   └── splits/
│       └── iu_split.csv        # Train/val/test UIDs
├── checkpoints/
│   ├── stage1/                 # best.pth, final.pth
│   ├── stage2/                 # image_classifier.pth, report_classifier.pth
│   └── stage3/                 # best_generator.pth, last_generator.pth, resume.pt
├── store/                      # FAISS index, cached features, train metadata
├── docs/                       # Architecture and pipeline documentation
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate conda environment
conda create -n sae_frag python=3.10
conda activate sae_frag

# 2. Install PyTorch with CUDA (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('wordnet')"
```

## Dataset

Download [IU X-Ray](https://openi.nlm.nih.gov/) and place it at `C:/Datasets/IU_Xray/` with the structure:

```
IU_Xray/
├── indiana_reports.csv
├── indiana_projections.csv
└── images/
    └── images_normalized/
        └── *.png
```

## Training Pipeline

All scripts are self-contained — no `PYTHONPATH` setup required.

### Stage 1 – Visual Encoder

```bash
python scripts/train/train_stage1.py
```

Trains `MultiViewBackbone` + `CrossModalAlignment` + `ProjectionHead`.  
Saves to `checkpoints/stage1/`.

### Stage 2 – Entity Classifiers

```bash
python scripts/train/train_stage2.py
```

Trains `SAEImageClassifier` from pseudo-labels produced by `ReportClassifier`.
Requires `checkpoints/stage2/report_classifier.pth` to already exist.  
Saves `image_classifier.pth` to `checkpoints/stage2/`.

### Stage 3 – Fact-Aware Retriever + Report Generator (FactMM-RAG)

```bash
# Step 1: Mine factually-informed positive report pairs (once, ~20 min)
#         Uses F1RadGraph entity-overlap similarity (Jain et al., 2021)
#         Output: store/factual_pairs.pkl
python scripts/prepare/mine_factual_pairs.py

# Step 2: Train the fact-aware multimodal retriever (FactMM-RAG, NAACL 2025)
#         InfoNCE loss: query=image, positive=(image+text) of factual pair
#         Output: checkpoints/stage1/factual_retriever.pth
python scripts/train/train_factual_retriever.py

# Step 3: Rebuild FAISS index using document encoder (image+text embeddings)
python scripts/prepare/build_index.py

# Step 4: Pre-compute and cache frozen model outputs (once, ~60-90 min)
python scripts/prepare/cache_features.py

# Step 5: Train hybrid report generator
python scripts/train/train_stage3.py

# Resume after interrupt
python scripts/train/train_stage3.py --resume
```

Saves checkpoints to `checkpoints/stage3/`.

> **Note**: Steps 1–2 implement the FactMM-RAG retriever training. If you skip
> them, build_index.py and cache_features.py fall back to the plain Stage-1
> image encoder (original behaviour).

## Evaluation

```bash
# Full test-set evaluation
python scripts/evaluate/evaluate.py

# Single sample inference
python scripts/evaluate/infer.py --idx 0
```

## Notes

- On Windows, always use `num_workers=0` in DataLoaders (already set in all scripts).
- The generator uses `bfloat16` autocast for memory efficiency (safe on Ampere+ GPUs).
- `cache_features.py` must be re-run if Stage-1 or Stage-2 checkpoints are updated.
