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

Stage 3 – Hybrid Report Generator
  FAISS retrieval → top-k candidate reports
  ReportVerifier  → re-ranks by cross-modal attention score
  HybridReportGenerator (flan-t5-base) → final report
```

## Results (IU X-Ray test set)

| Metric   | Score  |
|----------|--------|
| BLEU-1   | 0.256  |
| BLEU-4   | 0.057  |
| METEOR   | 0.235  |
| ROUGE-L  | 0.277  |

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

### Stage 3 – Report Generator

```bash
# Step 1: Build FAISS retrieval index (once)
python scripts/prepare/build_index.py

# Step 2: Pre-compute and cache frozen model outputs (once, ~60-90 min)
python scripts/prepare/cache_features.py

# Step 3: Train generator
python scripts/train/train_stage3.py

# Resume after interrupt
python scripts/train/train_stage3.py --resume
```

Saves checkpoints to `checkpoints/stage3/`.

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
