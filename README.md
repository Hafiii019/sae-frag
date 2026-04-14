# SAE-FRAG: Sparse Autoencoder Feature-Guided Radiology Report Generation

A multi-stage pipeline for automated chest X-ray report generation using multi-scale visual features, cross-modal alignment, RadGraph-based factual retrieval, and a medically pre-trained T5 decoder.

## Architecture

```
Stage 1 – Visual Encoder
  MultiViewBackbone (ResNet-101 → FPN → SAFE)
    └ SAFE queries P3 (28×28) for 4× finer detail than original SAFE@P4
    └ Additive dual-view fusion (SAENet eq.5): feat_frontal + feat_lateral
  + CrossModalAlignment (Bio-ClinicalBERT cross-attention)
  + ProjectionHead

Stage 2 – Entity Classifiers
  SAEImageClassifier   – predicts 14 CheXpert findings from image features
  ReportClassifier     – predicts 14 CheXpert findings from report text

Stage 3 (FactMM-RAG) – Fact-Aware Retrieval + Hybrid Report Generator
  Factual Pair Mining  → RadGraph entity-F1 eq.1 (Jain et al., 2021)
                          two-stage: Jaccard pre-filter → RadGraph F1 ≥ 0.3
  Factual Retriever    → InfoNCE on (image query, image+text document) pairs
  FAISS retrieval      → top-k factually-similar candidate reports
  ReportVerifier       → cross-modal attention re-ranking
  HybridReportGenerator (SciFive-base, medically pre-trained T5-base)
    └ Two-phase: freeze T5 (3 epochs) → full fine-tune
    └ Encoder: 49 visual + 4 entity + 128 retrieved + 25 prompt ≈ 206 tokens
    └ Beam search: num_beams=3, length_penalty=1.2, no_repeat_ngram_size=4
```

## Results (IU X-Ray test set)

| Metric                    | Before fixes | After fixes (target) |
|---------------------------|--------------|-----------------------|
| BLEU-1                    | 0.2778       | ~0.45–0.50            |
| BLEU-4                    | 0.0757       | ~0.15–0.18            |
| METEOR                    | 0.249        | ~0.22–0.24            |
| ROUGE-L                   | 0.2951       | ~0.35–0.40            |
| CheXBert F1 (micro)       | 0.4302       | ↑                    |
| Entity F1                 | 0.496        | ↑ (real RadGraph)     |

> **Note**: "After fixes" targets require a full pipeline rebuild with all changes applied.

## Key Design Changes (April 2025 Refactor)

| Change | Impact |
|--------|--------|
| Report targets capped at 60 words | Aligns with SAENet paper §4.1, stable generation |
| SAFE uses P3 (28×28) not P4 (14×14) | 4× better spatial resolution for small findings |
| Visual tokens pooled 196→49 before T5 | Fixes silent 512-token overflow bug |
| Retrieved text capped at 128 tokens | Total encoder sequence ≤ 206 tokens |
| SciFive-base replaces flan-t5-base | Medical vocabulary pre-training (+~0.05 BLEU) |
| Two-phase training (freeze → unfreeze) | Stable convergence on 6 GB GPU |
| RadGraph entity-F1 pair mining | Replaces Jaccard on binary labels (FactMM-RAG eq.1) |
| Additive view fusion (not weighted avg) | Matches SAENet paper eq.5 |

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
conda create -n ergonomics python=3.10
conda activate ergonomics

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Fix h5py BEFORE installing radgraph (Windows DLL requirement)
#    h5py 3.13 has a DLL incompatibility; 3.10–3.11 has stable Windows wheels
conda install h5py -c conda-forge   # installs a compatible version

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 6. Verify RadGraph is working
python scripts/prepare/_test_radgraph.py
```

> **h5py note (Windows):** If you see `ImportError: DLL load failed while importing defs`, run:
> `conda install h5py -c conda-forge` to replace the pip wheel with a conda one that bundles the correct HDF5 DLLs.

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
The fastest way to run is via the orchestrator:

```bash
conda activate ergonomics

# Run the full pipeline (skips stages whose checkpoints already exist)
python run_pipeline.py

# Resume from a specific stage
python run_pipeline.py --from mine_pairs

# Force-rerun one stage
python run_pipeline.py --only stage3 --force stage3
```

Or run individual stages manually:

### Stage 1 – Visual Encoder

```bash
python scripts/train/train_stage1.py
```

Trains `MultiViewBackbone` (ResNet-101 → FPN → SAFE@P3) + `CrossModalAlignment` + `ProjectionHead`.  
Saves to `checkpoints/stage1/`.

### Stage 2 – Entity Classifiers

```bash
python scripts/train/train_report_classifier.py   # text-side teacher
python scripts/train/train_stage2.py              # image-side student
```

Saves `report_classifier.pth` + `image_classifier.pth` to `checkpoints/stage2/`.

### Stage 3 – Fact-Aware Retriever + Report Generator (FactMM-RAG)

```bash
# Step 1: Mine factually-informed positive pairs using RadGraph entity-F1
#         Two-stage: Jaccard pre-filter (0.10) → entity-F1 ≥ 0.30
#         Output: store/factual_pairs.pkl  (+ store/radgraph_cache.json)
python scripts/prepare/mine_factual_pairs.py --delta 0.3 --top_k 2

# Step 2: Train fact-aware multimodal retriever (InfoNCE, in-batch negatives)
#         Output: checkpoints/stage1/factual_retriever.pth
python scripts/train/train_factual_retriever.py

# Step 3: Build FAISS index using document encoder (image+text embeddings)
#         Output: store/faiss_index.bin + store/train_reports.pkl
python scripts/prepare/build_index.py

# Step 4: Pre-compute and cache frozen outputs (49-token pooled, ~0.7 GB)
#         Must be re-run whenever Stage 1 or Stage 2 checkpoints change
#         Output: store/cache_train.pt + store/cache_val.pt
python scripts/prepare/cache_features.py

# Step 5: Train hybrid report generator (two-phase, 40 epochs)
#         Phase 1 (epochs 1–3): freeze T5, train projection layers only
#         Phase 2 (epoch 4+):   full fine-tune of SciFive-base
#         Output: checkpoints/stage3/best_generator.pth
python scripts/train/train_stage3.py

# Resume after interrupt
python scripts/train/train_stage3.py --resume
```

> **Important**: if you change `models.py` (e.g. SAFE resolution), you must
> re-run stages 1 → retriever → build_index → cache_features → stage3 in order.
> Delete `checkpoints/stage3/best_generator.pth` if switching generator backbone.

## Evaluation

```bash
# Full test-set evaluation (BLEU/ROUGE/CheXBert/Entity-F1)
python scripts/evaluate/evaluate.py

# Single sample inference
python scripts/evaluate/infer.py --idx 0
```

Results are written to `results/metrics.json` and `results/generated_reports.json`.

## Notes

- On Windows, always use `num_workers=0` in DataLoaders (already set in all scripts).
- The generator uses `bfloat16` autocast for memory efficiency (safe on Ampere+ GPUs).
- `cache_features.py` must be re-run if Stage-1 or Stage-2 checkpoints are updated.
- **RadGraph on Windows**: requires `h5py` from conda-forge (not pip wheel). See Setup above.
- **GPU memory**: all configs are tuned for 6 GB VRAM. Do not increase `BATCH_SIZE` or
  `max_length` values without checking the token budget.
