# 08 — Complete File Reference

## Root Level

| File | Description |
|------|-------------|
| `train.py` | **Stage 1** — trains MultiViewBackbone + CrossModalAlignment + ProjectionHeads using contrastive loss. Saves `checkpoints/best_stage1.pth`. |
| `train_sae.py` | Earlier version of Stage 1 training. Uses `IUXrayDataset` (single-view) instead of `IUXrayMultiViewDataset`. Saves per-epoch checkpoints to `checkpoint/`. |
| `check_cuda.py` | Quick utility to verify CUDA availability. |

---

## `configs/`

| File | Description |
|------|-------------|
| `config.py` | Central config: `DATA_ROOT`, `IMAGE_SIZE=224`, `BATCH_SIZE=4`, `NUM_EPOCHS=50`, `LR=3e-5`, `DEVICE=cuda`. |
| `__init__.py` | Package init. |

---

## `data/`

| File | Description |
|------|-------------|
| `dataset.py` | `IUXrayMultiViewDataset` — loads IU X-Ray images+reports, pairs frontal+lateral views, cleans text placeholders. Returns `(2,3,224,224)` tensor + cleaned report string. |
| `check_dataset.py` | Debugging utility to inspect dataset samples. |
| `__init__.py` | Package init. |

---

## `models/`

| File | Class | Role |
|------|-------|------|
| `backbone.py` | `ResNet101Backbone` | ImageNet-pretrained ResNet101 returning multi-scale features `C2(256), C3(512), C4(1024), C5(2048)`. |
| `fpn.py` | `FPN` | Feature Pyramid Network — lateral 1×1 convs + top-down upsampling → `P2/P3/P4/P5` all at 256 channels. |
| `safe.py` | `SAFE` | Spatial Attention Feature Enhancement — MHA where C5 queries attend to P5 keys/values → spatially enhanced feature map `(B,256,7,7)`. |
| `multiview_backbone.py` | `MultiViewBackbone` | Combines backbone+FPN+SAFE for both views, averages them → `(B,256,7,7)`. The core visual encoder used in all stages. |
| `alignment.py` | `CrossModalAlignment` | Cross-modal alignment: frozen ClinicalBERT encodes report text; MHA where image patches (Q) attend to text tokens (K,V) → `aligned_features(B,49,256)` + `cls_token(B,256)` + `attn_weights(B,49,L)`. |
| `projection.py` | `ProjectionHead` | MLP: `Linear(256,512) → ReLU → Linear(512,256) → L2-normalize`. Used to project image and text globals into contrastive embedding space. |
| `__init__.py` | — | Package init. |

---

## `utils/`

| File | Description |
|------|-------------|
| `losses.py` | `contrastive_loss(img_emb, txt_emb, temperature=0.2)` — symmetric InfoNCE loss. L2-normalizes inputs, computes B×B similarity matrix, cross-entropy against diagonal. |
| `__init__.py` | Package init. |

---

## `classification/`

| File | Description |
|------|-------------|
| `report_labeler.py` | `ReportClassifier` — ClinicalBERT CLS → `Linear(768,512) → ReLU → Dropout → Linear(512,14)`. Multi-label pathology classifier from report text. Pre-trained, weights in `report_classifier.pth`. |
| `sae_image_classifier.py` | `SAEImageClassifier` — Full SAEnet backbone + avg/max pooling concat + `Linear(512,256) → ReLU → Dropout → Linear(256,14)`. Multi-label pathology classifier from dual-view images. |
| `train_image_classifier.py` | **Stage 2b** — trains `SAEImageClassifier` using pseudo-labels from the frozen `ReportClassifier` (threshold 0.3). 20 epochs, lr=1e-4, BCEWithLogitsLoss. Saves `image_classifier.pth`. |
| `image_classifier.pth` | Saved weights for `SAEImageClassifier`. |
| `report_classifier.pth` | Saved weights for `ReportClassifier`. |

---

## `rag/`

| File | Description |
|------|-------------|
| `build_faiss_fast.py` | **Stage 3** — encodes all training samples with frozen Stage 1 backbone+projection, precomputes report entity vectors with `ReportClassifier`, saves `faiss_index.bin` + `train_reports.pkl`. |
| `hybrid_generator.py` | `HybridReportGenerator` — Flan-T5-base with `visual_proj(256→768)` and `entity_embed(14,768)`. Concatenates visual tokens + entity tokens + retrieved text tokens + prompt tokens as encoder input. Teacher-forced training, beam-search inference. |
| `train_hybrid_generator.py` | **Stage 4** — trains `HybridReportGenerator`. Loads frozen backbone+alignment+projection+image_classifier. For each batch: retrieve via FAISS → align features → verify entities → generator forward + backprop. 8 epochs, lr=1e-5, grad clip 1.0. |
| `evaluate_hybrid.py` | **Final evaluation** — runs full aligned inference pipeline on test split. Reports BLEU-1/2/3/4, METEOR, ROUGE-L. |
| `rag_inference.py` | Qualitative RAG inspection — for one test sample shows ground-truth report, top-3 retrieved reports, image entity vector, and verified overlaps. |
| `evaluate_generator.py` | Baseline evaluation — text-only RAG with vanilla T5-small (no visual tokens, no entities). |
| `entity_extractor.py` | `extract_entities(text)` — simple keyword lookup for 7 medical terms. Early prototype, not used in main pipeline. |
| `retriever.py` | `retrieve(images, top_k=3)` — standalone retrieval function using Stage 1 models + FAISS. |
| `t5_model.py` | Empty file (placeholder). |
| `hybrid_generator.pth` | Saved weights for `HybridReportGenerator`. |
| `t5_generator.pth` | Saved weights for the baseline T5 generator (used by `evaluate_generator.py`). |
| `faiss_index.bin` | FAISS flat L2 index of training image embeddings (generated by `build_faiss_fast.py`). |
| `train_reports.pkl` | Pickle list of `{report, entity_vector}` for each training sample. |

---

## `tools/`

| File | Description |
|------|-------------|
| `create_split.py` | Splits IU X-Ray UIDs 70/10/20 (train/val/test) with fixed seed. Saves `splits/iu_split.csv`. Run once before any training. |

---

## `splits/`

| File | Description |
|------|-------------|
| `iu_split.csv` | Column: `uid`, `split`. Defines which patient UIDs belong to each split. |

---

## `checkpoints/`

| File | Description |
|------|-------------|
| `best_stage1.pth` | Best Stage 1 checkpoint (lowest contrastive loss). Keys: `visual_model`, `alignment`, `proj_img`, `proj_txt`, `epoch`, `loss`. |
| `latest_stage1.pth` | Most recent Stage 1 checkpoint (every epoch). Same keys + `optimizer`. |
| `final_stage1.pth` | Stage 1 checkpoint after all epochs. Same keys (no optimizer). |

## `checkpoint/`

| File | Description |
|------|-------------|
| `sae_epoch_N.pth` | Per-epoch checkpoints from `train_sae.py` (older single-view training script). Keys: `backbone`, `alignment`, `proj_img`, `proj_txt`, `optimizer`. Note: uses key `backbone` not `visual_model`. |

---

## Checkpoint Key Name Mapping

Different scripts use different keys for the same visual encoder:

| Script | Saves visual encoder as | Loads via |
|--------|------------------------|-----------|
| `train.py` | `"visual_model"` | `checkpoint["visual_model"]` |
| `train_sae.py` | `"backbone"` | `checkpoint["backbone"]` |

All Stage 2/3/4 scripts load from `checkpoints/best_stage1.pth` using `"visual_model"`.

---

## Pre-trained Models Used

| Model | HuggingFace ID | Used In | Frozen? |
|-------|---------------|---------|---------|
| ClinicalBERT | `emilyalsentzer/Bio_ClinicalBERT` | `CrossModalAlignment`, `ReportClassifier` | Yes (both) |
| ResNet101 | torchvision pretrained | `ResNet101Backbone` | No (fine-tuned in Stage 1) |
| Flan-T5 Base | `google/flan-t5-base` | `HybridReportGenerator` | No (fine-tuned in Stage 4) |
