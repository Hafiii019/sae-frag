# 05 — Stage 3: Factual Pair Mining, FAISS Index, and Feature Cache

## Purpose

Build a dense vector database over the **entire training set**, trained with
**RadGraph entity-F1 factual supervision** (FactMM-RAG, NAACL 2025). At query
time the system retrieves the most factually-aligned training report to augment
the generator.

---

## Scripts (run in order)

```bash
# 1. Mine factual pairs with RadGraph entity-F1 (FactMM-RAG eq.1)
#    Two-stage: Jaccard pre-filter (0.10) → entity-F1 ≥ 0.30
#    Output: store/factual_pairs.pkl + store/radgraph_cache.json
python scripts/prepare/mine_factual_pairs.py --delta 0.3 --top_k 2

# 2. Train fact-aware multimodal retriever on mined pairs
#    Output: checkpoints/stage1/factual_retriever.pth
python scripts/train/train_factual_retriever.py

# 3. Build FAISS inner-product index using trained document encoder
#    Output: store/faiss_index.bin + store/train_reports.pkl
python scripts/prepare/build_index.py

# 4. Cache frozen outputs for Stage-3 generator training
#    Visual tokens pooled to 49 (7×7), cache size ~0.7 GB
#    Output: store/cache_train.pt + store/cache_val.pt
python scripts/prepare/cache_features.py
```

**Output files:**

| File | Contents |
|------|----------|
| `store/factual_pairs.pkl` | `{train_idx: [pos_idx, ...]}` mined by entity-F1 |
| `store/radgraph_cache.json` | RadGraph extraction cache (avoids re-running NER) |
| `store/faiss_index.bin` | FAISS flat inner-product index (256-dim, L2-normalised) |
| `store/train_reports.pkl` | List of `{report: str, entity_vector: Tensor}` |
| `store/cache_train.pt` | Pre-computed (49, 256) visual tokens + entity vectors |
| `store/cache_val.pt` | Val split cache |

---

## Step 1: RadGraph Entity-F1 Pair Mining

Implements FactMM-RAG equation 1:

$$s(q, d) = \frac{2\,|\hat{q} \cap \hat{d}|}{|\hat{q}| + |\hat{d}|}$$

where $\hat{q}$, $\hat{d}$ are sets of (token, label) tuples from RadGraph NER+RE.

**Two-stage strategy** (tractable on 6 GB GPU):
1. Jaccard pre-filter (threshold 0.10) on 14 CheXbert labels — fast, O(N²) vectorised
2. RadGraph entity-F1 on pre-filtered candidates only — O(N × candidates) set ops

Results are cached to `store/radgraph_cache.json` so re-runs are near-instant.

---

## Step 3: How the FAISS Index is Built

### Models Loaded (frozen from factual_retriever.pth or stage1/best.pth)

```python
visual_encoder = MultiViewBackbone()   # ResNet101 + FPN + SAFE@P3
proj_img       = ProjectionHead()      # 256→256, L2-normalised
alignment      = CrossModalAlignment() # Bio-ClinicalBERT cross-attention
proj_doc       = DocumentProjectionHead()  # doc embedding head
```

### Per-Sample Processing

```
For each training sample:
  1. visual_features = visual_encoder(images)       (B, 256, 28, 28)
  2. global_feat = visual_features.flatten(2).mean(dim=2)   (B, 256)
  3. img_emb = proj_img(global_feat)                (B, 256), L2-normalised
  4. doc_emb = proj_doc(alignment(visual_features, [report]).mean(1))  # if factual ckpt
  5. faiss_vec = doc_emb if factual else img_emb
```

---

## Step 4: Feature Cache (49-token pooling)

The cache pre-computes aligned features and entity vectors so Stage-3 training
never touches BERT or the visual backbone:

```
aligned_features (784, 256)  → adaptive_avg_pool2d → (49, 256)  stored
entity_vector   (14,)                                            stored
retrieved_text  str                                              stored
```

Pooling from 784→49 reduces cache from ~2.7 GB to ~0.7 GB and eliminates
per-step pooling overhead during training.

---

## How the Index is Built

### Models Loaded (all frozen from Stage 1)

```python
# Visual backbone
visual_encoder = MultiViewBackbone()
visual_encoder.load_state_dict(checkpoint["visual_model"])

# Projection head (maps 256→256, L2-normalized)
proj_img = ProjectionHead()
proj_img.load_state_dict(checkpoint["proj_img"])

# Report entity classifier (frozen, pre-trained)
report_classifier = ReportClassifier()
```

### Per-Sample Processing

```
For each training sample:
  1. visual_features = visual_encoder(images)       (B, 256, 28, 28)
  2. global_feat = visual_features.flatten(2).mean(dim=2)   (B, 256)
                   ↑ spatial average: one vector per image pair
  3. img_emb = proj_img(global_feat)                (B, 256), L2-normalized
  4. embeddings.append(img_emb)

  5. rep_entities = sigmoid(report_classifier(report)) > 0.5
  6. metadata.append({
         "report": report_text,
         "entity_vector": rep_entities  (1, 14)
     })
```

### Building the Index

```python
embeddings = np.vstack(embeddings).astype("float32")  # (N, 256)

index = faiss.IndexFlatL2(256)   # exact L2 nearest neighbor
index.add(embeddings)            # no training needed for Flat index

faiss.write_index(index, "rag/faiss_index.bin")
pickle.dump(metadata, open("rag/train_reports.pkl", "wb"))
```

`IndexFlatL2` performs exact exhaustive search — no approximation. For IU X-Ray's ~2,700 training samples this is fast enough. For larger datasets, consider `IndexIVFFlat` or `IndexHNSWFlat`.

---

## Retrieval at Query Time

```python
# Query embedding (same pipeline as indexing)
global_feat = visual_features.flatten(2).mean(dim=2)
img_emb = proj_img(global_feat)                        # (1, 256)
img_np = img_emb.cpu().numpy().astype("float32")

D, I = index.search(img_np, k=1)
# D: distances (1, k)    — smaller = more similar
# I: indices   (1, k)    — index into metadata list

retrieved_report = train_metadata[I[0][0]]["report"]
rep_entities     = train_metadata[I[0][0]]["entity_vector"]
```

**k=1** is used in the generator pipeline. `rag_inference.py` uses k=3 for exploration.

---

## Why Precompute Entity Vectors?

In the original `evaluate_hybrid.py`, `ReportClassifier` was called at inference time on the retrieved report:

```python
# OLD — wasteful and inconsistent
rep_logits = report_classifier([retrieved_report])
rep_entities = (torch.sigmoid(rep_logits) > 0.5).float()
```

This has two problems:
1. **Speed:** Runs ClinicalBERT on every test sample needlessly
2. **Consistency:** Stage 3 already computed these at build time with the same classifier

The correct approach (now implemented) reads directly from metadata:

```python
# CORRECT — precomputed at index build time
rep_entities = train_metadata[I[0][0]]["entity_vector"].to(device)
```

---

## Embedding Space

The FAISS index stores vectors produced by `proj_img`, which maps backbone features through a 2-layer MLP and L2-normalizes them. This means:

- L2 distance in the index ≈ 2 - 2·cosine_similarity (for unit vectors)
- Nearest neighbor = most cosine-similar image embedding
- The embedding was trained via **contrastive loss** to be similar across matched image-report pairs
- Images with similar pathologies should cluster together

---

## Index Statistics

Approximate for IU X-Ray:

| Split | Samples | Indexed? |
|-------|---------|---------|
| Train | ~2,770 | Yes |
| Val | ~396 | No |
| Test | ~793 | No |

Only train split is indexed — val and test are query-only.

---

## Metadata Structure

`train_reports.pkl` is a Python list of dicts:

```python
[
    {
        "report": "the heart is mildly enlarged. lung volumes are normal...",
        "entity_vector": tensor([[0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        #                         shape: (1, 14)
    },
    ...
]
```

`entity_vector` shape is `(1, 14)` because it was computed with batch_size=1. When used in the generator, it is concatenated:

```python
rep_entities_batch = torch.cat(rep_entities_batch).to(device)  # (B, 14)
```
