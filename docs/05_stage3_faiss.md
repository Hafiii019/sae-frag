# 05 — Stage 3: FAISS Knowledge Base

## Purpose

Build a dense vector database over the **entire training set**. At query time, given a test image, the system retrieves the most visually similar training case (report + entity vector).

This turns the training set into a **long-term memory** that the generator can consult.

---

## Script

```bash
python -m rag.build_faiss_fast
```

**Output files:**

| File | Contents |
|------|----------|
| `rag/faiss_index.bin` | FAISS flat L2 index of training image embeddings |
| `rag/train_reports.pkl` | List of `{report: str, entity_vector: Tensor}` per training sample |

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
For each training sample (batch_size=1):

  1. visual_features = visual_encoder(images)       (B, 256, 7, 7)
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
