# 06 — Stage 4: Hybrid Report Generation

## Goal

Train a **text decoder** that takes four distinct inputs and generates a complete radiology report:

1. **Region-aligned visual features** — what the image shows, conditioned on the retrieved report
2. **Verified entity vector** — confirmed pathology findings agreed by both image and report
3. **Retrieved report text** — the most factually-similar training case as context
4. **Task prompt** — instruction text guiding the generation style

---

## Scripts

```bash
# Training (two-phase: freeze backbone 3 epochs, then full fine-tune)
python scripts/train/train_stage3.py

# Inference only
python scripts/evaluate/infer.py

# Evaluation metrics
python scripts/evaluate/evaluate.py
```

---

## HybridReportGenerator Architecture

**File:** `src/rag/hybrid_generator.py`  
**Base model:** `razent/SciFive-base-Pubmed_PMC` (T5-base fine-tuned on PubMed + PMC, d_model=768)

SciFive is used instead of Flan-T5 because it is pre-trained on the same biomedical domain as radiology reports, giving better initialisation for clinical language generation.

### Additional Parameters

| Module | Input dim | Output dim | Purpose |
|--------|-----------|-----------|----------|
| `visual_proj` | 256 | 768 | Project aligned image features to T5 embedding space |
| `entity_embed` | 14 (vocab) | 768 | Learned embedding for each pathology label |

---

## Forward Pass — Encoder Input Construction

The T5 encoder receives a **concatenated sequence** of four token groups:

```
┌──────────────────────────────────────────────────────────────────┐
│  ENCODER INPUT SEQUENCE                                           │
│                                                                  │
│  [visual_tokens | entity_tokens | retrieved_tokens | prompt_tokens]│
│   (B, 49, 768)   (B, K, 768)    (B, ≤128, 768)    (B, T, 768)   │
│                                                                  │
│  Total: ~49 + 4 + 128 + 25 = ~206 tokens ≪ T5 limit of 512       │
└──────────────────────────────────────────────────────────────────┘
```

### Token Group 1: Visual Tokens

```python
# aligned_features arrive pooled from cache: (B, 49, 256)
visual_tokens = self.visual_proj(aligned_features)  # (B, 49, 256) → (B, 49, 768)
visual_mask   = torch.ones(B, 49)
```

The 49 (7×7) tokens arrive pre-pooled from the feature cache (adaptive avg-pool
of the original 784=28×28 aligned features). Each token covers one 7×7 macropatch
of the dual-view fused image.

### Token Group 2: Entity Tokens (Dynamic)

```python
entity_ids     = torch.arange(14).unsqueeze(0).expand(B, -1)   # (B, 14)
entity_embeds  = self.entity_embed(entity_ids)                   # (B, 14, 768)

# Per-sample: keep only ACTIVE (positive) entities
for b in range(B):
    active_embeds = entity_embeds[b][verified_entities[b].bool()]
    # If cardiomegaly=1, effusion=1, rest=0 → only 2 entity tokens
```

Padding via `nn.utils.rnn.pad_sequence` for the batch.

**Key design decision:** Only positive entities are included. This keeps the
sequence short when few findings are present and ensures entity tokens always
mean “this condition IS present.”

### Token Group 3: Retrieved Report Tokens

```python
retrieved_enc    = tokenizer(retrieved_texts, max_length=128, truncation=True, ...)
retrieved_embeds = t5.encoder.embed_tokens(retrieved_enc.input_ids)  # (B, ≤128, 768)
```

Capped at 128 tokens — a full retrieved report rarely exceeds this for 60-word
capped reports. Keeps total sequence at ~206 tokens, well under T5’s 512-position
limit.

### Token Group 4: Prompt Tokens

Example prompt:
> “Generate a detailed radiology report based on the chest X-ray regions, verified clinical findings, and retrieved context.”

---

## Forward Pass — Training Mode

```python
if target_texts is not None:
    target_enc = tokenizer(target_texts, max_length=128, ...)  # 60 words ≈ 90 tokens
    outputs = self.t5(
        inputs_embeds=encoder_inputs,
        attention_mask=attention_mask,
        labels=target_enc.input_ids         # teacher-forcing
    )
    return outputs.loss                     # cross-entropy
```

---

## Forward Pass — Inference Mode

```python
else:
    generated_ids = self.t5.generate(
        inputs_embeds=encoder_inputs,
        attention_mask=attention_mask,
        max_new_tokens=80,          # 60-word target ≈ 80 BPE tokens
        num_beams=3,                # beam search (SAENet paper), less VRAM than 4
        length_penalty=1.2,         # slightly favour longer complete sentences
        no_repeat_ngram_size=4,     # suppress 4-gram repetition
        early_stopping=True
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

---

## Training Configuration

**Script:** `scripts/train/train_stage3.py`

| Parameter | Value | Note |
|-----------|-------|------|
| Base model | `razent/SciFive-base-Pubmed_PMC` | Biomedical T5-base |
| Optimizer | AdamW | weight_decay=0.01 |
| LR (SciFive layers) | 2e-5 | Low: pre-trained weights |
| LR (new projections) | 1e-4 | Higher: random init |
| Epochs | 40 | 8 was too few |
| Warmup steps | 500 | Cosine annealing |
| Batch size | 2 | 6 GB VRAM |
| Grad clip | 1.0 | Stability |
| Patience | 8 | Early stopping window |
| `FREEZE_T5_EPOCHS` | 3 | Two-phase training |

### Two-Phase Training

- **Phase 1 (epochs 0–2):** SciFive backbone frozen, only `visual_proj` and `entity_embed` are updated. Projections learn to align visual/entity features to T5’s embedding space without corrupting pre-trained language priors.
- **Phase 2 (epoch 3+):** All parameters unfrozen. Full fine-tuning with differential learning rates (SciFive 2e-5, new layers 1e-4).

---

## Complete Training Step

```python
# All upstream models frozen, all no_grad
with torch.no_grad():

    # 1. Load pre-computed cache (avoids re-running BERT/backbone each step)
    aligned_features  = cache["aligned_features"]     # (B, 49, 256) pooled
    entity_vector     = cache["entity_vector"]         # (B, 14)
    retrieved_reports = cache["retrieved_text"]        # list[str]

# 2. Fact verification (image ∩ retrieved)
# (image classifier is run once per epoch in build_index, cached)
verified_entities = img_entities * rep_entities                    # (B, 14)

# 3. Generate and compute cross-entropy loss
loss = generator(
    aligned_features, verified_entities,
    retrieved_reports, [prompt] * B,
    target_texts=reports
)
loss.backward()
```
| `visual_proj` | 256 | 768 | Project aligned image features to T5 embedding space |
| `entity_embed` | 14 (vocab) | 768 | Learned embedding for each pathology label |

---

## Forward Pass — Encoder Input Construction

The T5 encoder receives a **concatenated sequence** of four token groups:

```
┌──────────────────────────────────────────────────────────────────┐
│  ENCODER INPUT SEQUENCE                                          │
│                                                                  │
│  [visual_tokens | entity_tokens | retrieved_tokens | prompt_tokens]│
│   (B, 49, 768)   (B, K, 768)    (B, ≤128, 768)    (B, T, 768)  │
│                                                                  │
│  Attention mask is concatenated correspondingly                  │
└──────────────────────────────────────────────────────────────────┘
```

### Token Group 1: Visual Tokens

```python
visual_tokens = self.visual_proj(aligned_features)  # (B, 49, 256) → (B, 49, 768)
visual_mask   = torch.ones(B, 49)
```

These represent the 7×7 = 49 image region features, already shaped by cross-attention with the retrieved report (from `CrossModalAlignment`). Each token corresponds to one spatial patch of the fused multi-view image.

### Token Group 2: Entity Tokens (Dynamic)

```python
entity_ids     = torch.arange(14).unsqueeze(0).expand(B, -1)   # (B, 14)
entity_embeds  = self.entity_embed(entity_ids)                   # (B, 14, 768)

# Per-sample: keep only ACTIVE (positive) entities
for b in range(B):
    active_embeds = entity_embeds[b][verified_entities[b].bool()]
    # If cardiomegaly=1, effusion=1, rest=0 → only 2 entity tokens
```

Padding via `nn.utils.rnn.pad_sequence` for the batch.

**Key design decision:** Only positive entities are included, not all 14. This keeps the sequence short when few findings are present and ensures entity tokens always mean "this condition IS present."

### Token Group 3: Retrieved Report Tokens

```python
retrieved_enc    = tokenizer(retrieved_texts, max_length=128, ...)
retrieved_embeds = t5.encoder.embed_tokens(retrieved_enc.input_ids)  # (B, ≤128, 768)
```

The retrieved report is embedded using T5's own token embeddings — the same embedding table used for the output vocabulary. This keeps the representation consistent with the model's language understanding.

### Token Group 4: Prompt Tokens

```python
prompt_enc    = tokenizer(prompt_texts, ...)
prompt_embeds = t5.encoder.embed_tokens(prompt_enc.input_ids)  # (B, T, 768)
```

Example prompt:
> "Generate a detailed radiology report based on the chest X-ray regions, verified clinical findings, and retrieved context."

---

## Forward Pass — Training Mode

```python
if target_texts is not None:
    target_enc = tokenizer(target_texts, max_length=128, ...)
    outputs = self.t5(
        inputs_embeds=encoder_inputs,
        attention_mask=attention_mask,
        labels=target_enc.input_ids         # teacher-forcing
    )
    return outputs.loss                     # cross-entropy
```

T5's standard training: the decoder sees the target shifted by one token (teacher-forcing). The loss is averaged cross-entropy over all target tokens.

---

## Forward Pass — Inference Mode

```python
else:
    generated_ids = self.t5.generate(
        inputs_embeds=encoder_inputs,
        attention_mask=attention_mask,
        max_length=150,
        num_beams=4,
        length_penalty=1.0
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

Beam search with 4 beams. `length_penalty=1.0` is neutral — no preference for shorter or longer outputs.

---

## Training Configuration

**Script:** `rag/train_hybrid_generator.py`

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-5 |
| Epochs | 8 |
| Batch size | 2 |
| Grad clip | 1.0 |

Low LR (1e-5) prevents the pre-trained Flan-T5 weights from diverging. Only `generator.parameters()` are updated — all upstream models (backbone, alignment, projection, classifiers) are frozen.

---

## Complete Training Step

```python
# All upstream models frozen, all no_grad
with torch.no_grad():

    # 1. Extract multi-scale spatial features
    visual_features = visual_encoder(images)             # (B, 256, 7, 7)

    # 2. Find most similar training case
    global_feat = visual_features.flatten(2).mean(dim=2) # (B, 256)
    img_emb = proj_img(global_feat)                       # (B, 256) L2-norm
    D, I = faiss_index.search(img_np, k=1)

    # 3. Retrieve text + pre-computed entity vector
    retrieved_reports = [metadata[I[b][0]]["report"] for b in range(B)]
    rep_entities = torch.cat([metadata[I[b][0]]["entity_vector"] for b in range(B)])

    # 4. Region features aligned to retrieved report
    #    image regions (Q) attend to retrieved report tokens (K, V)
    aligned_features, _, _ = alignment(visual_features, retrieved_reports)  # (B, 49, 256)

    # 5. Image-side entity predictions
    img_logits  = image_classifier(images)
    img_entities = (sigmoid(img_logits) > 0.5).float()            # (B, 14)

# 6. Fact verification
verified_entities = img_entities * rep_entities                    # (B, 14)

# 7. Generator forward (targets provided → training loss)
loss = generator(
    region_features=aligned_features,
    entity_vector=verified_entities,
    retrieved_texts=retrieved_reports,
    prompt_texts=prompts,
    target_texts=ground_truth_reports
)

# 8. Backprop with gradient clipping
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
optimizer.step()
```

---

## Why `aligned_features` Not Raw Backbone Features

The critical distinction between old and new pipeline:

| Feature Type | How Computed | What It Encodes |
|--------------|-------------|-----------------|
| Raw backbone: `feats.flatten(2).transpose(1,2)` | `visual_encoder(images)` → flatten | Pure visual appearance of the image |
| **Aligned features** (correct) | `alignment(visual_features, retrieved_report)` | Visual regions **cross-attended** to the retrieved report's clinical tokens — each region token has been updated to be aware of what the retrieved report describes |

When the same retrieved report is passed to both `alignment()` and `generator()`, the generator's retrieved text tokens and the visual tokens are **semantically coherent** — the visual tokens already reflect clinical concepts from that report.

---

## Output

- `rag/hybrid_generator.pth` — generator state dict saved after each epoch

Generator is re-trainable by running `train_hybrid_generator.py` again; it overwrites the same file each epoch.
