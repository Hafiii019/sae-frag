# 07 — Evaluation

## Running Evaluation

```bash
# Full evaluation: BLEU, METEOR, ROUGE-L, CheXBert F1, Entity F1
python scripts/evaluate/evaluate.py

# Inference only (write generated_reports.json, no metric print)
python scripts/evaluate/infer.py
```

---

## Evaluation Pipeline (`scripts/evaluate/evaluate.py`)

Runs the full inference pipeline over the **test split** and computes all metrics.

### Step-by-Step

```
For each test sample (image, reference_report):

  1. Load pre-computed cache (aligned_features 49×256, entity_vector 14)
  2. Run fact verification: verified = img_entities ⋅ rep_entities
  3. generated = generator(aligned_features, verified, [retrieved_report], [prompt])
  4. Compute BLEU-1/2/3/4, METEOR, ROUGE-L vs reference_report
  5. Compute CheXBert F1 (micro/macro) vs reference labels
  6. Compute RadGraph Entity F1 vs reference entities
```

---

## Metrics

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision: fraction of n-grams in the generated text that appear in the reference.

| Metric | Weights | What It Measures |
|--------|---------|------------------|
| BLEU-1 | (1,0,0,0) | Unigram (word-level) overlap |
| BLEU-2 | (0.5,0.5,0,0) | Bigram overlap |
| BLEU-3 | (0.33,0.33,0.33,0) | Trigram overlap |
| BLEU-4 | (0.25,0.25,0.25,0.25) | 4-gram overlap |

Smoothing (method1) is applied because 60-word reports can make unsmoothed BLEU-3/4 collapse to zero.

**Range:** 0–1. SAENet target: BLEU-1 ≈ 0.50, BLEU-4 ≈ 0.17.

---

### METEOR

Considers precision, recall, and **synonym matching** via WordNet. More robust than BLEU for paraphrased but semantically equivalent text.

**Range:** 0–1. Typical radiology range: 0.15–0.35.

---

### ROUGE-L

Finds the longest common subsequence between reference and generated text. `use_stemmer=True` strips word endings before matching.

**Range:** 0–1. Typical range: 0.20–0.40.

---

### CheXBert Clinical F1

Uses the CheXBert labeller to extract 14 binary pathology labels from both generated and reference reports, then computes:

- **Micro F1**: treats each label×sample pair equally — dominated by common negatives
- **Macro F1**: averages F1 per condition — sensitive to rare positives like "Pneumothorax"
- **Per-class F1**: one score per condition (useful for debugging hallucination)

This is the primary **clinical accuracy** signal — BLEU can be gamed by fluent but factually wrong text.

---

### RadGraph Entity F1 (FactMM-RAG eq.1)

Extracts (token, label) entity sets with RadGraph NER+RE from both generated and reference reports:

$$\text{Entity-F1}(g, r) = \frac{2\,|\hat{g} \cap \hat{r}|}{|\hat{g}| + |\hat{r}|}$$

Captures clinical entity precision/recall at the sub-sentence level. A report that lists the right anatomical findings in the wrong order still scores well here.

---

## Interpreting Results

```
==== FINAL RESULTS ====
BLEU-1:  0.XXX    ← word precision
BLEU-2:  0.XXX    ← phrase precision
BLEU-3:  0.XXX    ← clause precision
BLEU-4:  0.XXX    ← sentence-level precision
METEOR:  0.XXX    ← synonym-aware recall
ROUGE-L: 0.XXX    ← sequence-level recall
CheXBert Micro F1: 0.XXX
CheXBert Macro F1: 0.XXX
Entity F1: 0.XXX
```

Expected trends:
- BLEU-4 < BLEU-3 < BLEU-2 < BLEU-1 (longer n-grams harder to match exactly)
- METEOR > BLEU-1 (rewards recall; BLEU only rewards precision)
- ROUGE-L often between METEOR and BLEU-1
- CheXBert Micro F1 ≫ Macro F1 (negatives dominate; check Macro for clinical utility)

---

## Baseline Comparison

| System | Generator | Entity-F1 Pairs | Expected BLEU-4 |
|--------|-----------|-----------------|-----------------|
| RAG-only T5 (no visual) | Flan-T5-base | Jaccard | Lower |
| Full SAE-FRAG (this repo) | SciFive-base-Pubmed_PMC | RadGraph entity-F1 | Higher |

The hybrid system adds three novel components over the RAG-only baseline:
1. Region-aligned visual tokens (image understanding)
2. Fact-verified entity tokens (hallucination reduction)
3. ClinicalBERT-guided regional alignment (clinical semantics in visual features)

---

## Evaluation Pipeline (`rag/evaluate_hybrid.py`)

Runs the same inference pipeline as training (no target texts provided) over the entire **test split** (batch_size=1).

### Step-by-Step

```
For each test sample (image, reference_report):

  1. visual_features = visual_encoder(images)
  2. img_emb = proj_img(global_feat)
  3. (D, I) = faiss_index.search(img_emb, k=1)
  4. retrieved_report = metadata[I[0][0]]["report"]
     rep_entities     = metadata[I[0][0]]["entity_vector"]
  5. aligned_features = alignment(visual_features, [retrieved_report])
  6. img_entities     = sigmoid(image_classifier(images)) > 0.5
  7. verified         = img_entities * rep_entities
  8. generated = generator(
         aligned_features, verified,
         [retrieved_report], [prompt],
         target_texts=None
     )[0]
  9. Compute BLEU-1/2/3/4, METEOR, ROUGE-L
     against reference_report
```

---

## Metrics

All metrics compare the **generated report** against the **ground-truth report** string.

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision: what fraction of n-grams in the generated text appear in the reference.

| Metric | Weights | What It Measures |
|--------|---------|-----------------|
| BLEU-1 | (1,0,0,0) | Unigram (word-level) overlap |
| BLEU-2 | (0.5,0.5,0,0) | Bigram overlap |
| BLEU-3 | (0.33,0.33,0.33,0) | Trigram overlap |
| BLEU-4 | (0.25,0.25,0.25,0.25) | 4-gram overlap |

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1   # add-one smoothing for short texts
bleu4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
```

Smoothing is applied because medical reports can be short, making unsmoothed BLEU-3/4 collapse to zero.

**Range:** 0–1 (higher is better). Radiology report generation baselines typically score 0.05–0.20 on BLEU-4.

---

### METEOR (Metric for Evaluation of Translation with Explicit Ordering)

Considers precision, recall, and **synonym matching** via WordNet. More robust than BLEU for paraphrased but semantically equivalent text.

```python
from nltk.translate.meteor_score import meteor_score
nltk.download("wordnet")
score = meteor_score([ref_tokens], gen_tokens)
```

**Range:** 0–1 (higher is better). Typical range for radiology: 0.15–0.35.

---

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest Common Subsequence)

Finds the longest common subsequence between reference and generated text. Rewards generating the right words in the right order, regardless of gaps.

```python
from rouge_score import rouge_scorer
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
score = rouge.score(reference, generated)['rougeL'].fmeasure
```

`use_stemmer=True` strips word endings (runs, running → run) before matching.

**Range:** 0–1 (higher is better). Typical range: 0.20–0.40.

---

## Interpreting Results

```
==== FINAL RESULTS ====
BLEU-1:  0.XXX    ← word precision
BLEU-2:  0.XXX    ← phrase precision
BLEU-3:  0.XXX    ← clause precision
BLEU-4:  0.XXX    ← sentence-level precision
METEOR:  0.XXX    ← synonym-aware recall
ROUGE-L: 0.XXX    ← sequence-level recall
```

A well-performing system should show:
- BLEU-4 > BLEU-3 > BLEU-2 trend downward (expected — longer n-grams are harder)
- METEOR > BLEU-1 (METEOR rewards recall, BLEU only rewards precision)
- ROUGE-L often highest (LCS is more lenient than n-gram exact match)

---

## Other Evaluation Scripts

### `rag/rag_inference.py`

Non-metric inspection script. For one test sample, prints:
- The ground-truth report
- The top-3 retrieved reports from FAISS
- Image entity prediction vector
- Entity intersection (verified overlap) for each retrieved report

Use for **qualitative debugging** of retrieval quality and entity verification.

```bash
python -m rag.rag_inference
```

### `rag/evaluate_generator.py`

Evaluates an older T5-only generator (without visual tokens or entity vectors). Useful as a **RAG-only baseline**:

```
Image → FAISS → top-3 reports → T5(prompt + retrieved context) → generated report
```

Lacks: region-aligned features, entity verification, clinical cross-attention.

```bash
python -m rag.evaluate_generator
```

---

## Baseline Comparison

| System | Inputs to Decoder | Expected BLEU-4 |
|--------|--------------------|----------------|
| `evaluate_generator.py` (RAG-only T5) | Retrieved text only | Lower |
| `evaluate_hybrid.py` (Full SAE-FRAG) | Aligned features + entities + retrieved + prompt | Higher |

The hybrid system adds three novel components over the RAG-only baseline:
1. Region-aligned visual tokens (image understanding)
2. Fact-verified entity tokens (hallucination reduction)
3. ClinicalBERT-guided regional alignment (clinical semantics in visual features)
