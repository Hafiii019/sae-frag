"""Clinical evaluation metrics for radiology report generation.

Implements the standard metrics used in radiology NLG papers:

CheXBert Label F1 (micro + macro)
-----------------------------------
Both the generated and reference reports are labelled with the trained
ReportClassifier (ClinicalBERT → 14 CheXpert pathology classes).  Micro- and
macro-F1 are computed over all 14 classes.  This is equivalent to the
CheXBert metric used in CheXBert (Smit et al., 2020) and reported in
R2Gen, RGRG, and FactMM-RAG.

Entity F1 (Precision / Recall / F1 over RadGraph entities)
------------------------------------------------------------
Extracted entity strings for each generated and reference report are
compared as sets.  Precision, Recall and F1 are computed per sample and
averaged.  Falls back to keyword extraction if RadGraph is unavailable.

Both metrics are updated incrementally via ``update()`` and computed after
the loop via ``compute()``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = [
    "No Finding", "Cardiomegaly", "Pleural Effusion", "Pneumonia",
    "Pneumothorax", "Atelectasis", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Nodule", "Mass", "Hernia", "Infiltrate",
]
NUM_CLASSES = len(CLASS_NAMES)   # 14


class ClinicalMetrics:
    """Accumulate and compute CheXBert label F1 and entity F1.

    Parameters
    ----------
    report_classifier:
        Trained ``ReportClassifier`` instance (Bio_ClinicalBERT → 14 labels).
    radgraph_extractor:
        ``RadGraphExtractor`` instance (real RadGraph or keyword fallback).
    threshold:
        Sigmoid threshold for converting logits to binary labels (default 0.5).
    device:
        Torch device for inference.
    """

    def __init__(
        self,
        report_classifier,
        radgraph_extractor,
        threshold: float = 0.5,
        device=None,
    ):
        self.classifier  = report_classifier
        self.extractor   = radgraph_extractor
        self.threshold   = threshold
        self.device      = device or torch.device("cpu")

        # CheXBert accumulators: (N, 14) binary arrays
        self._gen_labels: List[np.ndarray]  = []
        self._ref_labels: List[np.ndarray]  = []

        # Entity F1 per-sample values
        self._entity_p: List[float] = []
        self._entity_r: List[float] = []
        self._entity_f: List[float] = []

    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, reference: str, generated: str) -> None:
        """Add one (reference, generated) pair to the accumulators."""

        # ── CheXBert labels ───────────────────────────────────────────
        gen_logits = self.classifier([generated]).to(self.device)
        ref_logits = self.classifier([reference]).to(self.device)

        gen_bin = (torch.sigmoid(gen_logits) >= self.threshold).cpu().numpy().astype(int)
        ref_bin = (torch.sigmoid(ref_logits) >= self.threshold).cpu().numpy().astype(int)

        self._gen_labels.append(gen_bin[0])   # (14,)
        self._ref_labels.append(ref_bin[0])   # (14,)

        # ── Entity F1 ─────────────────────────────────────────────────
        gen_entities = self._extract_entity_set(generated)
        ref_entities = self._extract_entity_set(reference)

        if not ref_entities and not gen_entities:
            p, r, f = 1.0, 1.0, 1.0
        elif not gen_entities or not ref_entities:
            p, r, f = 0.0, 0.0, 0.0
        else:
            tp = len(gen_entities & ref_entities)
            p  = tp / len(gen_entities)
            r  = tp / len(ref_entities)
            f  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        self._entity_p.append(p)
        self._entity_r.append(r)
        self._entity_f.append(f)

    def compute(self) -> Dict[str, float]:
        """Return all clinical scores as a flat dict.

        Raises
        ------
        RuntimeError
            If ``update()`` was never called.
        """
        if not self._gen_labels:
            raise RuntimeError("No samples accumulated. Call update() first.")

        gen = np.stack(self._gen_labels)   # (N, 14)
        ref = np.stack(self._ref_labels)   # (N, 14)

        results: Dict[str, float] = {}

        # ── Micro F1 ──────────────────────────────────────────────────
        tp_micro = (gen * ref).sum()
        fp_micro = (gen * (1 - ref)).sum()
        fn_micro = ((1 - gen) * ref).sum()
        p_micro  = tp_micro / (tp_micro + fp_micro + 1e-9)
        r_micro  = tp_micro / (tp_micro + fn_micro + 1e-9)
        f_micro  = 2 * p_micro * r_micro / (p_micro + r_micro + 1e-9)
        results["chexbert_precision_micro"] = round(float(p_micro), 4)
        results["chexbert_recall_micro"]    = round(float(r_micro), 4)
        results["chexbert_f1_micro"]        = round(float(f_micro), 4)

        # ── Macro F1 + per-class F1 ───────────────────────────────────
        per_class_f1 = []
        for i, name in enumerate(CLASS_NAMES):
            tp = (gen[:, i] * ref[:, i]).sum()
            fp = (gen[:, i] * (1 - ref[:, i])).sum()
            fn = ((1 - gen[:, i]) * ref[:, i]).sum()
            p_ = tp / (tp + fp + 1e-9)
            r_ = tp / (tp + fn + 1e-9)
            f_ = 2 * p_ * r_ / (p_ + r_ + 1e-9)
            per_class_f1.append(float(f_))
            key = name.lower().replace(" ", "_")
            results[f"chexbert_f1_{key}"] = round(float(f_), 4)

        results["chexbert_f1_macro"] = round(float(np.mean(per_class_f1)), 4)

        # ── Entity F1 ─────────────────────────────────────────────────
        results["entity_precision"] = round(float(np.mean(self._entity_p)), 4)
        results["entity_recall"]    = round(float(np.mean(self._entity_r)), 4)
        results["entity_f1"]        = round(float(np.mean(self._entity_f)), 4)

        return results

    def reset(self) -> None:
        """Clear all accumulators (call between evaluation runs)."""
        self._gen_labels.clear()
        self._ref_labels.clear()
        self._entity_p.clear()
        self._entity_r.clear()
        self._entity_f.clear()

    # ------------------------------------------------------------------

    def _extract_entity_set(self, text: str) -> set:
        """Return a lower-cased set of entity surface-form tokens."""
        result   = self.extractor.extract(text)
        entities = result.get("entities", {})
        return {
            v["tokens"].lower().strip()
            for v in entities.values()
            if isinstance(v, dict) and v.get("tokens")
        }
