"""RadGraph entity extractor for radiology reports.

Uses the ``radgraph`` pip package (wraps the PhysioNet RadGraph model) to
extract structured clinical entities (observations and anatomies with presence
labels) from free-text radiology reports.  Produces ClinicalBERT-based
embeddings of the extracted entity set for downstream similarity computations.

Falls back to keyword-based extraction transparently when ``radgraph`` is not
installed or when inference fails, so the rest of the pipeline stays runnable.

Entity labels (RadGraph taxonomy)
----------------------------------
OBS-DP  – Observation, Definitely Present
OBS-DA  – Observation, Definitely Absent
OBS-U   – Observation, Uncertain
ANAT-DP – Anatomy,      Definitely Present

Usage
-----
    extractor = RadGraphExtractor(cache_path="store/radgraph_cache.json")
    result    = extractor.extract("Cardiomegaly is present. No pneumothorax.")
    text      = extractor.to_entity_text(result)
    emb       = extractor.to_entity_embedding(text, bert_model, tokenizer)
    sim       = extractor.report_similarity(emb_a, emb_b)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback keyword vocabulary (used when RadGraph is unavailable)
# ---------------------------------------------------------------------------
_FALLBACK_TERMS: List[str] = [
    "pneumothorax",
    "effusion",
    "cardiomegaly",
    "atelectasis",
    "nodule",
    "opacity",
    "consolidation",
    "edema",
    "infiltrate",
    "fracture",
    "mass",
    "pleural",
    "pneumonia",
    "hernia",
]


class RadGraphExtractor:
    """Extract clinical entities from radiology reports using RadGraph.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected when *None*.
    cache_path:
        Optional path to a JSON file for persisting extraction results so the
        RadGraph model runs only once per unique report string.  The parent
        directory is created automatically on :meth:`save_cache`.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None            # lazy-loaded on first extract() call
        self._use_radgraph: bool = True
        self._cache: Dict[str, dict] = {}
        self._cache_path: Optional[Path] = (
            Path(cache_path) if cache_path else None
        )

        if self._cache_path and self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as fh:
                    self._cache = json.load(fh)
                logger.info(
                    "Loaded %d cached RadGraph results from %s",
                    len(self._cache),
                    self._cache_path,
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read RadGraph cache (%s).", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, report: str) -> dict:
        """Run RadGraph on *report* and return a structured entity dict.

        The returned dict follows the RadGraph annotation schema::

            {
                "text":     <original report text>,
                "entities": {
                    "<id>": {
                        "tokens":    <entity surface form>,
                        "label":     <OBS-DP | OBS-DA | OBS-U | ANAT-DP>,
                        "start_ix":  <int>,
                        "end_ix":    <int>,
                        "relations": [...]
                    },
                    ...
                }
            }

        Falls back to keyword matching when RadGraph is unavailable.
        Results are cached in-memory (and optionally on disk) to avoid
        repeated inference on identical report strings.

        Parameters
        ----------
        report:
            Raw radiology report text (findings + impression).

        Returns
        -------
        dict
        """
        report = _clean_text(report)
        if not report:
            return {"text": report, "entities": {}}

        if report in self._cache:
            return self._cache[report]

        result = (
            self._radgraph_extract(report)
            if self._use_radgraph
            else self._fallback_extract(report)
        )
        self._cache[report] = result
        return result

    def to_entity_text(self, result: dict) -> str:
        """Concatenate entity surface-form tokens into a single string.

        Parameters
        ----------
        result:
            Output of :meth:`extract`.

        Returns
        -------
        str
            Whitespace-joined entity surface forms, suitable for tokenisation
            by a language model.
        """
        entities = result.get("entities", {})
        tokens = [
            v["tokens"]
            for v in entities.values()
            if isinstance(v, dict) and v.get("tokens")
        ]
        return " ".join(tokens)

    def to_entity_embedding(
        self,
        entity_text: str,
        text_encoder: torch.nn.Module,
        tokenizer,
    ) -> torch.Tensor:
        """Return an L2-normalised ClinicalBERT mean-pool embedding.

        Parameters
        ----------
        entity_text:
            Output of :meth:`to_entity_text`.
        text_encoder:
            A ``transformers.AutoModel`` instance (e.g. Bio_ClinicalBERT)
            already moved to the correct device.
        tokenizer:
            Matching ``transformers.AutoTokenizer``.

        Returns
        -------
        torch.Tensor
            Shape ``(hidden_size,)`` on CPU, L2-normalised.
            Returns a zero vector when *entity_text* is empty.
        """
        hidden_size: int = text_encoder.config.hidden_size
        if not entity_text.strip():
            return torch.zeros(hidden_size)

        device = next(text_encoder.parameters()).device
        enc = tokenizer(
            entity_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            out = text_encoder(**enc)
            # Mean-pool over non-padding positions
            mask = enc["attention_mask"].unsqueeze(-1).float()   # (1, L, 1)
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = F.normalize(emb.squeeze(0), dim=-1)

        return emb.cpu()

    def report_similarity(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> float:
        """Cosine similarity in ``[-1, 1]`` between two entity embeddings.

        Parameters
        ----------
        emb_a, emb_b:
            Tensors of shape ``(hidden_size,)`` produced by
            :meth:`to_entity_embedding`.

        Returns
        -------
        float
        """
        a = F.normalize(emb_a.float().reshape(1, -1), dim=-1)
        b = F.normalize(emb_b.float().reshape(1, -1), dim=-1)
        return float(torch.mm(a, b.T).item())

    def save_cache(self) -> None:
        """Flush the in-memory extraction cache to *cache_path* (if set)."""
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, indent=2, ensure_ascii=False)
        logger.info(
            "Saved %d RadGraph results to %s",
            len(self._cache),
            self._cache_path,
        )

    @property
    def using_radgraph(self) -> bool:
        """``True`` when the real RadGraph model is loaded and active."""
        return self._use_radgraph and self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-initialise the RadGraph model on first call."""
        if self._model is not None:
            return
        try:
            from radgraph import RadGraph  # type: ignore[import]

            self._model = RadGraph(reward_level="partial")
            logger.info("RadGraph model loaded (device=%s).", self.device)
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"RadGraph unavailable ({exc}).\n"
                "Falling back to keyword-based entity extraction.\n"
                "This affects evaluate.py Entity F1 scores — results will be approximate.\n"
                "To fix: pip install radgraph  (requires transformers < 4.40)",
                stacklevel=2,
            )
            logger.warning("RadGraph unavailable (%s). Using keyword fallback.", exc)
            self._use_radgraph = False

    def _radgraph_extract(self, report: str) -> dict:
        self._load_model()
        if not self._use_radgraph:
            return self._fallback_extract(report)
        try:
            raw = self._model([report])
            # RadGraph returns a list; take the first (and only) annotation dict
            annotation: dict = raw[0] if isinstance(raw, (list, tuple)) else raw
            return _normalise_radgraph_output(annotation, report)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RadGraph inference error (%s).  Using keyword fallback for this sample.",
                exc,
            )
            self._use_radgraph = False
            return self._fallback_extract(report)

    def _fallback_extract(self, report: str) -> dict:
        """Keyword-matching fallback producing the same entity schema."""
        text_lower = report.lower()
        entities: Dict[str, dict] = {}
        entity_id = 0
        for term in _FALLBACK_TERMS:
            start = 0
            while True:
                idx = text_lower.find(term, start)
                if idx == -1:
                    break
                entities[str(entity_id)] = {
                    "tokens": term,
                    "label": "OBS-DP",
                    "start_ix": idx,
                    "end_ix": idx + len(term) - 1,
                    "relations": [],
                }
                entity_id += 1
                start = idx + 1
        return {"text": report, "entities": entities}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Collapse whitespace and strip surrounding spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _normalise_radgraph_output(annotation: dict, original_text: str) -> dict:
    """Reconcile different RadGraph API output versions to a single schema.

    Some versions of the library wrap the annotation under the input text
    string as a key; this function unwraps that layer.
    """
    if "entities" not in annotation:
        # Attempt one level of unwrapping
        for v in annotation.values():
            if isinstance(v, dict) and "entities" in v:
                annotation = v
                break
    annotation.setdefault("text", original_text)
    annotation.setdefault("entities", {})
    return annotation
