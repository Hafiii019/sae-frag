"""Report verification via CrossModalAlignment re-ranking.

Given a frozen ``CrossModalAlignment`` module and a list of FAISS-retrieved
candidate reports, ``ReportVerifier`` scores each candidate by measuring how
strongly the image's spatial patches (queries) attend to the report tokens
(keys / values) and returns the highest-scoring candidate as the *verified*
report.

Design rationale
----------------
``CrossModalAlignment`` runs multi-head cross-attention with image patches as
queries and ClinicalBERT report tokens as keys/values.  A high mean attention
weight signals that the report's clinical content is spatially grounded in the
image regions, making it a reliable context for the generation decoder.
Re-ranking the top-K FAISS candidates this way suppresses reports that are
visually similar in embedding space but clinically mismatched.

Usage
-----
    verifier = ReportVerifier(alignment=alignment_module, min_score=0.0)
    best_report, score = verifier.verify(image_features, candidate_reports)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReportVerifier:
    """Re-rank retrieved reports using cross-modal attention scores.

    Parameters
    ----------
    alignment:
        A ``CrossModalAlignment`` instance.  Will be set to eval mode and
        frozen on construction — no external freezing required.
    min_score:
        If every candidate scores below this threshold the verifier falls
        back to the FAISS rank-1 report (index 0 of *candidate_reports*).
        Set to ``0.0`` (default) to always re-rank without filtering.
    """

    def __init__(
        self,
        alignment: torch.nn.Module,
        min_score: float = 0.0,
    ) -> None:
        self.alignment = alignment
        self.min_score = min_score

        # Guarantee the alignment module is frozen in eval mode
        self.alignment.eval()
        for param in self.alignment.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        image_features: torch.Tensor,
        report: str,
    ) -> float:
        """Compute a cross-modal attention score for one (image, report) pair.

        Parameters
        ----------
        image_features:
            Shape ``(1, 256, 7, 7)`` — single-sample batch.
        report:
            Candidate radiology report text.

        Returns
        -------
        float
            Mean cross-modal attention weight in ``[0, 1]``.  Higher means
            the image patches attend more strongly to this report's tokens,
            indicating better clinical alignment.
        """
        if not report.strip():
            return 0.0

        aligned, cls_token, _ = self.alignment(image_features, [report])
        # Cosine similarity between global image embedding and text CLS token.
        # Produces scores in [-1, 1] — far more discriminative than mean
        # attention weight (~0.035 uniform baseline from old implementation).
        img_global = F.normalize(aligned.mean(dim=1), dim=-1)  # (1, 256)
        txt_global = F.normalize(cls_token,           dim=-1)  # (1, 256)
        return float(F.cosine_similarity(img_global, txt_global).item())

    @torch.no_grad()
    def verify(
        self,
        image_features: torch.Tensor,
        candidate_reports: List[str],
    ) -> Tuple[str, float]:
        """Select the best-matching report from *candidate_reports*.

        Scores every candidate independently via :meth:`score` and returns
        the one with the highest cross-modal attention score.  Falls back to
        the FAISS rank-1 entry when all scores are below ``self.min_score``
        or when only one candidate is provided.

        Parameters
        ----------
        image_features:
            Shape ``(1, 256, 7, 7)`` — single-sample batch.
        candidate_reports:
            Ordered list of FAISS-retrieved report strings (rank-1 first).

        Returns
        -------
        Tuple[str, float]
            ``(best_report, best_score)`` — the report that maximises the
            cross-modal attention score and the corresponding scalar score.

        Raises
        ------
        ValueError
            If *candidate_reports* is empty.
        """
        if not candidate_reports:
            raise ValueError("candidate_reports must not be empty.")

        # Single candidate: skip scoring loop
        if len(candidate_reports) == 1:
            s = self.score(image_features, candidate_reports[0])
            return candidate_reports[0], s

        scores: List[float] = [
            self.score(image_features, r) for r in candidate_reports
        ]

        for i, s in enumerate(scores):
            logger.debug("Candidate %d  score=%.4f", i, s)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        if best_score < self.min_score:
            logger.debug(
                "All scores below min_score=%.4f; falling back to FAISS rank-1.",
                self.min_score,
            )
            return candidate_reports[0], scores[0]

        return candidate_reports[best_idx], best_score
