"""Factual accuracy verification of RAG-generated radiology reports.

Analyses results/generated_reports.json and produces a per-sample + aggregate
factual accuracy report saved to results/factual_verification.json.

Three complementary checks are run for every non-empty sample:

1. Clinical-entity overlap
   A curated vocabulary of 50+ clinical findings is used to extract entity
   sets from both the reference and the generated report.  Per-sample
   Precision, Recall and F1 are computed, then averaged.

2. Hallucination / omission flags
   - hallucinations : entities present in *generated* but absent in *reference*
   - omissions      : entities present in *reference* but absent in *generated*

3. verify_score interpretation
   The CrossModalAlignment verify_score (already stored in the JSON) is
   bucketed:  HIGH (>= 0.03), MED (0..0.03), LOW (< 0).  A LOW score means
   the retrieved report was poorly grounded in the image — a proxy for
   retrieval quality independent of generation quality.

Usage
-----
    python scripts/evaluate/factual_verify.py
    python scripts/evaluate/factual_verify.py --input results/generated_reports.json
    python scripts/evaluate/factual_verify.py --top_errors 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# ── Clinical vocabulary ───────────────────────────────────────────────────────
# Ordered longest-first so multi-word terms match before their substrings.
_CLINICAL_TERMS: List[str] = sorted([
    # ── Pathological findings ──────────────────────────────────────
    "pleural effusion",
    "pulmonary edema",
    "pulmonary embolism",
    "pulmonary fibrosis",
    "pulmonary hypertension",
    "aortic dissection",
    "cardiac tamponade",
    "pericardial effusion",
    "hilar adenopathy",
    "mediastinal widening",
    "pneumoperitoneum",
    "pneumothorax",
    "cardiomegaly",
    "atelectasis",
    "consolidation",
    "emphysema",
    "infiltrate",
    "opacities",
    "opacity",
    "effusion",
    "granuloma",
    "nodule",
    "mass",
    "hernia",
    "fibrosis",
    "scoliosis",
    "kyphosis",
    "osteophyte",
    "fracture",
    "adenopathy",
    "calcification",
    "cardiomegaly",
    "edema",
    "pneumonia",
    # ── Devices / surgical changes ─────────────────────────────────
    "pacemaker",
    "defibrillator",
    "picc line",
    "catheter",
    "pacemaker lead",
    "sternal wires",
    "spinal rods",
    "surgical clips",
    # ── Descriptive / normal findings ─────────────────────────────
    "hyperexpanded",
    "hyperinflated",
    "flattened diaphragm",
    "tortuous aorta",
    "atherosclerosis",
    "atherosclerotic",
    "cardiomegaly",
    "normal heart size",
    "clear lungs",
    "no pneumothorax",
    "no effusion",
    "no consolidation",
    "no pleural effusion",
    "no focal consolidation",
    # ── Anatomy qualifiers ────────────────────────────────────────
    "bibasilar",
    "left lower lobe",
    "right lower lobe",
    "left upper lobe",
    "right upper lobe",
    "right middle lobe",
    "perihilar",
    "basilar",
], key=lambda t: -len(t))


def _extract_entities(text: str) -> Set[str]:
    """Return the set of clinical terms present in *text* (case-insensitive)."""
    text = text.lower()
    found: Set[str] = set()
    for term in _CLINICAL_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            found.add(term)
    return found


def _entity_scores(ref: str, gen: str) -> Tuple[float, float, float, Set[str], Set[str]]:
    """Return (precision, recall, f1, hallucinations, omissions)."""
    ref_ents = _extract_entities(ref)
    gen_ents = _extract_entities(gen)

    hallucinations = gen_ents - ref_ents   # in generated, NOT in reference
    omissions      = ref_ents - gen_ents   # in reference,  NOT in generated

    if not ref_ents and not gen_ents:
        return 1.0, 1.0, 1.0, hallucinations, omissions

    if not gen_ents:
        return 0.0, 0.0, 0.0, hallucinations, omissions

    if not ref_ents:
        return 0.0, 0.0, 0.0, hallucinations, omissions

    tp = len(gen_ents & ref_ents)
    p  = tp / len(gen_ents)
    r  = tp / len(ref_ents)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1, hallucinations, omissions


def _bucket_verify(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.03:
        return "HIGH"
    if score >= 0.0:
        return "MED"
    return "LOW"


def _faithfulness_verdict(
    f1: float,
    hallucinations: Set[str],
    omissions: Set[str],
    verify_bucket: str,
) -> str:
    """Single-word factual verdict for a sample."""
    if f1 >= 0.9 and not hallucinations:
        return "ACCURATE"
    if hallucinations and f1 < 0.5:
        return "HALLUCINATED"
    if omissions and f1 < 0.5:
        return "INCOMPLETE"
    if hallucinations:
        return "PARTIAL_HALLUCINATION"
    if omissions:
        return "PARTIAL_OMISSION"
    return "ACCEPTABLE"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, top_errors: int = 15) -> None:
    with open(input_path, encoding="utf-8") as f:
        samples: List[Dict] = json.load(f)

    per_sample_results = []
    precision_list, recall_list, f1_list = [], [], []
    verdict_counts: Dict[str, int] = {}
    hallucination_freq: Dict[str, int] = {}
    omission_freq: Dict[str, int] = {}
    verify_bucket_counts = {"HIGH": 0, "MED": 0, "LOW": 0, "unknown": 0}

    for s in samples:
        ref = (s.get("reference") or "").strip()
        gen = (s.get("generated") or "").strip()
        verify_score = s.get("verify_score")

        vbucket = _bucket_verify(verify_score)
        verify_bucket_counts[vbucket] += 1

        # Skip samples with empty reference or generated
        if not ref or not gen:
            per_sample_results.append({
                "sample_idx":   s.get("sample_idx"),
                "uid":          s.get("uid"),
                "verdict":      "EMPTY",
                "precision":    None,
                "recall":       None,
                "f1":           None,
                "hallucinations": [],
                "omissions":    [],
                "verify_score": verify_score,
                "verify_bucket": vbucket,
            })
            verdict_counts["EMPTY"] = verdict_counts.get("EMPTY", 0) + 1
            continue

        p, r, f1, hallucs, omits = _entity_scores(ref, gen)
        vbucket_for_verdict = vbucket
        verdict = _faithfulness_verdict(f1, hallucs, omits, vbucket_for_verdict)

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        for h in hallucs:
            hallucination_freq[h] = hallucination_freq.get(h, 0) + 1
        for o in omits:
            omission_freq[o] = omission_freq.get(o, 0) + 1

        per_sample_results.append({
            "sample_idx":    s.get("sample_idx"),
            "uid":           s.get("uid"),
            "verdict":       verdict,
            "precision":     round(p, 4),
            "recall":        round(r, 4),
            "f1":            round(f1, 4),
            "hallucinations": sorted(hallucs),
            "omissions":     sorted(omits),
            "verify_score":  verify_score,
            "verify_bucket": vbucket,
            "reference_snippet": ref[:120],
            "generated_snippet": gen[:120],
        })

    # ── Aggregate stats ───────────────────────────────────────────────────
    n_scored = len(precision_list)
    n_total  = len(samples)
    n_empty  = verdict_counts.get("EMPTY", 0)

    aggregate = {
        "total_samples":         n_total,
        "scored_samples":        n_scored,
        "empty_samples":         n_empty,
        "mean_entity_precision": round(sum(precision_list) / n_scored, 4) if n_scored else None,
        "mean_entity_recall":    round(sum(recall_list)    / n_scored, 4) if n_scored else None,
        "mean_entity_f1":        round(sum(f1_list)        / n_scored, 4) if n_scored else None,
        "verdict_distribution":  dict(sorted(verdict_counts.items())),
        "verify_score_buckets":  verify_bucket_counts,
        "top_hallucinated_terms": sorted(
            hallucination_freq.items(), key=lambda x: -x[1]
        )[:top_errors],
        "top_omitted_terms": sorted(
            omission_freq.items(), key=lambda x: -x[1]
        )[:top_errors],
    }

    output = {
        "aggregate": aggregate,
        "per_sample": per_sample_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SAE-FRAG  ·  Factual Accuracy Verification Report")
    print("=" * 68)
    print(f"  Total samples          : {n_total}")
    print(f"  Scored (non-empty)     : {n_scored}")
    print(f"  Empty (skipped)        : {n_empty}")
    print()
    print("  Clinical Entity Overlap (generated vs. reference)")
    print(f"    Precision  : {aggregate['mean_entity_precision']:.4f}")
    print(f"    Recall     : {aggregate['mean_entity_recall']:.4f}")
    print(f"    F1         : {aggregate['mean_entity_f1']:.4f}")
    print()
    print("  Verdict distribution")
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        pct = 100 * c / n_total
        bar = "█" * int(pct / 2)
        print(f"    {v:<25s}: {c:4d}  ({pct:5.1f}%)  {bar}")
    print()
    print("  CrossModal verify_score buckets")
    for b in ("HIGH", "MED", "LOW"):
        c = verify_bucket_counts[b]
        pct = 100 * c / n_total
        print(f"    {b:<6s}(>= {'0.03' if b=='HIGH' else '0.00' if b=='MED' else '-inf'})  : {c:4d}  ({pct:5.1f}%)")
    print()
    if hallucination_freq:
        print("  Most hallucinated terms (in generated, absent in reference)")
        for term, cnt in aggregate["top_hallucinated_terms"]:
            print(f"    {term:<35s}: {cnt}")
    print()
    if omission_freq:
        print("  Most omitted terms (in reference, absent in generated)")
        for term, cnt in aggregate["top_omitted_terms"]:
            print(f"    {term:<35s}: {cnt}")
    print()
    print(f"  Full report saved → {output_path}")
    print("=" * 68 + "\n")


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    parser = argparse.ArgumentParser(description="Factual verification of generated reports")
    parser.add_argument(
        "--input",
        default=os.path.join(root, "results", "generated_reports.json"),
        help="Path to generated_reports.json",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(root, "results", "factual_verification.json"),
        help="Output path for the verification report",
    )
    parser.add_argument(
        "--top_errors",
        type=int,
        default=15,
        help="Number of top hallucinated/omitted terms to display",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(args.input, args.output, args.top_errors)


if __name__ == "__main__":
    main()
