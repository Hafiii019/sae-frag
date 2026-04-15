"""
Stanza-based biomedical entity extraction for PKARG conditioning.

Implements Izhar et al. (2025) §3.3.2 — extracts four contextual categorical
entity types from the IU X-Ray reports using Stanza's clinical NER models:

    • Anatomy      (radiology package)    — organs, anatomical structures
    • Observation  (radiology package)    — radiological findings
    • Uncertainty  (radiology package)    — hedging terms (possible, likely, ...)
    • Problem      (i2b2 package)         — clinical problems / diagnoses

Keeps the TOP-20 most frequent entities per category from the training set,
then populates val/test sets with those same entities if they appear in the
report.  Following PKARG: sparse propagation during training/testing avoids
overfitting and limits vocab bleed.

Output:  store/entity_tags.json
    { "<uid>": "problems: effusion, consolidation; anatomies: lungs, heart; ...",
      ... }

Usage
-----
    conda activate ergonomics
    pip install stanza          # if not installed
    python scripts/prepare/extract_entities.py
    python scripts/prepare/extract_entities.py --top_k 20  # default

Notes
-----
  Stanza downloads clinical NER models (~300 MB) on first run.
  Models are cached at ~/stanza_resources/  (or STANZA_RESOURCES_DIR).
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from configs.config import Config
from data.dataset import IUXrayMultiViewDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Stanza entity type → output category label ────────────────────────────
# radiology package: ANATOMY, OBSERVATION, UNCERTAINTY
# i2b2 package:      PROBLEM
_TYPE_MAP = {
    "ANATOMY":      "anatomies",
    "OBSERVATION":  "observations",
    "UNCERTAINTY":  "uncertainties",
    "PROBLEM":      "problems",
}


def _load_stanza_pipelines():
    """Load Stanza NER pipelines; download models on first run."""
    try:
        import stanza
    except ImportError:
        log.error("stanza not installed. Run: pip install stanza")
        sys.exit(1)

    # processors= as a STRING limits loading to ONLY those two processors
    # (avoids loading lemma/pos/depparse which have broken mimic models).
    # package= as a DICT selects the model per processor.
    log.info("Downloading Stanza models (first run only, ~300 MB per model)...")
    stanza.download("en", processors="tokenize,ner",
                    package={"tokenize": "mimic", "ner": "radiology"},
                    logging_level="ERROR")
    stanza.download("en", processors="tokenize,ner",
                    package={"tokenize": "mimic", "ner": "i2b2"},
                    logging_level="ERROR")
    log.info("Building Stanza pipelines...")

    radiology_nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,ner",
        package={"tokenize": "mimic", "ner": "radiology"},
        logging_level="ERROR",
    )
    i2b2_nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,ner",
        package={"tokenize": "mimic", "ner": "i2b2"},
        logging_level="ERROR",
    )
    log.info("Stanza models loaded.")
    return radiology_nlp, i2b2_nlp


def _extract_entities(text: str, radiology_nlp, i2b2_nlp) -> dict:
    """Return {category: [entity_text, ...]} for one report text."""
    if not text or not text.strip():
        return {}

    entities: dict = defaultdict(list)

    # Radiology pipeline → Anatomy, Observation, Uncertainty
    doc = radiology_nlp(text)
    for ent in doc.entities:
        etype = ent.type.upper()
        if etype in ("ANATOMY", "OBSERVATION", "UNCERTAINTY"):
            cat = _TYPE_MAP[etype]
            norm = ent.text.lower().strip()
            if norm:
                entities[cat].append(norm)

    # i2b2 pipeline → Problem
    doc2 = i2b2_nlp(text)
    for ent in doc2.entities:
        if ent.type.upper() == "PROBLEM":
            norm = ent.text.lower().strip()
            if norm:
                entities["problems"].append(norm)

    return dict(entities)


def _format_entity_string(entity_dict: dict, top_entities: dict | None = None) -> str:
    """Format entity dict as a readable conditioning string.

    If top_entities is provided (train-set top-20 per category), only include
    entities that appear in that set (PKARG sparse propagation).

    Returns e.g.:
        "problems: effusion, consolidation; anatomies: lungs, heart; ..."
    """
    parts = []
    for cat in ("problems", "anatomies", "observations", "uncertainties"):
        if cat not in entity_dict:
            continue
        ents = entity_dict[cat]
        if top_entities is not None:
            allowed = top_entities.get(cat, set())
            ents = [e for e in ents if e in allowed]
        if ents:
            # Deduplicate while preserving order
            seen = set()
            unique = [e for e in ents if not (e in seen or seen.add(e))]
            parts.append(f"{cat}: {', '.join(unique)}")

    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-K most frequent entities per category to keep (default 20)")
    args = parser.parse_args()

    radiology_nlp, i2b2_nlp = _load_stanza_pipelines()

    # ── Process training set to find top-K vocab ──────────────────────────
    log.info("\nProcessing training set for entity vocabulary...")
    train_ds = IUXrayMultiViewDataset(Config.DATA_ROOT, split="train")

    category_counters: dict[str, Counter] = {
        "problems": Counter(),
        "anatomies": Counter(),
        "observations": Counter(),
        "uncertainties": Counter(),
    }

    train_entities: dict[str, dict] = {}  # uid → {cat: [ent, ...]}

    for sample in train_ds.samples:
        uid = str(sample["uid"])
        # Use the raw finding text (sample["report"] is already cleaned)
        text = sample["report"]
        ents = _extract_entities(text, radiology_nlp, i2b2_nlp)
        train_entities[uid] = ents
        for cat, ent_list in ents.items():
            category_counters[cat].update(ent_list)

    # Top-K most frequent per category (PKARG: sparse information propagation)
    top_entities: dict[str, set] = {}
    for cat, counter in category_counters.items():
        top_k_list = [e for e, _ in counter.most_common(args.top_k)]
        top_entities[cat] = set(top_k_list)
        log.info(f"  Top-{args.top_k} {cat}: {', '.join(top_k_list[:5])}{'...' if len(top_k_list) > 5 else ''}")

    # ── Build entity tag strings for all splits ───────────────────────────
    entity_tags: dict[str, str] = {}

    # Training set — limit to top-K vocab
    for sample in train_ds.samples:
        uid = str(sample["uid"])
        ents = train_entities.get(uid, {})
        entity_tags[uid] = _format_entity_string(ents, top_entities)

    # Val and test sets — extract + filter to top-K vocab
    for split in ("val", "test"):
        log.info(f"\nProcessing {split} set...")
        ds = IUXrayMultiViewDataset(Config.DATA_ROOT, split=split)
        for sample in ds.samples:
            uid = str(sample["uid"])
            if uid in entity_tags:
                continue
            text = sample["report"]
            ents = _extract_entities(text, radiology_nlp, i2b2_nlp)
            entity_tags[uid] = _format_entity_string(ents, top_entities)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(_ROOT, "store", "entity_tags.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entity_tags, f, indent=2, ensure_ascii=False)

    filled = sum(1 for v in entity_tags.values() if v)
    log.info(f"\nSaved {len(entity_tags)} entity tag strings "
             f"({filled} non-empty) → {os.path.relpath(out_path, _ROOT)}")

    # Sample output
    sample_uid = next(iter(entity_tags))
    log.info(f"  Example (uid={sample_uid}): \"{entity_tags[sample_uid]}\"")


if __name__ == "__main__":
    main()
