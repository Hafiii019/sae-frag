"""Ablation pipeline: Without SAFE (Spatial Attention Feature Enhancement).

Removes the cross-attention module that lets C5 semantic features query P4
detail features.  Without SAFE, MultiViewBackbone returns FPN P4 directly and
fuses the two views with equal weights instead of learned softmax weights.

What changes vs. full model
---------------------------
  Stage 1   Retrained — SAFE is inside MultiViewBackbone; the contrastive
            alignment embeddings change without it.
  Stage 2   Retrained — image classifier backbone also loses SAFE.
  Cache     Rebuilt   — visual / aligned features differ with the new backbone.
  Stage 3   Retrained — Stage-3 generator adapts to the SAFE-free features.
  Evaluate  Run       — backbone loaded without SAFE.

What is shared (not retrained)
-------------------------------
  Report classifier (train_report_classifier.py)
  Factual pair mining (mine_factual_pairs.py)
  Entity extraction  (extract_entities.py)
  Factual retriever  (train_factual_retriever.py)  -- uses Stage-1 that WAS
                       trained with SAFE, but the retriever index and FAISS
                       index are rebuilt against the new no-SAFE embeddings.
  FAISS index        Rebuilt to match the new backbone.

Checkpoints saved to
---------------------
  checkpoints/ablations/no_safe/stage1/
  checkpoints/ablations/no_safe/stage2/
  checkpoints/ablations/no_safe/          (Stage-3 best_generator.pth)
  store/ablations/no_safe/                (cache_train.pt / cache_val.pt)
  results/ablations/no_safe/              (metrics.json, generated_reports.json)

Usage
-----
  python ablations/pipeline_no_safe.py          # run all steps
  python ablations/pipeline_no_safe.py --from cache  # resume from cache step
  python ablations/pipeline_no_safe.py --only evaluate
"""

import argparse
import os
import subprocess
import sys
import time

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable

_ENV = os.environ.copy()
_ENV["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXP = "no_safe"

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"


def banner(text, color=CYAN):
    w = 70
    print(f"\n{color}{BOLD}{'-' * w}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'-' * w}{RESET}\n")


def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"


# ── Stage definitions ─────────────────────────────────────────────────────
# (id, label, script + args, skip-if-this-path-exists-or-None)
STAGES = [
    (
        "convert_models",
        "Download Bio_ClinicalBERT safetensors (skip if already done)",
        "scripts/prepare/convert_models.py",
        "models/bio_clinical_bert/model.safetensors",
    ),
    (
        "stage1",
        "Stage 1 — Contrastive Alignment WITHOUT SAFE",
        f"scripts/train/train_stage1.py --no_safe --exp_name {EXP}",
        f"checkpoints/ablations/{EXP}/stage1/best.pth",
    ),
    (
        "report_cls",
        "Stage 2a — Report Classifier (shared, skip if exists)",
        "scripts/train/train_report_classifier.py",
        "checkpoints/stage2/report_classifier.pth",
    ),
    (
        "stage2",
        "Stage 2b — Image Classifier WITHOUT SAFE",
        f"scripts/train/train_stage2.py --no_safe --exp_name {EXP}",
        f"checkpoints/ablations/{EXP}/stage2/image_classifier.pth",
    ),
    (
        "mine_pairs",
        "Factual Pair Mining (shared, skip if exists)",
        "scripts/prepare/mine_factual_pairs.py --delta 0.3 --top_k 2",
        "store/factual_pairs.pkl",
    ),
    (
        "extract_entities",
        "Entity Extraction (shared, skip if exists)",
        "scripts/prepare/extract_entities.py --top_k 20",
        "store/entity_tags.json",
    ),
    (
        "retriever",
        "Factual Retriever Training (shared, skip if exists)",
        "scripts/train/train_factual_retriever.py",
        "checkpoints/stage1/factual_retriever.pth",
    ),
    (
        "index",
        "Build FAISS Index (always re-run — fast)",
        "scripts/prepare/build_index.py",
        None,
    ),
    (
        "cache",
        "Cache Features WITHOUT SAFE",
        f"scripts/prepare/cache_features.py --no_safe --exp_name {EXP}",
        f"store/ablations/{EXP}/cache_train.pt",
    ),
    (
        "stage3",
        "Stage 3 — Report Generator (no_safe cache)",
        f"scripts/train/train_stage3.py --exp_name {EXP}",
        f"checkpoints/ablations/{EXP}/best_generator.pth",
    ),
    (
        "evaluate",
        "Evaluate WITHOUT SAFE",
        f"scripts/evaluate/evaluate.py --no_safe --exp_name {EXP}",
        None,
    ),
]

STAGE_IDS = [s[0] for s in STAGES]

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=f"SAE-FRAG ablation: {EXP}")
parser.add_argument("--from", dest="from_stage", choices=STAGE_IDS, default=None)
parser.add_argument("--only", dest="only_stage", choices=STAGE_IDS, default=None)
parser.add_argument("--force", dest="force_stages", choices=STAGE_IDS, nargs="+", default=[])
args = parser.parse_args()


def run_stage(stage_id, label, script_args, skip_path, force=False):
    parts  = script_args.split()
    script = parts[0]
    extra  = parts[1:]
    cmd    = [PYTHON, os.path.join(ROOT, script)] + extra

    if skip_path and not force:
        full = os.path.join(ROOT, skip_path)
        if os.path.exists(full):
            print(f"{YELLOW}  Skipping {label} — output exists:{RESET}")
            print(f"    {full}\n")
            return True

    banner(label)
    print(f"  Command: {' '.join(cmd)}\n")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=_ENV)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"\n{RED}{BOLD}  FAILED ({fmt_time(elapsed)}){RESET}\n")
        return False
    print(f"\n{GREEN}{BOLD}  Done ({fmt_time(elapsed)}){RESET}\n")
    return True


# ── Build run list ────────────────────────────────────────────────────────
if args.only_stage:
    run_list = [s for s in STAGES if s[0] == args.only_stage]
elif args.from_stage:
    idx      = STAGE_IDS.index(args.from_stage)
    run_list = STAGES[idx:]
else:
    run_list = list(STAGES)

# ── Run ───────────────────────────────────────────────────────────────────
banner(f"SAE-FRAG Ablation: {EXP}", GREEN)
print(f"  Removes   : SAFE (Spatial Attention Feature Enhancement)")
print(f"  Measures  : gain from C5->P4 cross-attention spatial enhancement")
print(f"  Stages    : {[s[0] for s in run_list]}\n")

pipeline_start = time.time()
results = {}

for stage_id, label, script, skip_path in run_list:
    force = stage_id in args.force_stages
    ok    = run_stage(stage_id, label, script, skip_path, force=force)
    results[stage_id] = ok
    if not ok:
        print(f"{RED}Pipeline stopped at '{stage_id}'.{RESET}")
        print(f"Fix the error, then resume with:")
        print(f"  python ablations/pipeline_no_safe.py --from {stage_id}\n")
        sys.exit(1)

banner("no_safe Ablation Complete", GREEN)
total = time.time() - pipeline_start
print(f"  Total time : {fmt_time(total)}")
print(f"  Results    : results/ablations/{EXP}/metrics.json")
print(f"\n  Run compare: python ablations/compare_ablations.py\n")
