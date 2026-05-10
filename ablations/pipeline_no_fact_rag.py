"""Ablation pipeline: Without Fact-Verified RAG.

Removes the FactMM-RAG factual retriever.  The FAISS index is still used for
retrieval, but the query encoder used to build it is the Stage-1 contrastive
checkpoint (image-only) rather than the fact-verified multimodal retriever.

What changes vs. full model
---------------------------
  Cache     Rebuilt   — forced to use stage1/best.pth (ignores
                         factual_retriever.pth if it exists).
  Stage 3   Retrained — generator adapts to the lower-quality retrieved reports.
  Evaluate  Run       — same full-model backbone; only Stage-3 weights differ.

What is NOT changed
-------------------
  Stage 1 / Stage 2 / FAISS index / report classifier — all shared.
  The factual retriever checkpoint is intentionally ignored (not deleted).

This isolates the gain from FactMM-RAG factual pair mining + retriever
training vs. using the simpler image-only contrastive retriever.

Checkpoints saved to
---------------------
  checkpoints/ablations/no_fact_rag/     (Stage-3 best_generator.pth)
  store/ablations/no_fact_rag/           (cache_train.pt / cache_val.pt)
  results/ablations/no_fact_rag/         (metrics.json, generated_reports.json)

Usage
-----
  python ablations/pipeline_no_fact_rag.py
  python ablations/pipeline_no_fact_rag.py --from stage3
  python ablations/pipeline_no_fact_rag.py --only evaluate
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

EXP = "no_fact_rag"

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
STAGES = [
    (
        "convert_models",
        "Download Bio_ClinicalBERT safetensors (skip if already done)",
        "scripts/prepare/convert_models.py",
        "models/bio_clinical_bert/model.safetensors",
    ),
    (
        "index",
        "Build FAISS Index (always re-run — fast)",
        "scripts/prepare/build_index.py",
        None,
    ),
    (
        "cache",
        "Cache Features WITHOUT fact-verified retriever",
        f"scripts/prepare/cache_features.py --no_fact_rag --exp_name {EXP}",
        f"store/ablations/{EXP}/cache_train.pt",
    ),
    (
        "stage3",
        "Stage 3 — Report Generator (no_fact_rag cache)",
        f"scripts/train/train_stage3.py --exp_name {EXP}",
        f"checkpoints/ablations/{EXP}/best_generator.pth",
    ),
    (
        "evaluate",
        "Evaluate WITHOUT fact-verified RAG",
        f"scripts/evaluate/evaluate.py --exp_name {EXP}",
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
print(f"  Removes   : Fact-Verified RAG retriever (uses Stage-1 contrastive encoder)")
print(f"  Measures  : gain from FactMM-RAG factual pair mining + retriever training")
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
        print(f"  python ablations/pipeline_no_fact_rag.py --from {stage_id}\n")
        sys.exit(1)

banner("no_fact_rag Ablation Complete", GREEN)
total = time.time() - pipeline_start
print(f"  Total time : {fmt_time(total)}")
print(f"  Results    : results/ablations/{EXP}/metrics.json")
print(f"\n  Run compare: python ablations/compare_ablations.py\n")
