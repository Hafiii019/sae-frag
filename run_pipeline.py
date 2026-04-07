"""Full SAE-FRAG training and evaluation pipeline.

Runs all stages in sequence.  Each stage checks if its output checkpoint
already exists and skips if so — re-running is safe and resumes from where
you left off.

Usage
-----
  python run_pipeline.py                  # run everything
  python run_pipeline.py --from stage2    # skip stage1, start from stage2
  python run_pipeline.py --only evaluate  # run one stage
  python run_pipeline.py --skip evaluate  # run all except evaluate

Stages (in order)
-----------------
  stage1       — visual-language contrastive alignment
  report_cls   — report classifier (text-side pseudo-label teacher)
  stage2       — image classifier (distilled from report_cls)
  mine_pairs   — factual pair mining for FactMM-RAG retriever
  retriever    — fact-aware multimodal dense retriever
  cache        — cache visual features to disk
  index        — build FAISS retrieval index
  stage3       — flan-t5-large report generator
  evaluate     — BLEU / ROUGE / ClinicalBERT metrics
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON = sys.executable

# ── Stage definitions ─────────────────────────────────────────────────────
# Each stage: (id, human label, script path, skip-if-exists path or None)
STAGES = [
    (
        "stage1",
        "Stage 1 — Visual-Language Alignment",
        "scripts/train/train_stage1.py",
        "checkpoints/stage1/best.pth",
    ),
    (
        "report_cls",
        "Stage 2a — Report Classifier",
        "scripts/train/train_report_classifier.py",
        "checkpoints/stage2/report_classifier.pth",
    ),
    (
        "stage2",
        "Stage 2b — Image Classifier",
        "scripts/train/train_stage2.py",
        "checkpoints/stage2/image_classifier.pth",
    ),
    (
        "mine_pairs",
        "Factual Pair Mining (FactMM-RAG)",
        "scripts/prepare/mine_factual_pairs.py --delta 0.5 --top_k 2",
        "store/factual_pairs.pkl",
    ),
    (
        "retriever",
        "Factual Retriever Training (FactMM-RAG)",
        "scripts/train/train_factual_retriever.py",
        "checkpoints/stage1/factual_retriever.pth",
    ),
    (
        "cache",
        "Cache Visual Features",
        "scripts/prepare/cache_features.py",
        "store/cache_train.pt",
    ),
    (
        "index",
        "Build FAISS Index",
        "scripts/prepare/build_index.py",
        None,   # always re-run (fast, ~30 s)
    ),
    (
        "stage3",
        "Stage 3 — Report Generator",
        "scripts/train/train_stage3.py",
        "checkpoints/stage3/best_generator.pth",
    ),
    (
        "evaluate",
        "Evaluation",
        "scripts/evaluate/evaluate.py",
        None,   # always run
    ),
]

STAGE_IDS = [s[0] for s in STAGES]

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--from",   dest="from_stage", choices=STAGE_IDS, default=None,
                    help="Start pipeline from this stage (skip all before it).")
parser.add_argument("--only",   dest="only_stage", choices=STAGE_IDS, default=None,
                    help="Run only this single stage.")
parser.add_argument("--skip",   dest="skip_stages", choices=STAGE_IDS, nargs="+", default=[],
                    help="Skip these stages.")
parser.add_argument("--force",  dest="force_stages", choices=STAGE_IDS, nargs="+", default=[],
                    help="Force-run these stages even if checkpoint exists.")
args = parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"

def banner(text, color=CYAN):
    width = 70
    print(f"\n{color}{BOLD}{'─' * width}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'─' * width}{RESET}\n")

def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

def run_stage(stage_id, label, script_args, skip_path, force=False):
    """Run a pipeline stage subprocess.  Returns True on success."""
    # Parse script path and any inline args
    parts  = script_args.split()
    script = parts[0]
    extra  = parts[1:]

    # Skip if checkpoint exists (and not forced)
    if skip_path and not force:
        full = os.path.join(ROOT, skip_path)
        if os.path.exists(full):
            print(f"{YELLOW}  ↷ Skipping {label} — checkpoint found:{RESET}")
            print(f"    {full}\n")
            return True

    banner(label)
    cmd = [PYTHON, os.path.join(ROOT, script)] + extra
    print(f"  Command: {' '.join(cmd)}\n")

    t0  = time.time()
    ret = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    if ret.returncode != 0:
        print(f"\n{RED}{BOLD}  ✗ {label} FAILED (exit code {ret.returncode}){RESET}")
        print(f"  Elapsed: {fmt_time(elapsed)}\n")
        return False

    print(f"\n{GREEN}{BOLD}  ✓ {label} done  ({fmt_time(elapsed)}){RESET}\n")
    return True


# ── Build execution list ──────────────────────────────────────────────────
if args.only_stage:
    run_list = [s for s in STAGES if s[0] == args.only_stage]
else:
    run_list = list(STAGES)
    if args.from_stage:
        idx = STAGE_IDS.index(args.from_stage)
        run_list = run_list[idx:]
    run_list = [s for s in run_list if s[0] not in args.skip_stages]

# ── Run ───────────────────────────────────────────────────────────────────
banner("SAE-FRAG Pipeline", GREEN)
print(f"  Stages to run: {[s[0] for s in run_list]}")
if args.force_stages:
    print(f"  Force-run:     {args.force_stages}")
print()

pipeline_start = time.time()
results = {}

for stage_id, label, script, skip_path in run_list:
    force = stage_id in args.force_stages
    ok = run_stage(stage_id, label, script, skip_path, force=force)
    results[stage_id] = ok
    if not ok:
        print(f"{RED}Pipeline stopped at '{stage_id}'.{RESET}")
        print("Fix the error above, then resume with:")
        print(f"  python run_pipeline.py --from {stage_id}\n")
        break

# ── Summary ───────────────────────────────────────────────────────────────
banner("Pipeline Summary", BOLD)
for sid, ok in results.items():
    icon  = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    label = next(s[1] for s in STAGES if s[0] == sid)
    print(f"  {icon}  {label}")

total = time.time() - pipeline_start
print(f"\n  Total time: {fmt_time(total)}\n")

if all(results.values()):
    print(f"{GREEN}{BOLD}  All stages completed successfully.{RESET}")
    print(f"  Results: {os.path.join(ROOT, 'results', 'metrics.json')}\n")
else:
    sys.exit(1)
