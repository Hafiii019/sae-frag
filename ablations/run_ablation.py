"""Ablation study runner for SAE-FRAG.

Each experiment re-trains Stage 3 and/or re-evaluates with a single component
removed.  Stages 1-2, the factual retriever, the FAISS index, and the feature
cache are *shared* across all ablations - only Stage 3 weights change.

Ablations
---------
  Tier 1 - Stage-3 retrain + evaluate (each ~2-4 h on one GPU)
    no_retrieval   Remove RAG: empty retrieved text -> measures retrieval gain
    no_entity      Zero entity vectors -> measures CheXbert conditioning gain
    no_impression  Skip impression section -> measures PKARG auxiliary gain
    flan_t5        flan-t5-base instead of SciFive-base -> measures medical-vocab gain
    no_two_phase   FREEZE_T5_EPOCHS=0 -> measures two-phase training gain

  Tier 2 - Evaluate only (no retraining, minutes per run)
    full_model     Baseline full model (reference for delta columns)
    beam1          Greedy decoding (num_beams=1)
    beam5          Wide beam search (num_beams=5)
    lp_1           Neutral length penalty (length_penalty=1.0)

Usage (run from the project root OR from this directory)
-----
  python ablations/run_ablation.py                      # run all
  python ablations/run_ablation.py --only no_retrieval  # single experiment
  python ablations/run_ablation.py --tier 2             # eval-only experiments
  python ablations/run_ablation.py --list               # list and exit
"""

import argparse
import os
import subprocess
import sys
import time

# Project root is one level above this file (ablations/)
ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable

# KMP_DUPLICATE_LIB_OK is also set inside each script before third-party imports,
# but propagating it here is belt-and-suspenders for any script we may add later.
_SUBPROCESS_ENV = os.environ.copy()
_SUBPROCESS_ENV["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _build_cmd(script_path: str, extra: list) -> list:
    return [PYTHON, script_path] + extra


# ── Colour helpers ────────────────────────────────────────────────────────────
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


# ── Experiment registry ───────────────────────────────────────────────────────
# Each experiment is a list of steps.  Every step is:
#   (label, script_with_args)
# Steps run sequentially; failure stops the experiment.
#
# Checkpoint dir : checkpoints/ablations/<exp_name>/   (set via --exp_name)
# Results dir    : results/ablations/<exp_name>/       (set via --exp_name)

EXPERIMENTS = {
    # ── Baseline (eval-only, saves to results/ablations/full_model/) ─────────
    "full_model": {
        "description": "Full model baseline (no ablation) - reference for delta columns",
        "tier": 2,
        "steps": [
            ("Evaluate full model",
             "scripts/evaluate/evaluate.py --exp_name full_model"),
        ],
    },

    # ── Tier 1 (Stage 3 retrain) ─────────────────────────────────────────────
    "no_retrieval": {
        "description": "No RAG - empty retrieved text at both train & eval",
        "tier": 1,
        "steps": [
            ("Train Stage 3 (no retrieval)",
             "scripts/train/train_stage3.py --exp_name no_retrieval --no_retrieval"),
            ("Evaluate (no retrieval)",
             "scripts/evaluate/evaluate.py --exp_name no_retrieval --no_retrieval"),
        ],
    },
    "no_entity": {
        "description": "No entity conditioning - zero CheXbert vectors at train & eval",
        "tier": 1,
        "steps": [
            ("Train Stage 3 (no entity)",
             "scripts/train/train_stage3.py --exp_name no_entity --no_entity"),
            ("Evaluate (no entity)",
             "scripts/evaluate/evaluate.py --exp_name no_entity --no_entity"),
        ],
    },
    "no_impression": {
        "description": "No impression section - removes PKARG auxiliary knowledge",
        "tier": 1,
        "steps": [
            ("Train Stage 3 (no impression)",
             "scripts/train/train_stage3.py --exp_name no_impression --no_impression"),
            ("Evaluate (no impression)",
             "scripts/evaluate/evaluate.py --exp_name no_impression --no_impression"),
        ],
    },
    "flan_t5": {
        "description": "flan-t5-base generator - removes medical domain pretraining",
        "tier": 1,
        "steps": [
            ("Train Stage 3 (flan-t5-base)",
             "scripts/train/train_stage3.py --exp_name flan_t5 "
             "--generator_model google/flan-t5-base"),
            ("Evaluate (flan-t5-base)",
             "scripts/evaluate/evaluate.py --exp_name flan_t5 "
             "--generator_model google/flan-t5-base"),
        ],
    },
    "no_two_phase": {
        "description": "No two-phase training - T5 unfrozen from epoch 0",
        "tier": 1,
        "steps": [
            ("Train Stage 3 (no freeze warmup)",
             "scripts/train/train_stage3.py --exp_name no_two_phase --freeze_epochs 0"),
            ("Evaluate (no two-phase)",
             "scripts/evaluate/evaluate.py --exp_name no_two_phase"),
        ],
    },

    # ── Component ablations (dedicated pipeline scripts) ─────────────────────
    # Each delegates to its own pipeline script which handles multi-stage
    # retraining.  Run them directly for finer-grained control:
    #   python ablations/pipeline_no_safe.py
    #   python ablations/pipeline_no_fact_rag.py
    #   python ablations/pipeline_no_region_align.py
    "no_safe": {
        "description": "No SAFE module - removes Spatial Attention Feature Enhancement (full retrain)",
        "tier": 0,
        "steps": [
            ("Full pipeline without SAFE",
             "ablations/pipeline_no_safe.py"),
        ],
    },
    "no_fact_rag": {
        "description": "No Fact-Verified RAG - uses Stage-1 contrastive retriever instead",
        "tier": 1,
        "steps": [
            ("Cache + Stage3 + Evaluate without fact-verified RAG",
             "ablations/pipeline_no_fact_rag.py"),
        ],
    },
    "no_region_align": {
        "description": "No Region-Aware Alignment - removes CrossModalAlignment",
        "tier": 1,
        "steps": [
            ("Cache + Stage3 + Evaluate without CrossModalAlignment",
             "ablations/pipeline_no_region_align.py"),
        ],
    },

    # ── Tier 2 (evaluate-only) ────────────────────────────────────────────────
    "beam1": {
        "description": "Greedy decoding (num_beams=1) - baseline full-model checkpoint",
        "tier": 2,
        "steps": [
            ("Evaluate (greedy, num_beams=1)",
             "scripts/evaluate/evaluate.py --exp_name beam1 --num_beams 1"),
        ],
    },
    "beam5": {
        "description": "Wide beam search (num_beams=5) - baseline full-model checkpoint",
        "tier": 2,
        "steps": [
            ("Evaluate (wide beam, num_beams=5)",
             "scripts/evaluate/evaluate.py --exp_name beam5 --num_beams 5"),
        ],
    },
    "lp_1": {
        "description": "Neutral length penalty (length_penalty=1.0)",
        "tier": 2,
        "steps": [
            ("Evaluate (length_penalty=1.0)",
             "scripts/evaluate/evaluate.py --exp_name lp_1 --length_penalty 1.0"),
        ],
    },
}

EXP_IDS = list(EXPERIMENTS.keys())

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--only",  dest="only_exp",    choices=EXP_IDS, default=None)
parser.add_argument("--skip",  dest="skip_exps",   choices=EXP_IDS, nargs="+", default=[])
parser.add_argument("--tier",  dest="tier_filter", type=int, choices=[0, 1, 2], default=None,
                    help="Run only experiments of this tier (0=full retrain, 1=stage3 retrain, 2=eval-only)")
parser.add_argument("--list",  action="store_true", help="List experiments and exit")
args = parser.parse_args()

if args.list:
    banner("SAE-FRAG Ablation Experiments", CYAN)
    for eid, exp in EXPERIMENTS.items():
        tier_tag = f"[Tier {exp['tier']}]"
        print(f"  {BOLD}{eid:<18}{RESET}  {tier_tag}  {exp['description']}")
    print()
    sys.exit(0)

# ── Build run list ────────────────────────────────────────────────────────────
if args.only_exp:
    run_list = [(args.only_exp, EXPERIMENTS[args.only_exp])]
else:
    run_list = [
        (eid, exp) for eid, exp in EXPERIMENTS.items()
        if eid not in args.skip_exps
        and (args.tier_filter is None or exp["tier"] == args.tier_filter)
    ]

# ── Runner ────────────────────────────────────────────────────────────────────

def run_step(label: str, script_args: str, exp_name: str) -> bool:
    parts  = script_args.split()
    script = parts[0]
    extra  = parts[1:]
    cmd    = _build_cmd(os.path.join(ROOT, script), extra)
    print(f"  {CYAN}>{RESET} {label}")
    print(f"    {' '.join(cmd)}\n")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=ROOT, env=_SUBPROCESS_ENV)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n{RED}{BOLD}  x FAILED  ({fmt_time(elapsed)}){RESET}\n")
        return False
    print(f"\n{GREEN}{BOLD}  ok Done  ({fmt_time(elapsed)}){RESET}\n")
    return True


def run_experiment(exp_id: str, exp: dict) -> bool:
    banner(f"[{exp_id}]  {exp['description']}  (Tier {exp['tier']})", CYAN)
    for label, script_args in exp["steps"]:
        ok = run_step(label, script_args, exp_id)
        if not ok:
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
banner("SAE-FRAG Ablation Study", GREEN)
print(f"  Experiments : {[e for e, _ in run_list]}")
print(f"  Results dir : results/ablations/<exp_name>/\n")

pipeline_start = time.time()
results: dict[str, bool] = {}

for exp_id, exp in run_list:
    ok = run_experiment(exp_id, exp)
    results[exp_id] = ok
    if not ok:
        print(f"{RED}Experiment '{exp_id}' failed.{RESET}")
        print(f"Fix the error above, then resume with:")
        print(f"  python ablations/run_ablation.py --only {exp_id}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
banner("Ablation Summary", BOLD)
for eid, ok in results.items():
    icon = f"{GREEN}ok{RESET}" if ok else f"{RED}x{RESET}"
    desc = EXPERIMENTS[eid]["description"]
    print(f"  {icon}  {BOLD}{eid:<18}{RESET}  {desc}")

total = time.time() - pipeline_start
print(f"\n  Total time: {fmt_time(total)}")

n_ok = sum(results.values())
print(f"  {n_ok}/{len(results)} experiments completed.\n")

if n_ok < len(results):
    print(f"{YELLOW}Run 'python ablations/compare_ablations.py' once all experiments are done.{RESET}\n")
    sys.exit(1)
else:
    print(f"{GREEN}{BOLD}  All ablations complete!{RESET}")
    print(f"  python ablations/compare_ablations.py\n")
