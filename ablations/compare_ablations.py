"""Print a comparison table of all ablation results vs. the full model baseline.

Reads results/metrics.json (full model) and results/ablations/*/metrics.json
(each ablation) and prints a formatted table with delta columns.

Usage
-----
  python ablations/compare_ablations.py
  python ablations/compare_ablations.py --metrics BLEU-4 ROUGE-L entity_f1
  python ablations/compare_ablations.py --csv  > ablations.csv
"""

import argparse
import json
import os
import sys

# Project root is one level above this file (ablations/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Metrics shown in the table (in display order)
DEFAULT_METRICS = [
    "BLEU-1", "BLEU-4",
    "ROUGE-L", "METEOR",
    "entity_f1",
    "chexbert_f1_micro",
    "fact_verify_f1",
    "CIDEr",
]

METRIC_LABELS = {
    "BLEU-1":           "BLEU-1",
    "BLEU-4":           "BLEU-4",
    "ROUGE-L":          "ROUGE-L",
    "METEOR":           "METEOR",
    "entity_f1":        "Entity F1",
    "chexbert_f1_micro":"CheXBert F1",
    "fact_verify_f1":   "Fact Verify F1",
    "CIDEr":            "CIDEr",
}

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"


def load_metrics(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delta_str(val: float | None, base: float | None, use_color: bool = True) -> str:
    if val is None or base is None:
        return "  n/a "
    d = val - base
    s = f"{d:+.4f}"
    if not use_color:
        return s
    if d > 0.001:
        return f"{GREEN}{s}{RESET}"
    if d < -0.001:
        return f"{RED}{s}{RESET}"
    return f"{YELLOW}{s}{RESET}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                        metavar="KEY",
                        help="Metric keys to display (default: %(default)s)")
    parser.add_argument("--csv", action="store_true",
                        help="Output CSV instead of the pretty table")
    parser.add_argument("--no_color", action="store_true",
                        help="Disable ANSI color codes")
    args = parser.parse_args()

    metrics_to_show: list[str] = args.metrics
    use_color = not args.csv and not args.no_color

    # -- Load results ----------------------------------------------------------
    ablations_dir = os.path.join(ROOT, "results", "ablations")

    # Prefer full_model from ablations/ (canonical); fall back to results/metrics.json
    baseline_candidates = [
        os.path.join(ablations_dir, "full_model", "metrics.json"),
        os.path.join(ROOT, "results", "metrics.json"),
    ]
    baseline = None
    for bp in baseline_candidates:
        baseline = load_metrics(bp)
        if baseline is not None:
            break

    if baseline is None:
        print("ERROR: no baseline found.")
        print("Run one of:")
        print("  python ablations/run_ablation.py --only full_model")
        print("  python run_pipeline.py")
        sys.exit(1)

    abl_names: list[str] = []
    if os.path.isdir(ablations_dir):
        abl_names = sorted(
            d for d in os.listdir(ablations_dir)
            if d != "full_model"
            and os.path.isfile(os.path.join(ablations_dir, d, "metrics.json"))
        )

    if not abl_names:
        print("No ablation results found in results/ablations/")
        print("Run:  python ablations/run_ablation.py --tier 2")
        sys.exit(0)

    all_rows: list[tuple[str, dict]] = [("full_model", baseline)]
    for name in abl_names:
        m = load_metrics(os.path.join(ablations_dir, name, "metrics.json"))
        if m is not None:
            all_rows.append((name, m))

    # -- CSV output ------------------------------------------------------------
    if args.csv:
        header = ["experiment"] + metrics_to_show
        print(",".join(header))
        for exp_name, m in all_rows:
            row = [exp_name]
            for k in metrics_to_show:
                v = m.get(k)
                row.append("" if v is None else str(v))
            print(",".join(row))
        return

    # -- Pretty table ----------------------------------------------------------
    col_w   = 10
    name_w  = 18
    metric_labels = [METRIC_LABELS.get(k, k) for k in metrics_to_show]

    # Header
    sep = "-" * (name_w + 2 + (col_w + 1) * len(metrics_to_show) * 2)
    if use_color:
        print(f"\n{CYAN}{BOLD}{sep}{RESET}")
        print(f"{CYAN}{BOLD}  SAE-FRAG  -  Ablation Comparison{RESET}")
        print(f"{CYAN}{BOLD}{sep}{RESET}")
    else:
        print(f"\n{sep}")
        print(f"  SAE-FRAG  -  Ablation Comparison")
        print(f"{sep}")

    # Column headers
    print(f"\n  {'Experiment':<{name_w}}", end="")
    print()

    # Metric label row
    print(f"  {'':>{name_w}}", end="")
    for label in metric_labels:
        print(f" {label:>{col_w}} {'':>{col_w}}", end="")
    print()
    print(f"  {'-' * name_w}", end="")
    for _ in metric_labels:
        print(f" {'-' * col_w} {'-' * col_w}", end="")
    print()

    # Data rows
    for exp_name, m in all_rows:
        is_baseline = (exp_name == "full_model")
        exp_label   = f"{BOLD}{exp_name}{RESET}" if use_color and is_baseline else exp_name
        print(f"  {exp_label:<{name_w + (len(BOLD) + len(RESET) if use_color and is_baseline else 0)}}", end="")
        for k in metrics_to_show:
            val  = m.get(k)
            base = baseline.get(k)
            val_str = f"{val:.4f}" if val is not None else "  n/a"
            dlt_str = ("  base " if is_baseline
                       else delta_str(val, base, use_color))
            print(f" {val_str:>{col_w}} {dlt_str:>{col_w}}", end="")
        print()

    print(f"\n  {sep}")
    print(f"  delta = value - full_model baseline")
    print(f"  Results: results/ablations/<exp_name>/metrics.json\n")

    # -- Best-per-metric summary -----------------------------------------------
    print(f"  Best per metric:")
    for k in metrics_to_show:
        vals = [(name, m.get(k)) for name, m in all_rows if m.get(k) is not None]
        if not vals:
            continue
        best_name, best_val = max(vals, key=lambda x: x[1])
        tag = " (baseline)" if best_name == "full_model" else ""
        label = METRIC_LABELS.get(k, k)
        print(f"    {label:<18}  {best_val:.4f}  [{best_name}{tag}]")
    print()


if __name__ == "__main__":
    main()
