"""One-time model download: Bio_ClinicalBERT safetensors edition.

transformers 5.x blocks torch.load of old .bin model files as a security
measure (CVE-2025-32434).  Bio_ClinicalBERT on HuggingFace only ships .bin;
the safetensors weights live in an open PR (#16).  This script downloads that
PR revision to models/bio_clinical_bert/ so all pipeline scripts can load
from disk without any torch.load safety issues.

Run once before the pipeline:
    python scripts/prepare/convert_models.py

Output: models/bio_clinical_bert/  (contains model.safetensors + config etc.)
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_DIR = os.path.join(_ROOT, "models", "bio_clinical_bert")

SENTINEL = os.path.join(LOCAL_DIR, "model.safetensors")


def main():
    if os.path.exists(SENTINEL):
        print(f"Bio_ClinicalBERT safetensors already present at:")
        print(f"  {os.path.relpath(LOCAL_DIR, _ROOT)}")
        print("Nothing to do.")
        return

    print("Downloading Bio_ClinicalBERT (safetensors) from HuggingFace PR #16...")
    print(f"Destination: {os.path.relpath(LOCAL_DIR, _ROOT)}")
    print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    # PR #16 of Bio_ClinicalBERT adds model.safetensors.
    # Using the PR revision avoids torch.load entirely.
    path = snapshot_download(
        repo_id="emilyalsentzer/Bio_ClinicalBERT",
        revision="refs/pr/16",
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.bin", "*.bin.index.json", "flax_model.*", "tf_model.*"],
    )

    if os.path.exists(SENTINEL):
        print(f"\nDone. Saved to: {path}")
        print("The pipeline will now use the local safetensors copy.")
    else:
        print(f"\nERROR: model.safetensors not found after download.")
        print("Fallback: convert from main branch .bin weights.")
        _convert_from_bin()


def _convert_from_bin():
    """Load .bin from main branch, re-save as safetensors locally."""
    from transformers import AutoModel, AutoTokenizer

    print("Loading Bio_ClinicalBERT (main branch, .bin) for conversion...")
    # This may still hit the torch.load check on torch < 2.6.
    # If it does, upgrade torch: pip install torch>=2.6
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    os.makedirs(LOCAL_DIR, exist_ok=True)
    model.save_pretrained(LOCAL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(LOCAL_DIR)
    print(f"Converted and saved to: {os.path.relpath(LOCAL_DIR, _ROOT)}")


if __name__ == "__main__":
    main()
