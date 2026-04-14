"""
Single-sample inference script for SAE-FRAG.

Loads all trained models and generates a report for one test-split image.
Prints the ground-truth reference, the retrieved report, and the generated report.

Usage
-----
    python scripts/evaluate/infer.py              # index 0
    python scripts/evaluate/infer.py --idx 42
"""

# ── Standard library ──────────────────────────────────────────────────────
import argparse
import logging
import os
import pickle
import sys

# ── Third-party ───────────────────────────────────────────────────────────
import faiss
import torch
import torch.nn.functional as F

# ── Project ───────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from classification.report_labeler import ReportClassifier
from classification.sae_image_classifier import SAEImageClassifier
from configs.config import Config
from data.dataset import IUXrayMultiViewDataset
from models import CrossModalAlignment, MultiViewBackbone, ProjectionHead
from rag.hybrid_generator import HybridReportGenerator
from rag.verifier import ReportVerifier

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


_FPN_KEY_REMAP = {
    "fpn.output_c2": "fpn.output_p2",
    "fpn.output_c3": "fpn.output_p3",
    "fpn.output_c4": "fpn.output_p4",
    "fpn.output_c5": "fpn.output_p5",
}


def _remap_fpn_keys(state_dict: dict) -> dict:
    """Translate legacy output_c* keys to output_p* when needed."""
    new_sd = {}
    for k, v in state_dict.items():
        for old, new in _FPN_KEY_REMAP.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        new_sd[k] = v
    return new_sd


# =============================================================================
# Helpers
# =============================================================================

def _load_stage1(device: torch.device) -> tuple:
    """Return (checkpoint, path) for the best available Stage-1 checkpoint."""
    candidates = [
        os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth"),
        os.path.join(_ROOT, "checkpoints", "stage1", "best.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return torch.load(path, map_location=device, weights_only=False), path
    raise FileNotFoundError("No Stage-1 checkpoint found.")


def _load_generator(device: torch.device) -> HybridReportGenerator:
    candidates = [
        os.path.join(_ROOT, "checkpoints", "stage3", "best_generator.pth"),
        os.path.join(_ROOT, "checkpoints", "stage3", "last_generator.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            model = HybridReportGenerator().to(device)
            model.load_state_dict(
                torch.load(path, map_location=device, weights_only=False),
                strict=False,
            )
            model.eval()
            return model
    raise FileNotFoundError("No Stage-3 generator checkpoint found.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0,
                        help="Test-split sample index (0 to 770).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load models ───────────────────────────────────────────────────────
    checkpoint, ckpt_path = _load_stage1(device)
    log.info(f"Stage-1 : {os.path.relpath(ckpt_path, _ROOT)}")

    visual_encoder = MultiViewBackbone().to(device)
    visual_encoder.load_state_dict(_remap_fpn_keys(checkpoint["visual_model"]))
    visual_encoder.eval()

    alignment = CrossModalAlignment().to(device)
    alignment.load_state_dict(checkpoint["alignment"], strict=False)
    alignment.eval()

    proj_img = ProjectionHead().to(device)
    proj_img.load_state_dict(checkpoint["proj_img"])
    proj_img.eval()

    image_classifier = SAEImageClassifier().to(device)
    image_classifier.load_state_dict(
        torch.load(
            os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"),
            map_location=device, weights_only=False,
        ),
        strict=False,
    )
    image_classifier.eval()

    report_classifier = ReportClassifier().to(device)
    report_classifier.load_state_dict(
        torch.load(
            os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"),
            map_location=device, weights_only=False,
        )
    )
    report_classifier.eval()

    generator = _load_generator(device)
    verifier  = ReportVerifier(alignment=alignment, min_score=0.0)

    # ── FAISS index + metadata ────────────────────────────────────────────
    faiss_index = faiss.read_index(os.path.join(_ROOT, "store", "faiss_index.bin"))
    with open(os.path.join(_ROOT, "store", "train_reports.pkl"), "rb") as f:
        train_metadata = pickle.load(f)

    # ── Load test sample ──────────────────────────────────────────────────
    test_dataset = IUXrayMultiViewDataset(Config.DATA_ROOT, split="test")
    if not 0 <= args.idx < len(test_dataset):
        raise IndexError(f"Index {args.idx} out of range (0-{len(test_dataset) - 1}).")

    image, reference = test_dataset[args.idx]
    log.info(f"\n--- Test sample {args.idx} / {len(test_dataset) - 1} ---")
    log.info(f"\nREFERENCE:\n{reference}\n")

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        x = image.unsqueeze(0).to(device)

        visual_features = visual_encoder(x)                        # (1, 256, 14, 14)

        # FAISS retrieval
        query = F.normalize(
            proj_img(visual_features.flatten(2).mean(dim=2)), dim=1
        ).cpu().numpy().astype("float32")
        faiss.normalize_L2(query)
        _, I = faiss_index.search(query, k=5)
        candidates = [train_metadata[i]["report"] for i in I[0]]

        # Verifier — pick best candidate
        retrieved_report, verify_score = verifier.verify(visual_features, candidates)

        # Region-aligned features
        aligned_features, _, _ = alignment(visual_features, [retrieved_report])

        # Soft-AND entity verification
        img_entities      = torch.sigmoid(image_classifier(x))
        rep_entities      = torch.sigmoid(report_classifier([retrieved_report]))
        verified_entities = img_entities * rep_entities

        # Generate
        rag_context = HybridReportGenerator.build_rag_retrieved_text([retrieved_report])[0]
        prompt      = HybridReportGenerator.build_entity_prompt(verified_entities.cpu())
        generated   = generator(
            region_features=aligned_features,
            entity_vector=verified_entities,
            retrieved_texts=[rag_context],
            prompt_texts=prompt,
            target_texts=None,
        )[0]

    log.info(f"RETRIEVED  (score={verify_score:.4f}):\n{retrieved_report}\n")
    log.info(f"GENERATED:\n{generated}")


if __name__ == "__main__":
    main()
import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pickle
import torch
import faiss
import torch.nn.functional as F

from configs.config                      import Config
from data.dataset                        import IUXrayMultiViewDataset
from models import MultiViewBackbone, CrossModalAlignment, ProjectionHead
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler       import ReportClassifier
from rag.hybrid_generator                import HybridReportGenerator
from rag.verifier                        import ReportVerifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = Config.DATA_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="Test-split index (0-770)")
args = parser.parse_args()

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
_FACTUAL_CKPT = os.path.join(_ROOT, "checkpoints", "stage1", "factual_retriever.pth")
_STAGE1_CKPT  = os.path.join(_ROOT, "checkpoints", "stage1", "best.pth")
_use_factual  = os.path.exists(_FACTUAL_CKPT)
_ckpt_path    = _FACTUAL_CKPT if _use_factual else _STAGE1_CKPT
if _use_factual:
    print("[infer] Using FactMM-RAG factual retriever checkpoint.")
ckpt = torch.load(_ckpt_path, map_location=DEVICE, weights_only=False)

enc = MultiViewBackbone().to(DEVICE)
enc.load_state_dict(ckpt["visual_model"])
enc.eval()

aln = CrossModalAlignment().to(DEVICE)
aln.load_state_dict(ckpt["alignment"])
aln.eval()

prj = ProjectionHead().to(DEVICE)
prj.load_state_dict(ckpt["proj_img"])
prj.eval()

ic = SAEImageClassifier().to(DEVICE)
ic.load_state_dict(torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"), map_location=DEVICE, weights_only=False), strict=False)
ic.eval()

rc = ReportClassifier().to(DEVICE)
rc.load_state_dict(torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "report_classifier.pth"), map_location=DEVICE, weights_only=False))
rc.eval()

gen = HybridReportGenerator().to(DEVICE)
_model_path = os.path.join(_ROOT, "checkpoints", "stage3", "best_generator.pth")
if not os.path.exists(_model_path):
    for _fb in [os.path.join(_ROOT, "checkpoints", "stage3", "last_generator.pth")]:
        if os.path.exists(_fb):
            _model_path = _fb
            break
print(f"Using model: {_model_path}")
gen.load_state_dict(
    torch.load(_model_path, map_location=DEVICE, weights_only=False),
    strict=False,
)
gen.eval()

idx_db   = faiss.read_index(os.path.join(_ROOT, "store", "faiss_index.bin"))
meta     = pickle.load(open(os.path.join(_ROOT, "store", "train_reports.pkl"), "rb"))
verifier = ReportVerifier(aln, min_score=0.0)

print("Models loaded.\n")

# ── Load one test image ───────────────────────────────────────────────────────
ds      = IUXrayMultiViewDataset(ROOT, split="test")
img, ref = ds[args.idx]

print(f"--- Test sample {args.idx} / {len(ds)-1} ---")
print(f"\nREFERENCE REPORT:\n{ref}\n")

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    x    = img.unsqueeze(0).to(DEVICE)
    feat = enc(x)

    q    = F.normalize(prj(feat.flatten(2).mean(2)), dim=1).cpu().numpy().astype("float32")
    faiss.normalize_L2(q)            # match IndexFlatIP index
    _, I = idx_db.search(q, 5)
    cands = [meta[i]["report"] for i in I[0]]

    rep, score = verifier.verify(feat, cands)
    print(f"RETRIEVED REPORT (verify_score={score:.4f}):\n{rep}\n")

    # FactMM-RAG: wrap retrieved report in context template
    rag_rep = HybridReportGenerator.build_rag_retrieved_text([rep])[0]

    af, _, _ = aln(feat, [rep])
    ev = torch.sigmoid(ic(x)) * torch.sigmoid(rc([rep]))

    out = gen(
        region_features=af,
        entity_vector=ev,
        retrieved_texts=[rag_rep],
        prompt_texts=["Generate a radiology report from this image:"],
    )

print(f"GENERATED REPORT:\n{out[0]}")
