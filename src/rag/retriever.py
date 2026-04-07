"""Fact-aware multimodal retriever (FactMM-RAG integration).

At inference the retriever encodes the query X-ray image and searches a
pre-built FAISS index for the most factually-similar training report.

Models are lazy-loaded on first ``retrieve()`` call so that importing this
module during training (before checkpoints exist) never crashes.

If checkpoints/stage1/factual_retriever.pth exists the retriever uses the
FactMM-RAG-trained query encoder.  Otherwise it falls back to the Stage-1
image encoder.

Usage
-----
    from rag.retriever import retrieve
    reports = retrieve(images, top_k=3)   # images: (B, 2, 3, 224, 224)
"""

# ── Standard library ──────────────────────────────────────────────────────
import logging
import os

# ── Third-party ───────────────────────────────────────────────────────────
import faiss
import torch
import torch.nn.functional as F

# ── Project ───────────────────────────────────────────────────────────────
from models import MultiViewBackbone, ProjectionHead

log = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FACTUAL_CKPT = os.path.join(ROOT, "checkpoints", "stage1", "factual_retriever.pth")
STAGE1_CKPT  = os.path.join(ROOT, "checkpoints", "stage1", "best.pth")
FAISS_PATH   = os.path.join(ROOT, "store", "faiss_index.bin")
REPORTS_PATH = os.path.join(ROOT, "store", "train_reports.pkl")

# ── Lazy state ────────────────────────────────────────────────────────────
_device      = None
_visual_model = None
_proj_img    = None
_index       = None
_report_db   = None


def _load():
    """Lazy-initialise all retriever components on first call."""
    global _device, _visual_model, _proj_img, _index, _report_db

    if _visual_model is not None:
        return   # already loaded

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose checkpoint: factual retriever > stage-1 > skip
    if os.path.exists(FACTUAL_CKPT):
        ckpt_path = FACTUAL_CKPT
        log.info("Retriever: using FactMM-RAG factual retriever checkpoint.")
    elif os.path.exists(STAGE1_CKPT):
        ckpt_path = STAGE1_CKPT
        log.info("Retriever: using Stage-1 image-only checkpoint (fallback).")
    else:
        raise FileNotFoundError(
            f"No retriever checkpoint found.\n"
            f"  Expected: {FACTUAL_CKPT}\n"
            f"  Fallback: {STAGE1_CKPT}\n"
            f"  Run train_stage1.py first."
        )

    checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=False)

    _visual_model = MultiViewBackbone().to(_device)
    _visual_model.load_state_dict(checkpoint["visual_model"], strict=False)
    _visual_model.eval()

    _proj_img = ProjectionHead().to(_device)
    _proj_img.load_state_dict(checkpoint["proj_img"])
    _proj_img.eval()

    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}\n"
            "Run: python scripts/prepare/build_index.py"
        )
    if not os.path.exists(REPORTS_PATH):
        raise FileNotFoundError(
            f"Report DB not found: {REPORTS_PATH}\n"
            "Run: python scripts/prepare/build_index.py"
        )

    _index     = faiss.read_index(FAISS_PATH)
    _report_db = torch.load(REPORTS_PATH, weights_only=False)


def retrieve(images: torch.Tensor, top_k: int = 3) -> list:
    """Retrieve the top-k factually-similar training reports for a batch of images.

    Parameters
    ----------
    images:
        Raw pixel tensors of shape ``(B, 2, 3, 224, 224)`` — the standard
        multi-view dataset output.  Visual encoding is done internally.
    top_k:
        Number of candidates to retrieve per sample.

    Returns
    -------
    list[str]
        Length B.  Rank-1 retrieved report text for each sample.
    """
    _load()

    with torch.no_grad():
        images = images.to(_device)

        # Visual encoding: (B,2,3,224,224) → (B,256,14,14) → pool → (B,256)
        visual_features = _visual_model(images)                    # (B, 256, 14, 14)
        global_feat     = visual_features.flatten(2).mean(dim=2)   # (B, 256)
        query_emb       = _proj_img(global_feat)                   # (B, 256) L2-normed
        query_np        = query_emb.cpu().float().numpy()
        faiss.normalize_L2(query_np)

        _D, I = _index.search(query_np, top_k)                    # (B, top_k)

        return [_report_db[I[b][0]]["report"] for b in range(images.size(0))]

