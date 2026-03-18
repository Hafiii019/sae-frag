import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pickle
import torch
import faiss
import torch.nn.functional as F

from data.dataset                        import IUXrayMultiViewDataset
from models.multiview_backbone           import MultiViewBackbone
from models.alignment                    import CrossModalAlignment
from models.projection                   import ProjectionHead
from classification.sae_image_classifier import SAEImageClassifier
from classification.report_labeler       import ReportClassifier
from rag.hybrid_generator                import HybridReportGenerator
from rag.verifier                        import ReportVerifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = os.environ.get("IU_XRAY_ROOT", "C:/Datasets/IU_Xray")

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="Test-split index (0-770)")
args = parser.parse_args()

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
ckpt = torch.load(os.path.join(_ROOT, "checkpoints", "stage1", "best.pth"), map_location=DEVICE, weights_only=False)

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
ic.load_state_dict(torch.load(os.path.join(_ROOT, "checkpoints", "stage2", "image_classifier.pth"), map_location=DEVICE, weights_only=False))
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
    _, I = idx_db.search(q, 5)
    cands = [meta[i]["report"] for i in I[0]]

    rep, score = verifier.verify(feat, cands)
    print(f"RETRIEVED REPORT (verify_score={score:.4f}):\n{rep}\n")

    af, _, _ = aln(feat, [rep])
    ev = torch.sigmoid(ic(x)) * torch.sigmoid(rc([rep]))

    out = gen(
        region_features=af,
        entity_vector=ev,
        retrieved_texts=[rep],
        prompt_texts=["Generate a detailed radiology report for the chest X-ray."],
    )

print(f"GENERATED REPORT:\n{out[0]}")
