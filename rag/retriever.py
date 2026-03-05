import torch
import faiss

from models.multiview_backbone import MultiViewBackbone
from models.alignment import CrossModalAlignment
from models.projection import ProjectionHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("checkpoints/best_stage1.pth", map_location=device)

visual_model = MultiViewBackbone().to(device)
alignment = CrossModalAlignment().to(device)
proj_img = ProjectionHead().to(device)

visual_model.load_state_dict(checkpoint["visual_model"])
alignment.load_state_dict(checkpoint["alignment"])
proj_img.load_state_dict(checkpoint["proj_img"])

visual_model.eval()
alignment.eval()
proj_img.eval()

index = faiss.read_index("rag_db/faiss_index.bin")
report_db = torch.load("rag_db/report_texts.pt")["reports"]


def retrieve(images, top_k=3):

    with torch.no_grad():

        images = images.to(device)

        visual_features = visual_model(images)
        aligned, text_cls, _ = alignment(visual_features, [""] * images.size(0))

        img_global = aligned.mean(dim=1)
        query_emb = proj_img(img_global).cpu().numpy()

        faiss.normalize_L2(query_emb)

        D, I = index.search(query_emb, top_k)

        retrieved = [report_db[i] for i in I[0]]

    return retrieved