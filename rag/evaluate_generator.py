import torch
import torch.nn.functional as F
import faiss
import pickle
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

from data.dataset import IUXrayMultiViewDataset
from models.multiview_backbone import MultiViewBackbone
from models.projection import ProjectionHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "C:/Datasets/IU_Xray"

# ==========================
# Load TEST dataset
# ==========================
test_dataset = IUXrayMultiViewDataset(ROOT, split="test")
loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ==========================
# Load FAISS + metadata
# ==========================
index = faiss.read_index("rag/faiss_index.bin")

with open("rag/train_reports.pkl", "rb") as f:
    train_samples = pickle.load(f)

# ==========================
# Load frozen visual encoder
# ==========================
visual_model = MultiViewBackbone().to(device)
proj_img = ProjectionHead().to(device)
visual_model.eval()
proj_img.eval()

# ==========================
# Load trained T5
# ==========================
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
t5.load_state_dict(torch.load("rag/t5_generator.pth", map_location=device))
t5.eval()

smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

bleu1 = []
bleu2 = []
bleu3 = []
bleu4 = []
meteor_scores = []
rougeL_scores = []

print("Evaluating...")

with torch.no_grad():

    for images, reports in tqdm(loader):

        images = images.to(device)

        visual_features = visual_model(images)
        img_global = visual_features.flatten(2).mean(dim=2)
        img_emb = proj_img(img_global)
        img_emb = F.normalize(img_emb, dim=1)
        img_emb_np = img_emb.cpu().numpy().astype("float32")

        _, I = index.search(img_emb_np, k=3)

        retrieved_texts = []
        for idx in I[0]:
            retrieved_texts.append(train_samples[int(idx)]["report"])

        prompt = "Instruction: Generate a detailed radiology report.\n\nRetrieved context:\n"
        for i, rt in enumerate(retrieved_texts):
            prompt += f"Report {i+1}: {rt}\n"
        prompt += "\nGenerate report:"

        enc = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        output_ids = t5.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_length=256,
            num_beams=4
        )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reference = reports[0]

        ref_tokens = reference.split()
        gen_tokens = generated.split()

        bleu1.append(sentence_bleu([ref_tokens], gen_tokens, weights=(1,0,0,0), smoothing_function=smooth))
        bleu2.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.5,0.5,0,0), smoothing_function=smooth))
        bleu3.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=smooth))
        bleu4.append(sentence_bleu([ref_tokens], gen_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))

        meteor_scores.append(meteor_score([reference.split()],
        generated.split()))
        rougeL_scores.append(rouge.score(reference, generated)['rougeL'].fmeasure)

print("\n==== FINAL RESULTS ====")
print("BLEU-1:", sum(bleu1)/len(bleu1))
print("BLEU-2:", sum(bleu2)/len(bleu2))
print("BLEU-3:", sum(bleu3)/len(bleu3))
print("BLEU-4:", sum(bleu4)/len(bleu4))
print("METEOR:", sum(meteor_scores)/len(meteor_scores))
print("ROUGE-L:", sum(rougeL_scores)/len(rougeL_scores))