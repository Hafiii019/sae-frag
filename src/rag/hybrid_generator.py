# ── Standard library ──────────────────────────────────────────────────────
import logging

# ── Third-party ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

log = logging.getLogger(__name__)


class HybridReportGenerator(nn.Module):

    # Number of entity conditioning tokens fed into the encoder.
    # 4 tokens give the T5 decoder 4× more entity context slots vs 1.
    N_ENTITY_TOKENS = 4

    CLASS_NAMES = [
        "No Finding", "Cardiomegaly", "Pleural Effusion", "Pneumonia",
        "Pneumothorax", "Atelectasis", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Nodule", "Mass", "Hernia", "Infiltrate",
    ]

    def __init__(self, num_entities=14, model_name="google/flan-t5-large"):
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        d_model = self.t5.config.d_model

        # Visual projection: LayerNorm → Linear → Dropout for stable training
        self.visual_norm = nn.LayerNorm(256)
        self.visual_proj = nn.Linear(256, d_model)
        self.visual_drop = nn.Dropout(0.1)

        # Entity projection: 2-layer MLP that maps the 14-class soft-AND vector
        # into N_ENTITY_TOKENS dense tokens.  An MLP learns richer non-linear
        # combinations of entity probabilities than a weighted embedding table.
        self.n_entity_tokens = self.N_ENTITY_TOKENS
        self.entity_proj = nn.Sequential(
            nn.Linear(num_entities, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.n_entity_tokens * d_model),
        )

    @staticmethod
    def build_entity_prompt(
        entity_vector: "torch.Tensor",
        threshold: float = 0.25,
    ) -> "list[str]":
        """Build entity-informed prompts from soft entity probability vectors.

        Follows the FactMM-RAG (NAACL 2025) RAG prompt template:
            "Here is a report of a related patient: '<retrieved_report>'
             Generate a radiology report from this image: <image>"

        The entity-aware prefix is prepended to give the T5 decoder
        additional diagnostic grounding beyond the retrieved report text.

        Parameters
        ----------
        entity_vector : Tensor of shape (B, 14) — soft-AND entity probabilities.
        threshold     : Minimum probability to include an entity in the prompt.

        Returns
        -------
        list[str] of length B, one per sample.
        """
        names = HybridReportGenerator.CLASS_NAMES
        prompts = []
        for ev in entity_vector:
            # Skip index 0 ("No Finding") — it's the absence class
            present = [
                names[i] for i, p in enumerate(ev.tolist())
                if i > 0 and p >= threshold
            ]
            if present:
                findings = ", ".join(present)
                prompt = (
                    f"Detected findings: {findings}. "
                    "Generate a radiology report from this image:"
                )
            else:
                prompt = "Generate a radiology report from this image:"
            prompts.append(prompt)
        return prompts

    @staticmethod
    def build_rag_retrieved_text(retrieved_texts: "list[str]") -> "list[str]":
        """Wrap retrieved reports in the FactMM-RAG prompt template.

        From Appendix A.2 of the paper:
            "Here is a report of a related patient: '<document>'"

        This is prepended to the image instruction so the T5 encoder sees
        the retrieved context before the generation directive.

        Parameters
        ----------
        retrieved_texts : list[str] of length B — rank-1 retrieved reports.

        Returns
        -------
        list[str] of length B — wrapped report strings fed to the encoder.
        """
        return [
            f"Here is a report of a related patient: \"{t}\""
            for t in retrieved_texts
        ]

    def forward(
        self,
        region_features,     # (B, 196, 256)  — 14×14 P4 tokens
        entity_vector,       # (B, 14)
        retrieved_texts,     # list[str]
        prompt_texts,        # list[str]
        target_texts=None
    ):

        device = region_features.device
        B = region_features.size(0)

        # ── 1. Visual tokens: LayerNorm → Linear → Dropout ───────────────────
        region_normed = self.visual_norm(region_features)           # (B, 196, 256)
        visual_tokens = self.visual_drop(
            self.visual_proj(region_normed)
        )                                                           # (B, 196, d_model)
        visual_mask = torch.ones(B, visual_tokens.size(1), device=device)

        # ── 2. Entity tokens: 2-layer MLP → N_ENTITY_TOKENS dense tokens ─────
        entity_tokens = self.entity_proj(entity_vector.float()).view(
            B, self.n_entity_tokens, -1
        )                                                           # (B, 4, d_model)
        entity_mask = torch.ones(B, self.n_entity_tokens, device=device)

        # ── 3. Retrieved text tokens ──────────────────────────────────────────
        retrieved_enc = self.tokenizer(
            retrieved_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
        ).to(device)
        retrieved_embeds = self.t5.encoder.embed_tokens(retrieved_enc.input_ids)
        retrieved_mask   = retrieved_enc.attention_mask

        # ── 4. Prompt tokens ──────────────────────────────────────────────────
        prompt_enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        prompt_embeds = self.t5.encoder.embed_tokens(prompt_enc.input_ids)
        prompt_mask   = prompt_enc.attention_mask

        # ── 5. Concatenate: visual | entity | retrieved | prompt ──────────────
        encoder_inputs = torch.cat(
            [visual_tokens, entity_tokens, retrieved_embeds, prompt_embeds], dim=1
        )
        attention_mask = torch.cat(
            [visual_mask, entity_mask, retrieved_mask, prompt_mask], dim=1
        )

        # ── 6. Training ───────────────────────────────────────────────────────
        if target_texts is not None:
            target_enc = self.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            labels = target_enc.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.t5(
                inputs_embeds=encoder_inputs,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            if self.training:
                log_probs  = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                smooth_loss = -log_probs.mean(dim=-1)
                mask        = (labels != -100).float()
                smooth_loss = (smooth_loss * mask).sum() / mask.sum()
                loss        = 0.9 * loss + 0.1 * smooth_loss

            return loss

        # ── 7. Inference ──────────────────────────────────────────────────────
        generated_ids = self.t5.generate(
            inputs_embeds=encoder_inputs,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_beams=6,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            min_length=20,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)