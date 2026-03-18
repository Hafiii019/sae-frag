import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class HybridReportGenerator(nn.Module):

    def __init__(self, num_entities=14, model_name="google/flan-t5-base"):
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        d_model = self.t5.config.d_model

        # Visual projection: LayerNorm → Linear → Dropout for stable training
        self.visual_norm = nn.LayerNorm(256)
        self.visual_proj = nn.Linear(256, d_model)
        self.visual_drop = nn.Dropout(0.1)

        # Entity embedding: learned embedding per pathology class.
        # Used as a soft-weighted sum (NOT boolean selection) so that soft-AND
        # probabilities are always meaningful — even small values contribute.
        self.entity_embed = nn.Embedding(num_entities, d_model)

    def forward(
        self,
        region_features,     # (B, 49, 256)
        entity_vector,       # (B, 14)
        retrieved_texts,     # list[str]
        prompt_texts,        # list[str]
        target_texts=None
    ):

        device = region_features.device
        B = region_features.size(0)

        # ======================================================
        # 1️⃣ Visual Tokens  (LayerNorm → Linear → Dropout)
        # ======================================================
        region_normed = self.visual_norm(region_features)           # (B, 49, 256)
        visual_tokens = self.visual_drop(
            self.visual_proj(region_normed)
        )                                                           # (B, 49, d_model)
        visual_mask = torch.ones(B, visual_tokens.size(1), device=device)

        # ======================================================
        # 2️⃣ Entity Token  (soft-weighted sum → 1 token per sample)
        # ------------------------------------------------------
        # entity_vector is a [0,1] probability vector (B, 14) from soft-AND
        # of image-classifier and report-classifier sigmoid outputs.
        # We compute a probability-weighted combination of the 14 learned
        # entity embeddings, producing ONE compact token per sample.
        # This is always well-defined (no boolean thresholding) and fully
        # differentiable, so gradients flow back through entity weights.
        # ======================================================
        entity_ids = torch.arange(
            entity_vector.size(1), device=device
        )                                                           # (14,)
        all_entity_embs = self.entity_embed(entity_ids)             # (14, d_model)

        # entity_vector: (B, 14), all_entity_embs: (14, d_model)
        # → weighted sum: (B, d_model) → unsqueeze → (B, 1, d_model)
        entity_token = torch.matmul(
            entity_vector.float(), all_entity_embs
        ).unsqueeze(1)                                              # (B, 1, d_model)
        entity_mask = torch.ones(B, 1, device=device)

        # ======================================================
        # 3️⃣ Retrieved Text Tokens
        # ======================================================
        retrieved_enc = self.tokenizer(
            retrieved_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        retrieved_embeds = self.t5.encoder.embed_tokens(
            retrieved_enc.input_ids
        )

        retrieved_mask = retrieved_enc.attention_mask

        # ======================================================
        # 4️⃣ Prompt Tokens
        # ======================================================
        prompt_enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        prompt_embeds = self.t5.encoder.embed_tokens(
            prompt_enc.input_ids
        )

        prompt_mask = prompt_enc.attention_mask

        # ======================================================
        # 5️⃣ Concatenate Everything
        # Order: visual regions | entity context | retrieved report | instruction
        # ======================================================
        encoder_inputs = torch.cat(
            [visual_tokens, entity_token, retrieved_embeds, prompt_embeds],
            dim=1
        )

        attention_mask = torch.cat(
            [visual_mask, entity_mask, retrieved_mask, prompt_mask],
            dim=1
        )

        # ======================================================
        # 6️⃣ Training Mode
        # ======================================================
        if target_texts is not None:

            target_enc = self.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Replace padding token ids with -100 so they are ignored by CE loss
            labels = target_enc.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.t5(
                inputs_embeds=encoder_inputs,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Label smoothing: reduces overconfident predictions, improves BLEU/ROUGE
            loss = outputs.loss
            if self.training:
                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                smooth_loss = -log_probs.mean(dim=-1)
                mask = (labels != -100).float()
                smooth_loss = (smooth_loss * mask).sum() / mask.sum()
                loss = 0.9 * loss + 0.1 * smooth_loss

            return loss

        # ======================================================
        # 7️⃣ Inference Mode
        # ======================================================
        else:

            generated_ids = self.t5.generate(
                inputs_embeds=encoder_inputs,
                attention_mask=attention_mask,
                max_new_tokens=300,
                num_beams=8,
                length_penalty=1.5,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                min_length=40,
                early_stopping=True,
            )

            return self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )