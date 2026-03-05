import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class HybridReportGenerator(nn.Module):

    def __init__(self, num_entities=14, model_name="google/flan-t5-base"):
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        d_model = self.t5.config.d_model

        # Visual projection (256 → 768)
        self.visual_proj = nn.Linear(256, d_model)

        # Entity embedding
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
        # 1️⃣ Visual Tokens
        # ======================================================
        visual_tokens = self.visual_proj(region_features)  # (B,49,768)
        visual_mask = torch.ones(B, visual_tokens.size(1), device=device)

        # ======================================================
        # 2️⃣ Dynamic Entity Tokens (ONLY active ones)
        # ======================================================
        entity_ids = torch.arange(entity_vector.size(1), device=device)
        entity_ids = entity_ids.unsqueeze(0).expand(B, -1)
        entity_embeddings = self.entity_embed(entity_ids)

        entity_tokens_list = []
        entity_mask_list = []

        for b in range(B):
            active = entity_vector[b].bool()
            active_embeds = entity_embeddings[b][active]

            if active_embeds.size(0) == 0:
                active_embeds = torch.zeros(1, entity_embeddings.size(-1), device=device)

            entity_tokens_list.append(active_embeds)
            entity_mask_list.append(torch.ones(active_embeds.size(0), device=device))

        entity_tokens = nn.utils.rnn.pad_sequence(
            entity_tokens_list,
            batch_first=True
        )

        entity_mask = nn.utils.rnn.pad_sequence(
            entity_mask_list,
            batch_first=True
        )

        # ======================================================
        # 3️⃣ Retrieved Text Tokens
        # ======================================================
        retrieved_enc = self.tokenizer(
            retrieved_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
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
        # ======================================================
        encoder_inputs = torch.cat(
            [visual_tokens, entity_tokens, retrieved_embeds, prompt_embeds],
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
                max_length=128
            ).to(device)

            outputs = self.t5(
                inputs_embeds=encoder_inputs,
                attention_mask=attention_mask,
                labels=target_enc.input_ids
            )

            return outputs.loss

        # ======================================================
        # 7️⃣ Inference Mode
        # ======================================================
        else:

            generated_ids = self.t5.generate(
                inputs_embeds=encoder_inputs,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=4,
                length_penalty=1.0
            )

            return self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )