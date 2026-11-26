import os
import torch
from torch import nn
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

class SmolVLMBackbone(nn.Module):
    def __init__(
            self,
            model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            seg_token: str = "<SEG>",
            compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_id = model_id
        self.seg_token = seg_token
        self.compute_dtype = compute_dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.add_tokens([seg_token], special_tokens=True)
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids(seg_token)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        base_vlm = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

        base_vlm.resize_token_embeddings(len(self.tokenizer))
        base_vlm = prepare_model_for_kbit_training(base_vlm)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.vlm = get_peft_model(base_vlm, lora_config)

        self.vlm.get_input_embeddings().weight.requires_grad = True
        self.vlm.get_output_embeddings().weight.requires_grad = True

        self.hidden_size = self.vlm.config.text_config.hidden_size

    @property
    def _unwrapped_vlm(self):
        """Resolves PEFT wrapping cleanly to access underlying text_model and lm_head."""
        return self.vlm.base_model.model if isinstance(self.vlm, PeftModel) else self.vlm

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Converts token IDs into LLM input embeddings."""
        return self.vlm.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Runs a forward pass directly through the causal text backbone using pre-fused
        multimodal embeddings [visual_embeds | adapted_text_prompts | conversation_embeds].
        """
        core = self._unwrapped_vlm

        outputs = core.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = core.lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }

    @torch.no_grad()
    def generate_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 64,
    ) -> tuple[list[str], torch.Tensor | None, torch.Tensor]:
        """Decode text and capture the state predicting the first <SEG> in each row."""
        self.eval()
        core = self._unwrapped_vlm
        cur_embeds = inputs_embeds
        batch_size, prompt_length, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        cur_mask = attention_mask if attention_mask is not None else torch.ones(
            (batch_size, prompt_length), dtype=torch.long, device=device
        )
        columns = torch.arange(prompt_length, device=device)
        last_positions = columns.masked_fill(cur_mask == 0, 0).amax(dim=1)
        rows = torch.arange(batch_size, device=device)
        eos_token_id = self.tokenizer.eos_token_id
        generated_ids = []
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        has_seg = torch.zeros_like(active)
        seg_hidden_states = inputs_embeds.new_zeros((batch_size, hidden_size))

        for _ in range(max_new_tokens):
            out = core.model.text_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                position_ids=(cur_mask.long().cumsum(dim=1) - 1).clamp_min(0),
                use_cache=False,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state[rows, last_positions]
            next_token = core.lm_head(last_hidden).argmax(dim=-1)
            next_token = next_token.masked_fill(~active, eos_token_id)
            generated_ids.append(next_token)

            is_seg = active & ~has_seg & (next_token == self.seg_token_id)
            seg_hidden_states = torch.where(is_seg[:, None], last_hidden, seg_hidden_states)
            has_seg = has_seg | is_seg

            cur_mask = torch.cat([cur_mask, active[:, None].to(cur_mask.dtype)], dim=1)
            active = active & (next_token != eos_token_id)
            if not active.any():
                break

            cur_embeds = torch.cat([cur_embeds, self.embed_tokens(next_token[:, None])], dim=1)
            last_positions = torch.full_like(last_positions, cur_embeds.shape[1] - 1)

        if not generated_ids:
            return [""] * batch_size, None, has_seg
        sequences = torch.stack(generated_ids, dim=1).tolist()
        sequences = [ids[:ids.index(eos_token_id) + 1] if eos_token_id in ids else ids for ids in sequences]
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=False)
        return texts, seg_hidden_states if has_seg.any() else None, has_seg

    def save_adapters(self, save_dir: str):
        """Saves only LoRA weights, tokenizer, and resized embedding layers."""
        os.makedirs(save_dir, exist_ok=True)
        self.vlm.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(
            {
                "embed_tokens": self.vlm.get_input_embeddings().state_dict(),
                "lm_head": self.vlm.get_output_embeddings().state_dict(),
            },
            os.path.join(save_dir, "seg_embeddings.pt"),
        )

    def load_adapters(self, save_dir: str):
        """Loads trained LoRA adapters and <SEG> token embeddings."""
        self.vlm = PeftModel.from_pretrained(self.vlm.base_model.model, save_dir)
        emb_ckpt = torch.load(os.path.join(save_dir, "seg_embeddings.pt"), map_location="cpu")
        self.vlm.get_input_embeddings().load_state_dict(emb_ckpt["embed_tokens"])
        self.vlm.get_output_embeddings().load_state_dict(emb_ckpt["lm_head"])

class VisualResidualProjector(nn.Module):
    def __init__(self, vis_dim: int, llm_dim: int, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.LayerNorm(vis_dim * 2),
            nn.Linear(vis_dim * 2, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        ).to(device)

    def forward(self, x_q: torch.Tensor, delta_x: torch.Tensor) -> torch.Tensor:
        # x_q, delta_x: [B, N_patches, vis_dim]
        fused = torch.cat([x_q, delta_x], dim=-1)  # [B, N_patches, 2 * vis_dim]
        return self.net(fused)                     # [B, N_patches, llm_dim]
