from collections.abc import Sequence
import torch
from torch import nn
import open_clip
from open_clip.transformer import VisionTransformer
from PIL import Image
from arch.vlm import SmolVLMBackbone, VisualResidualProjector
from arch.composite_features import CompositeFeatures
from arch.text_adapter import ClipPromptEncoder, QueryConditionedTextAdapter


class ad_model(nn.Module):
    def __init__(
            self,
            device: str = "cpu",
            composite_scales: tuple[int, ...] = (224, 112, 64),
            composite_weights: tuple[float, ...] | None = None,
            ):
        super().__init__()
        self.device = device
        self.composite_scales = composite_scales
        self.composite_weights = composite_weights if composite_weights is not None else (1.0,) * len(composite_scales)

        # initialize clip encoder
        CLIP_MODEL_ID = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_ID)
        self.clip_model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_ID)

        self.vis_dim = self.clip_model.visual.ln_post.bias.shape[0]

        # initialize other modules
        self.vlm_backbone = SmolVLMBackbone()

        self.vlm_hidden_size = self.vlm_backbone.hidden_size

        self.composite_features = CompositeFeatures(model=self.clip_model, preprocess=self.preprocess, scales=composite_scales, weights=self.composite_weights, device=device)

        self.clip_prompt_encoder = ClipPromptEncoder(clip_model=self.clip_model, tokenizer=self.tokenizer, device=device)

        self.visual_residual_projector = VisualResidualProjector(vis_dim=self.vis_dim, llm_dim=self.vlm_hidden_size, device=device)
        
        self.query_conditioned_text_adapter = QueryConditionedTextAdapter(llm_dim=self.vlm_hidden_size, device=device)

    def _build_prefix(
        self,
        query_img: Sequence[Image.Image],
        reference_imgs: Sequence[Sequence[Image.Image]],
        category: Sequence[str],
    ):
        """Shared between training and inference: builds [vis_llm | adapted_text_llm]."""
        query_features, _, absolute_residual, _ = self.composite_features(query_img, reference_imgs)
        prompt_encodings = self.clip_prompt_encoder(category)
        adapted_text, adapted_text_llm = self.query_conditioned_text_adapter(
            query_features, absolute_residual, prompt_encodings
        )
        vis_llm = self.visual_residual_projector(query_features, absolute_residual)

        # Concatenate continuous prefix: [B, N_patches + 2, llm_dim]
        prefix_embeds = torch.cat([vis_llm, adapted_text_llm], dim=1)
        return prefix_embeds, adapted_text, query_features

    def forward(
        self,
        query_img: Sequence[Image.Image],
        reference_imgs: Sequence[Sequence[Image.Image]],
        category: Sequence[str],
        input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        prefix_embeds, adapted_text, query_features = self._build_prefix(
            query_img, reference_imgs, category
        )
        B, prefix_len, _ = prefix_embeds.shape
        device = prefix_embeds.device

        # Embed text tokens and concatenate: [B, N + 2 + L, llm_dim]
        text_embeds = self.vlm_backbone.embed_tokens(input_ids.to(device))
        inputs_embeds = torch.cat([prefix_embeds.to(text_embeds.dtype), text_embeds], dim=1)

        # Prepend 1s to the attention mask for the continuous prefix
        prefix_mask = torch.ones((B, prefix_len), dtype=text_attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_mask, text_attention_mask.to(device)], dim=1)

        # Prepend -100 to labels so cross-entropy ignores the continuous prefix
        full_labels = None
        if labels is not None:
            ignore_prefix = torch.full((B, prefix_len), -100, dtype=labels.dtype, device=device)
            full_labels = torch.cat([ignore_prefix, labels.to(device)], dim=1)

        vlm_out = self.vlm_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )
        return vlm_out, adapted_text, query_features

    @torch.no_grad()
    def generate(
        self,
        query_img: Sequence[Image.Image],
        reference_imgs: Sequence[Sequence[Image.Image]],
        category: Sequence[str],
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
    ):
        prefix_embeds, _, _ = self._build_prefix(query_img, reference_imgs, category)
        B, prefix_len, _ = prefix_embeds.shape
        device = prefix_embeds.device

        prompt_embeds = self.vlm_backbone.embed_tokens(prompt_input_ids.to(device))
        inputs_embeds = torch.cat([prefix_embeds.to(prompt_embeds.dtype), prompt_embeds], dim=1)

        prefix_mask = torch.ones((B, prefix_len), dtype=prompt_attention_mask.dtype, device=device)
        full_mask = torch.cat([prefix_mask, prompt_attention_mask.to(device)], dim=1)

        texts, seg_hidden_states = self.vlm_backbone.generate_from_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
        )
        return texts, seg_hidden_states