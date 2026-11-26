from collections.abc import Sequence
import torch
from torch import nn
import open_clip
from PIL import Image, ImageOps
from arch.clip_mask_head import ClipMaskHead
from arch.vlm import SmolVLMBackbone, VisualResidualProjector
from arch.composite_features import CompositeFeatures
from arch.text_adapter import ClipPromptEncoder, QueryConditionedTextAdapter
import torchvision.transforms as T

def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    size = max(w, h)
    # Pad (left, top, right, bottom) -> right and bottom only
    padding = (0, 0, size - w, size - h)
    return ImageOps.expand(img, border=padding, fill=fill)

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
        self.clip_model.to(device=device, dtype=torch.bfloat16).eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_ID)

        norm = next(t for t in self.preprocess.transforms if isinstance(t, T.Normalize))

        self.preprocess = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            norm,
        ])

        visual = self.clip_model.visual
        self.vis_dim = visual.ln_post.bias.shape[0]
        clip_dim = visual.proj.shape[-1]

        # initialize other modules
        self.vlm_backbone = SmolVLMBackbone()

        self.vlm_hidden_size = self.vlm_backbone.hidden_size

        self.composite_features = CompositeFeatures(model=self.clip_model, preprocess=self.preprocess, scales=composite_scales, weights=self.composite_weights, device=device)

        self.clip_prompt_encoder = ClipPromptEncoder(clip_model=self.clip_model, tokenizer=self.tokenizer, device=device)

        self.visual_residual_projector = VisualResidualProjector(vis_dim=self.vis_dim, llm_dim=self.vlm_hidden_size, device=device)
        
        self.query_conditioned_text_adapter = QueryConditionedTextAdapter(
            vis_dim=self.vis_dim, text_dim=clip_dim, llm_dim=self.vlm_hidden_size, device=device
        )
        self.mask_head = ClipMaskHead(self.vlm_hidden_size, clip_dim, self.composite_weights).to(device)

    def _build_prefix(
            self,
            query_img: Sequence[Image.Image],
            reference_imgs: Sequence[Sequence[Image.Image]],
            category: Sequence[str],
            ):
        """Shared between training and inference: builds [vis_llm | adapted_text_llm]."""
        padded_query = [pad_to_square(img) for img in query_img]
        padded_refs = [[pad_to_square(img) for img in ref] for ref in reference_imgs]

        features = self.composite_features(padded_query, padded_refs)
        query_tokens = features.query_tokens.float()
        residual = features.residual.float()
        prompt_encodings = self.clip_prompt_encoder(category)
        adapted_text, adapted_text_llm = self.query_conditioned_text_adapter(
            query_tokens, residual, prompt_encodings
        )
        vis_llm = self.visual_residual_projector(query_tokens, residual)

        prefix_embeds = torch.cat([vis_llm, adapted_text_llm], dim=1)
        return prefix_embeds, adapted_text, features

    def _extract_seg_hidden(
            self,
            last_hidden_state: torch.Tensor,
            target_ids: torch.Tensor,
            prefix_len: int,
            ) -> torch.Tensor:
        """
        Extracts h_seg [B, llm_dim] at the position predicting <SEG>,
        matching generate_from_embeds in vlm.py.
        """
        matches = target_ids.to(last_hidden_state.device) == self.vlm_backbone.seg_token_id
        has_seg = matches.any(dim=1)
        if not has_seg.all():
            missing = (~has_seg).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"Samples {missing} are missing a target {self.vlm_backbone.seg_token} token.")
        prediction_positions = prefix_len + matches.long().argmax(dim=1) - 1
        rows = torch.arange(target_ids.shape[0], device=last_hidden_state.device)
        return last_hidden_state[rows, prediction_positions]

    def forward(
            self,
            query_img: Sequence[Image.Image],
            reference_imgs: Sequence[Sequence[Image.Image]],
            category: Sequence[str],
            input_ids: torch.Tensor,
            text_attention_mask: torch.Tensor,
            labels: torch.Tensor | None = None,
            mask_target_size: tuple[int, int] = (256, 256),
            ):
        prefix_embeds, adapted_text, features = self._build_prefix(
            query_img, reference_imgs, category
        )
        B, prefix_len, _ = prefix_embeds.shape
        device = prefix_embeds.device

        text_embeds = self.vlm_backbone.embed_tokens(input_ids.to(device))
        inputs_embeds = torch.cat([prefix_embeds.to(text_embeds.dtype), text_embeds], dim=1)

        prefix_mask = torch.ones((B, prefix_len), dtype=text_attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([prefix_mask, text_attention_mask.to(device)], dim=1)

        full_labels = None
        if labels is not None:
            ignore_prefix = torch.full((B, prefix_len), -100, dtype=labels.dtype, device=device)
            full_labels = torch.cat([ignore_prefix, labels.to(device)], dim=1)

        vlm_out = self.vlm_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

        target_ids = labels if labels is not None else input_ids
        target_ids = target_ids.to(device).masked_fill(text_attention_mask.to(device) == 0, -100)
        h_seg = self._extract_seg_hidden(vlm_out["last_hidden_state"], target_ids, prefix_len)
        pred_masks = self.mask_head(
            features.clip_patches, adapted_text, features.reference_distance, h_seg,
            grid_size=self.clip_model.visual.grid_size, target_size=mask_target_size,
        )

        return {
            "vlm_out": vlm_out,
            "pred_masks": pred_masks,
            "adapted_text": adapted_text,
            "query_features": features.query_tokens,
        }

    @torch.no_grad()
    def generate(
        self,
        query_img: Sequence[Image.Image],
        reference_imgs: Sequence[Sequence[Image.Image]],
        category: Sequence[str],
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        mask_target_size: tuple[int, int] = (1024, 1024),
    ):
        """Return decoded text, captured <SEG> states, and logits on the padded image canvas."""
        prefix_embeds, adapted_text, features = self._build_prefix(query_img, reference_imgs, category)
        B, prefix_len, _ = prefix_embeds.shape
        device = prefix_embeds.device

        prompt_embeds = self.vlm_backbone.embed_tokens(prompt_input_ids.to(device))
        inputs_embeds = torch.cat([prefix_embeds.to(prompt_embeds.dtype), prompt_embeds], dim=1)

        prefix_mask = torch.ones((B, prefix_len), dtype=prompt_attention_mask.dtype, device=device)
        full_mask = torch.cat([prefix_mask, prompt_attention_mask.to(device)], dim=1)

        texts, seg_hidden_states, has_seg = self.vlm_backbone.generate_from_embeds(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=max_new_tokens,
        )

        if seg_hidden_states is None:
            return texts, None, None
        mask_logits = self.mask_head(
            features.clip_patches, adapted_text, features.reference_distance, seg_hidden_states,
            grid_size=self.clip_model.visual.grid_size, target_size=mask_target_size,
        )
        mask_logits = mask_logits.masked_fill(~has_seg.to(mask_logits.device)[:, None, None, None], -100.0)
        return texts, seg_hidden_states, mask_logits
