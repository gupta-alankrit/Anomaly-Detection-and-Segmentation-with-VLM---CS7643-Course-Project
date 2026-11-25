import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel, SamProcessor


class PromptGenerator(nn.Module):
    """Maps VLM <SEG> hidden state [B, llm_dim] -> SAM sparse prompt [B, 1, 256]."""
    def __init__(self, llm_dim: int, sam_prompt_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, sam_prompt_dim),
            nn.GELU(),
            nn.Linear(sam_prompt_dim, sam_prompt_dim),
        )

    def forward(self, h_seg: torch.Tensor) -> torch.Tensor:
        return self.net(h_seg).unsqueeze(1)  # [B, 1, 256]

class DenseResidualPrompt(nn.Module):
    """Maps patch residuals [B, N, vis_dim] -> SAM dense prompt [B, 256, 64, 64]."""
    def __init__(self, vis_dim: int = 768, sam_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(vis_dim, sam_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(sam_dim, sam_dim, kernel_size=3, padding=1),
        )
        # Zero-init final conv so step 0 starts identically to SAM's default no_mask_embed
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, delta_x: torch.Tensor, grid_hw: tuple[int, int] = (64, 64)) -> torch.Tensor:
        B, N, C = delta_x.shape
        g = int(N ** 0.5)
        spatial = delta_x.transpose(1, 2).reshape(B, C, g, g)  # [B, 768, 14, 14]
        up = F.interpolate(spatial, size=grid_hw, mode="bilinear", align_corners=False)
        return self.net(up)  # [B, 256, 64, 64]

class SamMaskBranch(nn.Module):
    def __init__(self, llm_dim: int, model_id: str = "facebook/sam-vit-base", vis_dim: int = 768, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.vis_dim = vis_dim
        self.processor = SamProcessor.from_pretrained(model_id)
        self.sam = SamModel.from_pretrained(model_id).to(device)

        # Freeze image encoder & prompt encoder; keep mask_decoder trainable
        self.sam.vision_encoder.requires_grad_(False).eval()
        self.sam.prompt_encoder.requires_grad_(False).eval()
        self.sam.mask_decoder.requires_grad_(True)

        sam_dim = self.sam.config.vision_config.output_channels  # 256
        self.prompt_generator = PromptGenerator(llm_dim=llm_dim, sam_prompt_dim=sam_dim).to(device)
        # self.dense_prompt_gen = DenseResidualPrompt(vis_dim=vis_dim, sam_dim=sam_dim).to(device)

    @torch.no_grad()
    def encode_images(self, query_imgs) -> torch.Tensor:
        """Runs frozen SAM image encoder -> [B, 256, 64, 64]."""
        inputs = self.processor(images=query_imgs, return_tensors="pt").to(self.device)
        return self.sam.get_image_embeddings(inputs["pixel_values"])

    def forward(self, query_imgs, delta_x, h_seg: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        h_seg: [B, llm_dim] extracted from the VLM's <SEG> token position
        returns: mask logits of shape [B, 1, H, W]
        """
        image_embeds = self.encode_images(query_imgs)                        # [B, 256, 64, 64]
        image_pe = self.sam.get_image_wide_positional_embeddings()           # [1, 256, 64, 64]

        sparse_prompts = self.prompt_generator(h_seg)                        # [B, 1, 256]
        # SAM expects [B, num_prompts, num_tokens, 256]
        sparse_prompts = sparse_prompts.unsqueeze(1)

        # Standard SAM "no-mask" dense prompt embedding
        B = h_seg.shape[0]
        dense_prompts = (
            self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
            .expand(B, -1, image_embeds.shape[-2], image_embeds.shape[-1])
            # + self.dense_prompt_gen(delta_x, grid_hw=image_embeds.shape[-2:])
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeds,
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_prompts,
            dense_prompt_embeddings=dense_prompts,
            multimask_output=False,
        )
        # low_res_masks: [B, 1, 1, 256, 256] -> squeeze & upsample to target_size
        mask_logits = low_res_masks.squeeze(1)
        return F.interpolate(mask_logits, size=target_size, mode="bilinear", align_corners=False)