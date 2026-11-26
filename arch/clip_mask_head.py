import torch
import torch.nn.functional as F
from torch import nn


class ClipMaskHead(nn.Module):
    """Score CLIP patches using adapted class prompts and the VLM's <SEG> prediction."""

    def __init__(self, llm_dim: int, clip_dim: int, scale_weights: tuple[float, ...]):
        super().__init__()
        weights = torch.tensor(scale_weights, dtype=torch.float32)
        self.register_buffer("scale_weights", weights / weights.sum(), persistent=False)
        self.seg_norm = nn.LayerNorm(llm_dim)
        self.seg_projection = nn.Linear(llm_dim, clip_dim + 1)
        nn.init.normal_(self.seg_projection.weight, std=1e-3)
        nn.init.zeros_(self.seg_projection.bias)
        self.reference_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        clip_patches: torch.Tensor,
        adapted_text: torch.Tensor,
        reference_distance: torch.Tensor,
        seg_hidden: torch.Tensor,
        grid_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Convert [B, patches, scales, clip_dim] features into [B, 1, H, W] mask logits."""
        # The VLM adjusts the anomaly-vs-normal text direction and the image-wide bias.
        conditioning = self.seg_projection(self.seg_norm(seg_hidden.to(self.seg_norm.weight))).float()
        text_direction = adapted_text[:, 1].float() - adapted_text[:, 0].float()
        direction = text_direction + conditioning[:, :-1]

        # Combining normalized patch vectors by scale is equivalent to averaging their scores.
        patches = (clip_patches * self.scale_weights[None, None, :, None]).sum(dim=2)
        logits = 10.0 * (patches * direction[:, None, :]).sum(dim=-1)
        logits = logits + F.softplus(self.reference_weight) * reference_distance
        logits = logits + conditioning[:, -1:] - 2.0

        grid = logits.reshape(logits.shape[0], 1, *grid_size)
        return F.interpolate(grid, size=target_size, mode="bilinear", align_corners=False)
