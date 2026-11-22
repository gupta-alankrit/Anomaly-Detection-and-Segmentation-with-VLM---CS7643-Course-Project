"""Frozen, batched multiscale patch matching with OpenCLIP."""
import math
from collections.abc import Sequence

import open_clip
import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
import torchvision
from PIL import Image


def _scale_patch_tokens(visual: VisionTransformer, images: torch.Tensor) -> torch.Tensor:
    """Encode a resized image using interpolated positional embeddings."""
    height, width = images.shape[-2:]
    patch_height, patch_width = visual.patch_size
    grid_height, grid_width = height // patch_height, width // patch_width
    if (height, width) == tuple(visual.image_size):
        _, tokens = visual(images)
        return tokens

    original_height, original_width = visual.grid_size
    positions = visual.positional_embedding
    if positions.shape[0] != 1 + original_height * original_width:
        raise ValueError("Expected one class position and a spatial positional embedding")
    spatial_positions = positions[1:].reshape(original_height, original_width, -1).permute(2, 0, 1).unsqueeze(0)
    spatial_positions = F.interpolate(
        spatial_positions.float(), size=(grid_height, grid_width), mode="bicubic", align_corners=False
    ).to(positions.dtype)
    spatial_positions = spatial_positions.flatten(2).transpose(1, 2)
    scaled_positions = torch.cat((positions[:1].unsqueeze(0), spatial_positions), dim=1)

    tokens = visual.conv1(images).flatten(2).transpose(1, 2)
    class_token = visual.class_embedding.to(tokens.dtype).reshape(1, 1, -1).expand(tokens.shape[0], -1, -1)
    tokens = torch.cat((class_token, tokens), dim=1)
    tokens = visual.ln_pre(visual.patch_dropout(tokens + scaled_positions.to(tokens.dtype)))
    tokens = visual.transformer(tokens)
    _, tokens = visual._pool(tokens)
    if tokens.shape[1] != grid_height * grid_width:
        raise ValueError("Visual encoder did not return one token per image patch")
    return tokens


def _multiscale_embeddings(
    visual: VisionTransformer, images: torch.Tensor, scales: tuple[int, ...], weights: tuple[float, ...]
) -> torch.Tensor:
    patch_height, patch_width = visual.patch_size
    base_height, base_width = images.shape[-2:]
    base_grid = (base_height // patch_height, base_width // patch_width)
    features = []
    for scale, weight in zip(scales, weights):
        scaled = images if (scale, scale) == (base_height, base_width) else F.interpolate(
            images, size=(scale, scale), mode="bilinear", align_corners=False, antialias=True
        )
        tokens = _scale_patch_tokens(visual, scaled)
        grid_height, grid_width = scale // patch_height, scale // patch_width
        if tokens.shape[1] != grid_height * grid_width:
            raise ValueError("Visual encoder did not return one token per image patch")
        spatial = tokens.transpose(1, 2).reshape(images.shape[0], -1, grid_height, grid_width)
        if (grid_height, grid_width) != base_grid:
            spatial = F.interpolate(spatial.float(), size=base_grid, mode="bilinear", align_corners=False)
        vectors = spatial.flatten(2).transpose(1, 2)
        features.append(F.normalize(vectors.float(), dim=-1) * math.sqrt(weight))
    return F.normalize(torch.cat(features, dim=-1), dim=-1)

class CompositeFeatures(torch.nn.Module):
    """Extract query and best-matching reference patch features with frozen OpenCLIP.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: torchvision.transforms.transforms.Compose,
        device: str = "cpu",
        scales: Sequence[int] = (224, 112, 64),
        weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.model = model
        if not isinstance(self.model.visual, VisionTransformer):
            raise ValueError("Multiscale matching requires an OpenCLIP VisionTransformer")
        self.model.visual.output_tokens = True
        self.model.requires_grad_(False)
        self.model.to(device).eval()
        self.preprocess = preprocess

        visual = self.model.visual
        self.scales = tuple(scales)
        self.weights = tuple(weights) if weights is not None else (1.0,) * len(self.scales)
        if not self.scales or any(
            not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0
            or scale > min(visual.image_size)
            or any(scale % patch for patch in visual.patch_size)
            for scale in self.scales
        ):
            raise ValueError("Scales must be positive, no larger than the image, and divisible by both patch dimensions")
        if (len(self.weights) != len(self.scales)
                or any(not math.isfinite(weight) or weight < 0 for weight in self.weights)
                or not any(self.weights)):
            raise ValueError("Provide one finite non-negative weight per scale, with at least one positive weight")

    # def train(self, mode: bool = True):
    #     super().train(mode)
    #     self.model.eval()
    #     return self

    @torch.no_grad()
    def forward(
        self,
        query_imgs: Sequence[Image.Image],
        ref_imgs: Sequence[Sequence[Image.Image]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match each query against its own references.

        Inputs are batches of PIL images and per-query reference lists, respectively.
        Returns xq, xc, dx of shape [B, N, C] and best_sim of shape [B, N].
        Queries without references have zero xc, dx, and best_sim.
        """
        batch_size = len(query_imgs)
        if not batch_size or len(ref_imgs) != batch_size:
            raise ValueError("Provide a nonempty query batch and one reference list per query")

        counts = [len(refs) for refs in ref_imgs]
        images = [*query_imgs, *(image for refs in ref_imgs for image in refs)]
        device = next(self.model.parameters()).device
        imgs = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(device)
        visual = self.model.visual
        _, tokens = visual(imgs)
        xq = tokens[:batch_size]
        if not sum(counts):
            return xq, torch.zeros_like(xq), torch.zeros_like(xq), xq.new_zeros(xq.shape[:2])

        match = _multiscale_embeddings(visual, imgs, self.scales, self.weights)
        max_refs = max(counts)
        num_patches, channels = xq.shape[1:]
        ref_match = match.new_zeros((batch_size, max_refs, num_patches, match.shape[-1]))
        ref_tokens = tokens.new_zeros((batch_size, max_refs, num_patches, channels))
        valid = torch.zeros((batch_size, max_refs), dtype=torch.bool, device=device)
        offset = batch_size
        for index, count in enumerate(counts):
            ref_match[index, :count] = match[offset:offset + count]
            ref_tokens[index, :count] = tokens[offset:offset + count]
            valid[index, :count] = True
            offset += count

        sim = torch.einsum("bnd,bkmd->bnkm", match[:batch_size], ref_match)
        sim = sim.masked_fill(~valid[:, None, :, None], -torch.inf).flatten(2)
        best_sim, best = sim.max(dim=-1)
        xc = ref_tokens.flatten(1, 2).gather(1, best.unsqueeze(-1).expand(-1, -1, channels))
        has_refs = valid.any(dim=1)
        best_sim = torch.where(has_refs[:, None], best_sim, torch.zeros_like(best_sim))
        dx = torch.where(has_refs[:, None, None], (xq - xc).abs(), torch.zeros_like(xq))
        return xq, xc, dx, best_sim