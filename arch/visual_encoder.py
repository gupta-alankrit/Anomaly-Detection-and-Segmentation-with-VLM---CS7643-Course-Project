import open_clip
import torch
from torch import nn
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image
from collections.abc import Sequence

# TODO: create function for preprocessing batched images in a smart way

class visual_encoder_clip(nn.Module):
    def __init__(
            self,
            model_id="hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
            device="cpu"
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_id)
        self.model = self.model.to(device)
        self.model.visual.output_tokens = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        # images: [B, T, C, H, W], already preprocessed
        if images.ndim != 5:
            raise ValueError("Expected images with shape [B, T, C, H, W]")
        batch_size, sequence_length, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * sequence_length, channels, height, width)

        _, tokens = self.model.visual(flat_images)
        return tokens.reshape(batch_size, -1, tokens.shape[-1]) # [B, T*patches, width]