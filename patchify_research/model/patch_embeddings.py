"""Extract ViT patch embeddings from an image using LAION CLIP ViT-B/16."""
import argparse
import math
from collections.abc import Callable

import open_clip
import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_pil_image

MODEL_NAME = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K"


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device).eval()
    model.visual.output_tokens = True  # also return per-patch tokens
    return model, preprocess

@torch.no_grad()
def extract_patch_embeddings(image: Image.Image, model, preprocess):
    """Return (patch_embeddings [num_patches, dim], cls_embedding [dim], grid (h, w)).

    Patch tokens are the final-layer outputs after ln_post (width 768), before the
    CLIP projection. The CLS embedding is the projected 512-d image embedding.
    """
    device = next(model.parameters()).device
    x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)  # resize + center crop to 224
    pooled, tokens = model.visual(x)
    side = x.shape[-1] // model.visual.patch_size[0]
    # print("tokens of shape", tokens.shape)
    return tokens[0].cpu(), pooled[0].cpu(), (side, side)

@torch.no_grad()
def match_embeddings(source: Image.Image, query: Image.Image, model, preprocess):
    """Return the best-match indices of source that match each patch of query."""

    device = next(model.parameters()).device
    source = preprocess(source.convert("RGB")).unsqueeze(0).to(device)  # resize + center crop to 224
    query = preprocess(query.convert("RGB")).unsqueeze(0).to(device)  # resize + center crop to 224

    source_pooled, source_tokens = model.visual(source)
    query_pooled, query_tokens = model.visual(query)

    # trying cosine similarity using matrix multiplication
    s_norm = torch.nn.functional.normalize(source_tokens, dim=2)
    q_norm = torch.nn.functional.normalize(query_tokens, dim=2)
    return torch.argmax(s_norm @ q_norm.transpose(1,2), dim=1)


def _preprocessed_image(source_tensor, preprocess) -> Image.Image:
    normalizer = preprocess.transforms[-1]
    if not isinstance(normalizer, Normalize):
        raise ValueError("Expected preprocessing to end with a Normalize transform")

    mean = source_tensor.new_tensor(normalizer.mean)[:, None, None]
    std = source_tensor.new_tensor(normalizer.std)[:, None, None]
    return to_pil_image((source_tensor * std + mean).clamp(0, 1))


def _compose_patches(source_img, patch_size, positions) -> Image.Image:
    patch_height, patch_width = patch_size
    columns = source_img.width // patch_width
    rows = source_img.height // patch_height
    if source_img.width % patch_width or source_img.height % patch_height or len(positions) != rows * columns:
        raise ValueError("Patch positions do not match the preprocessed image dimensions")

    result = Image.new("RGB", source_img.size)
    for query_index, (source_x, source_y) in enumerate(positions):
        patch = source_img.crop((source_x, source_y, source_x + patch_width, source_y + patch_height))
        query_x = (query_index % columns) * patch_width
        query_y = (query_index // columns) * patch_height
        result.paste(patch, (query_x, query_y))
    return result


@torch.no_grad()
def create_patch_img(source: Image.Image, query: Image.Image, model, preprocess) -> Image.Image:
    """Arrange source patches in the positions of their best-matching query patches."""
    indices = match_embeddings(source, query, model, preprocess)[0].tolist()
    source_img = _preprocessed_image(preprocess(source.convert("RGB")), preprocess)
    patch_height, patch_width = model.visual.patch_size
    columns = source_img.width // patch_width
    positions = [((index % columns) * patch_width, (index // columns) * patch_height) for index in indices]
    return _compose_patches(source_img, model.visual.patch_size, positions)


def _offset_divisions(k: int, patch_size: tuple[int, int]) -> int:
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if k >= min(patch_size).bit_length():
        raise ValueError("2**k must divide both patch dimensions for whole-pixel offsets")
    divisions = 2 ** k
    if any(size % divisions for size in patch_size):
        raise ValueError("2**k must divide both patch dimensions for whole-pixel offsets")
    return divisions


def _match_offset_positions(
    source_batch: torch.Tensor,
    query_features: torch.Tensor,
    patch_size: tuple[int, int],
    divisions: int,
    encode: Callable[[torch.Tensor], torch.Tensor],
) -> list[tuple[int, int]]:
    patch_height, patch_width = patch_size
    height, width = source_batch.shape[-2:]
    rows, columns = height // patch_height, width // patch_width
    device = source_batch.device
    best_scores = torch.full((rows * columns,), -torch.inf, device=device)
    best_x = torch.zeros(rows * columns, dtype=torch.long, device=device)
    best_y = torch.zeros_like(best_x)

    for offset_y in range(0, patch_height, patch_height // divisions):
        for offset_x in range(0, patch_width, patch_width // divisions):
            shifted = F.pad(source_batch, (0, offset_x, 0, offset_y), mode="replicate")
            shifted = shifted[:, :, offset_y:offset_y + height, offset_x:offset_x + width]
            source_features = encode(shifted)
            if source_features.shape[1] != rows * columns:
                raise ValueError("Source patch count does not match the preprocessed image dimensions")
            scores = source_features[0] @ query_features.T
            valid_rows = rows - (offset_y > 0)
            valid_columns = columns - (offset_x > 0)
            valid = torch.zeros((rows, columns), dtype=torch.bool, device=device)
            valid[:valid_rows, :valid_columns] = True
            scores[~valid.flatten()] = -torch.inf
            scores, indices = scores.max(dim=0)
            improved = scores > best_scores
            best_scores = torch.where(improved, scores, best_scores)
            best_x = torch.where(improved, (indices % columns) * patch_width + offset_x, best_x)
            best_y = torch.where(improved, (indices // columns) * patch_height + offset_y, best_y)
    return list(zip(best_x.tolist(), best_y.tolist()))


@torch.no_grad()
def create_offset_patch_img(source: Image.Image, query: Image.Image, model, preprocess, k: int) -> Image.Image:
    """Match query patches against 4**k source grids with sub-patch pixel offsets.

    Offsets range from zero to one patch, exclusive, in steps of patch_size / 2**k.
    Patches extending outside the preprocessed source image are excluded.
    """
    patch_height, patch_width = model.visual.patch_size
    divisions = _offset_divisions(k, model.visual.patch_size)

    source_tensor = preprocess(source.convert("RGB"))
    query_tensor = preprocess(query.convert("RGB"))
    height, width = source_tensor.shape[-2:]
    if query_tensor.shape[-2:] != (height, width) or height % patch_height or width % patch_width:
        raise ValueError("Source and query dimensions must match and be divisible by the patch size")
    rows, columns = height // patch_height, width // patch_width
    device = next(model.parameters()).device
    source_batch = source_tensor.unsqueeze(0).to(device)
    _, query_tokens = model.visual(query_tensor.unsqueeze(0).to(device))
    if query_tokens.shape[1] != rows * columns:
        raise ValueError("Query patch count does not match the preprocessed image dimensions")
    query_norm = torch.nn.functional.normalize(query_tokens[0], dim=1)

    def encode(images: torch.Tensor) -> torch.Tensor:
        _, tokens = model.visual(images)
        return F.normalize(tokens, dim=2)

    positions = _match_offset_positions(source_batch, query_norm, model.visual.patch_size, divisions, encode)
    return _compose_patches(_preprocessed_image(source_tensor, preprocess), model.visual.patch_size, positions)


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


@torch.no_grad()
def create_multiscale_patch_img(
    source: Image.Image,
    query: Image.Image,
    model,
    preprocess,
    scales: tuple[int, ...] = (224, 112, 64),
    weights: tuple[float, ...] | None = None,
    k: int = 0,
) -> Image.Image:
    """Match query patches using concatenated, spatially aligned multiscale CLIP tokens.

    Scale weights specify the relative contributions to cosine similarity. Patch
    positions and output pixels always come from the original preprocessed source.
    k selects 4**k source grids with whole-pixel sub-patch offsets. Incomplete
    edge patches are excluded; padding supplies context for the encoder.
    """
    visual = model.visual
    if not isinstance(visual, VisionTransformer) or not visual.output_tokens:
        raise ValueError("Multiscale matching requires an OpenCLIP VisionTransformer with output_tokens enabled")
    source_tensor = preprocess(source.convert("RGB"))
    query_tensor = preprocess(query.convert("RGB"))
    height, width = source_tensor.shape[-2:]
    patch_height, patch_width = visual.patch_size
    divisions = _offset_divisions(k, visual.patch_size)
    if (query_tensor.shape != source_tensor.shape or (height, width) != tuple(visual.image_size)
            or height % patch_height or width % patch_width):
        raise ValueError("Preprocessed source and query must match the model image size and patch grid")
    if not scales or any(
        not isinstance(scale, int) or scale <= 0 or scale > min(height, width)
        or scale % patch_height or scale % patch_width for scale in scales
    ):
        raise ValueError("Scales must be positive, no larger than the image, and divisible by both patch dimensions")
    if weights is None:
        weights = (1.0,) * len(scales)
    if len(weights) != len(scales) or any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("Provide one finite non-negative weight per scale, with at least one positive weight")

    device = next(model.parameters()).device
    if k == 0:
        images = torch.stack((source_tensor, query_tensor)).to(device)
        features = _multiscale_embeddings(visual, images, scales, weights)
        indices = (features[0] @ features[1].T).argmax(dim=0).tolist()
        columns = width // patch_width
        positions = [((index % columns) * patch_width, (index // columns) * patch_height) for index in indices]
    else:
        source_batch = source_tensor.unsqueeze(0).to(device)
        query_features = _multiscale_embeddings(visual, query_tensor.unsqueeze(0).to(device), scales, weights)[0]

        def encode(images: torch.Tensor) -> torch.Tensor:
            return _multiscale_embeddings(visual, images, scales, weights)

        positions = _match_offset_positions(source_batch, query_features, visual.patch_size, divisions, encode)
    return _compose_patches(_preprocessed_image(source_tensor, preprocess), visual.patch_size, positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to source image supplying patches")
    parser.add_argument("query", help="Path to query image defining patch positions")
    parser.add_argument("--output", help="Optional path to save the composite image")
    parser.add_argument("--multiscale", action="store_true", help="Match using multiscale image embeddings")
    parser.add_argument(
        "--offset-level", type=int, default=0, metavar="K",
        help="Use 4**K source grids with sub-patch offsets, also with --multiscale (default: 0; max: 4)",
    )
    parser.add_argument("--scales", type=int, nargs="+", default=None, metavar="PIXELS",
                        help="Multiscale image sizes (default: 224 112 64)")
    parser.add_argument("--scale-weights", type=float, nargs="+", default=None, metavar="WEIGHT",
                        help="Relative cosine similarity weights, one per scale (default: equal)")
    args = parser.parse_args()
    if not args.multiscale and (args.scales is not None or args.scale_weights is not None):
        parser.error("--scales and --scale-weights require --multiscale")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device", device)
    model, preprocess = load_model(device)
    with Image.open(args.source) as source, Image.open(args.query) as query:
        if args.multiscale:
            scales = tuple(args.scales) if args.scales is not None else (224, 112, 64)
            weights = tuple(args.scale_weights) if args.scale_weights is not None else None
            image = create_multiscale_patch_img(source, query, model, preprocess, scales, weights, k=args.offset_level)
        elif args.offset_level == 0:
            image = create_patch_img(source, query, model, preprocess)
        else:
            image = create_offset_patch_img(source, query, model, preprocess, args.offset_level)
    if args.output:
        image.save(args.output)
        print(f"saved to {args.output}")
    else:
        print(f"created composite image: {image.width}x{image.height}")


if __name__ == "__main__":
    main()
