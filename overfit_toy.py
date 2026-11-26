"""Diagnose mask learning on one anomalous image with a pixel annotation."""

import argparse
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arch.clip_mask_head import ClipMaskHead
from arch.model import ad_model, pad_to_square
from evaluate import load_checkpoint, save_vis
from train import MASK_BCE_WEIGHT, MASK_DICE_WEIGHT, build_training_tokens, mask_loss_components


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(path: Path, image_size: tuple[int, int], mask_size: int = 256) -> torch.Tensor:
    with Image.open(path) as image:
        if image.size != image_size:
            raise ValueError(f"Mask size {image.size} must match query image size {image_size}.")
        mask = pad_to_square(image.convert("L")).resize((mask_size, mask_size), Image.Resampling.NEAREST)
        foreground = np.asarray(mask) > 0
    if not foreground.any() or foreground.all():
        raise ValueError("The mask must contain both foreground and background after preprocessing.")
    return torch.from_numpy(foreground)[None, None]


@torch.no_grad()
def report_mask(name: str, logits: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    probs = logits.float().sigmoid()
    predicted = probs >= 0.5
    iou = (predicted & target).sum() / (predicted | target).sum()
    print(
        f"{name} | foreground p={probs[target].mean().item():.4g} | "
        f"background p={probs[~target].mean().item():.4g} | max p={probs.max().item():.4g} | "
        f"IoU@0.5={iou.item():.3f}",
        flush=True,
    )
    return probs[0, 0].cpu().numpy()


def fit_mask_head(
    initial_head: ClipMaskHead,
    mask_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    grid_size: tuple[int, int],
    target: torch.Tensor,
    steps: int,
    lr: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    head = deepcopy(initial_head).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    targets = target.float()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(*mask_inputs, grid_size=grid_size, target_size=target.shape[-2:])
        bce, dice = mask_loss_components(logits, targets, pos_weight=pos_weight)
        loss = MASK_BCE_WEIGHT * bce + MASK_DICE_WEIGHT * dice
        if not torch.isfinite(loss):
            raise RuntimeError("Mask loss became non-finite.")
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return head(*mask_inputs, grid_size=grid_size, target_size=target.shape[-2:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True, help="Anomalous image")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask for the query, with nonzero defect pixels")
    parser.add_argument("--reference", type=Path, action="append", default=[], help="Normal image. May be repeated.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--prompt", default="Is there any anomaly in the image?")
    parser.add_argument("--target", help="Expected response, including <SEG>")
    parser.add_argument("--checkpoint", type=Path, help="Start from and diagnose a trained checkpoint")
    parser.add_argument("--steps", type=int, default=150, help="Head-only updates per loss setting")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation budget when testing a checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("toy_overfit_vis"))
    args = parser.parse_args()
    if args.steps < 1 or args.max_new_tokens < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Provide positive steps, learning rate, and generation budget.")

    query = load_rgb(args.query)
    target = load_mask(args.mask, query.size)
    references = [load_rgb(path) for path in args.reference]
    response = args.target or f"Yes. There is an anomaly in this {args.category}.\n<SEG>"

    device = "cuda"
    target = target.to(device)
    foreground_fraction = target.float().mean()
    print(f"Ground-truth foreground: {100 * foreground_fraction.item():.2f}% of mask pixels", flush=True)
    model = ad_model(device=device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device=device)

    tokenizer = model.vlm_backbone.tokenizer
    ids, attention, labels = build_training_tokens(tokenizer, [args.prompt], [response], device)
    grid_size = model.clip_model.visual.grid_size
    print(f"Mask patch grid: {grid_size[0]} × {grid_size[1]}", flush=True)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        prefix, adapted_text, features = model._build_prefix([query], [references], [args.category])
        _, seg_hidden = model._teacher_forced_vlm(prefix, ids, attention, labels)
        mask_inputs = features.clip_patches, adapted_text, features.reference_distance, seg_hidden
        before = model.mask_head(*mask_inputs, grid_size=grid_size, target_size=target.shape[-2:])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_image = target[0, 0].cpu().numpy()
    probs = report_mask("Teacher-forced before fitting", before, target)
    save_vis(query, target_image, probs, args.output_dir / "before.png")

    foreground_weight = (1.0 - foreground_fraction) / foreground_fraction
    for name, weight in (("training_loss", None), ("balanced_foreground", foreground_weight)):
        fitted = fit_mask_head(model.mask_head, mask_inputs, grid_size, target, args.steps, args.lr, weight)
        probs = report_mask(name, fitted, target)
        save_vis(query, target_image, probs, args.output_dir / f"{name}.png")

    if args.checkpoint:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            texts, _, generated = model.generate(
                [query], [references], [args.category], inputs["input_ids"], inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, mask_target_size=target.shape[-2:],
            )
        print(f"Generated response: {texts[0]!r}")
        if generated is None:
            print("Generation did not emit <SEG>. Evaluation would show an all-black mask.")
            probs = np.zeros(target_image.shape, dtype=np.float32)
        else:
            probs = report_mask("Generated mask from checkpoint", generated, target)
        save_vis(query, target_image, probs, args.output_dir / "generated.png")

    print(f"Saved query, annotation, and prediction strips to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
