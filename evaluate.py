import argparse
import os

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import re

from arch.model import ad_model, pad_to_square
from data.visa_dataset import CLASS_NAMES, VisADataset


def load_checkpoint(model: ad_model, ckpt_dir: str, device: str = "cuda"):
    """Loads LoRA adapters, <SEG> embeddings, and custom trainable heads."""
    vlm_dir = os.path.join(ckpt_dir, "vlm_adapters")
    heads_path = os.path.join(ckpt_dir, "custom_heads.pt")

    heads = torch.load(heads_path, map_location=device, weights_only=True)
    if "mask_head" not in heads:
        raise ValueError("Checkpoint is missing the CLIP mask head. Use a checkpoint saved by the updated trainer.")

    model.vlm_backbone.load_adapters(vlm_dir)
    model.visual_residual_projector.load_state_dict(heads["visual_residual_projector"])
    model.query_conditioned_text_adapter.load_state_dict(heads["query_conditioned_text_adapter"])
    model.mask_head.load_state_dict(heads["mask_head"])
    model.eval()


def parse_yes_no(text: str) -> int | None:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).lower()
    has_yes, has_no = "yes" in clean, "no" in clean
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return None


# def save_vis(query_img: Image.Image, gt_mask: np.ndarray, pred_prob: np.ndarray, out_path: str):
#     """Saves a 256x256 side-by-side strip: [Padded Query | GT Mask (white) | Pred Heatmap]."""
#     size = gt_mask.shape[0]
#     q_rgb = pad_to_square(query_img).resize((size, size)).convert("RGB")

#     # Explicitly map any positive GT pixel to pure white (255)
#     gt_uint8 = np.where(gt_mask > 0, 255, 0).astype(np.uint8)
#     gt_rgb = Image.fromarray(gt_uint8, mode="L").convert("RGB")

#     pred_rgb = Image.fromarray((np.clip(pred_prob, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")

#     strip = Image.new("RGB", (size * 3, size))
#     strip.paste(q_rgb, (0, 0))
#     strip.paste(gt_rgb, (size, 0))
#     strip.paste(pred_rgb, (size * 2, 0))
#     strip.save(out_path)


def save_vis(query_img: Image.Image, gt_mask: np.ndarray, pred_prob: np.ndarray, out_path: str):
    """Saves a 256x256 side-by-side strip: [Padded Query | GT Mask (white) | Pred Heatmap]."""
    size = gt_mask.shape[0]
    q_rgb = pad_to_square(query_img).resize((size, size)).convert("RGB")

    # Explicitly map any positive GT pixel to pure white (255)
    gt_uint8 = np.where(gt_mask > 0, 255, 0).astype(np.uint8)
    gt_rgb = Image.fromarray(gt_uint8, mode="L").convert("RGB")

    pred_rgb = Image.fromarray((np.clip(pred_prob, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")

    strip = Image.new("RGB", (size * 3, size))
    strip.paste(q_rgb, (0, 0))
    strip.paste(gt_rgb, (size, 0))
    strip.paste(pred_rgb, (size * 2, 0))
    strip.save(out_path)


def evaluate_category(
    model: ad_model,
    ds: VisADataset,
    category: str,
    device: str = "cuda",
    max_samples: int | None = None,
    vis_dir: str | None = None,
    num_vis: int = 4,
):
    tokenizer = model.vlm_backbone.tokenizer
    indices = list(range(len(ds)))
    if max_samples is not None and max_samples < len(indices):
        step = max(1, len(indices) // max_samples)
        indices = indices[::step][:max_samples]

    i_labels, i_scores = [], []
    p_labels, p_scores = [], []
    text_correct, text_total, text_unparsed = 0, 0, 0
    seg_emitted = 0

    if vis_dir:
        os.makedirs(os.path.join(vis_dir, category), exist_ok=True)
    saved_vis = {0: 0, 1: 0}  # Track normal (0) and anomalous (1) separately

    for idx in tqdm(indices, desc=f"Eval {category}", leave=False):
        sample = ds[idx]
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["user_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        gt_mask = sample["mask"].squeeze(0).numpy()  # [256, 256]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            texts, _, mask_logits = model.generate(
                query_img=[sample["query_img"]],
                reference_imgs=[sample["reference_imgs"]],
                category=[category],
                prompt_input_ids=inputs["input_ids"],
                prompt_attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                mask_target_size=gt_mask.shape,
            )

        gen_text = texts[0]
        gt_label = sample["is_anomalous"]

        if mask_logits is not None:
            seg_emitted += 1
            pred_prob = mask_logits[0, 0].float().sigmoid().cpu().numpy()
        else:
            pred_prob = np.zeros_like(gt_mask, dtype=np.float32)

        i_labels.append(gt_label)
        i_scores.append(float(pred_prob.max()))
        p_labels.append(gt_mask)
        p_scores.append(pred_prob)

        pred_yn = parse_yes_no(gen_text)
        text_total += 1
        if pred_yn is None:
            text_unparsed += 1
        elif pred_yn == gt_label:
            text_correct += 1

        if saved_vis[gt_label] < num_vis:
            gt_px = int((gt_mask > 0).sum())
            print(
                f"  [{category} | gt={gt_label} | gt_px={gt_px:4d} | max_p={pred_prob.max():.3f} | "
                f"seg={mask_logits is not None}] {gen_text!r}"
            )
            if vis_dir:
                out_path = os.path.join(vis_dir, category, f"sample_{idx}_gt{gt_label}.png")
                save_vis(sample["query_img"], gt_mask, pred_prob, out_path)
            saved_vis[gt_label] += 1

    i_labels_arr = np.asarray(i_labels)
    i_scores_arr = np.asarray(i_scores)
    p_labels_arr = np.asarray(p_labels, dtype=np.uint8).ravel()
    p_scores_arr = np.asarray(p_scores, dtype=np.float32).ravel()

    i_auroc = roc_auc_score(i_labels_arr, i_scores_arr) * 100 if len(np.unique(i_labels_arr)) > 1 else float("nan")
    p_aupr = average_precision_score(p_labels_arr, p_scores_arr) * 100 if p_labels_arr.sum() > 0 else float("nan")
    text_acc = (100.0 * text_correct / text_total) if text_total else 0.0
    normal_scores = i_scores_arr[i_labels_arr == 0]
    anomalous_scores = i_scores_arr[i_labels_arr == 1]

    return {
        "category": category,
        "n": text_total,
        "i_auroc": i_auroc,
        "p_aupr": p_aupr,
        "text_acc": text_acc,
        "unparsed": text_unparsed,
        "seg_rate": 100.0 * seg_emitted / text_total,
        "max_p_normal": float(normal_scores.mean()) if normal_scores.size else float("nan"),
        "max_p_anomalous": float(anomalous_scores.mean()) if anomalous_scores.size else float("nan"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="e.g. checkpoints/visa_run1/epoch_1")
    p.add_argument("--data_root", type=str, default="data/VisA")
    p.add_argument("--categories", nargs="+", default=["cashew", "pcb1"])
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=30, help="Max samples per category (None for all)")
    p.add_argument("--vis_dir", type=str, default="eval_vis", help="Directory to save mask strips")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    model = ad_model(device=device)
    load_checkpoint(model, args.ckpt_dir, device=device)

    results = []
    for cat in args.categories:
        if cat not in CLASS_NAMES:
            raise ValueError(f"Unknown category '{cat}'. Expected one of {CLASS_NAMES}")
        ds = VisADataset(
            root_dir=args.data_root,
            split="test",
            categories=[cat],
            k_shot=args.k_shot,
            mask_size=256,
            random_refs=False,  # Deterministic references for reproducible eval
        )
        res = evaluate_category(
            model=model,
            ds=ds,
            category=cat,
            device=device,
            max_samples=args.max_samples,
            vis_dir=args.vis_dir,
        )
        results.append(res)
        print(
            f"{cat:12s} (n={res['n']:3d}) | I-AUROC: {res['i_auroc']:5.2f}% | "
            f"P-AUPR: {res['p_aupr']:5.2f}% | Text Acc: {res['text_acc']:5.2f}% "
            f"(unparsed={res['unparsed']}) | <SEG> rate: {res['seg_rate']:5.1f}% | "
            f"Avg max_p (normal): {res['max_p_normal']:.3f} | "
            f"Avg max_p (anomalous): {res['max_p_anomalous']:.3f}"
        )

    if len(results) > 1:
        print("-" * 80)
        print(
            f"{'MEAN':12s}         | I-AUROC: {np.nanmean([r['i_auroc'] for r in results]):5.2f}% | "
            f"P-AUPR: {np.nanmean([r['p_aupr'] for r in results]):5.2f}% | "
            f"Text Acc: {np.mean([r['text_acc'] for r in results]):5.2f}%"
        )


if __name__ == "__main__":
    main()
