import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from arch.model import ad_model, pad_to_square
from data.visa_dataset import CLASS_NAMES, VisADataset


def load_checkpoint(model: ad_model, ckpt_dir: str, device: str = "cuda"):
    """Loads LoRA adapters, <SEG> embeddings, and custom trainable heads."""
    vlm_dir = os.path.join(ckpt_dir, "vlm_adapters")
    heads_path = os.path.join(ckpt_dir, "custom_heads.pt")

    model.vlm_backbone.load_adapters(vlm_dir)
    heads = torch.load(heads_path, map_location=device)
    model.visual_residual_projector.load_state_dict(heads["visual_residual_projector"])
    model.query_conditioned_text_adapter.load_state_dict(heads["query_conditioned_text_adapter"])
    model.sam_head.prompt_generator.load_state_dict(heads["sam_prompt_generator"])
    model.sam_head.sam.mask_decoder.load_state_dict(heads["sam_mask_decoder"])
    model.eval()


def parse_yes_no(text: str) -> int | None:
    """Returns 1 for Yes, 0 for No, or None if both/neither appear."""
    lower = text.lower()
    has_yes = "yes" in lower
    has_no = "no" in lower
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return None


def save_vis(query_img: Image.Image, gt_mask: np.ndarray, pred_prob: np.ndarray, out_path: str):
    """Saves a 256x256 side-by-side strip: [Padded Query | GT Mask | Pred Heatmap]."""
    size = gt_mask.shape[0]
    q_rgb = pad_to_square(query_img).resize((size, size)).convert("RGB")
    gt_rgb = Image.fromarray((gt_mask * 255).astype(np.uint8), mode="L").convert("RGB")
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
        # Stride evenly so we get a mix of normal and anomalous samples
        step = max(1, len(indices) // max_samples)
        indices = indices[::step][:max_samples]

    i_labels, i_scores = [], []
    p_labels, p_scores = [], []
    text_correct, text_total, text_unparsed = 0, 0, 0
    seg_emitted = 0

    if vis_dir:
        os.makedirs(os.path.join(vis_dir, category), exist_ok=True)
    saved_vis = 0

    for idx in tqdm(indices, desc=f"Eval {category}", leave=False):
        sample = ds[idx]
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["user_prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            texts, seg_hidden, sam_logits = model.generate(
                query_img=[sample["query_img"]],
                reference_imgs=[sample["reference_imgs"]],
                category=[category],
                prompt_input_ids=inputs["input_ids"],
                prompt_attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
            )

        gen_text = texts[0]
        gt_label = sample["is_anomalous"]
        gt_mask = sample["mask"].squeeze(0).numpy()  # [256, 256]

        if sam_logits is not None:
            seg_emitted += 1
            pred_logits = F.interpolate(
                sam_logits.float(), size=gt_mask.shape, mode="bilinear", align_corners=False
            )
            pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
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

        if saved_vis < num_vis:
            print(
                f"  [{category} | gt={gt_label} | max_p={pred_prob.max():.3f} | "
                f"seg={sam_logits is not None}] {gen_text!r}"
            )
            if vis_dir:
                out_path = os.path.join(vis_dir, category, f"sample_{idx}_gt{gt_label}.png")
                save_vis(sample["query_img"], gt_mask, pred_prob, out_path)
            saved_vis += 1

    i_labels_arr = np.asarray(i_labels)
    i_scores_arr = np.asarray(i_scores)
    p_labels_arr = np.asarray(p_labels, dtype=np.uint8).ravel()
    p_scores_arr = np.asarray(p_scores, dtype=np.float32).ravel()

    # Guard against single-class slices when --max_samples is tiny
    i_auroc = roc_auc_score(i_labels_arr, i_scores_arr) * 100 if len(np.unique(i_labels_arr)) > 1 else float("nan")
    p_aupr = average_precision_score(p_labels_arr, p_scores_arr) * 100 if p_labels_arr.sum() > 0 else float("nan")
    text_acc = (100.0 * text_correct / text_total) if text_total else 0.0

    return {
        "category": category,
        "n": text_total,
        "i_auroc": i_auroc,
        "p_aupr": p_aupr,
        "text_acc": text_acc,
        "unparsed": text_unparsed,
        "seg_rate": 100.0 * seg_emitted / text_total,
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
            f"(unparsed={res['unparsed']}) | <SEG> rate: {res['seg_rate']:5.1f}%"
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