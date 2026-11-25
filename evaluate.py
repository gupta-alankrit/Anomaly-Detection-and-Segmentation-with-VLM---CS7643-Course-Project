import argparse
from datetime import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

from arch.model import ad_model, pad_to_square
from data.visa_dataset import CLASS_NAMES, VisADataset

LABEL_NAMES = {0: "Normal", 1: "Anomalous", 2: "Unclassified"}


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
    model.sam_head.dense_prompt_gen.load_state_dict(heads["sam_dense_prompt_gen"])
    model.eval()


def parse_yes_no(text: str) -> int | None:
    """Returns 1 for Yes (Anomalous), 0 for No (Normal), or None if unclassified."""
    tokens = text.split()
    if not tokens:
        return None
    lower = tokens[0].lower()
    has_yes = "yes" in lower
    has_no = "no" in lower
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return None


def save_confusion_matrix_image(cm: np.ndarray, out_path: str, title: str = "Confusion Matrix"):
    """Renders and saves a 2x3 confusion matrix heatmap image."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    x_labels = ["Normal (0)", "Anomalous (1)", "Unclassified"]
    y_labels = ["Normal (0)", "Anomalous (1)"]

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="medium",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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
    vis_dir: str | None = "eval_vis",
    num_vis: int = 4,
    log_file: str | None = None,
    append_log: bool = False,
):
    tokenizer = model.vlm_backbone.tokenizer
    indices = list(range(len(ds)))
    if max_samples is not None and max_samples < len(indices):
        step = max(1, len(indices) // max_samples)
        indices = indices[::step][:max_samples]

    i_labels, i_scores = [], []
    p_labels, p_scores = [], []
    y_true_cls, y_pred_cls = [], []
    text_correct, text_total, text_unparsed = 0, 0, 0
    seg_emitted = 0

    if vis_dir:
        os.makedirs(os.path.join(vis_dir, category), exist_ok=True)
    saved_vis = {0: 0, 1: 0}  # Track normal (0) and anomalous (1) separately

    log_fh = None
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fh = open(log_file, "a" if append_log else "w", encoding="utf-8")
        log_fh.write(f"\n=== Category: {category} (n={len(indices)}) ===\n")

    try:
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
            gt_label = int(sample["is_anomalous"])
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
                pred_cls_id = 2  # Unclassified
                status = "UNCLASSIFIED"
            else:
                pred_cls_id = pred_yn
                if pred_yn == gt_label:
                    text_correct += 1
                    status = "CORRECT"
                else:
                    status = "WRONG"

            y_true_cls.append(gt_label)
            y_pred_cls.append(pred_cls_id)

            if log_fh is not None:
                img_ref = (
                    sample.get("img_path")
                    or sample.get("image_path")
                    or sample.get("filename")
                    or f"sample_{idx}"
                )
                log_fh.write(
                    f"[{category} | idx={idx:04d} | img={img_ref} | "
                    f"gt={LABEL_NAMES[gt_label]}({gt_label}) | "
                    f"pred={LABEL_NAMES[pred_cls_id]}({pred_cls_id}) | "
                    f"status={status} | max_p={pred_prob.max():.3f} | "
                    f"seg={sam_logits is not None}] "
                    f"output={gen_text!r}\n"
                )
                log_fh.flush()

            if saved_vis[gt_label] < num_vis:
                gt_px = int((gt_mask > 0).sum())
                print(
                    f"  [{category} | gt={gt_label} | gt_px={gt_px:4d} | max_p={pred_prob.max():.3f} | "
                    f"seg={sam_logits is not None}] {gen_text!r}"
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

        # Shape (2, 3): rows = [Actual Normal (0), Actual Anomalous (1)],
        # cols = [Pred Normal (0), Pred Anomalous (1), Pred Unclassified (2)]
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1, 2])[:2, :]

        cm_path = None
        if vis_dir:
            cm_path = os.path.join(vis_dir, category, "confusion_matrix.png")
            save_confusion_matrix_image(cm, cm_path, title=f"Confusion Matrix — {category}")

        if log_fh is not None and cm_path is not None:
            log_fh.write(f"Saved confusion matrix image: {cm_path}\n")
            log_fh.flush()

    finally:
        if log_fh is not None:
            log_fh.close()

    return {
        "category": category,
        "n": text_total,
        "i_auroc": i_auroc,
        "p_aupr": p_aupr,
        "text_acc": text_acc,
        "unparsed": text_unparsed,
        "seg_rate": 100.0 * seg_emitted / text_total,
        "confusion_matrix": cm,
        "cm_path": cm_path,
        "cm_counts": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "unclassified_normal": int(cm[0, 2]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "unclassified_anomalous": int(cm[1, 2]),
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True, help="e.g. checkpoints/visa_run1/epoch_1")
    p.add_argument("--data_root", type=str, default="data/VisA")
    p.add_argument("--categories", nargs="+", default=["cashew", "pcb1"])
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=30, help="Max samples per category (None for all)")
    p.add_argument("--vis_dir", type=str, default="eval_vis", help="Directory to save mask strips & confusion matrices")
    p.add_argument("--log_file", type=str, default="eval_vis/eval_log.txt", help="Path to save per-image text logs")
    return p.parse_args()


def main():
    args = parse_args()
    # device = "cuda"
    device = "mps"

    model = ad_model(device=device)
    load_checkpoint(model, args.ckpt_dir, device=device)

    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(
                f"Evaluation Run — {datetime.now().isoformat(timespec='seconds')}\n"
                f"Checkpoint: {args.ckpt_dir} | Categories: {args.categories} | "
                f"k_shot: {args.k_shot} | max_samples: {args.max_samples}\n"
            )

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
            log_file=args.log_file,
            append_log=True,
        )
        results.append(res)
        cm_note = f" | CM: {res['cm_path']}" if res["cm_path"] else ""
        print(
            f"{cat:12s} (n={res['n']:3d}) | I-AUROC: {res['i_auroc']:5.2f}% | "
            f"P-AUPR: {res['p_aupr']:5.2f}% | Text Acc: {res['text_acc']:5.2f}% "
            f"(unparsed={res['unparsed']}) | <SEG> rate: {res['seg_rate']:5.1f}%{cm_note}"
        )

    if len(results) > 1:
        total_cm = np.sum([r["confusion_matrix"] for r in results], axis=0)
        overall_cm_path = None
        if args.vis_dir:
            overall_cm_path = os.path.join(args.vis_dir, "overall_confusion_matrix.png")
            save_confusion_matrix_image(
                total_cm, overall_cm_path, title="Overall Confusion Matrix (All Categories)"
            )

        cm_note = f" | CM: {overall_cm_path}" if overall_cm_path else ""
        mean_line = (
            f"{'MEAN':12s}         | I-AUROC: {np.nanmean([r['i_auroc'] for r in results]):5.2f}% | "
            f"P-AUPR: {np.nanmean([r['p_aupr'] for r in results]):5.2f}% | "
            f"Text Acc: {np.mean([r['text_acc'] for r in results]):5.2f}%{cm_note}"
        )
        print("-" * 80)
        print(mean_line)

        if args.log_file:
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 80}\n{mean_line}\n")


if __name__ == "__main__":
    main()