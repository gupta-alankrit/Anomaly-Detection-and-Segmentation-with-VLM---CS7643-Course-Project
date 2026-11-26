import json
import os
import random
import re
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

SUBDATASETS = [
    "3cad",
    "cableinspect",
    "goodsad",
    "manta",
    "miad",
    "mpdd",
    "mvtec",
    "mvtec3d",
    "mvtecloco",
    "realiad",
    "realiad-d3",
    "texturead",
    "visa",
    "wfdd",
]

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def extract_cot(text: str) -> str:
    """Extracts chain-of-thought reasoning inside <think>...</think> tags."""
    match = _THINK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()

def _safe_open_rgb(path: str) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def format_target_response(
    cot: str,
    is_anomalous: int,
    cat_readable: str,
    decision_first: bool = True,
    max_cot_chars: int = 600,
) -> str:
    """Assembles Decision + CoT + <SEG> target string for SmolVLM."""
    cot_clean = " ".join(cot.split())
    if len(cot_clean) > max_cot_chars:
        # Trim cleanly at the last sentence/word boundary
        cot_clean = cot_clean[:max_cot_chars].rsplit(" ", 1)[0].rstrip(",;:") + "..."

    if is_anomalous:
        decision = f"Yes. There is an anomaly in this {cat_readable}."
    else:
        decision = f"No. There is no anomaly in this {cat_readable}."

    if decision_first:
        return f"{decision}\n<think>\n{cot_clean}\n</think>\n<SEG>"
    return f"<think>\n{cot_clean}\n</think>\n{decision} <SEG>"

def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    size = max(w, h)
    return ImageOps.expand(img, border=(0, 0, size - w, size - h), fill=fill)


def _index_gt_dir(
    gt_dir: str,
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], list[str]], dict[str, list[str]]]:
    """Indexes ground truth mask files by (rel_dir, clean_stem), (defect, clean_stem), and clean_stem."""
    by_rel_stem: dict[tuple[str, str], list[str]] = {}
    by_defect_stem: dict[tuple[str, str], list[str]] = {}
    by_stem: dict[str, list[str]] = {}

    if not os.path.isdir(gt_dir):
        return by_rel_stem, by_defect_stem, by_stem

    for dirpath, dirnames, filenames in os.walk(gt_dir):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, gt_dir)
        defect = rel_dir.split(os.sep)[0] if rel_dir != "." else ""
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            stem = os.path.splitext(fn)[0]
            clean_stem = stem[:-5] if stem.endswith("_mask") else stem
            by_rel_stem.setdefault((rel_dir, clean_stem), []).append(full)
            if defect:
                by_defect_stem.setdefault((defect, clean_stem), []).append(full)
            by_stem.setdefault(clean_stem, []).append(full)

    return by_rel_stem, by_defect_stem, by_stem


def _resolve_mask_path(
    query_path: str,
    test_dir: str,
    gt_idx: tuple[dict, dict, dict],
) -> str | None:
    """Resolves an anomalous query image path to its corresponding ground truth mask image."""
    by_rel_stem, by_defect_stem, by_stem = gt_idx
    rel_img = os.path.relpath(query_path, test_dir)
    rel_dir = os.path.dirname(rel_img)
    defect = rel_dir.split(os.sep)[0] if rel_dir else ""
    stem = os.path.splitext(os.path.basename(rel_img))[0]
    prefix = stem.split("_", 1)[0]

    cands = (
        by_rel_stem.get((rel_dir, stem))
        or by_defect_stem.get((defect, stem))
        or by_rel_stem.get((rel_dir, prefix))
        or by_defect_stem.get((defect, prefix))
        or by_stem.get(stem)
    )
    return cands[0] if cands else None


def _index_image_dir(
    base_dir: str,
) -> tuple[dict[tuple[str, str], list[str]], dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Indexes all files in a directory by (defect_subdir, stem), basename, and stem."""
    by_defect_stem: dict[tuple[str, str], list[str]] = {}
    by_basename: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}
    all_files: list[str] = []

    if not os.path.isdir(base_dir):
        return by_defect_stem, by_basename, by_stem, all_files

    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, base_dir)
        defect = rel_dir.split(os.sep)[0] if rel_dir != "." else ""
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            stem = os.path.splitext(fn)[0]
            all_files.append(full)
            by_basename.setdefault(fn, []).append(full)
            by_stem.setdefault(stem, []).append(full)
            if defect:
                by_defect_stem.setdefault((defect, stem), []).append(full)

    return by_defect_stem, by_basename, by_stem, all_files


def _resolve_query_path(
    name: str,
    is_anomalous: bool,
    test_idx: tuple[dict, dict, dict, list],
    test_good_idx: tuple[dict, dict, dict, list],
    train_good_idx: tuple[dict, dict, dict, list],
    good_idx: tuple[dict, dict, dict, list],
) -> str | None:
    """Resolves a text_annos.json `name` entry to an on-disk image path."""
    test_ds, test_bn, test_st, _ = test_idx
    _, tg_bn, tg_st, _ = test_good_idx
    _, tr_bn, tr_st, _ = train_good_idx
    _, gd_bn, gd_st, _ = good_idx

    bn = os.path.basename(name)
    stem = os.path.splitext(bn)[0]

    if not is_anomalous:
        if bn.startswith("train_good-"):
            clean_bn = bn[len("train_good-"):]
            clean_stem = os.path.splitext(clean_bn)[0]
            cands = tr_bn.get(clean_bn) or tr_st.get(clean_stem)
        elif bn.startswith("test_good-"):
            clean_bn = bn[len("test_good-"):]
            clean_stem = os.path.splitext(clean_bn)[0]
            cands = tg_bn.get(clean_bn) or tg_st.get(clean_stem)
        elif bn.startswith("good-"):
            clean_bn = bn[len("good-"):]
            clean_stem = os.path.splitext(clean_bn)[0]
            cands = gd_bn.get(clean_bn) or gd_st.get(clean_stem)
        else:
            cands = (
                tg_bn.get(bn)
                or tg_st.get(stem)
                or gd_bn.get(bn)
                or gd_st.get(stem)
                or tr_bn.get(bn)
                or tr_st.get(stem)
            )
        return cands[0] if cands else None

    if "/" not in name and "-" in name:
        defect, rest = name.split("-", 1)
        rest_stem = os.path.splitext(rest)[0]
        cands = test_ds.get((defect, rest_stem)) or test_bn.get(rest) or test_st.get(rest_stem)
    else:
        cands = test_bn.get(bn) or test_st.get(stem)
        if not cands and "-" in bn:
            defect, rest = bn.split("-", 1)
            rest_stem = os.path.splitext(rest)[0]
            cands = test_ds.get((defect, rest_stem)) or test_bn.get(rest) or test_st.get(rest_stem)

    return cands[0] if cands else None


def _resolve_ref_path(
    ref_name: str,
    train_good_idx: tuple[dict, dict, dict, list],
    good_idx: tuple[dict, dict, dict, list],
    test_good_idx: tuple[dict, dict, dict, list],
) -> str | None:
    """Resolves a text_annos.json `ref-name` entry to an on-disk normal reference image."""
    if not ref_name:
        return None
    _, tr_bn, tr_st, _ = train_good_idx
    _, gd_bn, gd_st, _ = good_idx
    _, tg_bn, tg_st, _ = test_good_idx

    bn = os.path.basename(ref_name)
    stem = os.path.splitext(bn)[0]
    cands = (
        tr_bn.get(bn)
        or tr_st.get(stem)
        or gd_bn.get(bn)
        or gd_st.get(stem)
        or tg_bn.get(bn)
        or tg_st.get(stem)
    )
    return cands[0] if cands else None


class MMRADDataset(Dataset):
    """PyTorch Dataset for the MMR-AD multimodal anomaly detection dataset.

    Loads query images, normal reference images, binary ground truth masks,
    bounding boxes (in MMR-AD's 448 short-edge resized coordinate space),
    binary anomaly classification labels (0 = normal, 1 = anomalous), and
    chain-of-thought reasoning text from MMR-AD's
    `ANNOS/<subdataset>/<category>/text_annos.json` files.
    """

    def __init__(
        self,
        root_dir: str = "data/MMR-AD",
        subdatasets: Sequence[str] | None = None,
        exclude_subdatasets: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        k_shot: int = 1,
        mask_size: int = 256,
        random_refs: bool = True,
        seed: int = 42,
        strict: bool = True,
    ):
        """
        Args:
            root_dir: Path to the MMR-AD root directory (containing `ANNOS/` and sub-dataset dirs).
            subdatasets: Optional whitelist of sub-dataset names to include (defaults to all 14).
            exclude_subdatasets: Optional sub-dataset names to exclude from the selected set.
            categories: Optional filter of category names (either `"<category>"` or `"<subdataset>/<category>"`).
            k_shot: Number of normal reference images to return per sample. When `k_shot >= 1`,
                the paired reference image referenced by the CoT text is always placed first;
                additional references (if `k_shot > 1`) are drawn from the category's normal pool.
            mask_size: Spatial resolution (H = W = mask_size) for returned binary mask tensors,
                aligned with `pad_to_square` (matching `VisADataset`).
            random_refs: Whether additional references beyond the paired ref are sampled randomly.
            seed: Random seed for reference sampling.
            strict: If True, raise an error when an annotated image/mask path cannot be resolved on disk.
        """
        self.root_dir = root_dir
        self.annos_dir = os.path.join(root_dir, "ANNOS")
        self.k_shot = k_shot
        self.mask_size = mask_size
        self.random_refs = random_refs
        self.rng = random.Random(seed)

        known = set(SUBDATASETS)
        if subdatasets is not None:
            unknown = set(subdatasets) - known
            if unknown:
                raise ValueError(
                    f"Unknown subdatasets: {sorted(unknown)}. Valid options: {SUBDATASETS}"
                )
            selected = [s for s in SUBDATASETS if s in set(subdatasets)]
        else:
            selected = list(SUBDATASETS)

        if exclude_subdatasets is not None:
            unknown_ex = set(exclude_subdatasets) - known
            if unknown_ex:
                raise ValueError(
                    f"Unknown exclude_subdatasets: {sorted(unknown_ex)}. Valid options: {SUBDATASETS}"
                )
            excluded = set(exclude_subdatasets)
            selected = [s for s in selected if s not in excluded]

        self.subdatasets: list[str] = selected
        allowed_categories = set(categories) if categories is not None else None

        self.samples: list[dict] = []
        self.normal_pool: dict[tuple[str, str], list[str]] = {}

        for sub in self.subdatasets:
            sub_anno_dir = os.path.join(self.annos_dir, sub)
            if not os.path.isdir(sub_anno_dir):
                if strict:
                    raise FileNotFoundError(f"Annotation directory not found: {sub_anno_dir}")
                continue

            for cat in sorted(os.listdir(sub_anno_dir)):
                cat_anno_dir = os.path.join(sub_anno_dir, cat)
                if not os.path.isdir(cat_anno_dir):
                    continue

                if allowed_categories is not None:
                    if cat not in allowed_categories and f"{sub}/{cat}" not in allowed_categories:
                        continue

                text_anno_path = os.path.join(cat_anno_dir, "text_annos.json")
                if not os.path.isfile(text_anno_path):
                    if strict:
                        raise FileNotFoundError(f"Missing text_annos.json: {text_anno_path}")
                    continue

                cat_img_dir = os.path.join(root_dir, sub, cat)
                test_dir = os.path.join(cat_img_dir, "test")
                test_idx = _index_image_dir(test_dir)
                test_good_idx = _index_image_dir(os.path.join(cat_img_dir, "test_good"))
                train_good_idx = _index_image_dir(os.path.join(cat_img_dir, "train_good"))
                good_idx = _index_image_dir(os.path.join(cat_img_dir, "good"))
                gt_idx = _index_gt_dir(os.path.join(cat_img_dir, "ground_truth"))

                # Normal reference pool comes from train_good/ (or good/ for cableinspect)
                pool = train_good_idx[3] if train_good_idx[3] else good_idx[3]
                self.normal_pool[(sub, cat)] = pool

                with open(text_anno_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)

                for entry in annotations:
                    bboxes = entry.get("bboxes") or []
                    is_anomalous = 1 if bboxes else 0
                    query_path = _resolve_query_path(
                        entry["name"],
                        bool(is_anomalous),
                        test_idx,
                        test_good_idx,
                        train_good_idx,
                        good_idx,
                    )
                    ref_path = _resolve_ref_path(
                        entry.get("ref-name", ""),
                        train_good_idx,
                        good_idx,
                        test_good_idx,
                    )
                    mask_path = (
                        _resolve_mask_path(query_path, test_dir, gt_idx)
                        if (is_anomalous and query_path is not None)
                        else None
                    )

                    if (
                        query_path is None
                        or (self.k_shot > 0 and ref_path is None)
                        or (is_anomalous and mask_path is None)
                    ):
                        if strict:
                            raise FileNotFoundError(
                                f"Failed to resolve paths for {sub}/{cat} entry "
                                f"name={entry.get('name')!r}, ref-name={entry.get('ref-name')!r}, "
                                f"mask_path={mask_path!r}"
                            )
                        continue

                    raw_text = entry.get("text", "")
                    cot = extract_cot(raw_text)

                    self.samples.append(
                        {
                            "subdataset": sub,
                            "category": cat,
                            "image_path": query_path,
                            "ref_image_path": ref_path,
                            "mask_path": mask_path,
                            "bboxes": bboxes,
                            "is_anomalous": is_anomalous,
                            "cot": cot,
                            "raw_text": raw_text,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_references(
        self,
        subdataset: str,
        category: str,
        paired_ref_path: str | None,
        exclude_path: str,
    ) -> list[Image.Image]:
        if self.k_shot <= 0:
            return []

        imgs: list[Image.Image] = []
        tried: set[str] = {exclude_path}

        if paired_ref_path is not None:
            tried.add(paired_ref_path)
            img = _safe_open_rgb(paired_ref_path)
            if img is not None:
                imgs.append(img)

        if len(imgs) < self.k_shot:
            pool = [p for p in self.normal_pool[(subdataset, category)] if p not in tried]
            if self.random_refs:
                self.rng.shuffle(pool)
            for p in pool:
                img = _safe_open_rgb(p)
                if img is not None:
                    imgs.append(img)
                    if len(imgs) == self.k_shot:
                        break

        return imgs

    def _load_mask(self, mask_path: str | None) -> torch.Tensor:
        """Returns binary float tensor [1, mask_size, mask_size] aligned with pad_to_square."""
        if mask_path is None:
            return torch.zeros((1, self.mask_size, self.mask_size), dtype=torch.float32)

        mask = Image.open(mask_path).convert("L")
        mask = pad_to_square(mask, fill=0)
        mask = mask.resize((self.mask_size, self.mask_size), resample=Image.NEAREST)
        mask_np = np.asarray(mask, dtype=np.float32)
        binary = (mask_np > 0).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict:
        for _ in range(10):  # Max retries before giving up
            item = self.samples[idx]
            try:
                query_img = Image.open(item["image_path"]).convert("RGB")
                ref_imgs = self._sample_references(
                    item["subdataset"],
                    item["category"],
                    paired_ref_path=item["ref_image_path"],
                    exclude_path=item["image_path"],
                )
                mask = self._load_mask(item["mask_path"])
                break
            except (UnidentifiedImageError, OSError) as e:
                print(f"[WARN] Skipping corrupt sample idx={idx} ({item['image_path']}): {e}")
                idx = self.rng.randrange(len(self.samples))
        else:
            raise RuntimeError("Exceeded max retries due to corrupt images.")

        sub = item["subdataset"]
        cat = item["category"]
        cat_readable = cat.replace("_", " ")

        query_img = Image.open(item["image_path"]).convert("RGB")
        ref_imgs = self._sample_references(
            sub,
            cat,
            paired_ref_path=item["ref_image_path"],
            exclude_path=item["image_path"],
        )
        mask = self._load_mask(item["mask_path"])

        user_prompt = (
            f"This is a photo of {cat_readable} for anomaly detection. "
            f"Is there any anomaly in the image?"
        )
        target_response = format_target_response(
            cot=item["cot"],
            is_anomalous=item["is_anomalous"],
            cat_readable=cat_readable,
            decision_first=False,
        )

        return {
            "query_img": query_img,
            "reference_imgs": ref_imgs,
            "subdataset": sub,
            "category": cat,
            "mask": mask,
            "bboxes": item["bboxes"],
            "is_anomalous": item["is_anomalous"],
            "cot": item["cot"],
            "raw_text": item["raw_text"],
            "user_prompt": user_prompt,
            "target_response": target_response,
            "image_path": item["image_path"],
            "ref_image_path": item["ref_image_path"],
            "mask_path": item["mask_path"],
        }


def mmr_ad_collate_fn(batch: list[dict]) -> dict:
    """Keeps PIL images/strings/bboxes as lists and stacks numeric mask and label tensors."""
    return {
        "query_imgs": [b["query_img"] for b in batch],
        "reference_imgs": [b["reference_imgs"] for b in batch],
        "subdatasets": [b["subdataset"] for b in batch],
        "categories": [b["category"] for b in batch],
        "masks": torch.stack([b["mask"] for b in batch], dim=0),  # [B, 1, H, W]
        "bboxes": [b["bboxes"] for b in batch],
        "is_anomalous": torch.tensor([b["is_anomalous"] for b in batch], dtype=torch.long),
        "cot": [b["cot"] for b in batch],
        "raw_texts": [b["raw_text"] for b in batch],
        "user_prompts": [b["user_prompt"] for b in batch],
        "target_responses": [b["target_response"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "ref_image_paths": [b["ref_image_path"] for b in batch],
        "mask_paths": [b["mask_path"] for b in batch],
    }


if __name__ == "__main__":
    ds = MMRADDataset(root_dir="data/MMR-AD", subdatasets=["mvtec", "wfdd"], k_shot=1, mask_size=256)
    n_anom = sum(s["is_anomalous"] for s in ds.samples)
    n_norm = len(ds) - n_anom
    print(
        f"Loaded {len(ds)} samples ({n_anom} anomalous, {n_norm} normal) "
        f"across {len(ds.normal_pool)} categories in {ds.subdatasets}."
    )

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=mmr_ad_collate_fn)
    batch = next(iter(loader))

    print("\n--- Batch Smoke Test ---")
    print("Subdatasets:     ", batch["subdatasets"])
    print("Categories:      ", batch["categories"])
    print("Labels (0/1):    ", batch["is_anomalous"].tolist())
    print("Query img sizes: ", [img.size for img in batch["query_imgs"]])
    print("Masks shape:     ", tuple(batch["masks"].shape), batch["masks"].dtype)
    print("Mask pos pixels: ", [int(m.sum().item()) for m in batch["masks"]])
    print("Bboxes per item: ", [len(b) for b in batch["bboxes"]])
    print("Refs per query:  ", [len(r) for r in batch["reference_imgs"]])
    print("Sample prompt:   ", batch["user_prompts"][0])
    print("Sample CoT[:160]:", repr(batch["cot"][0][:160]))
    print("Sample bboxes[0]:", batch["bboxes"][0])
