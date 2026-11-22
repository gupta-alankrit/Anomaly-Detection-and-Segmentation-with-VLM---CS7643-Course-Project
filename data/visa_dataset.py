import csv
import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]

DESCRIPTIONS = {
    "candle": "This is a photo of 4 candles for anomaly detection, every candle should be round, without any damage, flaw, defect, scratch, hole or broken part.",
    "capsules": "This is a photo of many small capsules for anomaly detection, every capsule is green, should be without any damage, flaw, defect, scratch, hole or broken part.",
    "cashew": "This is a photo of a cashew for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "chewinggum": "This is a photo of a chewinggum for anomaly detection, which should be white, without any damage, flaw, defect, scratch, hole or broken part.",
    "fryum": "This is a photo of a fryum for anomaly detection on green background, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "macaroni1": "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "macaroni2": "This is a photo of 4 macaronis for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb1": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb2": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb3": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pcb4": "This is a photo of pcb for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
    "pipe_fryum": "This is a photo of a pipe fryum for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part.",
}


def pad_to_square(img: Image.Image, fill: int = 0) -> Image.Image:
    w, h = img.size
    size = max(w, h)
    return ImageOps.expand(img, border=(0, 0, size - w, size - h), fill=fill)


class VisADataset(Dataset):
    def __init__(
        self,
        root_dir: str = "../data/VisA",
        csv_rel_path: str = "split_csv/1cls.csv",
        split: str = "test",  # "train" | "test" | "all"
        categories: Sequence[str] | None = None,
        k_shot: int = 1,
        mask_size: int = 256,
        random_refs: bool = True,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.k_shot = k_shot
        self.mask_size = mask_size
        self.random_refs = random_refs
        self.rng = random.Random(seed)

        allowed_classes = set(categories) if categories is not None else set(CLASS_NAMES)
        csv_path = os.path.join(root_dir, csv_rel_path)

        self.samples: list[dict] = []
        self.normal_pool: dict[str, list[str]] = {c: [] for c in allowed_classes}

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0] not in allowed_classes:
                    continue
                cat, row_split, rel_path = row[0], row[1], row[3]
                full_path = os.path.join(root_dir, rel_path)
                is_normal = "Normal" in rel_path.split("/")[-2]

                # Build normal reference pool from the train split
                if row_split == "train" and is_normal:
                    self.normal_pool[cat].append(full_path)

                if split == "all" or row_split == split:
                    if is_normal:
                        mask_path = None
                    else:
                        mask_path = full_path.replace("Images", "Masks").replace(".JPG", ".png")
                    self.samples.append({
                        "category": cat,
                        "image_path": full_path,
                        "mask_path": mask_path,
                        "is_anomalous": 0 if is_normal else 1,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_references(self, category: str, exclude_path: str) -> list[Image.Image]:
        if self.k_shot <= 0:
            return []
        pool = [p for p in self.normal_pool[category] if p != exclude_path]
        if len(pool) < self.k_shot:
            raise ValueError(f"Category '{category}' has only {len(pool)} normal refs, need {self.k_shot}.")
        chosen = self.rng.sample(pool, self.k_shot) if self.random_refs else pool[: self.k_shot]
        return [Image.open(p).convert("RGB") for p in chosen]

    def _load_mask(self, mask_path: str | None, img_size: tuple[int, int]) -> torch.Tensor:
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
        item = self.samples[idx]
        cat = item["category"]
        query_img = Image.open(item["image_path"]).convert("RGB")
        ref_imgs = self._sample_references(cat, exclude_path=item["image_path"])
        mask = self._load_mask(item["mask_path"], query_img.size)

        user_prompt = f"{DESCRIPTIONS[cat]} Is there any anomaly in the image?"
        if item["is_anomalous"]:
            target_response = f"Yes. There is an anomaly in this {cat}. <SEG>"
        else:
            target_response = f"No. There is no anomaly in this {cat}. <SEG>"

        return {
            "query_img": query_img,
            "reference_imgs": ref_imgs,
            "category": cat,
            "mask": mask,
            "is_anomalous": item["is_anomalous"],
            "user_prompt": user_prompt,
            "target_response": target_response,
            "image_path": item["image_path"],
        }


def visa_collate_fn(batch: list[dict]) -> dict:
    """Keeps PIL images/strings as lists and stacks numeric tensors."""
    return {
        "query_imgs": [b["query_img"] for b in batch],
        "reference_imgs": [b["reference_imgs"] for b in batch],
        "categories": [b["category"] for b in batch],
        "masks": torch.stack([b["mask"] for b in batch], dim=0),  # [B, 1, H, W]
        "is_anomalous": torch.tensor([b["is_anomalous"] for b in batch], dtype=torch.long),
        "user_prompts": [b["user_prompt"] for b in batch],
        "target_responses": [b["target_response"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }


if __name__ == "__main__":
    ds = VisADataset(root_dir="../data/VisA", split="test", k_shot=2, mask_size=256)
    print(f"Loaded {len(ds)} test samples across {len(ds.normal_pool)} categories.")
    for cat, refs in sorted(ds.normal_pool.items()):
        print(f"  {cat:12s}: {len(refs)} normal train refs")

    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=visa_collate_fn)
    batch = next(iter(loader))

    print("\n--- Batch Smoke Test ---")
    print("Categories:      ", batch["categories"])
    print("Labels (0/1):    ", batch["is_anomalous"].tolist())
    print("Masks shape:     ", tuple(batch["masks"].shape), batch["masks"].dtype)
    print("Mask unique vals:", torch.unique(batch["masks"]).tolist())
    print("Mask pos pixels: ", [int(m.sum().item()) for m in batch["masks"]])
    print("Refs per query:  ", [len(r) for r in batch["reference_imgs"]])
    print("Sample prompt:   ", batch["user_prompts"][0])
    print("Sample target:   ", batch["target_responses"][0])

    # Sanity check: normal images must have 0 positive mask pixels; anomalous must have > 0
    for i, label in enumerate(batch["is_anomalous"].tolist()):
        pos = int(batch["masks"][i].sum().item())
        if label == 0:
            assert pos == 0, f"Normal sample {batch['image_paths'][i]} has non-zero mask!"
        else:
            assert pos > 0, f"Anomalous sample {batch['image_paths'][i]} has empty mask!"
    print("\nAll batch assertions passed.")