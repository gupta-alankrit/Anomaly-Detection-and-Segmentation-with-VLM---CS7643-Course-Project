import argparse
import os

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch.model import ad_model
from arch.text_adapter import align_loss
from data.visa_dataset import CLASS_NAMES, VisADataset, visa_collate_fn


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return (1.0 - dice).mean()

def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return loss.mean()

class AnomalyVLMTrainer:
    def __init__(
        self,
        model: ad_model,
        lr_vlm: float = 5e-5,
        lr_adapters: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        lambda_text: float = 1.0,
        lambda_bce: float = 2.0,
        lambda_dice: float = 0.5,
        lambda_align: float = 0.7,
        grad_accum_steps: int = 4,
        device: str = "cuda",
        epochs: int = 10,
    ):
        self.model = model
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.lambda_text = lambda_text
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_align = lambda_align
        self.grad_accum_steps = grad_accum_steps
        self.tokenizer = model.vlm_backbone.tokenizer
        self.epochs = epochs

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        vlm_params = [p for p in model.vlm_backbone.parameters() if p.requires_grad]
        adapter_params = [
            p for p in (
                list(model.visual_residual_projector.parameters())
                + list(model.query_conditioned_text_adapter.parameters())
            )
            if p.requires_grad
        ]
        sam_params = (
            list(model.sam_head.prompt_generator.parameters())
            + list(model.sam_head.sam.mask_decoder.parameters())
        )

        self.trainable_params = vlm_params + adapter_params + sam_params
        sam_new_params = (
            list(model.sam_head.prompt_generator.parameters())
        )
        sam_decoder_params = list(model.sam_head.sam.mask_decoder.parameters())

        self.optimizer = bnb.optim.PagedAdamW8bit(
            [
                {"params": vlm_params, "lr": lr_vlm},            # e.g. 5e-5
                {"params": adapter_params, "lr": lr_adapters},   # e.g. 5e-4
                {"params": sam_new_params, "lr": lr_adapters},   # e.g. 5e-4
                {"params": sam_decoder_params, "lr": 1e-4},      # pretrained decoder
            ],
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

    def build_training_tokens(
        self,
        user_prompts: list[str],
        target_responses: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_ids_list = []
        labels_list = []

        for prompt, target in zip(user_prompts, target_responses):
            messages = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            target_str = target + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            target_ids = self.tokenizer(target_str, add_special_tokens=False)["input_ids"]

            full_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            full_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        pad_id = self.tokenizer.pad_token_id
        max_len = max(seq.size(0) for seq in full_ids_list)

        input_ids = torch.full((len(full_ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(full_ids_list), max_len), dtype=torch.long)
        labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(full_ids_list, labels_list)):
            seq_len = ids.size(0)
            input_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = lbl

        return input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

    def compute_loss(self, output: dict, gt_masks: torch.Tensor, is_anomalous: torch.Tensor) -> tuple[torch.Tensor, dict]:
        ce_loss = output["vlm_out"]["loss"]
        pred_masks = output["pred_masks"]

        # sfl = sigmoid_focal_loss(pred_masks.float(), gt_masks.float())
        bce = F.binary_cross_entropy_with_logits(pred_masks.float(), gt_masks.float())
        dice_loss = soft_dice_loss(pred_masks, gt_masks)

        # Project pooled query patches [B, 768] -> [B, 512] CLIP text space
        pooled_proj = output["query_features"].mean(dim=1).float() @ self.model.clip_model.visual.proj.float()
        a_loss = align_loss(pooled_proj, output["adapted_text"].float(), is_anomalous, scale=20)

        total_loss = (
            self.lambda_text * ce_loss
            + self.lambda_bce * bce
            + self.lambda_dice * dice_loss
            + self.lambda_align * a_loss
        )
        metrics = {
            "total": total_loss.item(),
            "ce": ce_loss.item(),
            "bce": bce.item(),
            "dice": dice_loss.item(),
            "align": a_loss.item(),
        }
        return total_loss, metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        self.model.clip_model.eval()
        self.model.sam_head.sam.vision_encoder.eval()
        self.model.sam_head.sam.prompt_encoder.eval()

        self.optimizer.zero_grad()
        running = {"total": 0.0, "ce": 0.0, "bce": 0.0, "dice": 0.0, "align": 0.0}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for step, batch in enumerate(pbar, start=1):
            input_ids, attn_mask, labels = self.build_training_tokens(
                batch["user_prompts"], batch["target_responses"]
            )
            gt_masks = batch["masks"].to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model(
                    query_img=batch["query_imgs"],
                    reference_imgs=batch["reference_imgs"],
                    category=batch["categories"],
                    input_ids=input_ids,
                    text_attention_mask=attn_mask,
                    labels=labels,
                    mask_target_size=(gt_masks.shape[-2], gt_masks.shape[-1]),
                )
                loss, metrics = self.compute_loss(output, gt_masks, batch["is_anomalous"].to(self.device))

            if not torch.isfinite(loss):
                print(f"\n[WARN] Non-finite loss at epoch {epoch} step {step}: {metrics}")
                self.optimizer.zero_grad()
                continue
            
            # Scale loss for gradient accumulation
            (loss / self.grad_accum_steps).backward()

            if step % self.grad_accum_steps == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            for k in running:
                running[k] += metrics[k]

            pbar.set_postfix(
                tot=f"{running['total'] / step:.3f}",
                ce=f"{running['ce'] / step:.3f}",
                bce=f"{running['bce'] / step:.3f}",
                dice=f"{running['dice'] / step:.3f}",
                align=f"{running['align'] / step:.3f}",
                max_p=f"{output['pred_masks'].sigmoid().max().item():.2f}"
            )

        return {k: v / len(dataloader) for k, v in running.items()}

    def save_checkpoint(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        # 1. Save LoRA + resized <SEG> embeddings
        self.model.vlm_backbone.save_adapters(os.path.join(save_dir, "vlm_adapters"))
        # 2. Save custom trainable heads only (excludes frozen CLIP, SAM encoder, 4-bit VLM)
        torch.save(
            {
                "visual_residual_projector": self.model.visual_residual_projector.state_dict(),
                "query_conditioned_text_adapter": self.model.query_conditioned_text_adapter.state_dict(),
                "sam_prompt_generator": self.model.sam_head.prompt_generator.state_dict(),
                "sam_mask_decoder": self.model.sam_head.sam.mask_decoder.state_dict(),
            },
            os.path.join(save_dir, "custom_heads.pt"),
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/VisA")
    p.add_argument("--save_dir", type=str, default="checkpoints/visa_run1")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=1)
    p.add_argument("--lr_vlm", type=float, default=5e-5)
    p.add_argument("--lr_adapters", type=float, default=1e-3)
    p.add_argument("--val_categories", nargs="+", default=["cashew", "pcb1"])
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    train_cats = [c for c in CLASS_NAMES if c not in set(args.val_categories)]
    print(f"Training categories ({len(train_cats)}): {train_cats}")
    print(f"Held-out categories ({len(args.val_categories)}): {args.val_categories}")

    # Note: VisA's 'test' split contains both normal and anomalous images (~1:1).
    # Normal references are still drawn from the 'train' split inside VisADataset.
    train_ds = VisADataset(
        root_dir=args.data_root,
        split="test",
        categories=train_cats,
        k_shot=args.k_shot,
        mask_size=256,
        random_refs=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=visa_collate_fn,
    )

    model = ad_model(device=device)
    trainer = AnomalyVLMTrainer(
        model=model,
        lr_vlm=args.lr_vlm,
        lr_adapters=args.lr_adapters,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        device=device,
    )

    for epoch in range(1, args.epochs + 1):
        avg = trainer.train_epoch(train_loader, epoch)
        trainer.scheduler.step()
        print(
            f"Epoch {epoch} done | total={avg['total']:.4f} "
            f"ce={avg['ce']:.4f} bce={avg['bce']:.4f} dice={avg['dice']:.4f}"
        )
        if epoch % 5 == 0 or epoch == args.epochs:
            trainer.save_checkpoint(os.path.join(args.save_dir, f"epoch_{epoch}"))


if __name__ == "__main__":
    main()