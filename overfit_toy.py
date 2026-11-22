import torch
from PIL import Image
from arch.model import ad_model
import bitsandbytes as bnb


class AnomalyVLMTrainer:
    def __init__(
        self,
        model: ad_model,
        lr_vlm: float = 2e-4,
        lr_adapters: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.tokenizer = model.vlm_backbone.tokenizer

        # Ensure a pad token exists for batching later
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Separate parameter groups: lower LR for LoRA/VLM, higher LR for new adapters
        vlm_params = [p for p in model.vlm_backbone.parameters() if p.requires_grad]
        adapter_params = [
            p for p in (
                list(model.visual_residual_projector.parameters())
                + list(model.query_conditioned_text_adapter.parameters())
            )
            if p.requires_grad
        ]
        sam_params = list(model.sam_head.prompt_generator.parameters()) + list(model.sam_head.sam.mask_decoder.parameters())

        self.optimizer = bnb.optim.PagedAdamW8bit(
            [
                {"params": vlm_params, "lr": lr_vlm},
                {"params": adapter_params, "lr": lr_adapters},
                {"params": sam_params, "lr": lr_adapters},
            ],
            weight_decay=weight_decay,
        )

    def build_training_tokens(
        self,
        user_prompts: list[str],
        target_responses: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Formats chat prompt + target response into padded input_ids, attention_mask,
        and labels (with -100 masking the user prompt and padding tokens).
        """
        full_ids_list = []
        labels_list = []

        for prompt, target in zip(user_prompts, target_responses):
            messages = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Append EOS so the model learns when to stop generating
            target_str = target + self.tokenizer.eos_token

            prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            target_ids = self.tokenizer(target_str, add_special_tokens=False)["input_ids"]

            full_ids = prompt_ids + target_ids
            # Mask out the prompt with -100 so CE loss only supervises the target response
            labels = [-100] * len(prompt_ids) + target_ids

            full_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Right-pad to the longest sequence in the batch
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

    def train_step(
        self,
        query_imgs: list[Image.Image],
        reference_imgs: list[list[Image.Image]],
        categories: list[str],
        user_prompts: list[str],
        target_responses: list[str],
    ) -> float:
        self.model.train()
        # Keep frozen CLIP in eval mode so dropout/norms don't drift
        self.model.clip_model.eval()
        self.model.sam_head.sam.vision_encoder.eval()

        input_ids, attn_mask, labels = self.build_training_tokens(user_prompts, target_responses)

        self.optimizer.zero_grad()

        # BF16 autocast matches SmolVLMBackbone's compute_dtype and avoids needing GradScaler
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(
                query_img=query_imgs,
                reference_imgs=reference_imgs,
                category=categories,
                input_ids=input_ids,
                text_attention_mask=attn_mask,
                labels=labels,
            )
            vlm_out = output["vlm_out"]
            loss = vlm_out["loss"]

        loss.backward()

        # Clip gradients across all trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

        self.optimizer.step()
        return loss.item()


def main():
    device = "cuda"
    print("Initializing model...")
    model = ad_model(device=device)
    trainer = AnomalyVLMTrainer(model=model, lr_vlm=2e-4, lr_adapters=1e-3, device=device)

    # 1. Load toy example
    query_img = [Image.open("toy_dataset/inspinerator_cut.jpg")]
    reference_imgs = [[Image.open("toy_dataset/inspinerator_good.jpg")]]
    category = ["pcb"]

    user_prompt = ["Is there any anomaly or defect in this pcb? Describe it if present."]
    target_response = ["Anomalous. There is a cut defect on the surface. <SEG>"]

    # 2. Overfit loop
    num_steps = 150
    print(f"Starting {num_steps}-step overfit test...")
    for step in range(1, num_steps + 1):
        loss = trainer.train_step(
            query_imgs=query_img,
            reference_imgs=reference_imgs,
            categories=category,
            user_prompts=user_prompt,
            target_responses=target_response,
        )
        if step == 1 or step % 5 == 0:
            print(f"Step {step:02d}/{num_steps} | Loss: {loss:.4f}")

    # 3. Verify generation & <SEG> token capture
    print("\nRunning inference check...")
    tokenizer = model.vlm_backbone.tokenizer
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt[0]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

    texts, seg_hidden_states = model.generate(
        query_img=query_img,
        reference_imgs=reference_imgs,
        category=category,
        prompt_input_ids=inputs["input_ids"],
        prompt_attention_mask=inputs["attention_mask"],
        max_new_tokens=64,
    )

    print("Generated text:", texts[0])
    print(
        "Captured <SEG> hidden state:",
        None if seg_hidden_states is None else tuple(seg_hidden_states.shape),
    )


if __name__ == "__main__":
    main()