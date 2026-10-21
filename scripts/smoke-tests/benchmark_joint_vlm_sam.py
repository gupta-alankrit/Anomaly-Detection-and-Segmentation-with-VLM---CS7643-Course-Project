"""

Step 4: PREPARATION ONLY: test joint Qwen3-VL LoRA -> SEG -> MLP -> SAM training.

This smoke test uses the existing cashew sample and short temporary responses in
configs/preparation_joint_targets.json. A normal reference and query are the VLM
inputs. Teacher-forced responses end in <SEG>; the last-layer hidden state AT
that input-token position is projected to one learned 256-D sparse SAM prompt.
SAM's frozen query-image features and its trainable mask decoder produce logits.
Response-only CE + mask BCE + soft Dice train the connected path.

Trainable: language-attention LoRA, only the new token's tied embedding/output
row (PEFT Trainable Tokens), projection MLP, SAM mask decoder. Frozen: all other
Qwen parameters, SAM image encoder and prompt encoder. CLIP/composites and
free-running text/SEG generation are not part of this initial compute test.

SAM's Sam.forward() and predictor APIs disable gradients, so this script calls
the encoders/decoder directly. SAM preprocessing and postprocessing retain the
original RGB/mask geometry. Nonzero VisA region IDs become a binary union; normal
targets are all zero. The user prompt contains original images and the fixed
instruction only; response text is teacher-forced solely for supervised training.

The script fetches pinned upstream SAM source with its LICENSE into data/vendor/
and a single official SAM ViT-B checkpoint into data/checkpoints/. Existing Qwen
weights in data/hf_cache/ are reused. No package installation is performed.
Everything here is preparation/testing; later production datasets, experiments,
and code will have a separate root when the project is reorganized.

Outputs: results/joint_compute_benchmark/<timestamp>/report.json and before/after
teacher-forced mask previews. No trained model is saved. Default: 8 examples,
microbatch 1, 20 measured + 2 warm-up updates, BF16 Qwen, FP32 SAM/MLP, rank-8
LoRA, 256 visual-token cap per image, and checkpointing enabled. SAM image
features are computed online by default; --sam-features cached measures the
variant that precomputes frozen features. Both modes retain encoder weights.

Usage (Conda environment cs7643-project)
--------------------------------------
    python scripts/benchmark_joint_vlm_sam.py --self-test
    python scripts/benchmark_joint_vlm_sam.py
    python scripts/benchmark_joint_vlm_sam.py --steps 100 --sam-features cached

--self-test uses tiny RANDOM Qwen and SAM architectures on CPU with real sample
images/masks. It validates interfaces and gradients, not pretrained quality or
GPU feasibility. The real GPU run loads pretrained Qwen and SAM ViT-B.

Upstream SAM: https://github.com/facebookresearch/segment-anything
Pinned code is reused unchanged under its Apache-2.0 LICENSE, copied beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import random
import statistics
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import requests
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from data.vendor.segment_anything_dca509fe793f601edb92606367a655c15ac00fdf.segment_anything.utils.transforms import ResizeLongestSide
from data.vendor.segment_anything_dca509fe793f601edb92606367a655c15ac00fdf.segment_anything.modeling import (
    ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer)

from benchmark_vlm_lora import CACHE, INSTRUCTION, MANIFEST, MODEL_ID, MODEL_REVISION, ROOT


SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
SAM_VENDOR = ROOT / "data/vendor" / f"segment-anything-{SAM_COMMIT}"
SAM_CHECKPOINT = ROOT / "data/checkpoints/sam_vit_b_01ec64.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
TARGET_FILE = ROOT / "configs/preparation_joint_targets.json"
SEG = "<SEG>"


def download(url: str, destination: Path) -> None:
    """Stream an asset to an atomic .part file, reusing a completed local file.

    Arguments
    ----------
    - url: str = Pinned public source URL; HTTP errors stop the operation.
    - destination: Path = File to create, with its parent directory if missing.

    Returns
    -------
    - None. An interrupted .part file is not mistaken for a complete download.
      Content length is checked when supplied by the server.
    """
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        expected = int(response.headers.get("Content-Length", 0))
        written = 0
        with temporary.open("wb") as target, tqdm(total=expected or None, unit="B", unit_scale=True,
                                                  desc=destination.name) as progress:
            for chunk in response.iter_content(1024 * 1024):
                target.write(chunk)
                written += len(chunk)
                progress.update(len(chunk))
        if expected and written != expected:
            raise IOError(f"Incomplete download: {written} != {expected}")
    temporary.replace(destination)


def load_sam_source():
    """Fetch/import pinned upstream SAM code while retaining its original license.

    Returns
    -------
    - module = Upstream segment_anything package. Only Python package files and
      root LICENSE/README are extracted. No environment or site-packages edits.
    """
    marker = SAM_VENDOR / "UPSTREAM.json"
    if not marker.exists():
        archive_path = ROOT / "data/vendor" / f"sam-{SAM_COMMIT}.zip"
        download(f"https://codeload.github.com/facebookresearch/segment-anything/zip/{SAM_COMMIT}", archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                parts = Path(member.filename).parts[1:]
                if not parts or member.is_dir():
                    continue
                relative = Path(*parts)
                keep = (parts[0] == "segment_anything" and relative.suffix == ".py") or str(relative) in {"LICENSE", "README.md"}
                destination = (SAM_VENDOR / relative).resolve()
                if keep and destination.is_relative_to(SAM_VENDOR.resolve()):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(member))
        marker.write_text(json.dumps({"repository": "https://github.com/facebookresearch/segment-anything",
                                      "commit": SAM_COMMIT, "license": "Apache-2.0"}, indent=2) + "\n")
    sys.path.insert(0, str(SAM_VENDOR))
    module = importlib.import_module("segment_anything")
    if not Path(module.__file__).resolve().is_relative_to(SAM_VENDOR.resolve()):
        raise RuntimeError("A different SAM package was already imported in this process")
    return module


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for recording the actual checkpoint used."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """Read positive test budgets; measured steps exclude warm-up updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--visual-tokens", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sam-features", choices=("online", "cached"), default="online")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.visual_tokens, args.rank) < 1 or args.warmup < 0:
        parser.error("Steps, visual-token budget, and rank must be positive; warmup nonnegative")
    return args


def prepare_examples(processor, sam, seg_id: int, visual_tokens: int) -> list[dict]:
    """Prepare image-only inputs and separate text/mask supervision on CPU.

    Arguments
    ----------
    - processor: Qwen3VLProcessor = Tokenizer with the added SEG special token.
    - sam: Sam = Provides its image encoder size and original resize convention.
    - seg_id: int = Added token ID; gathered only in the assistant response.
    - visual_tokens: int = Per-image VLM cap; SAM has its own 1024px resize.

    Returns
    -------
    - list[dict] = CPU VLM tensors, SEG position, resized SAM query (0..255),
      original-size binary mask [1,1,H,W], response, and original RGB path.
      All prompt tokens have CE label -100. Every response has exactly one SEG.
    """
    # from segment_anything.utils.transforms import ResizeLongestSide

    manifest = json.loads(MANIFEST.read_text())
    targets = json.loads(TARGET_FILE.read_text())["targets"]
    rows = {row["source_id"]: row for row in manifest["rows"]}
    if not set(targets).issubset(rows):
        raise ValueError("Temporary target identifiers missing from the downloaded sample")
    processor.image_processor.size = {"shortest_edge": min(4096, visual_tokens * 1024),
                                      "longest_edge": visual_tokens * 1024}
    resize = ResizeLongestSide(sam.image_encoder.img_size)
    examples = []
    for name, response in targets.items():
        row = rows[name]
        if row["query_path"] in row["reference_paths"]:
            raise ValueError("Query/reference overlap")
        with Image.open(MANIFEST.parent / row["query_path"]) as source:
            query = source.convert("RGB")
        with Image.open(MANIFEST.parent / row["reference_paths"][0]) as source:
            reference = source.convert("RGB")
        prompt = [{"role": "user", "content": [{"type": "image", "image": reference},
                   {"type": "image", "image": query},
                   {"type": "text", "text": INSTRUCTION + " End the response with " + SEG + "."}]}]
        prefix = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        messages = prompt + [{"role": "assistant", "content": [{"type": "text", "text": response}]}]
        batch = dict(processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                  return_dict=True, return_tensors="pt"))
        boundary = prefix["input_ids"].shape[1]
        if not torch.equal(batch["input_ids"][:, :boundary], prefix["input_ids"]):
            raise ValueError("Assistant response boundary mismatch")
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, :boundary] = -100
        positions = ((batch["input_ids"] == seg_id) & (batch["labels"] != -100)).nonzero()
        if positions.shape != (1, 2):
            raise ValueError("Expected exactly one supervised SEG token")
        rgb = np.asarray(query)
        # Binary union of true mask region IDs. A normal example has an empty
        # target at exactly its original query resolution.
        if row["label"]:
            with Image.open(MANIFEST.parent / row["mask_path"]) as source:
                mask = np.asarray(source) > 0
            if mask.shape != rgb.shape[:2] or not np.any(mask):
                raise ValueError(f"Invalid mask geometry/content: {name}")
        else:
            mask = np.zeros(rgb.shape[:2], dtype=bool)
        resized = resize.apply_image(rgb)
        examples.append({"name": name, "label": row["label"], "vlm": batch,
                         "seg_position": int(positions[0, 1]), "response": response,
                         "sam_image": torch.as_tensor(resized.copy()).permute(2, 0, 1).unsqueeze(0).float(),
                         "input_size": resized.shape[:2], "original_size": rgb.shape[:2],
                         "mask": torch.from_numpy(mask.copy()).float()[None, None],
                         "query_path": str(MANIFEST.parent / row["query_path"])})
    return examples


def attach_seg_lora(model, seg_id: int, rank: int, checkpointing: bool):
    """Train language LoRA and only the SEG token's tied embedding/output row.

    Returns
    -------
    - PEFT model = Frozen pretrained vision/base weights with small adapters.
      Raises if the token's input/output trainable rows are not tied correctly.
    """
    if seg_id >= model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(seg_id + 1)
    targets = [name for name, module in model.named_modules() if ".language_model." in name
               and isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    if not targets:
        raise ValueError("Missing language attention LoRA targets")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=2 * rank,
                        lora_dropout=0.0, bias="none", target_modules=targets,
                        trainable_token_indices=[seg_id])
    model = get_peft_model(model, config)
    incoming = model.get_input_embeddings().token_adapter
    outgoing = model.get_output_embeddings().token_adapter
    if outgoing.tied_adapter is not incoming:
        raise RuntimeError("SEG input and output trainable rows are not tied")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "lora_" not in name and "trainable_tokens_delta" not in name:
            raise RuntimeError(f"Unexpected trainable Qwen parameter: {name}")
        if parameter.requires_grad and "trainable_tokens_delta" in name:
            # Keep the one learned token row in FP32, like the LoRA adapters,
            # while the frozen full embedding matrix retains its base precision.
            parameter.data = parameter.data.float()
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


class JointTest(nn.Module):
    """Connect teacher-forced SEG states to the original SAM mask decoder.

    Image encoders and the SAM prompt encoder are frozen. SAM's native positional
    encoding and no-mask dense prompt are retained; the MLP supplies one learned
    sparse prompt rather than a point/box. Mask losses retain the Qwen graph.
    """

    def __init__(self, vlm, sam):
        """Register the VLM, SAM, and FP32 hidden_size -> 256 -> 256 projection."""
        super().__init__()
        self.vlm, self.sam = vlm, sam
        self.projection = nn.Sequential(nn.Linear(vlm.config.text_config.hidden_size, 256), nn.ReLU(), nn.Linear(256, 256))
        self.sam.requires_grad_(False)
        self.sam.eval()
        self.sam.mask_decoder.requires_grad_(True)
        self.sam.mask_decoder.train()

    def encode_sam(self, example: dict) -> torch.Tensor:
        """Return frozen query features [1,256,h,w], preserving SAM preprocessing."""
        with torch.no_grad():
            image = example["sam_image"].to(self.sam.device)
            return self.sam.image_encoder(self.sam.preprocess(image))

    def forward(self, example: dict) -> dict:
        """Compute CE, original-frame mask BCE/Dice, and teacher-forced mask logits.

        Arguments
        ----------
        - example: dict = One prepare_examples() record, optionally with a frozen
          cached_sam tensor. Mask/image coordinate metadata stays separate.

        Returns
        -------
        - dict = Scalar text/bce/dice/mask/total losses and original-size logits.
          All three loss coefficients are 1 for this engineering smoke test.
        """
        device = next(self.vlm.parameters()).device
        # Compute frozen SAM features before retaining the VLM graph, limiting
        # transient encoder memory. Online mode repeats this computation per step.
        features = example["cached_sam"].to(device) if "cached_sam" in example else self.encode_sam(example)
        batch = {key: value.to(device) for key, value in example["vlm"].items()}
        output = self.vlm(**batch, output_hidden_states=True, use_cache=False)
        state = output.hidden_states[-1][:, example["seg_position"], :]
        sparse_prompt = self.projection(state.float()).unsqueeze(1)
        with torch.no_grad():
            _, dense_prompt = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
            positional = self.sam.prompt_encoder.get_dense_pe()
        low_res, _ = self.sam.mask_decoder(image_embeddings=features, image_pe=positional,
                                           sparse_prompt_embeddings=sparse_prompt,
                                           dense_prompt_embeddings=dense_prompt,
                                           multimask_output=False)
        logits = self.sam.postprocess_masks(low_res, example["input_size"], example["original_size"])
        target = example["mask"].to(device)
        if logits.shape != target.shape:
            raise ValueError("Postprocessed mask/target geometry differs")
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = logits.sigmoid()
        dice = 1 - (2 * (probability * target).sum() + 1e-6) / (probability.sum() + target.sum() + 1e-6)
        mask_loss = bce + dice
        return {"total": output.loss.float() + mask_loss, "text": output.loss,
                "bce": bce, "dice": dice, "mask": mask_loss, "logits": logits}


def gradient_audit(model: JointTest, example: dict) -> dict:
    """Verify mask-only gradients reach LoRA, SEG row, MLP, and the mask decoder.

    Returns
    -------
    - dict[str, float] = Nonzero finite gradient norm for one used parameter per
      component. This uses mask loss alone, so CE cannot conceal a broken bridge.
      No optimizer update occurs; graph/tensors are released on return.
    """
    vlm_params = dict(model.vlm.named_parameters())
    representatives = {
        "vlm_lora_from_mask": next(p for n, p in vlm_params.items() if "lora_B" in n),
        "seg_row_from_mask": next(p for n, p in vlm_params.items() if "trainable_tokens_delta" in n and p.requires_grad),
        "projection_from_mask": model.projection[-1].weight,
        "sam_decoder_from_mask": model.sam.mask_decoder.output_hypernetworks_mlps[0].layers[-1].weight,
    }
    result = model(example)
    gradients = torch.autograd.grad(result["mask"], list(representatives.values()), allow_unused=True)
    norms = {}
    for (name, _), gradient in zip(representatives.items(), gradients):
        if gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
            raise RuntimeError(f"Mask-gradient bridge failed: {name}")
        norms[name] = float(gradient.float().norm())
    return norms


def build_test_models(processor, seg_id: int, args: argparse.Namespace, device: str) -> JointTest:
    """Construct pretrained GPU models or explicitly random tiny CPU models.

    Arguments
    ----------
    - processor: Qwen3VLProcessor = Includes the added SEG token in its vocabulary.
    - seg_id: int = Shared input/output token row adapted through PEFT.
    - args: Namespace = Rank, checkpointing, and self-test settings.
    - device: str = 'cuda' for the real run or 'cpu' for integration verification.

    Returns
    -------
    - JointTest = Qwen (BF16 on GPU), projection and SAM (FP32). Downloads the
      official SAM ViT-B weights only for a real run. Tiny weights are random.
    """
    source = load_sam_source()
    if args.self_test:
        # from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

        qwen_config = Qwen3VLConfig(
            text_config={"vocab_size": len(processor.tokenizer), "hidden_size": 64,
                         "intermediate_size": 128, "num_hidden_layers": 2,
                         "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                         "max_position_embeddings": 4096, "tie_word_embeddings": True,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [4, 6, 6], "mrope_interleaved": True}},
            vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 2, "out_hidden_size": 64, "patch_size": 16,
                           "spatial_merge_size": 2, "temporal_patch_size": 2,
                           "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1]},
            tie_word_embeddings=True,
        )
        qwen = Qwen3VLForConditionalGeneration(qwen_config)
        sam = Sam(
            image_encoder=ImageEncoderViT(img_size=64, patch_size=16, embed_dim=32, depth=2,
                                          num_heads=2, out_chans=256, use_rel_pos=True,
                                          global_attn_indexes=(1,), window_size=4),
            prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(4, 4),
                                         input_image_size=(64, 64), mask_in_chans=16),
            mask_decoder=MaskDecoder(transformer_dim=256,
                                     transformer=TwoWayTransformer(depth=2, embedding_dim=256,
                                                                  mlp_dim=512, num_heads=8),
                                     num_multimask_outputs=3),
            pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
        )
    else:
        print("Reusing Qwen weights; loading pretrained Qwen3-VL-2B", flush=True)
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE, dtype=torch.bfloat16,
            attn_implementation="sdpa")
        download(SAM_URL, SAM_CHECKPOINT)
        if file_sha256(SAM_CHECKPOINT) != SAM_SHA256:
            raise IOError("SAM checkpoint digest differs from the recorded official ViT-B asset")
        sam = source.sam_model_registry["vit_b"](checkpoint=None)
        sam.load_state_dict(torch.load(SAM_CHECKPOINT, map_location="cpu", weights_only=True))
    qwen = attach_seg_lora(qwen, seg_id, 2 if args.self_test else args.rank, args.gradient_checkpointing)
    return JointTest(qwen, sam).to(device)


def save_preview(model: JointTest, examples: list[dict], destination: Path) -> None:
    """Save teacher-forced query/ground-truth/predicted-probability panels.

    Arguments
    ----------
    - model: JointTest = Current weights, evaluated under no_grad without updates.
    - examples: list[dict] = Existing training-test examples; this is not held-out evaluation.
    - destination: Path = PNG output. Prediction colors always span [0,1].

    Returns
    -------
    - None. Decoder inputs use target SEG states; these panels do not test
      whether the VLM generates SEG or a correct description without forcing.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/mplconfig"))
    matplotlib.use("Agg")

    figure, axes = plt.subplots(len(examples), 3, figsize=(10, 2.8 * len(examples)), squeeze=False)
    with torch.no_grad():
        for row, example in enumerate(examples):
            result = model(example)
            with Image.open(example["query_path"]) as rgb:
                axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(example["name"] + (" (anomaly)" if example["label"] else " (normal)"))
            axes[row, 1].imshow(example["mask"][0, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("True binary mask")
            axes[row, 2].imshow(result["logits"].sigmoid()[0, 0].cpu(), cmap="magma", vmin=0, vmax=1)
            axes[row, 2].set_title("Teacher-forced prediction [0,1]")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def training_update(model: JointTest, example: dict, optimizer) -> dict:
    """Backpropagate CE+BCE+Dice and update LoRA/token/MLP/decoder parameters.

    Returns
    -------
    - dict[str,float] = Individual losses and total gradient norm. One example
      equals one optimizer update. Loss weights are all 1 for this test only.
    """
    optimizer.zero_grad(set_to_none=True)
    result = model(example)
    if not torch.isfinite(result["total"]):
        raise FloatingPointError("Nonfinite joint loss")
    result["total"].backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError("Missing or nonfinite joint gradients")
    optimizer.step()
    return {**{key: float(result[key].detach()) for key in ("total", "text", "bce", "dice")},
            "gradient_norm": float(norm)}


def main() -> None:
    """Run the preparation test, checking gradient connectivity before timing.

    A real GPU run records hardware/model/source identity and teacher-forced
    previews. A tiny CPU run checks interfaces and two optimizer updates without
    pretrained checkpoints. CUDA is checked before a real run downloads assets.
    """
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.self_test and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable here; run from the local cs7643-project terminal that passed the GPU benchmark")
    if args.self_test:
        torch.set_num_threads(2)
    elif not torch.cuda.is_bf16_supported():
        raise SystemExit("The real run expects a BF16-capable GPU")
    device = "cpu" if args.self_test else "cuda"
    initial_memory = None if args.self_test else torch.cuda.mem_get_info()
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE,
                                 allow_patterns=["*.json", "*.txt", "*.jinja"])
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [SEG]})
    seg_id = processor.tokenizer.convert_tokens_to_ids(SEG)
    model = build_test_models(processor, seg_id, args, device)
    examples = prepare_examples(processor, model.sam, seg_id, 4 if args.self_test else args.visual_tokens)
    if args.self_test:
        examples = [next(item for item in examples if item["label"]),
                    next(item for item in examples if not item["label"])]
    random.shuffle(examples)
    if args.sam_features == "cached":
        print("Precomputing frozen SAM query features", flush=True)
        for example in examples:
            example["cached_sam"] = model.encode_sam(example).cpu()
    audit = gradient_audit(model, next(item for item in examples if item["label"]))
    print("Mask-only gradient audit:", json.dumps(audit, indent=2), flush=True)
    counts = {"vlm_lora_and_seg": sum(p.numel() for p in model.vlm.parameters() if p.requires_grad),
              "projection": sum(p.numel() for p in model.projection.parameters()),
              "sam_mask_decoder": sum(p.numel() for p in model.sam.mask_decoder.parameters() if p.requires_grad)}
    print("Trainable parameter counts:", counts, flush=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.vlm.parameters() if p.requires_grad], "lr": 2e-5},
        {"params": model.projection.parameters(), "lr": 1e-4},
        {"params": [p for p in model.sam.mask_decoder.parameters() if p.requires_grad], "lr": 1e-4},
    ])
    if args.self_test:
        before = model.projection[-1].weight.detach().clone()
        for example in examples:
            print("Tiny random CPU update:", training_update(model, example, optimizer), flush=True)
        assert not torch.equal(before, model.projection[-1].weight.detach())
        assert all(p.grad is None for p in model.sam.image_encoder.parameters())
        assert all(p.grad is None for p in model.sam.prompt_encoder.parameters())
        assert all(p.grad is None for n, p in model.vlm.named_parameters() if ".visual." in n)
        print("PASS: random tiny CPU integration; pretrained GPU feasibility remains unmeasured for this joint path")
        return

    output = ROOT / "results/joint_compute_benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report = {"purpose": "preparation_compute_test_only", "model": MODEL_ID,
              "qwen_revision": MODEL_REVISION, "sam_code_commit": SAM_COMMIT,
              "sam_checkpoint_sha256": file_sha256(SAM_CHECKPOINT),
              "sam_checkpoint_url": SAM_URL, "settings": vars(args),
              "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
              "total_gib": initial_memory[1] / 2**30, "initial_free_gib": initial_memory[0] / 2**30,
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
              "trainable_counts": counts, "mask_only_gradient_norms": audit,
              "precision": {"qwen": "bfloat16", "sam_and_projection": "float32"},
              "sam_features": args.sam_features, "microbatch": 1,
              "loss_weights": {"text": 1, "bce": 1, "dice": 1},
              "scope": "teacher-forced joint training; intact reference; no CLIP/composite or free-running generation",
              "data_revision": json.loads(MANIFEST.read_text())["revision"],
              "target_file": str(TARGET_FILE),
              "examples": [{"name": item["name"], "label": item["label"], "response": item["response"],
                            "original_size": list(item["original_size"]), "sam_input_size": list(item["input_size"]),
                            "vlm_sequence_tokens": item["vlm"]["input_ids"].shape[1],
                            "vlm_image_grid_thw": item["vlm"]["image_grid_thw"].tolist()}
                           for item in examples], "updates": [], "status": "running"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    save_preview(model, examples, output / "before.png")
    report["setup_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["setup_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(args.warmup + args.steps):
            example = examples[step % len(examples)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            losses = training_update(model, example, optimizer)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            measured = step >= args.warmup
            row = {"name": example["name"], "label": example["label"],
                   "warmup": not measured, "seconds": elapsed, **losses}
            report["updates"].append(row)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{'measure' if measured else 'warmup'} {step+1} {example['name']}: "
                  f"CE={losses['text']:.4f} BCE={losses['bce']:.4f} Dice={losses['dice']:.4f} "
                  f"total={losses['total']:.4f} seconds/example={elapsed:.3f}", flush=True)
        measured_updates = [row for row in report["updates"] if not row["warmup"]]
        report.update({"status": "ok", "mean_seconds_per_example": statistics.mean(row["seconds"] for row in measured_updates),
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        report["overall_peak_allocated_gib"] = max(report["setup_peak_allocated_gib"], report["peak_allocated_gib"])
        report["overall_peak_reserved_gib"] = max(report["setup_peak_reserved_gib"], report["peak_reserved_gib"])
    except torch.cuda.OutOfMemoryError as error:
        report.update({"status": "out_of_memory", "error": str(error)})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    optimizer.zero_grad(set_to_none=True)
    save_preview(model, examples, output / "after.png")
    print(json.dumps({key: report[key] for key in ("status", "mean_seconds_per_example", "peak_allocated_gib", "peak_reserved_gib")}, indent=2))
    print(f"Report: {report_path}", flush=True)
    print(f"Teacher-forced previews: {output / 'before.png'} and {output / 'after.png'}", flush=True)


if __name__ == "__main__":
    main()
