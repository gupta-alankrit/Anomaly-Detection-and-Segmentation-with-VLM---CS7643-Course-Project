"""
Step 3: Measure two-image Qwen3-VL LoRA training cost on the small VisA sample.

The first image is an intact healthy reference; the second is the query. Only
language-attention LoRA matrices are trained. The pretrained visual encoder,
visual merger, embeddings, language base weights, and output head stay frozen.
This resource diagnostic precedes the composite builder and SAM integration.
It measures forward, backward, gradient accumulation, clipping, and AdamW steps.

Targets are a decision followed by a bounded excerpt of the existing generated
narrative. They exercise a realistic response-token budget but are NOT curated
short captions. The run produces resource measurements, not an anomaly-quality
evaluation or a model checkpoint. The SEG token, its trainable embedding, MLP,
and SAM decoder are not yet included; their joint cost must be measured later.

Defaults: BF16, SDPA attention, rank-8 LoRA, microbatch 1, accumulation 4,
256 and 512 maximum visual tokens per image, 2 warm-up + 10 measured optimizer
steps per setting. Images are processed before timing; measured time includes
CPU-to-GPU tensor transfer, but excludes image decoding and tokenization.

Outputs
-------
- results/compute_benchmark/<timestamp>/report.json: versions, freeze policy,
  actual input/response token counts, per-step times/losses, peak allocated and
  reserved VRAM, and OOM results. Weights are cached under data/hf_cache/.
- With --self-test: a tiny RANDOM Qwen3-VL architecture runs on CPU with the real
  tokenizer, real images, two-image tensors, and LoRA. This tests integration
  only; it gives no information about pretrained accuracy or GPU feasibility.

Usage (Conda environment cs7643-project)
--------------------------------------
    python scripts/benchmark_vlm_lora.py --self-test
    python scripts/benchmark_vlm_lora.py
    python scripts/benchmark_vlm_lora.py --visual-tokens 256 --steps 3 --warmup 1

Uses installed Transformers/PEFT APIs; no external training code is copied.
References: https://github.com/QwenLM/Qwen3-VL and
https://huggingface.co/docs/peft/en/package_reference/lora
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MANIFEST = ROOT / "data/mmrad_visa_sample/manifest.json"
CACHE = ROOT / "data/hf_cache"
INSTRUCTION = (
    "The first image is a healthy reference. Inspect the second image, which is "
    "the query. State whether it is normal or anomalous and describe any visible "
    "defect and its location."
)


def parse_args() -> argparse.Namespace:
    """Read resource-test settings and validate positive counts.

    Returns
    -------
    - argparse.Namespace = CLI settings. --steps excludes warm-up steps and
      counts optimizer updates; effective batch = 1 * --accumulation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Let, the original image size is HxW and the `visual-tokens` (budget cap) is N, the actual number of tokens
    # created, `N_a = floor(sqrt(HN/W)) x floor(sqrt(WN/H)) <= N` such that the image can be divided into `N_a`
    # number of 16x16 patches in groups of 2x2 (equivalently, 32x32 pixel patches). Hence, the downsampled image 
    # size is `(32 floor(sqrt(HN/W))) x (32 floor(sqrt(WN/H)))` and bicubic sampling compresses the original image 
    # to the downsampled image. 
    parser.add_argument("--visual-tokens", type=int, nargs="+", default=[256, 512],
        help="One or more positive visual-token (budget) caps per image so that 32x32 pixel patches can be made.",)
    
    parser.add_argument("--steps", type=int, default=10,
        help="Number of measured optimizer updates per `visual-tokens` setting, excluding warm-up updates.",)
    
    parser.add_argument("--warmup", type=int, default=2,
        help="Number of optimizer updates before timing measurements.",)
    
    parser.add_argument("--accumulation", type=int, default=4,
        help="Number of single-example forward/backward passes accumulated before each optimizer update." \
        "The only difference with the batch size is that in batch SGD, batch size number of data samples are" \
        "computed in parallel for weight update, while here `accumulation` number of data samples are processed" \
        "sequentially until the weight updates. This takes less GPU memory.",)
    
    parser.add_argument("--response-tokens", type=int, default=128,
        help="Maximum supervised response-text tokens per example for training, excluding chat terminators.",)
    
    parser.add_argument("--examples", type=int, default=4,
        help="Number of distinct query/reference pairs to cycle through, balanced normal/anomalous; must be even, " \
        "at least two, and available locally. This is similar to total number of data samples used in an epoch.",)
    
    parser.add_argument("--rank", type=int, default=8,
        help="Positive LoRA adapter rank; larger values add more trainable parameters and gives the adaptation more " \
        "capacity, but does not guarantee better results.",)
    
    parser.add_argument("--self-test", action="store_true",
        help="Run a tiny randomly initialized model on CPU with two local examples instead of the pretrained GPU benchmark.",)
    
    parser.add_argument("--learning-rate", type=float, default=2e-5,)

    parser.add_argument("--seed", type=int, default=42,)
    
    args = parser.parse_args()

    if min(args.steps, args.accumulation, args.response_tokens, args.rank, *args.visual_tokens) < 1:
        parser.error("Steps, accumulation, rank, and token budgets must be positive")
    if args.warmup < 0 or args.examples < 2 or args.examples % 2 or args.learning_rate <= 0:
        parser.error("Warmup must be nonnegative, examples positive/even, and learning rate positive")
    return args


def sample_rows(count: int, seed: int) -> tuple[Path, list[dict]]:
    """Select equal numbers of normal and anomalous development queries.

    Arguments
    ----------
    - count: int = Even number of distinct sample images to cycle through.
    - seed : int = Controls the order of the selected examples.

    Returns
    -------
    - (sample_root, rows): tuple[Path, list[dict]] = Root and manifest entries.
      Queries use the first candidate healthy reference. This does not establish
      a train/evaluation split; these cashew images are development data only.
    """
    # Load the information of the downloaded sample dataset.
    data = json.loads(MANIFEST.read_text())

    # Select (count // 2) normal and equal anomalous query images, then randomly shuffle all of them.
    selected = []
    for label in (0, 1):
        candidates = [row for row in data["rows"] if row["label"] == label]
        if len(candidates) < count // 2:
            raise ValueError("Requested more examples than the local sample provides")
        selected.extend(candidates[: count // 2])
    random.Random(seed).shuffle(selected)

    # Ensure that we have a total of `count` number of query (equal normal and anomalous) images.
    if len({row["query_path"] for row in selected}) != count:
        raise ValueError("Repeated query identities in benchmark sample")

    # Ensure no overlap between the training images and selected queries. 
    # And, a narrative text is present for all of them.
    for row in selected:
        if row["query_path"] in row["reference_paths"]:
            raise ValueError("Query/reference filename overlap")
        if not row.get("raw_narrative_for_inspection_only"):
            raise ValueError("Narratives missing; rerun inspect_visa_subset.py first")
        
    return MANIFEST.parent, selected


def prepare_batches(processor, sample_root: Path, rows: list[dict], visual_tokens: int,
                    response_tokens: int) -> tuple[list[dict], list[dict]]:
    """Prepare CPU tensors for response-only CE with two original RGB images.

    Arguments
    ----------
    - processor      : Qwen3VLProcessor = Official processor/tokenizer.
    - sample_root    : Path       = Root for manifest-relative image paths.
    - rows           : list[dict] = Selected sample records. Annotations are targets only.
    - visual_tokens  : int        = Upper budget per image, not guaranteed exact count.
    - response_tokens: int        = Maximum target-content tokens before chat terminator.

    Returns
    -------
    - (batches, metadata): CPU tensor dictionaries and actual sequence/grid counts.
      labels is [1, sequence_length]; every input/instruction/image token has
      label -100. Each batch has exactly one query and one healthy image.
    """
    # Qwen3-VL has 16px patches merged in 2x2 groups: one visual token per 32x32 image pixels. 
    # Pixel caps preserve aspect ratio and round-down (floor) to the grid.
    processor.image_processor.size = {
        "shortest_edge": min(64 * 64, visual_tokens * 32 * 32), # min. number of pixels.
        "longest_edge": visual_tokens * 32 * 32,                # max. number of pixels.
    }

    batches, metadata = [], []
    for row in rows:
        # Open the first healthy image as the `reference` image.
        with Image.open(sample_root / row["reference_paths"][0]) as source:
            reference = source.convert("RGB")

        # Open the `query` image.
        with Image.open(sample_root / row["query_path"]) as source:
            query = source.convert("RGB")

        # Build the user `prompt` by providing the healthy `reference` image, `query` image, and 
        # text instructions of the task.
        prompt = [{"role": "user", 
                   "content": [
                       {"type": "image", "image": reference}, 
                       {"type": "image", "image": query}, 
                       {"type": "text", "text": INSTRUCTION},
                ]}]

        # Build the ground-truth `target` response.
        decision = "anomalous" if row["label"] else "normal"
        target = f"Decision: {decision}. " + row["raw_narrative_for_inspection_only"]

        # Crop (encode in token-space, crop, decode in text-space) the `target` text-response.
        token_ids = processor.tokenizer.encode(target, add_special_tokens=False)[:response_tokens]
        target = processor.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Create the `complete` input as the user `propmt` + the assistant `target` response.
        complete = prompt + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]

        # Process -- text to tensor of token ids, image to pixel tensor -- the user `propmt` and `complete` input. 
        prefix = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt")
        batch = dict(processor.apply_chat_template(complete, tokenize=True, add_generation_prompt=False,
                                                  return_dict=True, return_tensors="pt"))
        prompt_length = prefix["input_ids"].shape[1]

        # Assert the exact token boundary. A silent mismatch would train on instruction/image positions 
        # or accidentally mask target tokens.
        if not torch.equal(batch["input_ids"][:, :prompt_length], prefix["input_ids"]):
            raise ValueError("Prompt is not an exact prefix of the complete training conversation")

        # Copy all the (text) token ids of the `complete` input as its `labels` and mark -100 for the token ids of 
        # `INSTRUCTIONS` so that the loss function ignores this text and only the assitant's answer is suprevised.
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, :prompt_length] = -100

        # Assert if there is at least one token whose label is not -100 so that the token can be used as the
        # ground-truth to supervise the VLM's answer.
        if not torch.any(batch["labels"] != -100):
            raise ValueError("No supervised response tokens")

        # Assert there are only two images: one `reference` and one `query`.
        if batch["image_grid_thw"].shape != (2, 3):
            raise ValueError("Expected exactly two processed images")

        # Record temporal (1 for image, >1 for video) and spatial (height, width) patch counts as a Python list
        # of lists for each image. 
        grids = batch["image_grid_thw"].tolist()

        # Record the metadata for each reference-query image pairs.
        metadata.append({
            "source_id": row["source_id"], 
            "label": row["label"],
            "query_path": row["query_path"], 
            "reference_path": row["reference_paths"][0],
            "total_tokens": batch["input_ids"].shape[1], 
            "prompt_tokens": prompt_length,
            "supervised_tokens": int((batch["labels"] != -100).sum()),
            "image_grid_thw": grids,
            "visual_tokens_per_image": [t * h * w // 4 for t, h, w in grids]
        })
        batches.append(batch)
    return batches, metadata


def attach_lora(model, rank: int):
    """Add LoRA only to language-attention query/key/value/output projections.

    Arguments
    ----------
    - model: Qwen3VLForConditionalGeneration = Loaded base model.
    - rank: int = Adapter rank; alpha is twice the rank and dropout is zero.

    Returns
    -------
    - (model, targets): PEFT model and exact fully qualified module names.
      Raises if any parameter outside LoRA is left trainable.
    """
    # Create a list of language attention module names of the selected VLM and raise an error if the list is empty.
    targets = [name for name, module in model.named_modules()
               if ".language_model." in name and isinstance(module, torch.nn.Linear)
               and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    if not targets:
        raise ValueError("Could not locate Qwen3-VL language attention modules")

    # Define configurations for parameter-efficient fine-tuning (PEFT), in this case LoRA (Low-Rank Adaptation) .
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=2 * rank,
                        lora_dropout=0.0, bias="none", target_modules=targets)

    # Wraps the selected VLM with LoRA such that only the `config.target_modules` weights are updated. Freesing 
    # the weights of all the other layers prevents catastrophic forgetting and the pre-trained VLM is fine-tuned
    # by inserting lightweight tensors such as `lora_A`, `lora_B` into specified target modules.
    model = get_peft_model(model, config)

    # Ensure that we only update the parameters with `lora_` in the name.
    if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise ValueError("Unexpected trainable base, visual, merger, or embedding parameters")

    # Disable the key/value caching during training. This is useful in inference to speed up the next token prediction
    # as it saves the past generated tokens' key/value, but all the tokens are predicted in parallel during training
    # with a causal attention-mask and loss is computed against the given ground-truth tokens. Hence, saving the cache
    # during training would just consume the memory and not even be used, so we disable it.
    model.config.use_cache = False

    # Gradient checkpointing: save less intermediate values during the forwards pass and recomputes the missing values
    # during backward pass. This is a trade-off between less memory and extra computation time.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load the model in the "training" mode.
    model.train()
    return model, targets


def optimizer_update(model, batches: list[dict], optimizer, device: str,
                     accumulation: int, offset: int) -> tuple[float, bool]:
    """Perform one real AdamW update after accumulating response CE gradients.

    Arguments
    ----------
    - model: PEFT model = Training-mode VLM with frozen pretrained components.
    - batches: list[dict] = CPU examples, including response-only labels.
    - optimizer: torch.optim.Optimizer = Contains trainable LoRA parameters only.
    - device: str = 'cuda' for the benchmark or 'cpu' for the tiny integration test.
    - accumulation: int = Number of microbatches (one image pair each) per update.
    - offset: int = Starting index, cycling through the finite example list.

    Returns
    -------
    - (mean_loss, finite_nonzero_gradient): tuple[float, bool]. Loss is mean CE
      across microbatches. Clip norm is checked before the optimizer step.
    """
    # Initialize the gradients to zero.
    optimizer.zero_grad(set_to_none=True)

    losses = []
    for index in range(accumulation):
        # Load the `batch` to the `device`.
        batch = {key: value.to(device) for key, value in batches[(offset + index) % len(batches)].items()}

        # Compute the loss for the whole microbatch and asset that it is finite.
        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite language loss")
        
        # Backpropogate the average loss per microbatch and record it (without its computation graph).
        (loss / accumulation).backward()
        losses.append(float(loss.detach()))
        
        del batch, loss # Delete the variables.

    # Find the list of trainable parameters.
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    # Gradient clipping regularization.
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    good_gradient = bool(torch.isfinite(norm) and norm > 0)
    if not good_gradient:
        raise FloatingPointError("LoRA gradients are nonfinite or all zero")

    optimizer.step()    # Update the weights for this step.
    return statistics.mean(losses), good_gradient


def tiny_cpu_test(processor, sample_root: Path, rows: list[dict]) -> dict:
    """Test two-image response masking and real LoRA updates with random weights.

    Arguments
    ----------
    - processor  : Qwen3VLProcessor = Official tokenizer and image preprocessing.
    - sample_root: Path             = Root for the existing real VisA images.
    - rows       : list[dict]       = One normal and one anomalous development query.

    Returns
    -------
    - dict = Test identity, losses, and processor token counts. Architecture is
      deliberately tiny, and no pretrained weights or GPU claims are involved.
    """
    # Set the number of CPU threads used for this taks in parallel.
    torch.set_num_threads(2)

    # Set the configuration of the Qwen3-VL model.
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": len(processor.tokenizer), 
            "hidden_size": 64,
            "intermediate_size": 128, 
            "num_hidden_layers": 2,
            "num_attention_heads": 2, 
            "num_key_value_heads": 1, 
            "head_dim": 32,
            "max_position_embeddings": 4096, 
            "tie_word_embeddings": True,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 6, 6], "mrope_interleaved": True}
        },
        vision_config={
            "depth": 2, 
            "hidden_size": 32, 
            "intermediate_size": 64,
            "num_heads": 2, 
            "out_hidden_size": 64, 
            "patch_size": 16,
            "spatial_merge_size": 2, 
            "temporal_patch_size": 2,
            "num_position_embeddings": 16, 
            "deepstack_visual_indexes": [0, 1]
        },
        tie_word_embeddings=True,
    )

    # Puts the VLM in the "training" mode with the defined LoRA configurations and rank.
    model, _ = attach_lora(Qwen3VLForConditionalGeneration(config), rank=2)

    # Create reference-query image `batches` and record the `metadata` per batch.
    batches, metadata = prepare_batches(processor, sample_root, rows[:2], 4, 24)

    # Instantiate the AdamW optimizer.
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)

    # Save the model's trainable weights `before` training.
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

    # Run the AdamW optimizer, update the weights, and record the mean loss for each of the two epochs.
    losses = [optimizer_update(model, batches, optimizer, "cpu", 2, 0)[0] for _ in range(2)]

    # Assert the model weights are changed after running the opimizer and raise error an error if they don't.
    changed = any(not torch.equal(before[name], parameter.detach())
                  for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert changed, "Optimizer did not change any LoRA parameters"

    # Also ensure that VLM's vision parameters (frozen) are not updated.
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if ".visual." in name)

    return {
        "test": "random_tiny_cpu_integration_only", 
        "losses": losses,
        "adapter_weights_changed": changed, 
        "frozen_visual_gradients_absent": True, 
        "samples": metadata
    }


def run_gpu_setting(model, batches: list[dict], args: argparse.Namespace) -> dict:
    """Measure complete optimizer updates after warm-up on the current GPU.

    Arguments
    ----------
    - model: PEFT model = CUDA-resident VLM in training mode.
    - batches: list[dict] = Prepared CPU examples from prepare_batches().
    - args: argparse.Namespace = Warm-up, measured step count, accumulation,
      and learning-rate settings from parse_args().

    Returns
    -------
    - result: dict = Per-update time/loss, examples per second, and peak allocated and
      reserved GiB. Timing excludes warm-up; memory peaks include warm-up.
      Both reported VRAM values belong to this PyTorch process, not other apps.
    """
    # Get the list of trainable parameters.
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate) # Instantiate the AdamW optimizer.
    torch.cuda.synchronize() # Let the GPU finish queued tasks before CPU going to the next line execution.
    torch.cuda.reset_peak_memory_stats() # Reset the peak GPU memory statistics.
    
    updates = []
    for step in range(args.warmup + args.steps):
        # Run one optimizer update and record the execution time.
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss, _ = optimizer_update(model, batches, optimizer, "cuda", args.accumulation,
                                   step * args.accumulation) # Run one optimizer update.
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Check if the current step is a warmup or measurable and record the loss and execution time.
        measured = step >= args.warmup
        print(f"{'measure' if measured else 'warmup'} {step + 1}: loss={loss:.4f}, "
              f"seconds/update={elapsed:.3f}", flush=True)
        if measured:
            updates.append({"seconds": elapsed, "loss": loss})

    # Compute the mean execution time for all optimizer updates (steps).
    mean_time = statistics.mean(update["seconds"] for update in updates)

    # Record the result for the GPU test.
    result = {
        "status": "ok", 
        "updates": updates, 
        "mean_seconds_per_optimizer_update": mean_time,
        "median_seconds_per_optimizer_update": statistics.median(update["seconds"] for update in updates),
        "examples_per_second": args.accumulation / mean_time,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
    }
    optimizer.zero_grad(set_to_none=True)   # Set the gradients to zero after completing all updaets.
    return result


def main() -> None:
    """Run the CPU integration check or write a real GPU resource report.

    Downloads tokenizer/config files as needed; GPU mode also downloads the one
    pinned pretrained checkpoint. GPU access is checked before any download.
    Every setting is written to the JSON report immediately after completion.
    """
    # Parse the script arguments and check the GPU availability.
    args = parse_args()
    torch.manual_seed(args.seed)
    if not args.self_test and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable here.")

    # Select two or "args.examples" number of query (half normal, half anomalous) images (`rows`) and their
    # path (`sample_root`) from the downloaded dataset.
    sample_root, rows = sample_rows(2 if args.self_test else args.examples, args.seed)

    # Find or download the necessary files, such as tokenizer settings, image-processor settings, and chat-template,
    # in "CACHE" that are necessary to run the pinned model. This filtered download excludes the model weights.
    processor_snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE,
                                           allow_patterns=["*.json", "*.txt", "*.jinja"])

    # Build an AutoProcessor() object that loads the necessary files, such as tokenizer settings, image-processor settings, 
    # and chat-template, which were found or downloaded from the previous line. This `processor` (the AutoProcessor() object)
    # is later used in prepare_batches() to preprocess and load the images, labels, or model weights in tensor-format.
    processor = AutoProcessor.from_pretrained(processor_snapshot, local_files_only=True)

    # Run a tiny test on CPU, if asked.
    if args.self_test:
        print(json.dumps(tiny_cpu_test(processor, sample_root, rows), indent=2))
        return

    # Ensures if our GPU has the BF16 data type. BF16: 16-bit float with half the memory of FP32 but retains the same
    # numerical range. Supported GPUs can compute with it efficiently.
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This initial benchmark expects a BF16-capable GPU")

    # Create the output directory, if not already been made.
    output = ROOT / "results/compute_benchmark" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    # Record the global free and total GPU memory in bytes.
    free, total = torch.cuda.mem_get_info()

    # Create a report and write it in a JSON file.
    report = {
        "model"                 : MODEL_ID, 
        "revision"              : MODEL_REVISION, 
        "settings"              : vars(args),
        "manifest"              : str(MANIFEST), 
        "data_revision"         : json.loads(MANIFEST.read_text())["revision"],
        "gpu"                   : torch.cuda.get_device_name(), 
        "total_gib"             : total / 2**30, # Memory in GiBs.
        "initial_free_gib"      : free / 2**30,  # Memory in GiBs.
        "python"                : sys.version,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "torchvision")},
        "cuda_runtime"          : torch.version.cuda, 
        "dtype"                 : "bfloat16", 
        "attention"             : "sdpa",
        "gradient_checkpointing": True, 
        "microbatch"            : 1, 
        "results"               : [],
        "scope"                 : "VLM LoRA only; intact reference + query; no CLIP, SEG, projection, or SAM",
        "target_policy"         : "Decision plus bounded uncurated source narrative; resource measurement only"
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Loading {MODEL_ID}; cache: {CACHE}", flush=True)

    # Load the pre-trained model (weights) to the GPU.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=CACHE, dtype=torch.bfloat16, attn_implementation="sdpa",).to("cuda")

    # Puts the VLM in the "training" mode with the given LoRA rank and add some more model information in the `report`.
    model, targets = attach_lora(model, args.rank)
    report["lora_target_modules"] = targets
    report["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    
    # Save only the small trainable LoRA weight tensors on CPU before any fine-tuning of the VLM. 
    # This snapshot is restored for each `visual-tokens` budget.
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    print(f"Trainable LoRA parameters: {report['trainable_parameters']:,}", flush=True)

    for budget in args.visual_tokens:
        # Restore the initial LoRA weights so this setting starts independently
        # of updates made while measuring earlier token budgets.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(initial[name].to(parameter.device))
        print(f"Preparing visual-token cap {budget} per image", flush=True)

        # Create reference-query image `batches` and record the `metadata` per batch.
        batches, metadata = prepare_batches(processor, sample_root, rows, budget, args.response_tokens)

        # Get the result of fine-tuning the VLM on GPU.
        try:
            result = run_gpu_setting(model, batches, args)
        except torch.cuda.OutOfMemoryError as error:
            result = {
                "status": "out_of_memory", 
                "error": str(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30
            }

        # Update the `report` with the GPU test `result` and write it in a JSON file.
        result.update({"visual_token_cap_per_image": budget, "samples": metadata})
        report["results"].append(result)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in {"samples", "updates"}}, indent=2))
        
        model.zero_grad(set_to_none=True)
        del batches
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
