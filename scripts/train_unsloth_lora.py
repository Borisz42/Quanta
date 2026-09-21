#!/usr/bin/env python3
"""Unsloth QLoRA Fine-Tuning Pipeline for QUANTA S-Expression Transducer (Phase 6.3).

Trains Qwen 3.5 4B/2B or Gemma 4 in 4-bit precision within the RTX 3070 8GB VRAM envelope:
- 4-bit base weights (load_in_4bit=True)
- LoRA adapters: r=16, alpha=32, target linear modules
- Gradient checkpointing + cosine LR schedule (lr=2e-4)
- 8-bit AdamW optimizer with batch size 2 and gradient accumulation 4-8
- Saves fine-tuned adapter weights to models/quanta-slm-lora/

Usage:
    python scripts/train_unsloth_lora.py --dataset-path data/training/sexpr_train.jsonl --dry-run
    python scripts/train_unsloth_lora.py --model-name qwen3.5-4b --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure Windows stdout handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from core.artifacts import require_artifacts, MissingArtifactError

logger = logging.getLogger(__name__)

# Standard instruction formatting template for S-expression transduction
TRANSDUCTION_PROMPT_TEMPLATE = """Below is an instruction that describes a semantic transduction task, paired with input discourse text. Write a canonical GBNF S-expression that represents the Entity-Event Directed Acyclic Graph (ASG).

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_training_model_name(model_name: Optional[str]) -> str:
    """Resolve user-friendly model aliases to HF repo identifiers."""
    name = (model_name or "qwen3.5-4b").strip().lower()
    mapping = {
        "qwen3.5-4b": "unsloth/Qwen2.5-3B-Instruct",
        "qwen-4b": "unsloth/Qwen2.5-3B-Instruct",
        "qwen3.5-2b": "unsloth/Qwen2.5-1.5B-Instruct",
        "qwen-2b": "unsloth/Qwen2.5-1.5B-Instruct",
        "ultra-low-vram": "unsloth/Qwen2.5-1.5B-Instruct",
        "gemma-4": "unsloth/gemma-2-2b-it",
        "gemma": "unsloth/gemma-2-2b-it",
        "default": "unsloth/Qwen2.5-3B-Instruct",
    }
    return mapping.get(name, model_name or "unsloth/Qwen2.5-3B-Instruct")


def format_dataset_sample(instruction: str, input_text: str, output_text: str, eos_token: str = "") -> str:
    """Format a single training sample using the canonical prompt template."""
    return TRANSDUCTION_PROMPT_TEMPLATE.format(
        instruction=instruction.strip(),
        input=input_text.strip(),
        output=output_text.strip(),
    ) + eos_token


def train_lora(
    dataset_path: Union[str, Path],
    model_name: str = "qwen3.5-4b",
    output_dir: Union[str, Path] = "models/quanta-slm-lora",
    checkpoint_dir: Union[str, Path] = "models/quanta-slm-checkpoints",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    max_steps: int = 60,
    epochs: Optional[int] = None,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Configures and runs Unsloth QLoRA fine-tuning for S-expression transduction."""
    t0 = time.perf_counter()
    data_file = Path(dataset_path).resolve()
    out_dir = Path(output_dir).resolve()
    chk_dir = Path(checkpoint_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_file}")

    resolved_model = resolve_training_model_name(model_name)

    # Load dataset from JSONL
    samples: List[Dict[str, Any]] = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError(f"Training dataset is empty: {data_file}")

    config_summary = {
        "model_name": model_name,
        "resolved_model": resolved_model,
        "dataset_path": str(data_file),
        "total_samples": len(samples),
        "output_dir": str(out_dir),
        "checkpoint_dir": str(chk_dir),
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": TARGET_MODULES,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "epochs": epochs,
        "seed": seed,
        "dry_run": dry_run,
        "load_in_4bit": True,
    }

    # If dry run mode, validate prompt formatting and exit cleanly
    if dry_run:
        sample_prompt = format_dataset_sample(
            instruction=samples[0].get("instruction", ""),
            input_text=samples[0].get("input", ""),
            output_text=samples[0].get("output", ""),
            eos_token="<|im_end|>",
        )
        # Write config metadata
        config_path = out_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_summary, f, indent=2)

        config_summary["sample_formatted_prompt"] = sample_prompt[:300] + "..."
        config_summary["status"] = "dry_run_success"
        config_summary["elapsed_time"] = time.perf_counter() - t0
        return config_summary

    # Full Unsloth training execution
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    logger.info(f"Loading base model '{resolved_model}' in 4-bit precision...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    logger.info(f"Configuring LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    # Format text entries
    formatted_texts: List[str] = []
    eos = getattr(tokenizer, "eos_token", "") or ""
    for s in samples:
        formatted_texts.append(
            format_dataset_sample(
                instruction=s.get("instruction", ""),
                input_text=s.get("input", ""),
                output_text=s.get("output", ""),
                eos_token=eos,
            )
        )

    hf_dataset = Dataset.from_dict({"text": formatted_texts})

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps if epochs is None else -1,
        num_train_epochs=epochs if epochs is not None else 1.0,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=seed,
        output_dir=str(chk_dir),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    logger.info("Executing LoRA training loop...")
    train_stats = trainer.train()

    logger.info(f"Saving fine-tuned adapter weights to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Save training metadata
    config_summary["train_loss"] = getattr(train_stats, "training_loss", None)
    config_summary["status"] = "training_success"
    config_summary["elapsed_time"] = time.perf_counter() - t0

    config_path = out_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)

    return config_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune local SLM via Unsloth QLoRA for QUANTA S-Expression Transduction (Phase 6.3)."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/training/sexpr_train.jsonl",
        help="Path to instruction JSONL dataset (default: data/training/sexpr_train.jsonl)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen3.5-4b",
        help="Base model identifier or preset (default: qwen3.5-4b, options: qwen-2b, gemma-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/quanta-slm-lora",
        help="Directory to save fine-tuned LoRA adapters (default: models/quanta-slm-lora)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/quanta-slm-checkpoints",
        help="Directory for training checkpoints (default: models/quanta-slm-checkpoints)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA scaling alpha (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (default: 2)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Maximum training steps (default: 60)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides max-steps if set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run configuration and prompt formatting check without training weights",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("QUANTA Unsloth QLoRA Fine-Tuning Pipeline (Phase 6.3)")
    print("=" * 70)
    print(f"  Base Model:          {args.model_name}")
    print(f"  Dataset:             {args.dataset_path}")
    print(f"  Adapter Output:      {args.output_dir}")
    print(f"  Checkpoints:         {args.checkpoint_dir}")
    print(f"  LoRA Config:         r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Batch Envelope:      batch={args.batch_size}, accum={args.gradient_accumulation_steps}")
    print(f"  Max Steps / LR:      steps={args.max_steps}, lr={args.learning_rate}")
    print(f"  Dry-Run Mode:        {args.dry_run}")
    print("=" * 70)

    require_artifacts(args.dataset_path, component="UnslothQLoRAPipeline")

    try:
        res = train_lora(
            dataset_path=args.dataset_path,
            model_name=args.model_name,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            batch_size=args.batch_size,
            grad_accum_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            epochs=args.epochs,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print("\nTraining procedure finished successfully!")
        print(f"  Status:             {res.get('status')}")
        print(f"  Resolved Model:     {res.get('resolved_model')}")
        print(f"  Total Samples:      {res.get('total_samples')}")
        print(f"  Effective Batch:    {res.get('effective_batch_size')}")
        print(f"  Elapsed Time:       {res.get('elapsed_time', 0):.2f}s")
        print("=" * 70)
        return 0
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
