#!/usr/bin/env python3
"""QLoRA fine-tuning of Gemma 4 E4B on MP-IDB Falciparum.

Designed to run on Colab Free T4 (16 GB, no bf16, no flash-attention).
The script auto-detects bf16 support and falls back to fp16 + paged_adamw_8bit
on Turing-class GPUs.

Inputs:  JSONL splits from src/data/preprocess.py.
Outputs: a LoRA adapter directory ready for inference.

Example (Colab):
    python src/training/train_lora.py \\
        --data data/processed \\
        --output runs/gemma4-e4b-falciparum-r1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-4-E4B-it"

SYSTEM_PROMPT = (
    "You are a clinical microscopy assistant analyzing thin blood smear images "
    "for Plasmodium falciparum. Detect parasites and return their bounding boxes "
    "(normalized to [0, 1000]) followed by a short clinical report describing "
    "the stages and count observed. Ground every claim in the detection block."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory holding train.jsonl / val.jsonl.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the LoRA adapter and trainer state.",
    )
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training samples for a smoke test.",
    )
    return p.parse_args()


def detect_precision() -> tuple[torch.dtype, bool]:
    """Return (compute_dtype, use_bf16). T4 has no bf16; fall back to fp16."""
    if not torch.cuda.is_available():
        print("[!] No CUDA — this script needs a GPU. Aborting.", file=sys.stderr)
        sys.exit(1)
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True
    print("[i] GPU lacks bf16 support (likely T4). Using fp16 + paged_adamw_8bit.")
    return torch.float16, False


def format_sample(sample: dict) -> dict:
    """Map a JSONL record to the OpenAI-style chat schema TRL expects."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["prompt"]},
                    {"type": "image", "image": sample["image_path"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["target"]}],
            },
        ]
    }


def build_collate_fn(processor):
    """Vision-SFT collator: pack messages into model inputs and mask non-text tokens."""
    pad_id = processor.tokenizer.pad_token_id
    # Special-token attribute names changed between processor versions; resolve once.
    image_token_ids: list[int] = []
    for attr in ("image_token_id", "boi_token_id", "eoi_token_id"):
        tid = getattr(processor.tokenizer, attr, None)
        if tid is not None:
            image_token_ids.append(tid)

    def collate(examples):
        texts, images = [], []
        for ex in examples:
            text = processor.apply_chat_template(
                ex["messages"], add_generation_prompt=False, tokenize=False
            ).strip()
            # Load image lazily — JSONL stores absolute paths
            image_msgs = [
                c
                for m in ex["messages"]
                if isinstance(m["content"], list)
                for c in m["content"]
                if c.get("type") == "image"
            ]
            imgs = [Image.open(c["image"]).convert("RGB") for c in image_msgs]
            texts.append(text)
            images.append(imgs)

        batch = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        labels = batch["input_ids"].clone()
        labels[labels == pad_id] = -100
        for tid in image_token_ids:
            labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    return collate


def load_splits(data_dir: Path, max_train: int | None):
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    for p in (train_path, val_path):
        if not p.exists():
            sys.exit(f"Missing split: {p}. Run src/data/preprocess.py first.")
    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "val": str(val_path)},
    )
    if max_train is not None:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    ds = ds.map(format_sample, remove_columns=ds["train"].column_names)
    return ds["train"], ds["val"]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    compute_dtype, use_bf16 = detect_precision()
    print(f"[i] compute_dtype={compute_dtype}, use_bf16={use_bf16}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )

    print(f"[i] Loading processor + model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=compute_dtype,
        device_map="auto",
        quantization_config=quant,
        attn_implementation="eager",  # T4 has no flash-attn
    )
    model.config.use_cache = False  # required for gradient checkpointing

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    train_ds, val_ds = load_splits(args.data, args.max_train_samples)
    print(f"[i] train={len(train_ds)} val={len(val_ds)}")

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if not use_bf16 else "adamw_torch_fused",
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="tensorboard",
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=build_collate_fn(processor),
    )

    print("[i] Starting training")
    trainer.train()
    trainer.save_model(str(args.output))
    print(f"[i] Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
