#!/usr/bin/env python3
"""Run inference with a fine-tuned Gemma 4 E4B LoRA adapter on an MP-IDB image.

Loads the base model in 4-bit NF4 (matching training), attaches the LoRA
adapter, generates the detection block, parses the JSON, denormalizes
coordinates from the [0, 1000] grid back to pixel space, and saves a
visualization PNG with boxes overlaid. Designed for Colab T4 — same
quant/precision settings as src/training/train_lora.py.

Two input modes:
  --image PATH                          (no ground truth overlay)
  --test-jsonl PATH --index N           (uses image_path + boxes_abs from record)

Example (Colab):
    python src/inference/predict.py \\
        --adapter runs/gemma4-e4b-falciparum-r1 \\
        --test-jsonl data/processed/test.jsonl \\
        --index 0 \\
        --output pred_0.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_ID_DEFAULT = "google/gemma-4-E4B-it"
GRID = 1000

# MUST match src/training/train_lora.py:SYSTEM_PROMPT and
# src/data/preprocess.py:PROMPT — duplicated to avoid cross-package imports
# under `python src/inference/predict.py` on Colab.
SYSTEM_PROMPT = (
    "You are a clinical microscopy assistant analyzing thin blood smear images "
    "for Plasmodium falciparum. Detect parasites and return their bounding boxes "
    "(normalized to [0, 1000]) followed by a short clinical report describing "
    "the stages and count observed. Ground every claim in the detection block."
)
USER_PROMPT = "Detect Plasmodium parasites and describe findings."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter directory.")
    p.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image; no ground truth.")
    src.add_argument(
        "--test-jsonl",
        type=Path,
        help="JSONL produced by src/data/preprocess.py; --index picks the record.",
    )
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="Visualization PNG path.")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--image-max-size",
        type=int,
        default=768,
        help="Match training default — resizes longest edge before processor().",
    )
    p.add_argument(
        "--max-soft-tokens",
        type=int,
        default=140,
        help="Match training default — visual token budget after pooling.",
    )
    p.add_argument("--force-fp16", action="store_true")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Generation repetition penalty (1.0 = off). Mitigates degenerate "
        "near-duplicate boxes on undertrained models / dense images.",
    )
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=8,
        help="Block exact n-gram repetitions during generation (0 = off).",
    )
    return p.parse_args()


BOX_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")


def detect_precision(force_fp16: bool) -> torch.dtype:
    if not torch.cuda.is_available():
        sys.exit("[!] No CUDA — inference on 4-bit Gemma 4 needs a GPU.")
    if force_fp16:
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_input(args: argparse.Namespace) -> tuple[Path, dict | None]:
    """Return (image_path, ground_truth_record_or_None)."""
    if args.image is not None:
        return args.image, None
    with args.test_jsonl.open() as f:
        for i, line in enumerate(f):
            if i == args.index:
                rec = json.loads(line)
                return Path(rec["image_path"]), rec
    sys.exit(f"[!] Index {args.index} out of range in {args.test_jsonl}")


def build_messages(image_path: Path) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image", "image": str(image_path)},
            ],
        },
    ]


def parse_detection_block(text: str) -> dict[str, Any] | None:
    """Find the first balanced {...} that json-decodes to a dict containing 'boxes'.

    Falls back to regex-extracting individual [x, y, w, h] arrays if no
    balanced JSON is found — covers truncated output and degenerate
    repetition-loop generations that never closed the outer braces.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "boxes" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)

    matches = BOX_RE.findall(text)
    if matches:
        boxes = [[int(a), int(b), int(c), int(d)] for a, b, c, d in matches]
        return {"boxes": boxes, "count": len(boxes), "_salvaged": True}
    return None


def extract_report(text: str) -> str | None:
    marker = "Report:"
    idx = text.find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker) :].strip()


def denormalize_box(box: list[int], W: int, H: int, grid: int) -> list[int]:
    nx, ny, nw, nh = box
    return [
        round(nx * W / grid),
        round(ny * H / grid),
        round(nw * W / grid),
        round(nh * H / grid),
    ]


def render(
    image: Image.Image,
    pred_boxes: list[list[int]],
    gt_boxes: list[list[int]] | None,
    output: Path,
) -> None:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    if gt_boxes:
        for x, y, w, h in gt_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    for x, y, w, h in pred_boxes:
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    args = parse_args()
    compute_dtype = detect_precision(args.force_fp16)
    print(f"[i] compute_dtype={compute_dtype}")

    image_path, gt_record = load_input(args)
    if not image_path.exists():
        sys.exit(f"[!] Image not found: {image_path}")
    print(f"[i] Image: {image_path}")

    print(f"[i] Loading processor + base: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor.image_processor, "max_soft_tokens"):
        processor.image_processor.max_soft_tokens = args.max_soft_tokens

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=compute_dtype,
        device_map="auto",
        quantization_config=quant,
        attn_implementation="sdpa",
    )
    print(f"[i] Attaching LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    image_full = Image.open(image_path).convert("RGB")
    W_full, H_full = image_full.size
    image_small = image_full.copy()
    image_small.thumbnail((args.image_max_size, args.image_max_size))

    messages = build_messages(image_path)
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ).strip()
    device = next(model.parameters()).device
    inputs = processor(
        text=[text], images=[[image_small]], return_tensors="pt", padding=True
    ).to(device)

    print("[i] Generating...")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("\n=== Raw generation ===")
    print(gen_text)
    print("=== /generation ===\n")

    detection = parse_detection_block(gen_text)
    if detection is None:
        sys.exit("[!] Could not parse a detection block from output.")

    pred_boxes_norm: list[list[int]] = detection.get("boxes", [])
    pred_count = detection.get("count")
    salvaged = detection.get("_salvaged", False)
    salvage_tag = "  (salvaged from malformed output)" if salvaged else ""
    print(f"[i] Parsed {len(pred_boxes_norm)} boxes, count={pred_count}{salvage_tag}")

    report = extract_report(gen_text)
    if report:
        print(f"[i] Report: {report}")

    pred_boxes_abs = [denormalize_box(b, W_full, H_full, GRID) for b in pred_boxes_norm]
    gt_boxes_abs = gt_record["boxes_abs"] if gt_record else None
    if gt_boxes_abs is not None:
        print(
            f"[i] Ground truth: {len(gt_boxes_abs)} boxes, count={gt_record['count']}"
        )

    render(image_full, pred_boxes_abs, gt_boxes_abs, args.output)
    print(f"[i] Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
