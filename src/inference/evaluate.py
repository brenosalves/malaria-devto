#!/usr/bin/env python3
"""Quantitative evaluation of a fine-tuned Gemma 4 adapter on MP-IDB test split.

Runs inference over every record in test.jsonl and computes:
- parse_rate              % of outputs where the JSON detection block parsed
- salvage_rate_of_parses  % of parses that fell back to regex (truncated output)
- count_mae               mean absolute error between predicted and GT parasite count
- count_exact_rate        % of images where pred_count == gt_count
- recall / precision / f1 @ IoU >= {0.3, 0.5, 0.7}  (greedy matching, micro-averaged)
- mean_iou_of_matches_at_50

Writes a per-image + aggregate JSON to --output. Reuses model loading
and parsing from src/inference/predict.py.

Example (Colab):
    python src/inference/evaluate.py \\
        --adapter /content/drive/MyDrive/malaria-devto/runs/gemma4-e4b-falciparum-r2 \\
        --test-jsonl data/processed/test.jsonl \\
        --output eval_r2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Reuse helpers from predict.py (sibling module).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import (  # noqa: E402
    GRID,
    MODEL_ID_DEFAULT,
    build_messages,
    denormalize_box,
    detect_precision,
    parse_detection_block,
)

IOU_THRESHOLDS = (0.3, 0.5, 0.7)
PRIMARY_IOU = 0.5  # used for precision / F1 / mean matched-IoU summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir.")
    p.add_argument(
        "--test-jsonl",
        type=Path,
        required=True,
        help="Test split JSONL produced by src/data/preprocess.py.",
    )
    p.add_argument("--output", type=Path, required=True, help="JSON report path.")
    p.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--image-max-size",
        type=int,
        default=768,
        help="Match training default — pre-resizes longest edge.",
    )
    p.add_argument("--max-soft-tokens", type=int, default=140)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=8)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N records (smoke test).",
    )
    p.add_argument("--force-fp16", action="store_true")
    return p.parse_args()


def iou_xywh(a: list[int], b: list[int]) -> float:
    """IoU between two axis-aligned boxes in [x, y, w, h] format."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    gt_boxes: list[list[int]],
    pred_boxes: list[list[int]],
    iou_thresh: float,
) -> list[tuple[int, int, float]]:
    """Greedy IoU matching. Each GT and pred matched at most once."""
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            v = iou_xywh(g, p)
            if v >= iou_thresh:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for v, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        out.append((gi, pi, v))
    return out


def load_model(args: argparse.Namespace, compute_dtype: torch.dtype):
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
    print(f"[i] Attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    return processor, model


def predict_one(
    processor,
    model,
    image_path: Path,
    image_max_size: int,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> tuple[int, int, str]:
    """Run one inference. Returns (full_width, full_height, generated_text)."""
    image_full = Image.open(image_path).convert("RGB")
    W, H = image_full.size
    image_small = image_full.copy()
    image_small.thumbnail((image_max_size, image_max_size))
    messages = build_messages(image_path)
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ).strip()
    device = next(model.parameters()).device
    inputs = processor(
        text=[text], images=[[image_small]], return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    gen_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return W, H, gen_text


def evaluate_record(
    rec: dict, processor, model, args: argparse.Namespace
) -> dict[str, Any]:
    image_path = Path(rec["image_path"])
    gt_boxes_abs: list[list[int]] = rec["boxes_abs"]
    gt_count: int = rec["count"]
    image_id: str = rec["image_id"]

    if not image_path.exists():
        return {"image_id": image_id, "error": f"Image not found: {image_path}"}

    W, H, gen_text = predict_one(
        processor,
        model,
        image_path,
        args.image_max_size,
        args.max_new_tokens,
        args.repetition_penalty,
        args.no_repeat_ngram_size,
    )
    detection = parse_detection_block(gen_text)
    parsed = detection is not None
    salvaged = bool(detection.get("_salvaged", False)) if parsed else False

    base_result: dict[str, Any] = {
        "image_id": image_id,
        "gt_count": gt_count,
        "parsed": parsed,
        "salvaged": salvaged,
        "raw_output_head": gen_text[:500],
    }

    if not parsed:
        base_result["pred_count"] = None
        return base_result

    pred_boxes_norm = detection.get("boxes", [])
    pred_count = int(detection.get("count", len(pred_boxes_norm)))
    pred_boxes_abs = [denormalize_box(b, W, H, GRID) for b in pred_boxes_norm]
    base_result["pred_count"] = pred_count
    base_result["n_pred_boxes"] = len(pred_boxes_abs)

    for thresh in IOU_THRESHOLDS:
        matches = greedy_match(gt_boxes_abs, pred_boxes_abs, thresh)
        base_result[f"tp_iou_{int(thresh * 100)}"] = len(matches)

    primary_matches = greedy_match(gt_boxes_abs, pred_boxes_abs, PRIMARY_IOU)
    base_result["mean_iou_tp_at_50"] = (
        sum(m[2] for m in primary_matches) / len(primary_matches)
        if primary_matches
        else None
    )
    return base_result


def aggregate(per_image: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_image)
    parsed = [r for r in per_image if r.get("parsed")]
    n_parsed = len(parsed)
    n_salvaged = sum(1 for r in parsed if r.get("salvaged"))

    diffs = [
        abs(r["gt_count"] - r["pred_count"])
        for r in parsed
        if r.get("pred_count") is not None
    ]
    count_mae = sum(diffs) / len(diffs) if diffs else None
    count_exact = sum(1 for d in diffs if d == 0)
    count_exact_rate = count_exact / n_parsed if n_parsed else 0.0

    summary: dict[str, Any] = {
        "n_images": n,
        "n_parsed": n_parsed,
        "n_salvaged": n_salvaged,
        "parse_rate": n_parsed / n if n else 0.0,
        "salvage_rate_of_parses": n_salvaged / n_parsed if n_parsed else 0.0,
        "count_mae": count_mae,
        "count_exact_rate": count_exact_rate,
    }

    for thresh in IOU_THRESHOLDS:
        key = f"tp_iou_{int(thresh * 100)}"
        total_tp = sum(r.get(key, 0) for r in parsed)
        total_gt = sum(r["gt_count"] for r in parsed)
        total_pred = sum(r.get("n_pred_boxes", 0) for r in parsed)
        recall = total_tp / total_gt if total_gt else 0.0
        precision = total_tp / total_pred if total_pred else 0.0
        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )
        suffix = f"_iou_{int(thresh * 100)}"
        summary[f"recall{suffix}"] = recall
        summary[f"precision{suffix}"] = precision
        summary[f"f1{suffix}"] = f1
        summary[f"total_tp{suffix}"] = total_tp

    matched_ious = [
        r["mean_iou_tp_at_50"]
        for r in parsed
        if r.get("mean_iou_tp_at_50") is not None
    ]
    summary["mean_iou_of_matches_at_50"] = (
        sum(matched_ious) / len(matched_ious) if matched_ious else None
    )
    return summary


def _fmt(v: float | None, spec: str = ".3f") -> str:
    return "n/a" if v is None else format(v, spec)


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Aggregate ===")
    print(f"  Images evaluated:        {summary['n_images']}")
    print(
        f"  Parse rate:              {summary['parse_rate']:.1%}  "
        f"({summary['n_parsed']}/{summary['n_images']})"
    )
    print(
        f"  Salvage rate of parses:  {summary['salvage_rate_of_parses']:.1%}  "
        f"({summary['n_salvaged']}/{summary['n_parsed']})"
    )
    print(f"  Count MAE:               {_fmt(summary['count_mae'], '.2f')}")
    print(f"  Count exact rate:        {summary['count_exact_rate']:.1%}")
    for thresh in IOU_THRESHOLDS:
        i = int(thresh * 100)
        print(
            f"  @ IoU>=0.{i:02d}:  TP={summary[f'total_tp_iou_{i}']:>4}  "
            f"R={summary[f'recall_iou_{i}']:.3f}  "
            f"P={summary[f'precision_iou_{i}']:.3f}  "
            f"F1={summary[f'f1_iou_{i}']:.3f}"
        )
    print(
        f"  Mean IoU of matches@0.5: {_fmt(summary['mean_iou_of_matches_at_50'])}"
    )


def main() -> None:
    args = parse_args()
    if not args.test_jsonl.exists():
        sys.exit(f"[!] Test JSONL not found: {args.test_jsonl}")

    records = [
        json.loads(line)
        for line in args.test_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]
    print(f"[i] Evaluating {len(records)} test records.")

    compute_dtype = detect_precision(args.force_fp16)
    print(f"[i] compute_dtype={compute_dtype}")
    processor, model = load_model(args, compute_dtype)

    per_image: list[dict[str, Any]] = []
    t0 = time.time()
    for i, rec in enumerate(records):
        t_start = time.time()
        result = evaluate_record(rec, processor, model, args)
        per_image.append(result)
        elapsed = time.time() - t_start
        gt = result.get("gt_count", "?")
        pred = result.get("pred_count") if result.get("parsed") else "?"
        parsed_tag = "OK" if result.get("parsed") else "PARSE-FAIL"
        salv = " (salvaged)" if result.get("salvaged") else ""
        print(
            f"  [{i + 1:>3}/{len(records)}]  {result.get('image_id', '?'):<32}  "
            f"gt={gt:<3}  pred={pred!s:<3}  {parsed_tag}{salv}  ({elapsed:.1f}s)"
        )

    total_time = time.time() - t0
    print(f"\n[i] Done in {total_time:.1f}s ({total_time / max(len(records), 1):.1f}s/image)")

    summary = aggregate(per_image)
    print_summary(summary)

    output = {
        "args": {
            "adapter": str(args.adapter),
            "model_id": args.model_id,
            "image_max_size": args.image_max_size,
            "max_soft_tokens": args.max_soft_tokens,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "limit": args.limit,
        },
        "summary": summary,
        "per_image": per_image,
        "wall_time_seconds": total_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\n[i] Saved evaluation to {args.output}")


if __name__ == "__main__":
    main()
