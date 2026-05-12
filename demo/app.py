#!/usr/bin/env python3
"""Gradio demo for Malar.IA — upload a thin blood smear, get parasite boxes
and a clinical-style report.

Loads the base Gemma 4 E4B model in 4-bit + LoRA adapter once at startup
(eager), then serves Gradio UI for interactive inference. Reuses parsing /
rendering logic from src/inference/predict.py.

Environment variables:
    MALARIA_ADAPTER         Path to LoRA adapter dir.
                            Default: runs/gemma4-e4b-falciparum-r1
    MALARIA_MODEL_ID        Base model id. Default: google/gemma-4-E4B-it
    MALARIA_IMAGE_MAX_SIZE  Longest edge after pre-resize (must match training).
                            Default: 768
    MALARIA_MAX_SOFT_TOKENS Visual token budget. Default: 140

Run:
    python demo/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

# Make src/ importable so we can reuse predict.py's helpers.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from src.inference.predict import (  # noqa: E402
    GRID,
    MODEL_ID_DEFAULT,
    build_messages,
    denormalize_box,
    detect_precision,
    extract_report,
    parse_detection_block,
)

ADAPTER_PATH = os.environ.get("MALARIA_ADAPTER", "runs/gemma4-e4b-falciparum-r1")
MODEL_ID = os.environ.get("MALARIA_MODEL_ID", MODEL_ID_DEFAULT)
IMAGE_MAX_SIZE = int(os.environ.get("MALARIA_IMAGE_MAX_SIZE", "768"))
MAX_SOFT_TOKENS = int(os.environ.get("MALARIA_MAX_SOFT_TOKENS", "140"))
MAX_NEW_TOKENS = int(os.environ.get("MALARIA_MAX_NEW_TOKENS", "1024"))


def _load_model():
    """Eager-load base + adapter at app startup."""
    print(f"[i] Loading processor + base: {MODEL_ID}")
    compute_dtype = detect_precision(force_fp16=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if hasattr(processor.image_processor, "max_soft_tokens"):
        processor.image_processor.max_soft_tokens = MAX_SOFT_TOKENS

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=compute_dtype,
        device_map="auto",
        quantization_config=quant,
        attn_implementation="sdpa",
    )
    print(f"[i] Attaching LoRA adapter: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    device = next(model.parameters()).device
    print(f"[i] Ready. Device: {device}")
    return processor, model, device


PROCESSOR, MODEL, DEVICE = _load_model()


def _render_boxes(image: Image.Image, boxes_abs: list[list[int]]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for x, y, w, h in boxes_abs:
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=4)
    return img


def predict(image: Image.Image | None) -> tuple[Image.Image | None, dict, str]:
    """Run inference and return (annotated_image, detection_dict, report_text)."""
    if image is None:
        return None, {"error": "No image uploaded."}, ""

    image_full = image.convert("RGB")
    W, H = image_full.size
    image_small = image_full.copy()
    image_small.thumbnail((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE))

    # build_messages takes a path purely for the chat template; the actual
    # image bytes come from the PIL object we pass to PROCESSOR() below.
    messages = build_messages(Path("/uploaded.png"))
    text = PROCESSOR.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    ).strip()
    inputs = PROCESSOR(
        text=[text], images=[[image_small]], return_tensors="pt", padding=True
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=8,
        )
    gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
    gen_text = PROCESSOR.tokenizer.decode(gen_ids, skip_special_tokens=True)

    detection = parse_detection_block(gen_text)
    if detection is None:
        return image_full, {"error": "Could not parse detection block.", "raw": gen_text}, ""

    boxes_norm: list[list[int]] = detection.get("boxes", [])
    boxes_abs = [denormalize_box(b, W, H, GRID) for b in boxes_norm]
    annotated = _render_boxes(image_full, boxes_abs)
    report = extract_report(gen_text) or ""

    return annotated, detection, report


DESCRIPTION = """
# Malar.IA — *Plasmodium falciparum* detection with Gemma 4

Fine-tuned **Google Gemma 4 E4B** (4-bit QLoRA on Colab Free T4) to detect malaria parasites in thin blood smear microscopy and emit a short clinical-style report. Submission to the **dev.to Gemma 4 Challenge** (Build category).

**How to use:** upload a thin blood smear image (Giemsa stain, ~100× magnification). The model returns bounding boxes plus a short clinical description grounded in those boxes.
""".strip()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Malar.IA") as ui:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Thin blood smear")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Detections", type="pil")
                out_report = gr.Textbox(label="Clinical report", lines=4)
                out_json = gr.JSON(label="Detection block")
        btn.click(predict, inputs=[inp], outputs=[out_img, out_json, out_report])
    return ui


if __name__ == "__main__":
    build_ui().launch()
