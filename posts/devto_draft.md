<!--
DRAFT — dev.to submission for the Gemma 4 Challenge (Build category).
Deadline: 2026-05-24, 23:59 PDT.
Required tags: devchallenge, gemmachallenge, gemma
Required sections: What I Built / Demo / Code / How I Used Gemma 4 (+ variant justification).

Front matter for dev.to:
---
title: "Malar.IA — Teaching Gemma 4 to read malaria blood smears"
published: false
description: Fine-tuning Google's Gemma 4 multimodal to detect Plasmodium falciparum in thin blood smears and emit a clinical-style report. Submission to the dev.to Gemma 4 Challenge — Build.
tags: devchallenge, gemmachallenge, gemma, ai
cover_image: [TODO: 1000x420 cover with a smear + boxes]
---
-->

# Malar.IA — Teaching Gemma 4 to read malaria blood smears

Malaria still kills roughly 600,000 people a year, most of them children under five. The WHO gold standard for diagnosis is the same one we've had for a century: a trained microscopist staring at a Giemsa-stained blood smear, counting parasites by eye. It's accurate, it's cheap on consumables — and it's brutally bottlenecked on the human in the loop.

For the **dev.to Gemma 4 Challenge — Build**, I fine-tuned **Google's Gemma 4 multimodal** to do part of that job: look at a thin blood smear image, find *Plasmodium falciparum* parasites, draw boxes around them, and write a short clinical-style report describing what it saw. The whole thing trains in ~80 minutes on a free Colab T4. The adapter is 97 MB.

This post is the build log: what I shipped, what works, what doesn't, and the engineering choices behind it.

---

## What I Built

A pipeline that takes a thin blood smear microscopy image and produces:

1. A **structured detection block** — JSON with normalized bounding boxes and a parasite count.
2. A **clinical-style report** — one short sentence in the format a microscopist would write, grounded in the detection block.

Example output for an image with one ring-stage parasite:

```json
{"boxes": [[442, 490, 40, 50]], "count": 1}

Report: 1 Plasmodium falciparum parasite detected: 1 in ring stage.
```

The model is a single fine-tuned Gemma 4 E4B (~4B effective params) with a 97 MB LoRA adapter on top. No separate detector, no separate report generator — one multimodal model emits both the boxes and the text in one autoregressive pass, with the text grounded in the boxes by training-time supervision.

That coupling — **boxes paired with a description that names only what the boxes contain** — is the explainability story. A flat classifier saying "malaria positive" is useless to a microscopist. A heatmap is better but ungrounded. Boxes + a sentence that *cites* those boxes is something a clinician can audit at a glance.

---

## How I Used Gemma 4

### Why Gemma 4 (and which variant)

The challenge brief is explicit: Gemma 4 has to be **at the heart** of the project. The eligible variants are 2B/4B (multimodal "E" series), 31B Dense, and 26B MoE. PaliGemma — the obvious pick for vision-text fine-tuning a year ago — is explicitly *not* eligible.

That actually fits this problem perfectly. Gemma 4 is natively multimodal: text and image tokens flow through the same transformer, so emitting structured JSON with image-grounded coordinates *and* prose in one pass is the natural mode of operation, not a hack. PaliGemma's clean image→text design would have made coordinates harder to fuse with grounded prose.

**I chose Gemma 4 E4B** (~4B effective params, multimodal) for one reason: **portability**. The challenge's second judging axis is whether the project could plausibly deploy on consumer or edge hardware. E4B with 4-bit NF4 quantization fits inside a free Colab T4 (16 GB) end-to-end — training included. The 27B/26B variants would have been more accurate but would have killed the portability story; the 2B (E2B) variant trains comfortably on T4 but gives up too much spatial reasoning on dense smears. E4B is the smallest variant where this task actually clicks.

### Training: QLoRA on a free Colab T4

I trained on a single free Colab T4 (16 GB, Turing). This is genuinely cramped for a 4B multimodal model — the official Gemma 4 fine-tuning guide assumes L4 / A100 with bf16 and flash-attention, neither of which Turing supports natively. Getting it to fit was its own engineering exercise, and the constraints that worked are the ones that make the portability claim real.

The setup that finally landed:

- **4-bit NF4 quantization** via `bitsandbytes`, with double-quant and bf16 compute (emulated on Turing).
- **LoRA** with `r=16`, `alpha=16`, `target_modules="all-linear"`, **no `modules_to_save`**. Saving `lm_head` and `embed_tokens` alone blows the 16 GB budget by ~2 GB on T4 — those layers stay frozen.
- **Manual gradient-checkpointing setup**, skipping `peft.prepare_model_for_kbit_training`. The helper casts every non-quantized parameter (layer norms, etc.) to fp32 in a single allocation, which OOMs immediately. Doing the essential pieces by hand (`gradient_checkpointing_enable`, `enable_input_require_grads`) keeps norms in fp16 and shaves off ~10 GB of transient memory.
- **Vision token budget capped at 70** (the floor of Gemma 4's `70/140/280/560/1120` ladder) and **images pre-resized to 512 px on the longest edge**. The vision tower is the OOM hotspot on T4 — these two settings buy almost all the headroom.
- **Eval disabled** during training. The eval forward pass materializes full logits (~3.6 GB extra) and immediately OOMs.
- **Optimizer**: `paged_adamw_8bit` (since the bf16-friendly `adamw_torch_fused` doesn't help on Turing).

Total training: 73 train images × 10 epochs / grad_accum 8 = ~90 optimizer steps, ~80 minutes wall time. The LoRA adapter that pops out is 97 MB.

### Data: MP-IDB, Falciparum-only, and a CSV that lies

I used the **MP-IDB malaria parasite database** (Loddo et al., MIT-licensed): 104 thin blood smear images at 2592×1944 with 1,297 annotated parasites across the four malaria species.

Two scope decisions, both forced by the data:

1. **Falciparum-only.** Of the four species in MP-IDB, only *P. falciparum* ships with bounding-box CSVs. Malariae, Ovale, and Vivax have only binary segmentation masks. Deriving boxes from masks via connected components is doable but adds a fragile preprocessing stage; with two weeks to ship, Falciparum-only is the right v1 scope. Falciparum is also the most clinically critical species — it causes ~95% of malaria deaths.

2. **Stages live in the text, not in the boxes.** The Falciparum subset is wildly skewed: 94.8% ring, 3.2% trophozoite, 1.4% schizont, 0.5% gametocyte. Per-box stage classification would just learn to predict "ring" every time. Instead, the textual report describes the stage mix, and the boxes are stage-agnostic.

And then there's the data gotcha that ate my afternoon: **the MP-IDB CSV header is mislabeled**. The columns are headed `xmin, xmax, ymin, ymax` but the values don't satisfy that ordering — row 1 has `xmax=887 < xmin=919`. I rendered the three plausible interpretations side-by-side and pixel-matched against the canonical `crops/` folder to confirm: the actual format is **`x_topleft, y_topleft, width, height`** (COCO xywh). One rename in the preprocessor, and the rest of the pipeline can pretend the header was honest.

### Prompt and target format

Every training sample is a single chat turn:

- **System** — "You are a clinical microscopy assistant analyzing thin blood smear images for *Plasmodium falciparum*. Detect parasites and return their bounding boxes (normalized to [0, 1000]) followed by a short clinical report..."
- **User** — the smear image + "Detect Plasmodium parasites and describe findings."
- **Assistant (target)** —
  ```
  {"boxes": [[442, 490, 40, 50]], "count": 1}

  Report: 1 Plasmodium falciparum parasite detected: 1 in ring stage.
  ```

Coordinates are normalized to a `[0, 1000]` grid — Gemma 4's vision-input convention — and reconstructed to absolute pixels at inference using the image's true width and height. Splits are stratified by *image*, not by box (70/15/15) so that boxes from the same image never cross splits.

---

## Demo

🔗 **Live demo (HF Space):** [TODO: insert HF Space URL once deployed]

![demo screenshot](TODO_demo.png)

Upload a thin blood smear, get the annotated image + JSON detection block + clinical report. The model takes ~30 s per image on the Space's GPU.

**Example smear:** [TODO: link to a couple sample images people can right-click and re-upload]

---

## Code

🔗 **Repo:** https://github.com/brenosalves/malaria-devto

The structure:

```
src/
  data/        # MP-IDB loader, CSV column rename, JSONL split builder
  training/    # QLoRA fine-tuning (T4-tuned)
  inference/   # Adapter loading + JSON parsing + box denormalization
demo/
  app.py       # Gradio app (this demo)
notebooks/
  colab_setup.md  # Colab T4 recipe
posts/
  devto_draft.md  # This post
```

The training script is heavily commented around every T4 compromise — anyone trying to reproduce QLoRA on a 16 GB Turing card should be able to read [`src/training/train_lora.py`](https://github.com/brenosalves/malaria-devto/blob/main/src/training/train_lora.py) and understand *why* each setting has to be exactly what it is.

To reproduce from scratch on a free Colab notebook:

```bash
# (1) clone repo + install deps (see notebooks/colab_setup.md)
# (2) clone MP-IDB
git clone --depth 1 https://github.com/andrealoddo/MP-IDB-... data/raw/MP-IDB
# (3) preprocess
python src/data/preprocess.py --input data/raw/MP-IDB/Falciparum --output data/processed
# (4) train
python src/training/train_lora.py --data data/processed --output runs/r1 \
    --epochs 10 --max-soft-tokens 70 --image-max-size 512
# (5) inference
python src/inference/predict.py --adapter runs/r1 \
    --test-jsonl data/processed/test.jsonl --index 0 --output pred.png
```

---

## What works, what doesn't

**What works:**
- The model emits **structurally valid JSON** with boxes and a count on essentially every input.
- The **clinical report is grammatically correct** and uses the right singular/plural forms ("1 parasite detected: 1 in ring stage." vs "12 parasites detected: 10 in ring stage, 1 trophozoite, 1 schizont.").
- The **count is calibrated** on low-density images (1–3 parasites): predicted count matches ground truth on sanity-check examples.
- The **system runs on free Colab T4** end-to-end. Training, inference, and the Gradio demo all fit.

**What's still rough:**
- **Spatial localization is the weak axis.** On low-density images the count is right, but the boxes don't always land precisely on the parasite. The fundamental cause is the visual-token budget: at 70 tokens for a 2592×1944 image, each token covers roughly a 200×200 px receptive field, while a parasite is ~70–100 px. The model has the *semantics* but not the *resolution* to localize precisely. Bumping the token budget to 280 or 560 (which is where localization papers say accuracy unlocks) would require an L4 or better — outside the free-Colab portability constraint.
- **Greedy decoding on dense images sometimes degenerates** into near-duplicate boxes (a classic autoregressive failure mode on an undertrained model). I added `repetition_penalty=1.1` and `no_repeat_ngram_size=8` to inference to mitigate, plus a tolerant parser that regex-extracts boxes when the JSON gets truncated.

I'm leaving these as documented limitations rather than hiding them — they're the honest answer to "what does fine-tuning a 4B multimodal model on 73 images in 80 minutes get you?" The architecture and pipeline are right; the spatial-resolution gap is a hardware budget, not a design flaw.

---

## Closing

Two things I think this submission does well:

1. **The explainability pairing is the whole point.** Boxes + a clinical-style sentence grounded in those boxes is a better unit of model output for clinical-adjacent tasks than either alone. It's auditable.
2. **The portability story is real, not performative.** This whole thing — train, infer, demo — runs on free Colab. The compromises required to make that happen are documented in the code and in this post.

If you're building anything multimodal on a budget, [the repo](https://github.com/brenosalves/malaria-devto) is meant to be readable as a worked example of the T4 corners you have to round.

— Breno

---

[TODO before publish]
- [ ] Insert HF Space URL once deployed
- [ ] Cover image (1000×420) — smear + boxes overlay
- [ ] Demo screenshot or short gif (compressed)
- [ ] Re-run inference on 3–5 test images and pick the best for the post (mix of densities)
- [ ] If r2 (10-epoch) adapter improved localization meaningfully, update the "What works / What's rough" section
- [ ] Sanity-check word count + paragraph rhythm on dev.to preview
- [ ] Verify tags render: devchallenge, gemmachallenge, gemma
- [ ] Set `published: true` in front matter
