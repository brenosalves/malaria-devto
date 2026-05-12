# Colab T4 setup — fine-tuning Gemma 4 E4B on MP-IDB

Open a new Colab notebook with **Runtime → Change runtime type → T4 GPU** selected (Free tier is fine). Paste cells in order.

## 1. Authenticate & clone

```python
# Authenticate Hugging Face (needed to download google/gemma-4-E4B-it)
from huggingface_hub import notebook_login
notebook_login()
```

```bash
%cd /content
!rm -rf malaria-devto && git clone https://github.com/<YOUR_GH_USER>/malaria-devto.git
%cd /content/malaria-devto
```

## 2. Install training deps

```bash
%pip install -q -U \
    "transformers>=4.50" \
    "accelerate>=1.0" \
    "peft>=0.13" \
    "trl>=0.12" \
    "bitsandbytes>=0.44" \
    "datasets>=3.0" \
    pillow pandas
```

Restart the runtime if prompted by Colab (`Runtime → Restart session`).

## 3. Clone dataset & preprocess

```bash
!mkdir -p data/raw
!git clone --depth 1 \
  https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis.git \
  data/raw/MP-IDB

!python src/data/preprocess.py \
    --input data/raw/MP-IDB/Falciparum \
    --output data/processed
```

## 4. Smoke test (3 samples, 1 epoch) — verify everything wires up

```bash
!python src/training/train_lora.py \
    --data data/processed \
    --output runs/smoke \
    --epochs 1 \
    --max-train-samples 3
```

If this completes without OOM, you're clear to launch the real run.

## 5. Real training run

```bash
!python src/training/train_lora.py \
    --data data/processed \
    --output runs/gemma4-e4b-falciparum-r1 \
    --epochs 3 \
    --grad-accum 8
```

Expected: ~1.5–3 hours on T4 for 73 images × 3 epochs.

## 6. Persist the adapter to Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
dst = '/content/drive/MyDrive/malaria-devto/runs/gemma4-e4b-falciparum-r1'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree('runs/gemma4-e4b-falciparum-r1', dst)
print('saved to', dst)
```

---

## If you OOM on T4

In order of preference:

1. **Cut visual token budget.** The processor's image budget defaults to a high value. Force the lower tier by passing `images_kwargs={"max_num_visual_tokens": 560}` to the processor in `build_collate_fn` (or 280 if 560 still OOMs).
2. **Lower LoRA rank.** Try `--lora-r 8 --lora-alpha 8`.
3. **Drop to E2B.** `--model-id google/gemma-4-E2B-it` — half the params, more headroom, slightly weaker on fine detail.
4. **Switch to Colab Pro.** L4 (24 GB) lets you keep all the settings above and is ~3× faster.

## Notes for T4 specifically

- The script auto-detects bf16 and falls back to **fp16 + paged_adamw_8bit** on Turing.
- Flash-attention is **disabled** (`attn_implementation="eager"`); the official Gemma 4 fine-tuning guide assumes L4/A100 and uses flash-attn.
- The `gradient_checkpointing=True` setting is mandatory on T4 — without it you will OOM.
