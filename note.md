# Malar.IA: Gemma 4 for *Plasmodium falciparum* Detection

Fine-tune **Gemma 4 multimodal** to detect *Plasmodium falciparum* parasites in thin blood smear images and emit a clinical-style report. Submission to the dev.to **Gemma 4 Challenge** (Build category, deadline 2026-05-24).

## 🛠 Tech Stack
*   **Model:** Google **Gemma 4** (multimodal) — variant 4B (or 2B for portability)
*   **Framework:** Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-/8-bit)
*   **Domain:** Digital Pathology / Malaria Microscopy

## 🧪 Project Specifics
*   **Task:** Image → text. Binary parasite detection (boxes) + natural-language report listing stage distribution and count.
*   **Input:** Thin blood smear microscopy (Giemsa stain), 2592×1944 px, ~100× magnification.
*   **Dataset:** [MP-IDB](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis), Falciparum subset only — 104 images, 1297 annotated parasites (94.8% ring, 3.2% trophozoite, 1.4% schizont, 0.5% gametocyte). MIT licensed.
*   **Output schema (prompted):**
    ```
    {"boxes": [[x, y, w, h], ...], "count": 12}
    Report: 12 Plasmodium falciparum parasites detected, predominantly
    ring-stage (10), with 1 trophozoite and 1 schizont.
    ```
*   **Why falciparum-only:** Other 3 species in MP-IDB ship only segmentation masks (no bbox CSV); deriving boxes via connected components would risk the deadline. Falciparum is the most clinically critical species (~95% of malaria deaths).

## ⚠️ Dataset Gotchas
*   **CSV header is misleading.** `mp-idb-falciparum.csv` advertises columns `xmin, xmax, ymin, ymax` but the actual format is **`x_topleft, y_topleft, width, height`** (COCO-style xywh). Confirmed empirically via pixel-match against the official `crops/` folder. The preprocess step must rename these columns.
*   **No PII** in filenames, but datasets are public and patient-anonymous regardless.

## ⌨️ Development Commands
*   **Setup:** `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
*   **Explore:** `python src/data/explore.py --data-root data/raw/MP-IDB --species Falciparum`
*   **Preprocess:** `python src/data/preprocess.py --input data/raw/MP-IDB/Falciparum --output data/processed`
*   **Fine-tune:** `python src/training/train_lora.py --config configs/gemma4_4b_lora.yaml`
*   **Inference:** `python src/inference/predict.py --image test.jpg`
*   **Demo:** `python demo/app.py`

## 📏 Standards & Guidelines
*   **Code Style:** PEP 8; type hints on function signatures.
*   **Detection:** Every box carries a confidence score; the textual report must only describe stages/counts grounded in the detection block (no hallucinated findings).
*   **Coordinates:** Normalize boxes to `[0, 1000]` (Gemma 4 vision input convention) for training; convert back at inference. Round-trip test for any new dataset.
*   **Naming:** `snake_case` for vars/files, `PascalCase` for classes.

## 📂 Project Structure
*   `src/data/` — Preprocessing + prompt formatting.
*   `src/training/` — LoRA fine-tuning loop.
*   `src/inference/` — Adapter loading + parsing.
*   `configs/` — Hyperparameters.
*   `demo/` — Gradio app for the submission demo.
*   `data/` — Git-ignored. MP-IDB clone at `data/raw/MP-IDB/`.

## 🎯 Challenge Goals (Gemma 4 Challenge — Build)
*   **Explainability:** Boxes paired with a clinical-style description of what was found.
*   **Portability:** 4-/8-bit quantization + LoRA. Justify variant (2B vs 4B) explicitly in the submission post.
*   **Deliverables:** dev.to post with tags `devchallenge`, `gemmachallenge`, `gemma`; sections "What I Built", "Demo", "Code", "How I Used Gemma 4"; demo video or deployed link; public repo.
