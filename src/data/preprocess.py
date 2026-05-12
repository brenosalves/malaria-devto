#!/usr/bin/env python3
"""Preprocess MP-IDB Falciparum into JSONL splits for Gemma 4 fine-tuning.

Reads the (mislabeled) Falciparum CSV, renames columns to the truthful
`x, y, w, h` schema, groups annotations per image, normalizes coordinates
to a `[0, 1000]` grid, builds prompt/target pairs, and writes
train/val/test JSONL files with an image-level 70/15/15 split.

Example:
    python src/data/preprocess.py \\
        --input data/raw/MP-IDB/Falciparum \\
        --output data/processed
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

PROMPT = "Detect Plasmodium parasites and describe findings."

# (singular, plural) for each MP-IDB stage abbreviation.
# 'ring' uses "ring stage" as an invariant noun phrase.
STAGE_NAMES: dict[str, tuple[str, str]] = {
    "ring": ("in ring stage", "in ring stage"),
    "tro": ("trophozoite", "trophozoites"),
    "schi": ("schizont", "schizonts"),
    "game": ("gametocyte", "gametocytes"),
}

# CSV header lies — actual format is (x_topleft, y_topleft, width, height).
COLUMN_RENAME = {"xmin": "x", "xmax": "y", "ymin": "w", "ymax": "h"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to MP-IDB/Falciparum directory.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for train/val/test JSONL.",
    )
    p.add_argument(
        "--grid",
        type=int,
        default=1000,
        help="Normalization grid size (default 1000).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    return p.parse_args()


def normalize_box(
    box: tuple[int, int, int, int], width: int, height: int, grid: int
) -> list[int]:
    x, y, w, h = box
    return [
        round(x * grid / width),
        round(y * grid / height),
        round(w * grid / width),
        round(h * grid / height),
    ]


def denormalize_box(
    box: list[int], width: int, height: int, grid: int
) -> list[int]:
    nx, ny, nw, nh = box
    return [
        round(nx * width / grid),
        round(ny * height / grid),
        round(nw * width / grid),
        round(nh * height / grid),
    ]


def describe_stages(counts: Counter[str]) -> str:
    """Render stage counts as a clinical-style sentence."""
    total = sum(counts.values())
    parasite_word = "parasite" if total == 1 else "parasites"
    # Sort by descending count for natural "predominantly X" ordering
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    parts: list[str] = []
    for stage, n in items:
        sing, plur = STAGE_NAMES[stage]
        word = sing if n == 1 else plur
        parts.append(f"{n} {word}")
    return f"{total} Plasmodium falciparum {parasite_word} detected: {', '.join(parts)}."


def build_target(boxes_norm: list[list[int]], counts: Counter[str]) -> str:
    detection = {"boxes": boxes_norm, "count": sum(counts.values())}
    report = describe_stages(counts)
    return f"{json.dumps(detection)}\n\nReport: {report}"


def split_image_ids(
    image_ids: list[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, set[str]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        sys.exit(f"Ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    rng = random.Random(seed)
    shuffled = image_ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    train = set(shuffled[:n_train])
    val = set(shuffled[n_train : n_train + n_val])
    test = set(shuffled[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def build_records(input_dir: Path, grid: int) -> list[dict]:
    csv_path = input_dir / "mp-idb-falciparum.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).rename(columns=COLUMN_RENAME)
    missing = {"x", "y", "w", "h"} - set(df.columns)
    if missing:
        sys.exit(f"Renamed CSV missing expected columns: {missing}")

    records: list[dict] = []
    for filename, group in df.groupby("filename"):
        img_path = input_dir / "img" / filename
        if not img_path.exists():
            print(f"  [skip] image missing: {img_path}")
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        boxes_abs = [
            (int(r.x), int(r.y), int(r.w), int(r.h)) for r in group.itertuples()
        ]
        boxes_norm = [normalize_box(b, W, H, grid) for b in boxes_abs]
        counts = Counter(group["parasite_type"])
        records.append(
            {
                "image_id": Path(filename).stem,
                "image_path": str(img_path.resolve()),
                "image_width": W,
                "image_height": H,
                "prompt": PROMPT,
                "target": build_target(boxes_norm, counts),
                "boxes_abs": [list(b) for b in boxes_abs],
                "boxes_norm": boxes_norm,
                "stage_counts": dict(counts),
                "count": int(sum(counts.values())),
            }
        )
    return records


def round_trip_check(records: list[dict], grid: int, n: int = 3) -> None:
    """Verify abs -> norm -> abs is stable within a few pixels."""
    print("\nRound-trip check (abs -> norm -> abs):")
    max_err = 0
    for r in records[:n]:
        W, H = r["image_width"], r["image_height"]
        for abs_box, norm_box in zip(r["boxes_abs"][:2], r["boxes_norm"][:2]):
            recovered = denormalize_box(norm_box, W, H, grid)
            err = max(abs(a - b) for a, b in zip(abs_box, recovered))
            max_err = max(max_err, err)
            print(
                f"  {r['image_id']:30}  abs={abs_box}  norm={norm_box}  "
                f"recovered={recovered}  max_err={err}px"
            )
    # Quantization error scales as max(W, H) / grid. For 2592x1944 / 1000 ~ 2.6 px.
    threshold = max(records[0]["image_width"], records[0]["image_height"]) / grid + 1
    if max_err > threshold:
        sys.exit(f"Round-trip error {max_err}px exceeds threshold {threshold:.1f}px")
    print(f"  OK — max error {max_err}px within tolerance {threshold:.1f}px")


def write_splits(
    records: list[dict],
    splits: dict[str, set[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {r["image_id"]: r for r in records}
    for split_name, ids in splits.items():
        out = output_dir / f"{split_name}.jsonl"
        with out.open("w") as f:
            for image_id in sorted(ids):
                if image_id not in by_id:
                    continue
                f.write(json.dumps(by_id[image_id]) + "\n")
        print(f"  Wrote {out}: {len(ids)} images")


def summarize_splits(
    records: list[dict], splits: dict[str, set[str]]
) -> None:
    by_id = {r["image_id"]: r for r in records}
    print("\nSplit summary:")
    print(f"  {'split':<7} {'images':>7} {'boxes':>7} {'ring':>5} {'tro':>4} {'schi':>5} {'game':>5}")
    for split_name, ids in splits.items():
        stage_total: Counter[str] = Counter()
        box_total = 0
        for image_id in ids:
            r = by_id.get(image_id)
            if r is None:
                continue
            box_total += r["count"]
            stage_total.update(r["stage_counts"])
        print(
            f"  {split_name:<7} {len(ids):>7} {box_total:>7} "
            f"{stage_total['ring']:>5} {stage_total['tro']:>4} "
            f"{stage_total['schi']:>5} {stage_total['game']:>5}"
        )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input dir not found: {args.input}")

    records = build_records(args.input, args.grid)
    print(f"\nBuilt {len(records)} image records from MP-IDB Falciparum.")
    print(f"Sample target:\n  {records[0]['target']}")

    round_trip_check(records, args.grid)

    image_ids = [r["image_id"] for r in records]
    splits = split_image_ids(image_ids, tuple(args.ratios), args.seed)
    summarize_splits(records, splits)

    print(f"\nWriting JSONL to {args.output}/")
    write_splits(records, splits, args.output)


if __name__ == "__main__":
    main()
