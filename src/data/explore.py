#!/usr/bin/env python3
"""Validate MP-IDB bounding-box format by rendering competing hypotheses.

The CSV header reads `filename,parasite_type,xmin,xmax,ymin,ymax`, but the
values don't satisfy that ordering (e.g. row 1 has xmax=887 < xmin=919).
This script renders each plausible interpretation side-by-side so we can
pick the right one visually.

Example:
    python src/data/explore.py --data-root data/raw/MP-IDB \\
        --species Falciparum --output data/explore_falciparum.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

SPECIES = ["Falciparum", "Malariae", "Ovale", "Vivax"]


@dataclass
class BoxHypothesis:
    name: str
    description: str
    decode: Callable[[float, float, float, float], tuple[float, float, float, float]]


HYPOTHESES: list[BoxHypothesis] = [
    BoxHypothesis(
        name="H1: literal header (xmin, xmax, ymin, ymax)",
        description="Trust header as written. Expected to fail when xmax < xmin.",
        decode=lambda a, b, c, d: (a, c, b, d),
    ),
    BoxHypothesis(
        name="H2: top-left xywh (x, y, w, h)",
        description="Columns are actually x_topleft, y_topleft, width, height.",
        decode=lambda a, b, c, d: (a, b, a + c, b + d),
    ),
    BoxHypothesis(
        name="H3: center xywh (xc, yc, w, h)",
        description="Columns are x_center, y_center, width, height (YOLO-style, absolute pixels).",
        decode=lambda a, b, c, d: (a - c / 2, b - d / 2, a + c / 2, b + d / 2),
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to MP-IDB root (folder containing Falciparum/, Vivax/, etc.).",
    )
    p.add_argument("--species", default="Falciparum", choices=SPECIES)
    p.add_argument(
        "--image",
        default=None,
        help="Specific image filename (e.g. '1305121398-0001-R_S.jpg'). Defaults to first in CSV.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("explore_bbox_hypotheses.png"),
        help="Where to save the side-by-side comparison PNG.",
    )
    return p.parse_args()


def load_annotations(species_dir: Path, species: str) -> pd.DataFrame:
    csv_path = species_dir / f"mp-idb-{species.lower()}.csv"
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def summarize(df: pd.DataFrame, species: str) -> None:
    print(f"\n=== {species} summary ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique images:     {df['filename'].nunique()}")
    print(f"Parasite types:    {df['parasite_type'].value_counts().to_dict()}")
    print("Column ranges:")
    for col in ("xmin", "xmax", "ymin", "ymax"):
        s = df[col]
        print(f"  {col:5}  min={s.min():>6}  max={s.max():>6}  median={s.median():>6}")


def render_hypothesis(
    ax, img: Image.Image, rows: pd.DataFrame, hyp: BoxHypothesis
) -> None:
    ax.imshow(img)
    ax.set_title(f"{hyp.name}\n{hyp.description}", fontsize=9)
    ax.axis("off")
    for _, row in rows.iterrows():
        x0, y0, x1, y1 = hyp.decode(
            row["xmin"], row["xmax"], row["ymin"], row["ymax"]
        )
        rect = mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0 - 4,
            row["parasite_type"],
            color="lime",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"),
        )


def main() -> None:
    args = parse_args()
    species_dir = args.data_root / args.species
    if not species_dir.exists():
        sys.exit(f"Species dir not found: {species_dir}")

    df = load_annotations(species_dir, args.species)
    summarize(df, args.species)

    filename = args.image or df["filename"].iloc[0]
    rows = df[df["filename"] == filename]
    if rows.empty:
        sys.exit(f"No annotations for {filename!r} in CSV.")

    img_path = species_dir / "img" / filename
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")
    img = Image.open(img_path)

    print(f"\nSample image: {filename}")
    print(f"  Dimensions (W x H): {img.size}")
    print(f"  Boxes in image:     {len(rows)}")
    print("  Raw annotation rows:")
    for _, r in rows.iterrows():
        print(
            f"    {r['parasite_type']:6}  "
            f"xmin={r['xmin']:>5}  xmax={r['xmax']:>5}  "
            f"ymin={r['ymin']:>5}  ymax={r['ymax']:>5}"
        )

    fig, axes = plt.subplots(1, len(HYPOTHESES), figsize=(6 * len(HYPOTHESES), 6))
    if len(HYPOTHESES) == 1:
        axes = [axes]
    for ax, hyp in zip(axes, HYPOTHESES):
        render_hypothesis(ax, img, rows, hyp)
    fig.suptitle(
        f"{args.species} — {filename}  ({img.size[0]} x {img.size[1]})", fontsize=11
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"\nSaved comparison: {args.output}")
    print(
        "Pick the hypothesis where boxes land on parasites consistently across all annotations."
    )


if __name__ == "__main__":
    main()
