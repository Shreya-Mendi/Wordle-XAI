"""
Synthetic data visualizer — run this to inspect training data before training.

Usage:
    cd backend/vision
    python preview_data.py

Outputs:
    data_preview.png  — 4×10 grid showing 10 random samples per class
    data_stats.txt    — label distribution + color mean/std per class

This lets you verify the synthetic tiles look visually correct before training.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Reuse generation logic from train.py
from train import make_tile, TILE_COLORS, LETTERS
from model import STATE_CLASSES


def make_preview_grid(n_per_class: int = 10, tile_size: int = 64) -> Image.Image:
    """
    Generate a grid image: 4 rows (one per class) × n_per_class columns.
    Each cell is a randomly generated tile for that class.
    """
    n_classes = len(STATE_CLASSES)
    padding = 6
    label_width = 90
    cell = tile_size + padding

    total_w = label_width + n_per_class * cell + padding
    total_h = n_classes * cell + padding + 30  # +30 for title row

    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 22))
    draw = ImageDraw.Draw(canvas)

    # Title
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    draw.text((8, 8), "Synthetic Wordle Tile Training Data Preview", fill=(200, 200, 200), font=title_font)

    for row_idx, state in enumerate(STATE_CLASSES):
        y_offset = 30 + row_idx * cell + padding // 2

        # Row label
        color_rgb = TILE_COLORS[state]
        draw.rectangle(
            [(4, y_offset + 4), (label_width - 8, y_offset + tile_size - 4)],
            fill=(30, 30, 32),
            outline=color_rgb,
            width=2,
        )
        draw.text(
            (8, y_offset + tile_size // 2 - 8),
            state.upper(),
            fill=color_rgb,
            font=label_font,
        )

        # Sample tiles
        for col_idx in range(n_per_class):
            letter = random.choice(LETTERS) if state != "empty" else ""
            tile = make_tile(state, letter, size=tile_size, jitter=0.15)

            x_offset = label_width + col_idx * cell + padding // 2
            canvas.paste(tile, (x_offset, y_offset))

    return canvas


def compute_class_stats(n_samples: int = 100) -> dict:
    """
    Compute mean and std of pixel values for each class.
    Lets you verify colour jitter is within expected bounds.
    """
    stats = {}
    for state in STATE_CLASSES:
        pixels = []
        for _ in range(n_samples):
            letter = random.choice(LETTERS) if state != "empty" else ""
            tile = make_tile(state, letter, jitter=0.15)
            arr = np.array(tile).astype(np.float32)
            pixels.append(arr.mean(axis=(0, 1)))  # shape (3,)

        pixels_np = np.array(pixels)  # (n_samples, 3)
        stats[state] = {
            "mean_rgb": pixels_np.mean(axis=0).tolist(),
            "std_rgb":  pixels_np.std(axis=0).tolist(),
            "expected_base_rgb": list(TILE_COLORS[state]),
        }
    return stats


def verify_label_integrity() -> list[str]:
    """
    Sanity-check: generate one tile per class and verify it was created
    with the right background colour (within jitter tolerance).
    Samples a corner region to avoid the centered letter pixels.
    """
    issues = []
    for state in STATE_CLASSES:
        letter = "A" if state != "empty" else ""
        tile = make_tile(state, letter, jitter=0.0)  # no jitter for exact check
        arr = np.array(tile)
        # Sample top-left corner (away from border=2px and letter which is centered)
        corner = arr[6:16, 6:16].mean(axis=(0, 1))
        expected = np.array(TILE_COLORS[state], dtype=float)

        # Allow up to 20 RGB units of difference (anti-aliasing)
        diff = np.abs(corner - expected).max()
        if diff > 20:
            issues.append(
                f"  WARN [{state}]: corner pixel {corner.astype(int)} "
                f"differs from expected {TILE_COLORS[state]} by {diff:.1f}"
            )
        else:
            issues.append(
                f"  OK   [{state}]: corner pixel {corner.astype(int)} "
                f"≈ expected {TILE_COLORS[state]}  (Δ={diff:.1f})"
            )
    return issues


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    preview_path = os.path.join(output_dir, "data_preview.png")
    stats_path   = os.path.join(output_dir, "data_stats.txt")

    print("=" * 60)
    print("  Synthetic Data Inspector")
    print("=" * 60)
    print()

    # 1. Colour integrity check
    print("[1/3] Verifying tile colour integrity (no jitter)...")
    checks = verify_label_integrity()
    for line in checks:
        print(line)
    print()

    # 2. Per-class pixel statistics
    print("[2/3] Computing per-class pixel statistics (100 samples)...")
    stats = compute_class_stats(n_samples=100)
    lines = ["Per-class pixel statistics (mean/std over 100 random tiles)\n"]
    lines.append(f"{'Class':<10} {'Mean R':>8} {'Mean G':>8} {'Mean B':>8}   {'Std R':>7} {'Std G':>7} {'Std B':>7}   Expected RGB")
    lines.append("-" * 85)
    for state, s in stats.items():
        m = s["mean_rgb"]
        sd = s["std_rgb"]
        base = s["expected_base_rgb"]
        row = (
            f"{state:<10} "
            f"{m[0]:>8.1f} {m[1]:>8.1f} {m[2]:>8.1f}   "
            f"{sd[0]:>7.1f} {sd[1]:>7.1f} {sd[2]:>7.1f}   "
            f"({base[0]}, {base[1]}, {base[2]})"
        )
        lines.append(row)
        print("  " + row)

    with open(stats_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  → Stats written to: {stats_path}")
    print()

    # 3. Visual grid
    print("[3/3] Generating visual preview grid (4 classes × 10 tiles)...")
    grid = make_preview_grid(n_per_class=10, tile_size=64)
    grid.save(preview_path)
    print(f"  → Preview saved to: {preview_path}")
    print()

    print("=" * 60)
    print("  INTERPRETATION GUIDE")
    print("=" * 60)
    print()
    print("  data_preview.png  — open this image to visually verify:")
    print("    Row 1 (CORRECT): green background + white letter")
    print("    Row 2 (PRESENT): yellow background + white letter")
    print("    Row 3 (ABSENT):  dark gray background + white letter")
    print("    Row 4 (EMPTY):   near-black background, no letter")
    print()
    print("  What to look for:")
    print("    ✓ Each row should have the right background colour")
    print("    ✓ Letters should be centered, readable, and varied (A-Z)")
    print("    ✓ Slight colour variation between tiles (that's the jitter)")
    print("    ✓ EMPTY tiles should be dark with no visible letter")
    print()
    print("  If anything looks wrong:")
    print("    - Check TILE_COLORS dict in train.py matches real Wordle colours")
    print("    - Reduce jitter (default 0.15) if colours bleed across classes")
    print("    - Check font loading if letters look wrong")
    print()
    print(f"  Classes: {STATE_CLASSES}")
    print(f"  Training default: 2000 tiles/class = 8000 total")
    print(f"  Val split: 10% = 800 val / 7200 train")


if __name__ == "__main__":
    main()
