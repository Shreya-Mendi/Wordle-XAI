"""
Generate synthetic Wordle screenshot examples for the UI.

Produces 4 PNG files in frontend/examples/:
  example_1_opener.png  — empty board (ask for opener)
  example_2_early.png   — 1 guess (CRANE, mixed feedback)
  example_3_mid.png     — 2 guesses (making progress)
  example_4_late.png    — 3 guesses (close to answer)

The screenshots are sized like an iPhone (390×844) so the preprocessor's
fallback grid-detection heuristic maps them correctly.

Usage:
    cd backend/vision
    python generate_examples.py
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Canvas / layout constants  (matches preprocessor.py heuristic exactly)
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 390, 844

# Grid bounds derived from preprocessor fallback:
#   grid_w = int(IMG_W * 0.72) = 280
#   grid_h = int(IMG_H * 0.43) = 362
#   gx = (IMG_W - 280) // 2    = 55
#   gy = int(IMG_H * 0.13)     = 109
GRID_X = 55
GRID_Y = 109
GRID_W = 280
GRID_H = 362
ROWS, COLS = 6, 5

TILE_W = GRID_W // COLS   # 56
TILE_H = GRID_H // ROWS   # 60
GAP = 4                    # gap between tiles (cosmetic only)

# Official Wordle dark-mode colors
BG_COLOR        = (18,  18,  19)   # canvas background
TILE_EMPTY_BG   = (18,  18,  19)   # #121213
TILE_EMPTY_BORDER = (58, 58, 60)   # visible border on empty tiles
TILE_CORRECT    = (83,  141,  78)  # #538d4e
TILE_PRESENT    = (181, 159,  59)  # #b59f3b
TILE_ABSENT     = (58,   58,  60)  # #3a3a3c
HEADER_TEXT_COLOR = (255, 255, 255)

STATE_COLORS = {
    "correct": TILE_CORRECT,
    "present": TILE_PRESENT,
    "absent":  TILE_ABSENT,
    "empty":   TILE_EMPTY_BG,
}


def _get_font(size: int):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_board(board: list[list[tuple[str, str]]]) -> Image.Image:
    """
    board: 6 rows × 5 cols of (letter, state).
    state is one of: 'correct', 'present', 'absent', 'empty'.
    """
    img = Image.new("RGB", (IMG_W, IMG_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header — "WORDLE" title
    title_font = _get_font(28)
    title = "WORDLE"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tx = (IMG_W - (bbox[2] - bbox[0])) // 2
    ty = 28
    draw.text((tx, ty), title, fill=HEADER_TEXT_COLOR, font=title_font)

    # Thin separator line below header
    draw.line([(0, 80), (IMG_W, 80)], fill=(58, 58, 60), width=1)

    # Letter font for tiles
    letter_font = _get_font(int(TILE_H * 0.52))

    for row_idx, row in enumerate(board):
        for col_idx, (letter, state) in enumerate(row):
            x0 = GRID_X + col_idx * TILE_W + GAP // 2
            y0 = GRID_Y + row_idx * TILE_H + GAP // 2
            x1 = x0 + TILE_W - GAP
            y1 = y0 + TILE_H - GAP

            tile_color = STATE_COLORS[state]
            draw.rectangle([(x0, y0), (x1, y1)], fill=tile_color)

            # Border: filled tiles get a slightly lighter rim; empty tiles get a dim border
            if state == "empty":
                draw.rectangle([(x0, y0), (x1, y1)], outline=TILE_EMPTY_BORDER, width=2)
            else:
                border_c = tuple(min(255, c + 20) for c in tile_color)
                draw.rectangle([(x0, y0), (x1, y1)], outline=border_c, width=2)

            # Draw letter
            if letter and state != "empty":
                tw_px = x1 - x0
                th_px = y1 - y0
                lbbox = draw.textbbox((0, 0), letter, font=letter_font)
                lw = lbbox[2] - lbbox[0]
                lh = lbbox[3] - lbbox[1]
                lx = x0 + (tw_px - lw) // 2 - lbbox[0]
                ly = y0 + (th_px - lh) // 2 - lbbox[1]
                draw.text((lx, ly), letter, fill=(255, 255, 255), font=letter_font)

    return img


def _empty_row():
    return [("", "empty")] * 5


# ---------------------------------------------------------------------------
# Define the 4 example boards
# ---------------------------------------------------------------------------

# Game being played towards the target TRITE
# Colours chosen to be visually varied and illustrative.

EXAMPLES = [
    {
        "filename": "example_1_opener.png",
        "label":    "Empty board",
        "caption":  "No guesses yet — ask for best opener",
        "board": [_empty_row()] * 6,
    },
    {
        "filename": "example_2_early.png",
        "label":    "After 1 guess",
        "caption":  "CRANE — R present, E correct",
        "board": [
            [("C","absent"), ("R","present"), ("A","absent"), ("N","absent"), ("E","correct")],
            _empty_row(), _empty_row(), _empty_row(), _empty_row(), _empty_row(),
        ],
    },
    {
        "filename": "example_3_mid.png",
        "label":    "After 2 guesses",
        "caption":  "CRANE + STORE — R & E locked in",
        "board": [
            [("C","absent"),  ("R","present"), ("A","absent"), ("N","absent"), ("E","correct")],
            [("S","absent"),  ("T","present"), ("O","absent"), ("R","correct"), ("E","correct")],
            _empty_row(), _empty_row(), _empty_row(), _empty_row(),
        ],
    },
    {
        "filename": "example_4_late.png",
        "label":    "After 3 guesses",
        "caption":  "Three in — narrowing fast",
        "board": [
            [("C","absent"),  ("R","present"), ("A","absent"),  ("N","absent"),  ("E","correct")],
            [("S","absent"),  ("T","present"), ("O","absent"),  ("R","correct"), ("E","correct")],
            [("B","absent"),  ("R","correct"), ("I","absent"),  ("T","present"), ("E","correct")],
            _empty_row(), _empty_row(), _empty_row(),
        ],
    },
]


# ---------------------------------------------------------------------------
# Generate and save
# ---------------------------------------------------------------------------

def main():
    # Resolve output directory relative to this script
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent.parent / "frontend" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving example screenshots to: {out_dir}")

    for ex in EXAMPLES:
        img = draw_board(ex["board"])
        path = out_dir / ex["filename"]
        img.save(path)
        print(f"  ✓ {ex['filename']}  ({ex['label']})")

    print(f"\nDone — {len(EXAMPLES)} examples generated.")
    print("They will appear as clickable thumbnails in the UI under the drop zone.")


if __name__ == "__main__":
    main()
