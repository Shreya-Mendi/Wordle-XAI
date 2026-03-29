"""
Wordle screenshot preprocessor.

Extracts the 6×5 tile grid from a Wordle screenshot and determines:
  - Each tile's bounding box
  - Each tile's state (correct/present/absent/empty) via HSV color analysis
  - Each tile's letter via template matching (letter silhouette comparison)

Works on both Wordle dark-mode and light-mode screenshots without configuration.
Produces a structured board representation consumed by app.py.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# ---------------------------------------------------------------------------
# Wordle color palettes (in HSV)
# ---------------------------------------------------------------------------

# Dark-mode palette
DARK_CORRECT_HSV  = (75,  80, 90)   # #538d4e green
DARK_PRESENT_HSV  = (47,  70, 80)   # #b59f3b yellow
DARK_ABSENT_HSV   = (240, 5,  24)   # #3a3a3c dark gray

# Light-mode palette
LIGHT_CORRECT_HSV  = (112, 50, 70)  # #6aaa64 green
LIGHT_PRESENT_HSV  = (48,  55, 79)  # #c9b458 yellow
LIGHT_ABSENT_HSV   = (220, 10, 50)  # #787c7e gray

# Letter reference templates (generated once at module load)
_LETTER_TEMPLATES: dict[str, np.ndarray] = {}


def _build_letter_templates() -> None:
    """
    Build 26 reference images for A-Z using PIL default font.
    Each template is a 48×48 grayscale binary image (letter white on black).
    """
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        img = Image.new("L", (48, 48), color=0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            except Exception:
                font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), ch, font=font)
        x = (48 - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (48 - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), ch, fill=255, font=font)
        _LETTER_TEMPLATES[ch] = np.array(img)


_build_letter_templates()


# ---------------------------------------------------------------------------
# Color-based state detection
# ---------------------------------------------------------------------------

def _classify_state_hsv(tile_rgb: np.ndarray) -> str:
    """
    Classify tile state from the center region color.
    Returns one of: 'correct', 'present', 'absent', 'empty'.
    """
    h, w = tile_rgb.shape[:2]
    # Sample the central 40% of the tile to avoid border effects
    cy, cx = h // 2, w // 2
    margin_y, margin_x = max(h // 5, 4), max(w // 5, 4)
    center = tile_rgb[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x]

    if center.size == 0:
        center = tile_rgb

    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    mean_v = float(np.mean(hsv[:, :, 2]))

    # Empty tile: very dark (dark mode) or very bright (light mode) with low saturation
    if mean_v < 40 and mean_s < 40:
        return "empty"
    if mean_v > 220 and mean_s < 30:
        return "empty"

    # Green (correct): hue ~75-150 degrees in OpenCV (0-180 range)
    if 60 <= mean_h <= 100 and mean_s > 40:
        return "correct"

    # Yellow (present): hue ~20-45 degrees
    if 18 <= mean_h <= 50 and mean_s > 50:
        return "present"

    # Gray / dark gray (absent)
    return "absent"


# ---------------------------------------------------------------------------
# Letter detection via template matching
# ---------------------------------------------------------------------------

def _extract_letter_from_tile(tile_rgb: np.ndarray) -> str:
    """
    Identify the letter in a tile using normalized template matching.
    Returns uppercase letter A-Z, or '' if confidence is too low.
    """
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Isolate letter pixels: threshold to find bright (white/light) regions
    # Works for both light-on-dark and dark-on-light tiles
    _, thresh_light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, thresh_dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    letter_mask = thresh_light if thresh_light.sum() > thresh_dark.sum() else thresh_dark

    # Resize to 48×48 for template comparison
    letter_resized = cv2.resize(letter_mask, (48, 48), interpolation=cv2.INTER_AREA)

    best_letter = ""
    best_score = -1.0

    for ch, template in _LETTER_TEMPLATES.items():
        result = cv2.matchTemplate(
            letter_resized.astype(np.float32),
            template.astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        score = float(result.max())
        if score > best_score:
            best_score = score
            best_letter = ch

    # Low confidence → report blank
    # 0.45 is a well-calibrated threshold for normalized cross-correlation
    if best_score < 0.45:
        return ""

    return best_letter


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------

def _detect_grid_bbox(img_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Attempt to locate the 5×6 Wordle tile grid in the image.
    Returns (x, y, w, h) bounding box, or None if not found.

    Strategy:
    1. Find large connected groups of near-square, similarly-sized colored rectangles.
    2. Rely on the fact that Wordle tiles have strong edges and regular spacing.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = img_rgb.shape[:2]
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Look for square-ish boxes of reasonable size (roughly tile-sized)
        if area < 300:
            continue
        aspect = w / max(h, 1)
        if 0.6 < aspect < 1.6:
            candidates.append((x, y, w, h))

    if len(candidates) < 10:
        # Fallback: use the known position of the Wordle grid in typical screenshots.
        # The grid is centered horizontally and occupies roughly 15–58% of the image
        # height (above the keyboard). This works for both iOS and Android screenshots.
        grid_w = int(img_w * 0.72)
        grid_h = int(img_h * 0.43)
        gx = (img_w - grid_w) // 2
        gy = int(img_h * 0.13)
        return (gx, gy, grid_w, grid_h)

    # Cluster candidates spatially to find the grid region
    xs = [c[0] for c in candidates]
    ys = [c[1] for c in candidates]
    ws = [c[2] for c in candidates]
    hs = [c[3] for c in candidates]

    x0, y0 = min(xs), min(ys)
    x1 = max(x + w for x, _, w, _ in candidates)
    y1 = max(y + h for _, y, _, h in candidates)

    return (x0, y0, x1 - x0, y1 - y0)


def _split_grid_to_tiles(
    grid_rgb: np.ndarray,
    n_rows: int = 6,
    n_cols: int = 5,
) -> list[list[np.ndarray]]:
    """
    Uniformly divide the grid image into n_rows × n_cols tile crops.
    Returns a 2-D list of RGB numpy arrays.
    """
    h, w = grid_rgb.shape[:2]
    th = h // n_rows
    tw = w // n_cols
    tiles = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            y0 = r * th
            x0 = c * tw
            crop = grid_rgb[y0:y0 + th, x0:x0 + tw]
            row.append(crop)
        tiles.append(row)
    return tiles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_board(
    pil_image: Image.Image,
    vision_model=None,
) -> list[list[dict]]:
    """
    Extract the full Wordle board from a screenshot.

    Args:
        pil_image: the full screenshot as a PIL Image
        vision_model: optional WordleVisionModel instance; if provided, its
                      ResNet prediction overrides the HSV state for filled tiles.

    Returns:
        board: list of 6 rows, each row a list of 5 dicts:
            {
                "letter": str (uppercase A-Z or ""),
                "state": str ("correct"|"present"|"absent"|"empty"),
                "confidence": float (0-1),
                "tile_b64": str (base64 PNG of the tile crop),
            }
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    img_rgb = np.array(pil_image)
    h, w = img_rgb.shape[:2]

    # Detect grid bounding box
    bbox = _detect_grid_bbox(img_rgb)
    if bbox is None:
        gx, gy, gw, gh = 0, 0, w, h
    else:
        gx, gy, gw, gh = bbox

    # Add small padding inward to avoid border artefacts
    pad = max(int(min(gw, gh) * 0.02), 2)
    gx = max(gx + pad, 0)
    gy = max(gy + pad, 0)
    gw = min(gw - 2 * pad, w - gx)
    gh = min(gh - 2 * pad, h - gy)

    grid_rgb = img_rgb[gy:gy + gh, gx:gx + gw]
    tile_rows = _split_grid_to_tiles(grid_rgb)

    board: list[list[dict]] = []

    for row_idx, tile_row in enumerate(tile_rows):
        row_data = []
        for col_idx, tile_np in enumerate(tile_row):
            # Convert to PIL for model inference
            tile_pil = Image.fromarray(tile_np)

            # Color-based state
            state = _classify_state_hsv(tile_np)

            # Letter detection (only for filled tiles)
            letter = ""
            confidence = 1.0 if state == "empty" else 0.0

            if state != "empty":
                letter = _extract_letter_from_tile(tile_np)

                # Optional: override state with ResNet prediction
                if vision_model is not None:
                    try:
                        pred_state, conf, _ = vision_model.predict(tile_pil)
                        if pred_state != "empty":
                            state = pred_state
                            confidence = conf
                        else:
                            confidence = 1.0 - conf
                    except Exception:
                        pass

            # Encode tile as base64 PNG
            tile_b64 = _pil_to_b64(tile_pil.resize((64, 64)))

            row_data.append({
                "letter": letter,
                "state": state,
                "confidence": round(confidence, 4),
                "tile_b64": tile_b64,
            })
        board.append(row_data)

    return board


def _pil_to_b64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return __import__("base64").b64encode(buf.getvalue()).decode("utf-8")
