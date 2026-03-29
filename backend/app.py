"""
Wordle XAI Agent — Flask Backend

API Endpoints:
    GET  /api/health          — system readiness check
    POST /api/analyze         — vision model: screenshot → board state + Grad-CAM
    POST /api/suggest         — NLP solver: board state → word suggestions + saliency
    POST /api/manual-suggest  — solver only (no screenshot needed)
"""

import os
import sys
import time
import base64
import io
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, origins=["*"])

# ---------------------------------------------------------------------------
# Model initialization (singletons loaded once at startup)
# ---------------------------------------------------------------------------

_vision_model = None
_gradcam = None
_solver = None
_models_loaded = False


def _load_models():
    global _vision_model, _gradcam, _solver, _models_loaded

    try:
        # Add backend directory to path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from vision.model import WordleVisionModel
        from vision.gradcam import TileGradCAM
        from player.solver import WordleSolver

        checkpoint = os.path.join(backend_dir, "vision", "checkpoints", "best_model.pth")
        _vision_model = WordleVisionModel(checkpoint_path=checkpoint)
        _gradcam = TileGradCAM(_vision_model.model)
        _solver = WordleSolver()
        _models_loaded = True
        print("[App] All models loaded successfully.")
    except Exception as e:
        print(f"[App] Warning: model loading failed — {e}")
        traceback.print_exc()
        _models_loaded = False


_load_models()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _decode_image(request) -> Image.Image | None:
    """
    Extract a PIL Image from either:
      - multipart form-data with 'image' field
      - JSON body with 'image_b64' field
    """
    if request.content_type and "multipart" in request.content_type:
        file = request.files.get("image")
        if file is None:
            return None
        return Image.open(file.stream).convert("RGB")

    data = request.get_json(silent=True) or {}
    b64str = data.get("image_b64", "")
    if not b64str:
        return None

    # Strip data URL prefix if present
    if "," in b64str:
        b64str = b64str.split(",", 1)[1]

    image_bytes = base64.b64decode(b64str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        # True only when a trained checkpoint was actually loaded from disk
        "vision_ready": _vision_model is not None and getattr(_vision_model, "loaded_checkpoint", False),
        "solver_ready": _solver is not None,
        "models_loaded": _models_loaded,
        "checkpoint_loaded": _vision_model is not None and getattr(_vision_model, "loaded_checkpoint", False),
    })


@app.post("/api/analyze")
def analyze():
    """
    Analyze a Wordle screenshot.

    Input: multipart image upload (field 'image') OR JSON {'image_b64': '...'}
    Output: board state + per-tile Grad-CAM overlays
    """
    t0 = time.perf_counter()

    pil_image = _decode_image(request)
    if pil_image is None:
        return jsonify({"error": "No image provided. Send 'image' file or 'image_b64' JSON."}), 400

    try:
        from vision.preprocessor import extract_board
        board = extract_board(pil_image, vision_model=_vision_model)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    # Generate Grad-CAM for each filled tile
    if _gradcam is not None:
        for row in board:
            for tile in row:
                if tile["state"] == "empty" or not tile.get("tile_b64"):
                    tile["gradcam_b64"] = ""
                    continue
                try:
                    tile_bytes = base64.b64decode(tile["tile_b64"])
                    tile_pil = Image.open(io.BytesIO(tile_bytes)).convert("RGB")
                    _heatmap, overlay, pred_idx, conf = _gradcam.explain(tile_pil)
                    tile["gradcam_b64"] = TileGradCAM_encode(overlay)
                    tile["cam_confidence"] = round(conf, 3)
                except Exception as e:
                    tile["gradcam_b64"] = ""
    else:
        for row in board:
            for tile in row:
                tile["gradcam_b64"] = ""

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify({"board": board, "elapsed_ms": elapsed})


def TileGradCAM_encode(pil_image) -> str:
    from vision.gradcam import TileGradCAM
    return TileGradCAM.pil_to_b64(pil_image)


@app.post("/api/suggest")
def suggest():
    """
    Get word suggestions given the current board state.

    Input JSON:
        {
          "board": [[{"letter": "R", "state": "correct"}, ...], ...],
          "num_suggestions": 5
        }
    Output: entropy-ranked word list with saliency scores
    """
    t0 = time.perf_counter()

    if _solver is None:
        return jsonify({"error": "Solver not loaded"}), 503

    data = request.get_json(silent=True) or {}
    board_rows = data.get("board", [])
    n = int(data.get("num_suggestions", 5))

    # Empty board is valid — solver returns entropy-sorted opener recommendations
    try:
        result = _solver.get_suggestions(board_rows, n=n)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Solver error: {e}"}), 500

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    result["elapsed_ms"] = elapsed
    return jsonify(result)


@app.post("/api/manual-suggest")
def manual_suggest():
    """
    Suggest words from a manually-specified game state (no screenshot).

    Input JSON:
        {
          "guesses": [
            {"word": "CRANE", "result": ["absent","correct","present","absent","absent"]},
            ...
          ],
          "num_suggestions": 5
        }
    """
    t0 = time.perf_counter()

    if _solver is None:
        return jsonify({"error": "Solver not loaded"}), 503

    data = request.get_json(silent=True) or {}
    guesses = data.get("guesses", [])
    n = int(data.get("num_suggestions", 5))

    # Convert guesses to board format
    board_rows = []
    for guess in guesses:
        word = guess.get("word", "").lower()
        result = guess.get("result", [])
        if len(word) != 5 or len(result) != 5:
            continue
        row = [{"letter": word[i], "state": result[i]} for i in range(5)]
        board_rows.append(row)

    try:
        suggestions = _solver.get_suggestions(board_rows, n=n)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Solver error: {e}"}), 500

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    suggestions["elapsed_ms"] = elapsed
    return jsonify(suggestions)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[App] Starting Wordle XAI server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
