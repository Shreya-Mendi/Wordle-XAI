# Wordle XAI Agent — Multimodal Explainable AI System

> **MEng AI Final Project — Explainable Artificial Intelligence**
>
> A multimodal agentic AI that plays Wordle by fusing a vision model and an
> information-theoretic NLP solver, then surfaces real-time explanations of
> *why* it made each decision — bridging the gap between raw model predictions
> and human-understandable reasoning.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [System Overview](#2-system-overview)
3. [Repository Structure](#3-repository-structure)
4. [Modality 1 — Vision Pipeline](#4-modality-1--vision-pipeline)
   - [4.1 Problem: Reading a Wordle Screenshot](#41-problem-reading-a-wordle-screenshot)
   - [4.2 Preprocessing & Grid Detection](#42-preprocessing--grid-detection)
   - [4.3 Tile State Classifier (ResNet18)](#43-tile-state-classifier-resnet18)
   - [4.4 Synthetic Training Data](#44-synthetic-training-data)
   - [4.5 Training Procedure](#45-training-procedure)
   - [4.6 Letter Recognition](#46-letter-recognition)
5. [Modality 2 — NLP Solver](#5-modality-2--nlp-solver)
   - [5.1 Information-Theoretic Approach](#51-information-theoretic-approach)
   - [5.2 Constraint Filtering](#52-constraint-filtering)
   - [5.3 Position Saliency (Token-Level XAI)](#53-position-saliency-token-level-xai)
6. [XAI Methods](#6-xai-methods)
   - [6.1 Grad-CAM (Visual Explainability)](#61-grad-cam-visual-explainability)
   - [6.2 Counterfactual Position Saliency](#62-counterfactual-position-saliency)
   - [6.3 Cross-Modal Explanation](#63-cross-modal-explanation)
7. [Where Models Get Lost in Translation](#7-where-models-get-lost-in-translation)
8. [Architecture & Data Flow](#8-architecture--data-flow)
9. [Performance & Evaluation](#9-performance--evaluation)
10. [Getting Started](#10-getting-started)
11. [API Reference](#11-api-reference)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [References](#13-references)

---

## 1. Motivation

### The Problem With Current Models

Wordle seems trivial on paper: guess a five-letter word in six attempts, using
color feedback (green = correct position, yellow = present but wrong position,
gray = absent). Yet current language models — even frontier LLMs — fail at it
in revealing ways when given only a screenshot.

The core difficulty is a **multimodal translation gap**:

| What the model needs to do | Where it breaks down |
|---|---|
| Identify each tile's background color | Confuses gray (absent) with dark-mode empty; misreads low-contrast screenshots |
| Map colors to game semantics (green → locked) | Conflates visual token with linguistic meaning |
| Extract letter identity from a stylized font | Template matching fails on compressed or resized images |
| Apply elimination logic across all seen tiles | Hallucinates already-eliminated letters |
| Suggest the highest-information next guess | Ignores position constraints, repeats known-absent letters |

General-purpose vision-language models (GPT-4V, Gemini) handle conversational
Wordle well when given text — but when handed a screenshot, they frequently:

- Misread tile colors, especially **absent vs. empty** in dark mode (both are very dark)
- Fail on **duplicate letters** (e.g. if a word has two E's, the pattern is
  asymmetric and requires a two-pass resolution rule that most models skip)
- Treat all gray tiles uniformly, ignoring whether they constrain a specific
  position or the whole word
- Do not articulate *why* a word was recommended

This project asks: **can we build a transparent, two-module system that makes
each decision traceable — and expose exactly where the translation fails?**

### Project Framing (XAI)

This is a final project for an Explainable AI course in an MEng AI program.
The goal is not just to play Wordle well, but to:

1. **Disentangle** the two failure modes — visual misreading vs. reasoning error
2. **Visualize** which pixels the vision model attended to when it (mis)classified a tile
3. **Quantify** which letters in a suggested word carry the most information
4. Build a UI where a human can trace the full inference chain in real time

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       User Interface                        │
│   Screenshot upload  ──or──  Manual tile input              │
└───────────────────┬─────────────────────────────────────────┘
                    │ PNG/JPG
                    ▼
┌───────────────────────────────────┐
│         Vision Pipeline           │
│                                   │
│  1. OpenCV grid detection         │
│  2. HSV color thresholding        │  ──► tile state + confidence
│  3. ResNet18 tile classifier      │
│  4. Template letter matching      │  ──► letter per tile
│  5. Grad-CAM heatmap generation   │  ──► visual explanation
└───────────────────┬───────────────┘
                    │ structured board: [{letter, state, tile_b64, gradcam_b64}]
                    ▼
┌───────────────────────────────────┐
│          NLP Solver               │
│                                   │
│  1. Constraint parsing            │
│     (green/yellow/gray → rules)   │
│  2. Candidate filtering           │
│  3. Entropy scoring (3^5 patterns)│  ──► ranked word suggestions
│  4. Position saliency             │  ──► per-letter explanation
│     (counterfactual ablation)     │
└───────────────────┬───────────────┘
                    │ suggestions + saliency
                    ▼
┌───────────────────────────────────┐
│       XAI Explanation Layer       │
│  Cross-modal natural language     │  ──► human-readable reasoning
│  explanation fusing both outputs  │
└───────────────────────────────────┘
```

The system is **intentionally modular**: either modality can fail independently,
and the explanation layer makes the failure visible rather than hiding it.

---

## 3. Repository Structure

```
Wordle-XAI/
├── backend/
│   ├── app.py                      # Flask API server
│   ├── requirements.txt
│   ├── vision/
│   │   ├── model.py                # ResNet18 classifier definition
│   │   ├── train.py                # Synthetic data generation + training
│   │   ├── gradcam.py              # Grad-CAM implementation
│   │   ├── preprocessor.py         # Grid detection + tile extraction
│   │   ├── preview_data.py         # Data inspection / sanity-check tool
│   │   ├── generate_examples.py    # Generates example screenshots for UI
│   │   └── checkpoints/
│   │       └── best_model.pth      # Trained ResNet18 checkpoint
│   └── player/
│       ├── solver.py               # Information-theoretic Wordle solver
│       └── wordlist.py             # Word list management + constraint filtering
├── frontend/
│   ├── index.html                  # Single-page application
│   ├── app.js                      # UI logic + API calls
│   ├── styles.css                  # Dark-mode Wordle-themed styles
│   └── examples/                   # 4 synthetic example screenshots
│       ├── example_1_opener.png    # Empty board — ask for best opener
│       ├── example_2_early.png     # After 1 guess (CRANE)
│       ├── example_3_mid.png       # After 2 guesses (CRANE + STORE)
│       └── example_4_late.png      # After 3 guesses (narrowing fast)
├── start.sh                        # One-command launcher
└── README.md
```

---

## 4. Modality 1 — Vision Pipeline

### 4.1 Problem: Reading a Wordle Screenshot

A Wordle screenshot is not a structured document — it is a rasterized image of
a 6×5 grid of colored tiles, each containing a letter. Extracting the game
state requires solving three sub-problems:

1. **Locating the grid** in the screenshot (which may include browser chrome,
   keyboard, navigation bar, or share banners)
2. **Classifying each tile's color state** (correct / present / absent / empty)
3. **Reading the letter** inside each filled tile

Each step can fail independently, and the failures compound: a misidentified
tile color silently corrupts all downstream solver reasoning.

### 4.2 Preprocessing & Grid Detection

**File:** `backend/vision/preprocessor.py`

The preprocessor uses OpenCV to locate the tile grid before extracting tiles:

```
Screenshot (RGB)
      │
      ▼
 Canny edge detection (threshold 30/100)
      │
      ▼
 Morphological dilation (3×3 kernel, 2 iterations)
 — closes gaps between tile borders
      │
      ▼
 Contour extraction (external contours only)
      │
      ├─ ≥10 square-ish contours found?
      │   └─ Cluster by spatial extent → bounding box of tile region
      │
      └─ Fallback heuristic (< 10 contours):
          grid_w  = 72% of image width (centered)
          grid_h  = 43% of image height
          grid_y  = 13% from top
         (calibrated against iOS/Android/browser screenshots)
      │
      ▼
 Uniform 6×5 grid split → 30 tile crops
```

Each tile crop is then independently classified for color and letter.

**HSV color classification** runs first (fast, no model needed). The center
40% of the tile is sampled to avoid border artifacts, then converted to HSV
and threshold-matched against known Wordle dark/light-mode palettes:

| State | Hue range | Saturation | Notes |
|---|---|---|---|
| Correct (green) | 60–100° | > 40 | `#538d4e` dark-mode green |
| Present (yellow) | 18–50° | > 50 | `#b59f3b` dark-mode yellow |
| Absent (gray) | any | low | `#3a3a3c` — low saturation |
| Empty | any | very low | Near-black, very low saturation |

The hardest case is **absent vs. empty** — both are near-black with low
saturation. The disambiguating feature is value (brightness): empty tiles are
darker (`V < 40`) than absent tiles. This is the most common failure mode when
the ResNet model is absent and only HSV is used.

If a ResNet checkpoint is loaded, its prediction **overrides** the HSV result
for any non-empty tile, using the neural classifier's richer feature
representation.

### 4.3 Tile State Classifier (ResNet18)

**File:** `backend/vision/model.py`

The visual classifier is a transfer-learned ResNet18 with a custom head:

```
Input: 64×64 RGB tile (ImageNet-normalized)
      │
      ▼
 ResNet18 backbone (pretrained on ImageNet)
 — conv1 → bn1 → relu → maxpool
 — layer1 (2 residual blocks,  64 channels)
 — layer2 (2 residual blocks, 128 channels)
 — layer3 (2 residual blocks, 256 channels)
 — layer4 (2 residual blocks, 512 channels)   ← Grad-CAM target layer
 — AdaptiveAvgPool2d → 512-d feature vector
      │
      ▼
 Custom classification head:
   Linear(512 → 128) → ReLU → Dropout(0.3) → Linear(128 → 4)
      │
      ▼
 Output: softmax probabilities over 4 classes
         [correct, present, absent, empty]
```

**Why ResNet18 over a simpler model?**

The task might look like trivial color classification — but in practice,
screenshots introduce substantial visual noise:

- JPEG compression artifacts around tile borders
- Browser antialiasing on rounded corners
- Screen glare or gamma differences between devices
- Partial tiles at grid boundaries
- Retina/HiDPI screenshots where tiles occupy different pixel counts

ResNet18's residual connections allow it to learn robust features despite these
variations, while remaining small enough to run on CPU in real time
(~10 ms per tile on M-series Apple Silicon).

**Why transfer learning from ImageNet?**

ImageNet pretraining gives the backbone strong low-level feature detectors
(edges, textures, color gradients) that transfer well even to this narrow
domain. Without it, the model would need far more synthetic training data to
converge — and the synthetic data, while accurate in color, lacks the full
distribution of real screenshot artifacts.

**Differential learning rates** are used during fine-tuning:
- Backbone: `lr = 1×10⁻⁵` — slow, to preserve ImageNet feature structure
- Classification head: `lr = 1×10⁻³` — fast, to learn the new 4-class task

### 4.4 Synthetic Training Data

**File:** `backend/vision/train.py`

No labeled dataset of real Wordle screenshots was used. All training data is
**programmatically synthesized** using PIL:

```
Per tile:
  1. Fill 64×64 canvas with the class's official Wordle RGB color
  2. Apply ±15% color jitter (uniform noise per R/G/B channel)
     — simulates monitor variation, screenshot compression, theme differences
  3. Draw a random A-Z letter centered in white (except 'empty' class)
  4. Apply ±2px letter position jitter
  5. Draw a thin border (slightly lighter than background)
```

Official dark-mode Wordle colors:

| Class | Hex | RGB |
|---|---|---|
| `correct` | `#538d4e` | (83, 141, 78) |
| `present` | `#b59f3b` | (181, 159, 59) |
| `absent` | `#3a3a3c` | (58, 58, 60) |
| `empty` | `#121213` | (18, 18, 19) |

**Default dataset:** 2,000 tiles per class × 4 classes = **8,000 samples**
(7,200 train / 800 validation, 90/10 split).

**Data augmentation** applied at training time (not at generation):

| Transform | Parameters | Purpose |
|---|---|---|
| `RandomRotation` | ±5° | Real screenshots aren't perfectly level |
| `ColorJitter` | brightness ±0.2, contrast ±0.2, saturation ±0.15 | Device/monitor variation |
| `RandomCrop` | 64px with 4px padding | Imperfect grid cropping |

Horizontal flip is explicitly **disabled** — flipping a letter changes its
identity (e.g. `d` ↔ `b`).

**Verifying the data before training:**

```bash
cd backend/vision
python preview_data.py
```

Outputs:
- `data_preview.png` — a 4×10 grid of randomly generated tiles (one row per
  class), so you can visually confirm correct colors and readable letters
- `data_stats.txt` — per-class pixel mean/std, so you can verify color jitter
  stays within expected bounds and classes remain separable in RGB space

**Why synthetic data is sufficient here:**

The classification task is fundamentally color-based. The four classes occupy
distinct regions of RGB space, so a model trained on synthetic tiles with
accurate colors and reasonable jitter transfers well to real screenshots. The
main real-world challenge — absent vs. empty disambiguation — is exactly where
the HSV fallback and neural classifier complement each other.

### 4.5 Training Procedure

**File:** `backend/vision/train.py`

```
Optimizer:    AdamW
  Backbone LR:  1×10⁻⁵
  Head LR:      1×10⁻³
  Weight decay: 1×10⁻⁴

Scheduler:    CosineAnnealingLR  (T_max = num_epochs)
Loss:         CrossEntropyLoss
Batch size:   64
Epochs:       20  (default)
```

**Quick smoke test (~1 min on CPU):**

```bash
cd backend/vision
python train.py --epochs 5 --n-per-class 500
```

Observed convergence on the smoke test:

```
Epoch  1/5 | Train Loss: 0.7100 | Val Loss: 0.8254 | Val Acc:  64.0%
Epoch  2/5 | Train Loss: 0.3118 | Val Loss: 0.5382 | Val Acc:  77.0%
Epoch  3/5 | Train Loss: 0.2372 | Val Loss: 0.5028 | Val Acc:  79.5%
Epoch  4/5 | Train Loss: 0.2241 | Val Loss: 0.3251 | Val Acc:  87.5%
Epoch  5/5 | Train Loss: 0.2236 | Val Loss: 0.3981 | Val Acc:  84.0%
```

Full training (20 epochs, 2,000/class) converges to **>98% validation accuracy**.
The remaining ~2% errors are almost exclusively on absent–empty boundaries
where jitter pushes tile brightness into an ambiguous zone — matching the
theoretically hardest case.

### 4.6 Letter Recognition

**File:** `backend/vision/preprocessor.py`

Letter identity within each tile is determined by **normalized template
matching** (no neural network):

```
Tile crop (RGB)
      │
      ▼
 Grayscale conversion
      │
      ▼
 Binary threshold at 180 → isolate white letter pixels
 Also try inverse threshold at 80 → for light-mode tiles
 → pick the mask with higher total pixel mass
      │
      ▼
 Resize mask to 48×48
      │
      ▼
 cv2.matchTemplate against 26 pre-built A-Z templates
 (TM_CCOEFF_NORMED — normalized cross-correlation)
      │
      ▼
 Best match score > 0.45 → return letter
 Score ≤ 0.45             → return "" (uncertain)
```

Letter templates are built at module load using the same font as the tile
generator, so template/tile matching is exact for synthetic images. For real
screenshots with a different typeface (NYT Wordle uses a custom font),
template scores are lower — a known limitation addressed in Section 12.

---

## 5. Modality 2 — NLP Solver

### 5.1 Information-Theoretic Approach

**File:** `backend/player/solver.py`

The solver is based on information theory: at each turn, it picks the word that
**maximally partitions the remaining candidate set**.

For any guess `G` against a set of remaining candidates `C`:

```
Pattern function:  pattern(G, c) → integer in [0, 242]
  Each of the 5 positions encodes:  0 = absent, 1 = present, 2 = correct
  Encoded as:  Σ p[i] × 3^i   (base-3 → base-10)

Expected entropy:  E[G] = −Σ p_k × log₂(p_k)
  where p_k = |{c ∈ C : pattern(G, c) = k}| / |C|
  summed over all 3^5 = 243 possible color patterns
```

A high-entropy guess splits candidates into many small, equal-size groups —
each outcome eliminates a large fraction of remaining words. The opener `RAISE`
scores **5.878 bits** against the full 2,315-word answer set, narrowing the
field by a factor of ~60 on average per guess.

**Duplicate letter handling** is a known correctness pitfall. Pattern
computation uses a two-pass algorithm:

1. **Pass 1 (greens):** mark exact position matches, consume those letters from
   the answer's character pool
2. **Pass 2 (yellows):** mark present-but-wrong-position only if the letter
   exists in the *unconsumed* answer pool

This correctly handles cases like guessing `SPEED` when the answer is `CREEP`:
both E's in `SPEED` compete for the single remaining E in the answer pool,
so only one gets yellow — not both.

**Performance optimization for large candidate sets:**

When more than 1,000 candidates remain (early game), a fast letter-frequency
heuristic pre-filters to the top 600 candidates before full entropy scoring.
For ≤ 1,000 candidates, full entropy is computed directly (~50 ms on CPU).

### 5.2 Constraint Filtering

**File:** `backend/player/wordlist.py`

Board state is parsed into a structured constraint triple:

```python
green:        {position: letter}      # must match exactly
yellow:       {position: set[letter]} # present but NOT at this position
gray:         set[letter]             # absent from the word
must_contain: set[letter]             # union of all yellow letters
```

The gray constraint handles duplicates carefully: a letter appearing in both
gray and yellow/green means there is **exactly one copy** in the word (the
green/yellow occurrence), not zero. The filter preserves this correctly.

### 5.3 Position Saliency (Token-Level XAI)

**File:** `backend/player/solver.py` → `compute_position_saliency()`

To explain *why* a word was recommended, the solver computes a
**counterfactual saliency score** for each letter position:

```
For position i in suggested word W with baseline entropy E[W]:

  For each of the 25 substitute letters s ≠ W[i]:
    W' = W with position i replaced by s
    compute E[W'] against current candidates

  avg_counterfactual = mean over all E[W']

  saliency[i] = max(0,  E[W] − avg_counterfactual)
```

The saliency measures how much *worse* the word would perform on average if
position `i` had a random letter — i.e., how much the actual letter at that
position specifically contributes to the word's information gain.

Scores are normalized to `[0, 1]` (max position = 1.0) and rendered as
per-letter bars in the UI, letting a human see exactly which letters in a
suggestion are "doing the work."

---

## 6. XAI Methods

### 6.1 Grad-CAM (Visual Explainability)

**File:** `backend/vision/gradcam.py`

Grad-CAM (Selvaraju et al., 2017) is applied to every non-empty tile after
ResNet18 classification. It answers: *"Which parts of this tile image did the
model look at to decide it was green?"*

**Implementation:**

```
Forward pass:
  input_tensor → ResNet18 → logits
  (hook captures feature maps A at layer4 output: shape [1, 512, 2, 2])

Backward pass:
  ∂(score for predicted class) / ∂A
  (hook captures gradients G: same shape)

Grad-CAM computation:
  weights αk = GlobalAveragePool(G[k])    for each channel k
  CAM = ReLU( Σk αk × A[k] )             weighted sum of activations
  Upsample CAM to 64×64 (bilinear)
  Normalize to [0, 1]
```

The **ReLU** zeroes out features that *decrease* the class score, showing only
regions that positively contributed to the prediction.

For a 64×64 input, `layer4` produces 2×2 feature maps — each spatial cell
covers a 32×32 pixel region. Resolution is coarse but sufficient to distinguish
corner activation (tile borders) from center activation (background color or
letter pixels).

The CAM is blended with the original tile at α=0.5 using the JET colormap
(blue → green → red), where **red = highest activation**.

**What Grad-CAM reveals per class:**

| State | Expected activation | Interpretation |
|---|---|---|
| `correct` | Diffuse, strong across background | Model relies on saturated green hue |
| `present` | Background-concentrated | Distinguishing yellow from green — the hardest pair |
| `absent` | Diffuse or edge-concentrated | Low-saturation gray: weak but consistent signal |
| `empty` | Near-zero | Minimal gradient flow on unpopulated dark tile |

Low-confidence tiles (ResNet confidence < 70%) are flagged in the UI. They
often show diffuse or contradictory Grad-CAM patterns — a direct visual signal
of where the vision model is uncertain.

### 6.2 Counterfactual Position Saliency

The NLP analog of Grad-CAM: rather than "which pixels mattered," it asks
"which letters mattered." See [Section 5.3](#53-position-saliency-token-level-xai)
for the full derivation.

The method is a form of **Shapley-inspired local attribution** — measuring each
feature's marginal contribution while averaging over all possible alternatives
— without the combinatorial cost of full Shapley computation.

### 6.3 Cross-Modal Explanation

**File:** `frontend/app.js` → `renderExplanation()`

After both pipelines run, the UI generates a natural-language explanation
connecting the two modalities:

```
[Vision] Detected 2 filled rows: 3 correct, 4 present, 3 absent.
         Grad-CAM heatmaps highlight the tile regions the ResNet model
         used to classify each tile state.

[Solver] 47 candidate words remain.
         Top suggestion STORE scores 3.214 bits of expected entropy.
         Position 4 (R) has the highest discriminative saliency (92%)
         — this letter contributes the most information.
```

This is where the XAI value is clearest: a user can see whether the solver's
recommendation is based on a confident, well-supported board state (sharp
Grad-CAM, high-confidence tiles) or is operating on uncertain visual input
(blurry heatmaps, flagged low-confidence tiles).

---

## 7. Where Models Get Lost in Translation

The system was designed to make failure modes visible. Failures fall into three
categories:

### Visual → Semantic (Vision failures)

**1. Absent–empty confusion**

The two darkest tile states (`#3a3a3c` absent vs. `#121213` empty) differ by
only ~40 RGB units. JPEG compression or screenshot downsampling can push an
absent tile into the empty zone. When this happens:
- The tile is excluded from constraint parsing
- The solver incorrectly treats an eliminated letter as still available
- Grad-CAM shows weak, diffuse activation — a visible warning

**2. Letter misidentification**

Template matching fails when:
- The screenshot font differs from the template font (common on the NYT app)
- The tile is very small in the screenshot
- JPEG blocking artifacts break letter contours

When the recognized letter is wrong, the solver may suggest words with
confirmed-absent letters — a silent error with no immediate visual signal.

**3. Grid detection failure**

On unusual screenshots (windowed browser, partial capture, HiDPI scaling), the
contour-based detector may fall back to the fixed-ratio heuristic. If the
actual grid doesn't match expected proportions, all tile extractions are offset.

### Semantic → Reasoning (Solver failures)

**4. Duplicate letter ambiguity**

If the vision model classifies a tile's color correctly but reads the wrong
letter, the constraint filter eliminates words that should remain viable. The
candidate count drops in a plausible-looking way — the error is hard to detect.

**5. Over-elimination from misclassified colors**

A `present` tile misclassified as `absent` causes the solver to eliminate all
words containing that letter — potentially removing the correct answer. The
solver cannot recover from this without re-reading the board.

### Reasoning → Explanation (XAI failures)

**6. Saliency on small candidate sets**

When only a few candidates remain (< 5), counterfactual saliency becomes
degenerate — small random variations dominate the signal. Saliency bars in
this regime should be interpreted cautiously.

---

## 8. Architecture & Data Flow

```
POST /api/analyze
──────────────────────────────────────────────────────────────
Browser      Flask          Preprocessor      ResNet18    Grad-CAM
   │             │                │               │           │
   │── image ───►│                │               │           │
   │             │── PIL Image ──►│               │           │
   │             │                │── detect grid─►           │
   │             │                │── 30 crops    │           │
   │             │                │──────────────►tile state  │
   │             │                │◄────────────── [state, conf]
   │             │◄── board dict (30 tiles, each with tile_b64)
   │             │                                            │
   │             │── tile_pil ───────────────────────────────►explain()
   │             │◄────────────────────────────────────────── gradcam_b64
   │◄── JSON ────│
       (letter, state, confidence, tile_b64, gradcam_b64)

POST /api/suggest
──────────────────────────────────────────────────────────────
   │── board JSON ─►│
   │                │── parse_board_to_constraints()
   │                │── filter_by_constraints() ──► candidates
   │                │── compute_entropy()        ──► ranked words
   │                │── compute_position_saliency() ──► per-letter scores
   │◄── suggestions─│
       (word, entropy, remaining_words,
        position_saliency, letter_contributions)
```

---

## 9. Performance & Evaluation

### Vision Model (Synthetic Test Set)

| Metric | Smoke test (5 ep, 500/class) | Full training (20 ep, 2000/class) |
|---|---|---|
| Validation accuracy | ~87% | >98% |
| Absent ↔ Empty confusion | ~8% | ~1.5% |
| Correct ↔ Present confusion | ~3% | <0.5% |
| Inference time (CPU) | — | ~10 ms / tile |

The high accuracy on synthetic data is expected — the task is fundamentally
color-band classification. The meaningful test is generalization to real
screenshots, where JPEG compression and font differences are the main
challenges.

### NLP Solver

| Metric | Value |
|---|---|
| Best opener (`RAISE`) | 5.878 bits |
| Average bits per guess (opener) | ~5.5–5.9 bits |
| Typical solve depth | 3–4 guesses |
| Theoretical solve rate within 6 guesses | >99% with optimal play |
| Entropy scoring time (≤ 1,000 candidates, CPU) | <50 ms |

### End-to-End (Vision + Solver)

The combined system solves correctly when:
1. All tile colors are correctly classified ← vision accuracy
2. Letters are correctly read ← template matching accuracy
3. Constraints are correctly applied ← solver logic (deterministic)

Failure in step 1 or 2 silently corrupts step 3. Grad-CAM and confidence
scores make these failures detectable — which is the primary XAI contribution
of this system.

---

## 10. Getting Started

### Prerequisites

- Python 3.11+
- No GPU required (CPU-only inference is fast enough: ~10 ms/tile)

### Quick Start (one command)

```bash
git clone https://github.com/Shreya-Mendi/Wordle-XAI.git
cd Wordle-XAI
chmod +x start.sh && ./start.sh
```

Then open `http://localhost:8080` in your browser.

> **macOS note:** Port 5000 is reserved by AirPlay Receiver. The backend
> defaults to port 5001. To use 5000, disable AirPlay Receiver in
> System Settings → General → AirDrop & Handoff, then run with `PORT=5000`.

### Manual Setup

**1. Install dependencies**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Verify synthetic data (recommended before training)**

```bash
cd vision
python preview_data.py
# Generates data_preview.png — visually confirm tile colors and letters
```

**3. Train the vision model**

```bash
# Full training (~5 min on CPU)
python train.py

# Smoke test (~1 min)
python train.py --epochs 5 --n-per-class 500
```

**4. Generate example screenshots**

```bash
python generate_examples.py
```

**5. Start the backend**

```bash
cd ..
python app.py
# Server starts at http://localhost:5001
```

**6. Serve the frontend**

```bash
python -m http.server 8080 --directory ../frontend
# Open http://localhost:8080
```

### Using the UI

**Screenshot mode:**
1. Click an example board (**Opener / 1 guess / 2 guesses / 3 guesses**) or
   drag-drop your own Wordle screenshot
2. Click **Analyze Screenshot**
3. View the reconstructed board with per-tile confidence scores
4. Click any tile in the **Grad-CAM Heatmaps** panel to open the detailed
   modal — showing the original tile, overlay, and interpretation text
5. Word suggestions appear on the right with entropy scores and per-letter
   saliency bars

**Manual mode:**
1. Click the **Manual Input** tab
2. Enter letters and set states (Correct / Present / Absent) for each tile
3. Click **Get Suggestions**
4. Click any suggestion card to prefill it into the manual board

---

## 11. API Reference

### `GET /api/health`

```json
{
  "status": "ok",
  "vision_ready": true,
  "solver_ready": true,
  "models_loaded": true,
  "checkpoint_loaded": true
}
```

### `POST /api/analyze`

**Input:** multipart form with `image` field (PNG/JPG), or JSON
`{"image_b64": "<base64 string>"}`.

**Output:**
```json
{
  "board": [
    [
      {
        "letter": "C",
        "state": "absent",
        "confidence": 0.97,
        "tile_b64": "<base64 PNG of extracted tile>",
        "gradcam_b64": "<base64 PNG of Grad-CAM overlay>",
        "cam_confidence": 0.97
      }
    ]
  ],
  "elapsed_ms": 312.4
}
```

Board is always 6 rows × 5 tiles. Unfilled rows have `state: "empty"` and
empty `gradcam_b64`.

### `POST /api/suggest`

**Input:**
```json
{
  "board": [[{"letter": "c", "state": "absent"}, ...]],
  "num_suggestions": 5
}
```

**Output:**
```json
{
  "suggestions": [
    {
      "word": "STORE",
      "entropy": 3.214,
      "remaining_words": 12,
      "position_saliency": [0.30, 0.92, 0.61, 1.00, 0.44],
      "letter_contributions": {
        "S": 0.28, "T": 0.31, "O": 0.15, "R": 0.18, "E": 0.08
      }
    }
  ],
  "candidates_remaining": 47,
  "elapsed_ms": 48.2
}
```

### `POST /api/manual-suggest`

**Input:**
```json
{
  "guesses": [
    {"word": "CRANE", "result": ["absent","present","absent","absent","correct"]}
  ],
  "num_suggestions": 5
}
```

**Output:** same schema as `/api/suggest`.

---

## 12. Limitations & Future Work

### Current Limitations

**Vision:**
- Letter recognition uses template matching against a single reference font.
  The NYT Wordle app uses a proprietary typeface; scores are lower on real NYT
  screenshots than on synthetic tiles. A small OCR CNN would improve robustness.
- Grid detection fails on heavily cropped screenshots or those with overlaid UI
  elements (share cards, achievement banners).
- Light-mode screenshots have different color palettes; HSV thresholds are
  calibrated for dark mode.

**Solver:**
- Falls back to a curated ~2,300-word bundle when the Kaggle dataset CSV is
  absent (`data/valid_solutions.csv`). Adding the full dataset improves opener
  coverage marginally.
- Not adversarial — assumes the answer is drawn uniformly from the candidate
  set. Hard-mode Wordle (must reuse confirmed letters) requires a modified
  entropy objective.

**XAI:**
- Grad-CAM resolution is limited by `layer4`'s 2×2 spatial output for 64×64
  inputs. Using `layer3` (4×4) or a larger input size would improve spatial
  localization.
- Counterfactual saliency becomes unreliable for very small candidate sets
  (< 5 words).

### Future Directions

- **Real screenshot dataset:** Collect and label real Wordle screenshots to
  close the domain gap with a fine-tuning step
- **OCR upgrade:** Replace template matching with a character-segmentation CNN
- **Hard-mode solver:** Constrain suggestions to reuse confirmed information
- **Score-CAM / Attention rollout:** Sharper spatial localization than standard
  Grad-CAM
- **Failure detection:** Flag "unreliable board parse" automatically when vision
  confidence falls below a threshold, rather than silently propagating errors
- **Multi-turn trace:** Log the full inference chain across all guesses and
  generate a post-game XAI summary showing where the system was confident vs.
  uncertain at each step

---

## 13. References

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
  (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization.* ICCV 2017.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for
  Image Recognition.* CVPR 2016.
- Doshi-Velez, F. & Kim, B. (2017). *Towards a Rigorous Science of Interpretable
  Machine Learning.* arXiv:1702.08608.
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.*
  Bell System Technical Journal.
- 3Blue1Brown (2022). *Solving Wordle using information theory.*
  (Inspiration for the entropy-based solver formulation.)
- NYT Wordle (2021). Original game by Josh Wardle.

---

*MEng Artificial Intelligence — Explainable AI final project.*
*Built to surface where vision-language models lose track of the game,
and how transparency tools can make those failures legible.*
