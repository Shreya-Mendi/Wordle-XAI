/**
 * Wordle XAI Agent — Frontend Application
 *
 * Orchestrates:
 *  1. Screenshot upload → POST /api/analyze → board + Grad-CAM
 *  2. Manual board input → POST /api/manual-suggest → suggestions
 *  3. Rendering: tile board with flip animations, Grad-CAM overlays,
 *     word suggestion cards with per-letter saliency bars,
 *     cross-modal XAI explanation text
 */

const API_BASE = "http://localhost:5001";

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  boardData: null,
  suggestions: null,
  selectedTile: null,
  modelsReady: false,
  healthInterval: null,
};

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initDragDrop();
  initExamples();
  initManualBoard();
  initTabs();
  initModal();
  pollHealth();
});

// ── Health polling ────────────────────────────────────────────────────────────
function pollHealth() {
  const badge = $("#statusBadge");
  badge.classList.add("loading");
  badge.querySelector(".status-text").textContent = "Loading models…";

  state.healthInterval = setInterval(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/health`);
      const d = await r.json();
      if (d.vision_ready && d.solver_ready) {
        clearInterval(state.healthInterval);
        badge.classList.remove("loading");
        badge.classList.add("ready");
        badge.querySelector(".status-text").textContent = "Models Ready";
        state.modelsReady = true;
        $("#analyzeBtn").disabled = false;
      }
    } catch {
      badge.classList.remove("loading");
      badge.classList.add("error");
      badge.querySelector(".status-text").textContent = "Server offline";
      clearInterval(state.healthInterval);
    }
  }, 2000);
}

// ── Example screenshots ───────────────────────────────────────────────────────
function initExamples() {
  document.querySelectorAll(".example-thumb").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const src = btn.dataset.src;
      if (!src) return;

      // Highlight selected
      document.querySelectorAll(".example-thumb").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Fetch the image and convert to a File so handleFile works unchanged
      const resp = await fetch(src);
      const blob = await resp.blob();
      const filename = src.split("/").pop();
      const file = new File([blob], filename, { type: blob.type });
      handleFile(file);
    });
  });
}

// ── Drag & Drop / File Upload ─────────────────────────────────────────────────
function initDragDrop() {
  const zone = $("#dropZone");
  const input = $("#fileInput");

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleFile(file);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0]);
  });

  $("#analyzeBtn").addEventListener("click", () => {
    if (state._currentFile) analyzeScreenshot(state._currentFile);
  });
}

function handleFile(file) {
  state._currentFile = file;
  // Clear example highlight if user uploads their own file
  if (!file.name.startsWith("example_")) {
    document.querySelectorAll(".example-thumb").forEach(b => b.classList.remove("active"));
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = $("#boardPreview");
    preview.src = e.target.result;
    preview.parentElement.classList.add("visible");
    $("#analyzeBtn").disabled = !state.modelsReady;
    $("#dropZone").querySelector(".upload-hint").textContent = file.name;
  };
  reader.readAsDataURL(file);
}

// ── Analyze screenshot ────────────────────────────────────────────────────────
async function analyzeScreenshot(file) {
  setLoading("#visionPanel", true, "Running vision model…");
  $("#analyzeBtn").disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", file);

    const r = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });
    const data = await r.json();

    if (data.error) throw new Error(data.error);

    state.boardData = data.board;
    renderBoard(data.board);
    renderGradcam(data.board);

    // Fire suggestion request
    fetchSuggestions(data.board);
  } catch (err) {
    showError(`Vision model error: ${err.message}`);
  } finally {
    setLoading("#visionPanel", false);
    $("#analyzeBtn").disabled = false;
  }
}

// ── Fetch suggestions ─────────────────────────────────────────────────────────
async function fetchSuggestions(boardRows) {
  setLoading("#suggestionPanel", true, "Computing word suggestions…");

  try {
    const r = await fetch(`${API_BASE}/api/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board: boardRows, num_suggestions: 5 }),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);

    state.suggestions = data;
    renderSuggestions(data);
    renderExplanation(state.boardData, data);
  } catch (err) {
    showError(`Solver error: ${err.message}`);
  } finally {
    setLoading("#suggestionPanel", false);
  }
}

// ── Manual board input ────────────────────────────────────────────────────────
function initManualBoard() {
  const container = $("#manualBoard");
  for (let r = 0; r < 6; r++) {
    const row = document.createElement("div");
    row.className = "manual-row";
    row.dataset.row = r;

    for (let c = 0; c < 5; c++) {
      const input = document.createElement("input");
      input.type = "text";
      input.maxLength = 1;
      input.className = "manual-letter-input";
      input.dataset.row = r;
      input.dataset.col = c;
      input.placeholder = "·";

      input.addEventListener("input", (e) => {
        e.target.value = e.target.value.toUpperCase().replace(/[^A-Z]/g, "");
        if (e.target.value && c < 4) {
          const next = container.querySelector(
            `input[data-row="${r}"][data-col="${c + 1}"]`
          );
          if (next) next.focus();
        }
      });

      input.addEventListener("keydown", (e) => {
        if (e.key === "Backspace" && !e.target.value && c > 0) {
          const prev = container.querySelector(
            `input[data-row="${r}"][data-col="${c - 1}"]`
          );
          if (prev) { prev.focus(); prev.value = ""; }
        }
      });

      const select = document.createElement("select");
      select.className = "state-select";
      select.dataset.row = r;
      select.dataset.col = c;
      [
        ["empty",   "—"],
        ["correct", "🟩 Correct"],
        ["present", "🟨 Present"],
        ["absent",  "⬛ Absent"],
      ].forEach(([val, label]) => {
        const opt = document.createElement("option");
        opt.value = val;
        opt.textContent = label;
        select.appendChild(opt);
      });

      const cell = document.createElement("div");
      cell.style.display = "flex";
      cell.style.flexDirection = "column";
      cell.style.gap = "3px";
      cell.appendChild(input);
      cell.appendChild(select);
      row.appendChild(cell);
    }
    container.appendChild(row);
  }

  $("#manualSubmitBtn").addEventListener("click", submitManualBoard);
  $("#manualClearBtn").addEventListener("click", clearManualBoard);
}

function submitManualBoard() {
  const container = $("#manualBoard");
  const boardRows = [];

  for (let r = 0; r < 6; r++) {
    const row = [];
    let rowEmpty = true;
    for (let c = 0; c < 5; c++) {
      const letter = container.querySelector(
        `input[data-row="${r}"][data-col="${c}"]`
      ).value.toLowerCase();
      const stateEl = container.querySelector(
        `select[data-row="${r}"][data-col="${c}"]`
      );
      const tileState = stateEl ? stateEl.value : "empty";
      row.push({ letter, state: tileState });
      if (tileState !== "empty" && letter) rowEmpty = false;
    }
    if (!rowEmpty) boardRows.push(row);
  }

  if (boardRows.length === 0) {
    showError("Enter at least one guess row.");
    return;
  }

  state.boardData = boardRows;
  renderBoard(boardRows);
  fetchSuggestions(boardRows);
}

function clearManualBoard() {
  $$("#manualBoard input").forEach(el => el.value = "");
  $$("#manualBoard select").forEach(el => el.value = "empty");
  $("#wordleBoard").innerHTML = "<div class='placeholder'><span class='placeholder-icon'>📋</span>Board will appear here</div>";
  $("#gradcamGrid").innerHTML = "<div class='placeholder'><span class='placeholder-icon'>🔥</span>Grad-CAM heatmaps will appear after analysis</div>";
  $("#suggestionsList").innerHTML = "<div class='placeholder'><span class='placeholder-icon'>💡</span>Suggestions will appear here</div>";
  $("#explanationBox").textContent = "";
  state.boardData = null;
  state.suggestions = null;
}

// ── Render board ──────────────────────────────────────────────────────────────
function renderBoard(boardData) {
  const container = $("#wordleBoard");
  container.innerHTML = "";

  boardData.forEach((row) => {
    const rowEl = document.createElement("div");
    rowEl.className = "board-row";

    row.forEach((tile, colIdx) => {
      const tileEl = document.createElement("div");
      tileEl.className = "tile";
      tileEl.dataset.state = tile.state || "empty";
      tileEl.textContent = (tile.letter || "").toUpperCase();

      if (tile.state && tile.state !== "empty") {
        tileEl.classList.add("reveal");
        tileEl.style.animationDelay = `${colIdx * 0.1}s`;
      }

      rowEl.appendChild(tileEl);
    });

    container.appendChild(rowEl);
  });

  // Pad to 6 rows
  const filledRows = boardData.length;
  for (let r = filledRows; r < 6; r++) {
    const rowEl = document.createElement("div");
    rowEl.className = "board-row";
    for (let c = 0; c < 5; c++) {
      const tileEl = document.createElement("div");
      tileEl.className = "tile";
      tileEl.dataset.state = "empty";
      rowEl.appendChild(tileEl);
    }
    container.appendChild(rowEl);
  }
}

// ── Render Grad-CAM ───────────────────────────────────────────────────────────
function renderGradcam(boardData) {
  const container = $("#gradcamGrid");
  container.innerHTML = "";

  let hasCams = false;

  boardData.forEach((row, rowIdx) => {
    row.forEach((tile, colIdx) => {
      const wrapper = document.createElement("div");
      wrapper.className = "cam-tile";
      wrapper.dataset.state = tile.state || "empty";
      wrapper.dataset.letter = (tile.letter || "").toUpperCase();
      wrapper.dataset.row = rowIdx;
      wrapper.dataset.col = colIdx;

      const imgSrc = tile.gradcam_b64
        ? `data:image/png;base64,${tile.gradcam_b64}`
        : (tile.tile_b64 ? `data:image/png;base64,${tile.tile_b64}` : null);

      if (imgSrc) {
        hasCams = true;
        const img = document.createElement("img");
        img.src = imgSrc;
        img.alt = `Row ${rowIdx + 1}, Col ${colIdx + 1}`;
        wrapper.appendChild(img);

        const label = document.createElement("div");
        label.className = "cam-label";
        label.textContent = (tile.letter || "?").toUpperCase();
        wrapper.appendChild(label);

        wrapper.addEventListener("click", () =>
          openCamModal(tile, rowIdx, colIdx)
        );
      } else {
        wrapper.style.background = "var(--bg-secondary)";
        wrapper.style.opacity = "0.3";
      }

      container.appendChild(wrapper);
    });
  });

  if (!hasCams) {
    container.innerHTML =
      "<div class='placeholder'><span class='placeholder-icon'>🔥</span>Upload a screenshot to see Grad-CAM</div>";
  }
}

// ── Render suggestions ────────────────────────────────────────────────────────
function renderSuggestions(data) {
  const container = $("#suggestionsList");
  container.innerHTML = "";

  const candidateBar = $("#candidatesBar");
  if (candidateBar) {
    candidateBar.innerHTML = `
      <span>Remaining candidates:</span>
      <span class="candidates-count">${data.candidates_remaining ?? "—"}</span>
      <span style="color:var(--color-text-dim)">words</span>
    `;
  }

  if (!data.suggestions || data.suggestions.length === 0) {
    container.innerHTML =
      "<div class='placeholder'><span class='placeholder-icon'>🤔</span>No suggestions (no candidates remaining)</div>";
    return;
  }

  data.suggestions.forEach((s, idx) => {
    const card = document.createElement("div");
    card.className = "suggestion-card";
    card.style.animationDelay = `${idx * 0.08}s`;

    // Header
    const header = document.createElement("div");
    header.className = "suggestion-header";

    const wordEl = document.createElement("span");
    wordEl.className = "suggestion-word";
    wordEl.textContent = s.word;

    const entropyBadge = document.createElement("span");
    entropyBadge.className = "entropy-badge";
    entropyBadge.title = "Expected information gain in bits";
    entropyBadge.textContent = `${s.entropy} bits`;

    header.appendChild(wordEl);
    header.appendChild(entropyBadge);
    card.appendChild(header);

    // Saliency bars (one per letter position)
    if (s.position_saliency && s.position_saliency.length === 5) {
      const salRow = document.createElement("div");
      salRow.className = "saliency-row";

      s.position_saliency.forEach((sal, i) => {
        const cell = document.createElement("div");
        cell.className = "saliency-cell";

        const letterLabel = document.createElement("div");
        letterLabel.className = "saliency-letter";
        letterLabel.textContent = s.word[i] || "";

        const barWrap = document.createElement("div");
        barWrap.className = "saliency-bar-wrap";
        barWrap.title = `Position ${i + 1} saliency: ${(sal * 100).toFixed(0)}%`;

        const barFill = document.createElement("div");
        barFill.className = "saliency-bar-fill";
        barFill.style.width = `${Math.round(sal * 100)}%`;
        // Colour: green (high) → blue (low)
        const hue = Math.round(140 * sal + 200 * (1 - sal));
        barFill.style.setProperty("--bar-color", `hsl(${hue}, 70%, 52%)`);
        barFill.style.background = `hsl(${hue}, 70%, 52%)`;

        barWrap.appendChild(barFill);
        cell.appendChild(letterLabel);
        cell.appendChild(barWrap);
        salRow.appendChild(cell);
      });

      card.appendChild(salRow);
    }

    // Meta
    const meta = document.createElement("div");
    meta.className = "suggestion-meta";
    meta.innerHTML = `
      <span>Worst case: <b>${s.remaining_words}</b> words left</span>
    `;
    card.appendChild(meta);

    // Click to prefill
    card.addEventListener("click", () => {
      prefillWord(s.word);
    });

    container.appendChild(card);
  });
}

// Highlight the selected word in the manual board
function prefillWord(word) {
  // Find the first empty row in the manual board
  for (let r = 0; r < 6; r++) {
    const firstInput = document.querySelector(
      `#manualBoard input[data-row="${r}"][data-col="0"]`
    );
    if (firstInput && !firstInput.value) {
      for (let c = 0; c < 5; c++) {
        const inp = document.querySelector(
          `#manualBoard input[data-row="${r}"][data-col="${c}"]`
        );
        if (inp) inp.value = (word[c] || "").toUpperCase();
      }
      // Switch to manual tab
      activateTab("manual");
      break;
    }
  }
}

// ── Explanation ───────────────────────────────────────────────────────────────
function renderExplanation(boardData, suggData) {
  const box = $("#explanationBox");
  if (!box) return;

  const filledTiles = (boardData || [])
    .flat()
    .filter((t) => t.state && t.state !== "empty");

  const greenCount = filledTiles.filter((t) => t.state === "correct").length;
  const yellowCount = filledTiles.filter((t) => t.state === "present").length;
  const grayCount = filledTiles.filter((t) => t.state === "absent").length;
  const filledRows = (boardData || []).filter((r) =>
    r.some((t) => t.state && t.state !== "empty")
  ).length;

  const best = suggData?.suggestions?.[0];
  const remaining = suggData?.candidates_remaining ?? "—";

  let html = `<span class="tag-vision">[Vision]</span> `;

  if (filledRows > 0) {
    html += `Detected <span class="highlight">${filledRows} filled row${filledRows > 1 ? "s" : ""}</span>: `;
    if (greenCount) html += `<span class="highlight" style="color:var(--color-correct)">${greenCount} correct</span>, `;
    if (yellowCount) html += `<span class="highlight" style="color:var(--color-present)">${yellowCount} present</span>, `;
    if (grayCount) html += `<span class="highlight">${grayCount} absent</span>. `;
    html += `Grad-CAM heatmaps highlight the tile regions the ResNet model used to classify each tile state. `;
  } else {
    html += `No filled rows detected — showing opener recommendations. `;
  }

  html += `<br><span class="tag-nlp">[Solver]</span> `;
  html += `<span class="highlight">${remaining}</span> candidate words remain. `;

  if (best) {
    html += `Top suggestion <span class="highlight" style="color:var(--color-present)">${best.word}</span> scores `;
    html += `<span class="highlight">${best.entropy} bits</span> of expected entropy. `;

    const maxSal = Math.max(...(best.position_saliency || [0]));
    const maxIdx = (best.position_saliency || []).indexOf(maxSal);
    if (maxIdx >= 0) {
      html += `Position ${maxIdx + 1} (<b>${best.word[maxIdx]}</b>) has the highest discriminative saliency `;
      html += `(${(maxSal * 100).toFixed(0)}%) — this letter contributes the most information. `;
    }
  }

  box.innerHTML = html;
}

// ── Grad-CAM Modal ────────────────────────────────────────────────────────────
function initModal() {
  const backdrop = $("#camModal");
  if (!backdrop) return;

  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) closeModal();
  });

  $("#modalClose")?.addEventListener("click", closeModal);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

function openCamModal(tile, row, col) {
  const backdrop = $("#camModal");
  if (!backdrop) return;

  const original = tile.tile_b64
    ? `data:image/png;base64,${tile.tile_b64}`
    : "";
  const gradcam = tile.gradcam_b64
    ? `data:image/png;base64,${tile.gradcam_b64}`
    : "";

  const letter = (tile.letter || "?").toUpperCase();
  const stateLabel = { correct: "Correct", present: "Present", absent: "Absent", empty: "Empty" }[tile.state] || tile.state;
  const stateIcon  = { correct: "✅", present: "🟨", absent: "⬛", empty: "—" }[tile.state] || "";

  // Confidence: cam_confidence is ResNet confidence; tile.confidence is HSV fallback
  const camConf = tile.cam_confidence;
  const hsvConf = tile.confidence;
  const confDisplay = camConf != null
    ? `${(camConf * 100).toFixed(1)}% (ResNet)`
    : (hsvConf != null ? `${(hsvConf * 100).toFixed(1)}% (HSV)` : "—");

  // Uncertainty: flag low-confidence predictions
  const isUncertain = camConf != null && camConf < 0.70;
  const isHighConf  = camConf != null && camConf >= 0.95;

  $("#modalTitle").textContent = `Tile [${row + 1}, ${col + 1}] — "${letter}" — ${stateLabel} ${stateIcon}`;
  $("#modalOriginal").src = original || "";
  $("#modalGradcam").src = gradcam || original || "";

  // Header row
  let desc = `<strong>Letter:</strong> ${letter} &nbsp;|&nbsp; `;
  desc += `<strong>State:</strong> <span style="color:var(--color-${tile.state ?? 'text'})">${stateLabel}</span> &nbsp;|&nbsp; `;
  desc += `<strong>Model confidence:</strong> ${confDisplay}`;

  if (isUncertain) {
    desc += ` &nbsp;<span style="color:#e67e22;font-weight:bold">[Low confidence — model uncertain]</span>`;
  } else if (isHighConf) {
    desc += ` &nbsp;<span style="color:var(--color-correct)">[High confidence]</span>`;
  }
  desc += `<br><br>`;

  // Grad-CAM explanation — tied to actual model output
  if (tile.gradcam_b64) {
    desc += `<strong>Grad-CAM interpretation:</strong> The heatmap shows which pixel regions of this tile most strongly activated ResNet18's last convolutional layer (layer4) for the <em>${stateLabel}</em> class prediction. `;

    if (tile.state === "correct") {
      desc += `For <span style="color:var(--color-correct)">Correct (green)</span> tiles, the model attends primarily to the saturated green background. `;
      desc += `Warm (red/yellow) heatmap regions indicate the highest activation — these are the areas the model weighted most heavily when assigning ${confDisplay} confidence to the Correct class. `;
    } else if (tile.state === "present") {
      desc += `For <span style="color:var(--color-present)">Present (yellow)</span> tiles, activations concentrate on the yellow background hue. `;
      desc += `The model must distinguish yellow from green — heatmap intensity reflects how discriminative each region is between the Present and Correct classes. `;
    } else if (tile.state === "absent") {
      desc += `For <strong>Absent (gray)</strong> tiles, the model classifies by the absence of chromatic signal. `;
      desc += `Diffuse or edge-concentrated activations are expected — the model relies on the low-saturation gray tone as a negative cue. `;
      if (isUncertain) {
        desc += `The low confidence (${confDisplay}) suggests the tile color may be ambiguous between absent and empty in this screenshot. `;
      }
    } else {
      desc += `Empty tiles produce minimal gradient flow — the model's convolutional features respond weakly to the unpopulated dark tile background. `;
    }

    desc += `<br><br><em>Note: bright heatmap areas = high gradient magnitude → model most reliant on those pixels for this classification.</em>`;
  } else {
    desc += `<em>Grad-CAM not available — ResNet model checkpoint not loaded. State was classified by HSV color thresholding only.</em>`;
  }

  $("#modalDescription").innerHTML = desc;
  backdrop.classList.add("open");
}

function closeModal() {
  $("#camModal")?.classList.remove("open");
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function initTabs() {
  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });
}

function activateTab(tabName) {
  $$(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabName));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.dataset.tab === tabName));
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function setLoading(panelSel, active, message = "Processing…") {
  const panel = $(panelSel);
  if (!panel) return;
  let overlay = panel.querySelector(".loading-overlay");
  if (!overlay) return;
  overlay.querySelector(".loading-msg").textContent = message;
  overlay.classList.toggle("active", active);
}

function showError(msg) {
  console.error(msg);
  const toast = document.createElement("div");
  toast.style.cssText = `
    position:fixed; bottom:24px; right:24px;
    background:#c0392b; color:#fff;
    padding:12px 18px; border-radius:8px;
    font-size:0.85rem; z-index:999;
    max-width:360px; box-shadow:0 4px 16px rgba(0,0,0,0.5);
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}
