#!/bin/bash
# Wordle XAI Agent — Quick Start Script

set -e

echo "============================================"
echo "  Wordle XAI Agent — Multimodal AI System"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Please install Python 3.11+."
  exit 1
fi

PYTHON=$(command -v python3)
echo "[1/3] Python: $($PYTHON --version)"

# Install dependencies
echo "[2/3] Installing backend dependencies..."
cd "$(dirname "$0")/backend"
$PYTHON -m pip install -r requirements.txt -q

# Start server
echo "[3/3] Starting Flask server on http://localhost:5000"
echo ""
echo "  → Open frontend/index.html in your browser"
echo "  → Or: python -m http.server 8080 --directory frontend"
echo ""
echo "  [Ctrl+C to stop]"
echo ""

$PYTHON app.py
