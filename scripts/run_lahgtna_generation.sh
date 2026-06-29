#!/usr/bin/env bash
# Saudi TTS data generation via Lahgtna-OmniVoice (text + audio only).
#
# Step 1: gather 100K Saudi sentences  →  data/saudi_100k.txt
# Step 2: synthesize audio with OmniVoice  →  data/synthetic_data/ (+ HF upload)
#
# Prerequisites:
#   - NVIDIA GPU + driver (nvidia-smi works)
#   - pip install git+https://github.com/Oddadmix/Lahgtna-OmniVoice.git
#   - pip install torch soundfile tqdm huggingface_hub
#   - huggingface-cli login   (if uploading to HF_DATASET_REPO)
#
# Usage:
#   bash scripts/run_lahgtna_generation.sh
#   GPUS=0,1 bash scripts/run_lahgtna_generation.sh   # multi-GPU
#
# Run detached:
#   setsid nohup bash scripts/run_lahgtna_generation.sh > generation.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-./.venv/bin/python}"
if [ ! -x "$PY" ]; then PY=python3; fi

ts() { date "+%F %T"; }

echo "[$(ts)] === Step 1/2: Gather 100K Saudi text ==="
$PY scripts/gather_saudi_text.py

LINES=$(wc -l < data/saudi_100k.txt)
echo "[$(ts)] Text file: $LINES lines in data/saudi_100k.txt"
if [ "$LINES" -lt 99000 ]; then
    echo "[$(ts)] ❌ Expected ~100K sentences, got $LINES"
    exit 1
fi

echo "[$(ts)] === Step 2/2: Generate audio with Lahgtna-OmniVoice ==="
$PY scripts/generate_lahgetna_data.py

echo "[$(ts)] Done. Output: data/synthetic_data/metadata.csv"
