#!/usr/bin/env bash
# Saudi-TTS end-to-end driver: waits for synthetic generation to finish,
# prepares the training data, then launches XTTS-v2 fine-tuning.
#
# Run detached so it survives terminal/session close:
#   setsid nohup bash scripts/run_saudi_pipeline.sh > pipeline.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/home/ai/saudi-tts-finetune/.venv-train/bin/python

ts() { date "+%F %T"; }

echo "[$(ts)] Pipeline started. Waiting for synthetic generation to finish …"

# 1. Wait for the generation process to exit (resume-safe; user may restart it)
while pgrep -f "scripts/generate_lahgetna_data.py" > /dev/null; do
    sleep 120
done

# Sanity: require ≥95% of the 50K utterances before proceeding
N=$(wc -l < data/synthetic_data/metadata.csv 2>/dev/null || echo 0)
N=$((N + $(wc -l < data/synthetic_data/metadata_gpu0.csv 2>/dev/null || echo 0)))
echo "[$(ts)] Generation stopped. metadata rows: $N"
if [ "$N" -lt 47500 ]; then
    echo "[$(ts)] ❌ Only $N/49998 utterances generated — generation died early."
    echo "          Restart it (python3 scripts/generate_lahgetna_data.py resumes),"
    echo "          then re-run this script."
    exit 1
fi

# 2. Wait for LibriSpeech download if still running
while pgrep -f "download_librispeech" > /dev/null; do
    echo "[$(ts)] Waiting for LibriSpeech download …"
    sleep 120
done
if [ ! -d data/raw/librispeech_hf ]; then
    echo "[$(ts)] ⚠️  data/raw/librispeech_hf missing — training will be Arabic-only."
fi

# 3. Prepare datasets (LibriSpeech 20h cap + synthetic resample to 22.05 kHz)
echo "[$(ts)] Preparing datasets …"
$PY scripts/prepare_dataset.py || { echo "[$(ts)] ❌ prepare_dataset failed"; exit 1; }

ROWS=$(wc -l < data/processed/synthetic/metadata.csv 2>/dev/null || echo 0)
echo "[$(ts)] Processed synthetic rows: $ROWS"
if [ "$ROWS" -lt 45000 ]; then
    echo "[$(ts)] ❌ Too few processed rows ($ROWS) — aborting before training."
    exit 1
fi

# Free disk for checkpoints: drop re-downloadable LibriSpeech raw copies
# (only after processed English data verifiably exists)
EN_ROWS=$(wc -l < data/processed/librispeech/metadata.csv 2>/dev/null || echo 0)
if [ "$EN_ROWS" -gt 1000 ]; then
    echo "[$(ts)] Cleaning LibriSpeech raw caches (processed rows: $EN_ROWS) …"
    rm -rf data/raw/librispeech_hf
    rm -rf ~/.cache/huggingface/hub/datasets--openslr--librispeech_asr
fi
df -h / | tail -1

# 4. Fine-tune XTTS-v2
echo "[$(ts)] Starting XTTS-v2 fine-tuning …"
$PY scripts/train.py
echo "[$(ts)] Pipeline finished (exit $?)."
