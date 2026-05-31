"""
Dataset preprocessing pipeline for Leva-TTS.

For every dataset:
  1. Load audio  →  resample to 22 050 Hz mono
  2. VAD trim (librosa, top_db = 30)
  3. Duration filter: 0.5 s – 15 s
  4. SNR filter: ≥ 15 dB
  5. Loudness-normalize to −23 LUFS
  6. Text normalization (numbers, currency, punctuation)
  7. Diacritization via CATTEncoderOnly (skipped if text already has tashkeel)
  8. Write XTTS-compatible metadata.csv  (id|transcript|transcript)

Diacritics note
---------------
Training data that ships without tashkeel (OmniLingual, CommonVoice, raw ASC)
is automatically diacritized by CATTEncoderOnly at this step. Data that
already contains tashkeel (checked by `has_diacritics()`) is passed through
unchanged.  This means the model always trains on diacritized transcripts and
generalises correctly to undiacritized runtime input (which the TextProcessor
also diacritizes at inference time).

Usage
-----
Edit variables at the bottom, then:
    python -m leva_tts.data.preprocess
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)

TARGET_SR   = 22_050
MIN_DUR_S   = 0.5
MAX_DUR_S   = 15.0
TARGET_LUFS = -23.0
MIN_SNR_DB  = 15.0

# Lazy-loaded singletons
_TEXT_PROC = None
def _get_text_processor():
    global _TEXT_PROC
    if _TEXT_PROC is None:
        from leva_tts.text.processor import TextProcessor
        _TEXT_PROC = TextProcessor(verbose=False)
    return _TEXT_PROC


# ── Audio utilities ───────────────────────────────────────────────────────────

def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    wav, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return wav.astype(np.float32), sr


def load_audio_array(array: np.ndarray, orig_sr: int) -> np.ndarray:
    wav = np.array(array, dtype=np.float32)
    if orig_sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=TARGET_SR)
    return wav


def trim_silence(wav: np.ndarray, top_db: int = 30) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed


def loudness_normalize(wav: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-9:
        return wav
    current_lufs = 20.0 * np.log10(rms + 1e-9)
    gain_linear  = 10.0 ** ((target_lufs - current_lufs) / 20.0)
    wav = wav * gain_linear
    peak = np.max(np.abs(wav))
    if peak > 0.99:
        wav *= 0.99 / peak
    return wav


def estimate_snr(wav: np.ndarray, frame_ms: int = 30) -> float:
    frame_len = int(TARGET_SR * frame_ms / 1000)
    frames    = [wav[i:i + frame_len] for i in range(0, len(wav) - frame_len, frame_len)]
    if not frames:
        return 0.0
    rms_frames  = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    noise_floor = np.percentile(rms_frames, 10) + 1e-9
    return float(20 * np.log10(np.max(rms_frames) / noise_floor))


def save_wav(wav: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, TARGET_SR, subtype="PCM_16")


def write_metadata(out_dir: Path, rows: list[tuple[str, str]]):
    """Write XTTS-compatible metadata.csv: id|transcript|transcript"""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for wav_id, text in rows:
            w.writerow([wav_id, text, text])
    logger.info(f"  Wrote {len(rows)} entries → {meta}")


# ── Text normalization + diacritization ───────────────────────────────────────

def normalize_and_diacritize(text: str, already_diacritized: bool = False, lang: str = 'ar') -> str:
    """
    Normalize text and apply partial diacritization (homographs only).

    No CATT dependency.  Text passes through:
      1. Unicode cleanup + number verbalization (via TextProcessor stages)
      2. Levantine lexicon CSV overrides (dialect words get correct diacritics)
      3. Partial diacritization: only known ambiguous homographs get tashkeel

    This produces transcripts that the model can handle both with and without
    diacritics.  Any remaining pronunciation issues are fixed via the lexicon
    CSV (data/levantine_lexicon.csv) without re-processing the audio.
    """
    # English — skip all Arabic normalization
    if lang == "en":
        import re as _re
        return _re.sub(r"\s+", " ", text.lower()).strip()

    norm = _get_text_processor()

    # Stage 1-3: unicode cleanup, numbers, arabic preproc
    s = norm._stage_unicode(text)
    s = norm._stage_numbers(s)
    s = norm._stage_arabic_preproc(s)

    # Apply Levantine lexicon overrides (no CATT — partial diacritics only)
    from leva_tts.text.lexicon import apply_lexicon
    s = apply_lexicon(s)

    # Apply homograph rules from JSON
    # s = _apply_homographs(s)

    return re.sub(r"\s+", " ", s).strip()


# Homograph rules loaded lazily
# _HOMOGRAPHS: dict | None = None

# def _get_homographs() -> dict:
#     global _HOMOGRAPHS
#     if _HOMOGRAPHS is None:
#         import json, re as _re
#         hp = Path("data/homographs.json")
#         if hp.exists():
#             raw = json.loads(hp.read_text(encoding="utf-8"))
#             _HOMOGRAPHS = {
#                 k: v["diacritized"]
#                 for k, v in raw.items()
#                 if not k.startswith("_")
#             }
#         else:
#             _HOMOGRAPHS = {}
    return _HOMOGRAPHS

# def _apply_homographs(text: str) -> str:
#     """Replace bare homograph words with their partially-diacritized form."""
#     hg = _get_homographs()
#     if not hg:
#         return text
#     tokens = re.split(r"(\s+|[،؟؛,\.!?\-:]+)", text)
#     result = []
#     _strip = re.compile(r"[ً-ٰٟ]")
#     for tok in tokens:
#         bare = _strip.sub("", tok.strip())
#         result.append(hg.get(bare, tok))
#     return "".join(result)


# ── Common sample processor ───────────────────────────────────────────────────

def process_sample(
    wav:       np.ndarray,
    text:      str,
    wav_id:    str,
    out_dir:   Path,
    lang:      str   = "ar",
    diacritized: bool = False,
    skip_snr:  bool  = False,
) -> Optional[tuple[str, str]]:
    """
    Full pipeline for a single audio+text pair.
    Returns (wav_id, cleaned_text) on success, None if filtered out.
    """
    try:
        if len(wav) == 0:
            return None
        wav = trim_silence(wav)
        dur = len(wav) / TARGET_SR
        if not (MIN_DUR_S <= dur <= MAX_DUR_S):
            return None
        if not skip_snr and estimate_snr(wav) < MIN_SNR_DB:
            return None
        wav = loudness_normalize(wav)
        save_wav(wav, out_dir / "wavs" / f"{wav_id}.wav")
        clean = normalize_and_diacritize(text)
        if not clean.strip():
            return None
        return wav_id, clean
    except Exception as e:
        logger.debug(f"Sample {wav_id} skipped: {e}")
        return None


# ── Per-dataset processors ────────────────────────────────────────────────────

def process_asc(raw_dir: Path, out_dir: Path):
    """
    Arabic Speech Corpus (halabi2016/arabic_speech_corpus)
    Columns: audio (dict: array float32 @ 48 kHz), text, orthographic, phonetic
    ~3.7 h South Levantine Arabic (Damascene). Text is NOT diacritized.

    Handles three possible layouts produced by download_asc():
      A) Arrow/Dataset format  (save_to_disk) — load_from_disk()
      B) Parquet files         (snapshot_download) — pd.read_parquet()
      C) Raw WAV + TXT files   (manual download) — _process_asc_direct()
    """
    # ── Layout A: Arrow dataset (save_to_disk) ────────────────────────────
    dataset_state = raw_dir / "dataset_info.json"
    if dataset_state.exists():
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(raw_dir))
            _process_asc_from_hf(ds, out_dir)
            return
        except Exception as e:
            logger.warning(f"load_from_disk failed ({e}), trying Parquet fallback")

    # ── Layout B: Parquet files (snapshot_download) ───────────────────────
    parquet_files = list(raw_dir.rglob("*.parquet"))
    if parquet_files:
        logger.info(f"Loading ASC from {len(parquet_files)} Parquet file(s)")
        _process_asc_from_parquet(parquet_files, out_dir)
        return

    # ── Layout C: Raw WAV + TXT ───────────────────────────────────────────
    wav_files = list(raw_dir.rglob("*.wav"))
    if wav_files:
        logger.info(f"Loading ASC from {len(wav_files)} raw WAV files")
        _process_asc_direct(raw_dir, out_dir)
        return

    logger.error(
        f"No recognised ASC layout found in {raw_dir}\n"
        f"  Expected: dataset_info.json (Arrow), *.parquet, or *.wav + *.txt\n"
        f"  Re-run: python scripts/prepare_dataset.py with SKIP_DOWNLOAD=False"
    )


def _process_asc_from_hf(ds, out_dir: Path):
    """Process ASC from a HuggingFace Dataset object (Arrow layout)."""
    rows, skip = [], 0
    for i, sample in enumerate(tqdm(ds, desc="ASC")):
        audio = sample.get("audio") or {}
        array = np.array(audio.get("array", []), dtype=np.float32)
        sr    = int(audio.get("sampling_rate", TARGET_SR))
        text  = (sample.get("orthographic") or sample.get("text") or "").strip()
        if not text or array.size == 0:
            skip += 1; continue
        wav    = load_audio_array(array, sr)
        result = process_sample(wav, text, f"asc_{i:06d}", out_dir, lang="ar")
        if result: rows.append(result)
        else:      skip += 1
    write_metadata(out_dir, rows)
    logger.info(f"ASC (Arrow): {len(rows)} kept, {skip} skipped → {out_dir}")


def _process_asc_from_parquet(parquet_files: list, out_dir: Path):
    """
    Process ASC from raw Parquet files (snapshot_download layout).

    Parquet audio columns in HF datasets are stored as structs:
      {"bytes": <raw audio bytes>, "path": "filename.wav"}
    or as pre-decoded arrays.  We handle both.
    """
    import io
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pip install pandas")

    rows, skip = [], 0
    for pq_file in tqdm(parquet_files, desc="ASC (Parquet)"):
        try:
            df = pd.read_parquet(pq_file)
        except Exception as e:
            logger.warning(f"Cannot read {pq_file.name}: {e}")
            continue

        for i, row in df.iterrows():
            # Text columns
            text = ""
            for col in ("orthographic", "text", "sentence"):
                val = row.get(col, None)
                if val and isinstance(val, str) and val.strip():
                    text = val.strip()
                    break
            if not text:
                skip += 1; continue

            # Audio column
            audio_col = row.get("audio", None)
            wav = None
            if isinstance(audio_col, dict):
                if "array" in audio_col and audio_col["array"] is not None:
                    arr = np.array(audio_col["array"], dtype=np.float32)
                    sr  = int(audio_col.get("sampling_rate", 48000))
                    wav = load_audio_array(arr, sr)
                elif "bytes" in audio_col and audio_col["bytes"]:
                    # Decode raw bytes via soundfile
                    try:
                        import soundfile as sf
                        buf = io.BytesIO(audio_col["bytes"])
                        arr, sr = sf.read(buf, dtype="float32", always_2d=False)
                        wav = load_audio_array(arr, int(sr))
                    except Exception as e:
                        logger.debug(f"Audio bytes decode failed: {e}")
                        skip += 1; continue
            elif isinstance(audio_col, bytes):
                try:
                    import soundfile as sf
                    buf = io.BytesIO(audio_col)
                    arr, sr = sf.read(buf, dtype="float32", always_2d=False)
                    wav = load_audio_array(arr, int(sr))
                except Exception as e:
                    logger.debug(f"Audio bytes decode failed: {e}")
                    skip += 1; continue

            if wav is None or wav.size == 0:
                skip += 1; continue

            uid    = f"asc_pq_{pq_file.stem}_{i}"
            result = process_sample(wav, text, uid, out_dir, lang="ar")
            if result: rows.append(result)
            else:      skip += 1

    write_metadata(out_dir, rows)
    logger.info(f"ASC (Parquet): {len(rows)} kept, {skip} skipped → {out_dir}")


def _process_asc_direct(raw_dir: Path, out_dir: Path):
    """Fallback for raw ASC WAV+TXT layout."""
    rows, skip = [], 0
    for wav_src in tqdm(list(raw_dir.rglob("*.wav")), desc="ASC (direct)"):
        txt_path = wav_src.with_suffix(".txt")
        if not txt_path.exists():
            txt_path = wav_src.parent / (wav_src.stem + ".txt")
        if not txt_path.exists():
            skip += 1; continue
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
            wav, _  = load_audio(wav_src)
            result  = process_sample(wav, text, wav_src.stem, out_dir, lang="ar")
            if result:
                rows.append(result)
            else:
                skip += 1
        except Exception as e:
            logger.debug(f"ASC direct {wav_src.name}: {e}")
            skip += 1
    write_metadata(out_dir, rows)
    logger.info(f"ASC direct: {len(rows)} kept, {skip} skipped")


def _decode_audio_column(ds):
    """
    Force-decode the 'audio' column in a HuggingFace dataset.

    After load_from_disk(), audio features are stored as
    {"path": "...", "bytes": None, "array": None} — not yet decoded.
    Casting to Audio(sampling_rate=None) triggers decoding so that
    each sample's audio dict contains a real float32 'array'.
    """
    try:
        from datasets import Audio
        if "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(sampling_rate=None))
        return ds
    except Exception as e:
        logger.warning(f"Audio cast failed ({e}); will attempt raw byte fallback per-sample")
        return ds


def _get_audio_array(sample: dict) -> tuple:
    """
    Extract (wav_array, sample_rate) from an HF audio sample dict.
    Handles decoded arrays, raw bytes, and file-path references.
    """
    import io as _io

    audio = sample.get("audio") or {}
    if not isinstance(audio, dict):
        return None, TARGET_SR

    sr = int(audio.get("sampling_rate", TARGET_SR) or TARGET_SR)

    # 1. Decoded array (happy path after cast_column)
    arr = audio.get("array")
    if arr is not None:
        a = np.asarray(arr, dtype=np.float32)
        if a.size > 0:
            return a, sr

    # 2. Raw bytes (some HF datasets store audio this way)
    raw_bytes = audio.get("bytes")
    if raw_bytes:
        try:
            a, sr2 = sf.read(_io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            return a.astype(np.float32), int(sr2)
        except Exception as e:
            logger.debug(f"Audio bytes decode failed: {e}")

    # 3. File path (Arrow cache or local WAV)
    path = audio.get("path")
    if path:
        fp = Path(path)
        if fp.exists():
            try:
                a, sr2 = librosa.load(str(fp), sr=None, mono=True)
                return a.astype(np.float32), int(sr2)
            except Exception as e:
                logger.debug(f"Audio path load failed ({fp}): {e}")

    return None, TARGET_SR


def _vad_split_audio(wav: np.ndarray, sr: int,
                     max_dur: float = MAX_DUR_S, top_db: int = 28):
    """
    Split a long recording into segments ≤ max_dur seconds using silence detection.
    Returns list of (start_sample, end_sample) tuples.
    """
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return [(0, len(wav))]

    max_samples = int(max_dur * sr)
    min_samples = int(MIN_DUR_S * sr)
    chunks = []
    seg_start = int(intervals[0][0])
    seg_end   = int(intervals[0][1])

    for start, end in intervals[1:]:
        start, end = int(start), int(end)
        if (seg_end - seg_start) + (end - start) <= max_samples:
            seg_end = end                # extend current chunk
        else:
            if seg_end - seg_start >= min_samples:
                chunks.append((seg_start, seg_end))
            seg_start, seg_end = start, end

    if seg_end - seg_start >= min_samples:
        chunks.append((seg_start, seg_end))

    return chunks if chunks else [(0, min(len(wav), max_samples))]


def process_omnilingual_subset(raw_dir: Path, out_dir: Path, subset_key: str,
                               align_device: str = "cpu"):
    """
    facebook/omnilingual-asr-corpus — apc_Arab (all 3 splits concatenated).

    Dataset specifics:
      - Text column : 'raw_text'  (Levantine Arabic, with diacritics)
      - Duration    : mean ~68 s, max ~98 s  (paragraph-level recordings)

    Processing strategy — CTC Forced Alignment:
      1. For each recording, run ctc-forced-aligner (MMS-300M) to obtain
         word-level timestamps without re-transcribing.
      2. Split the diacritized raw_text at sentence boundaries.
      3. Map each sentence to its audio span from the word timestamps.
      4. Extract audio segments — cuts happen at word boundaries, not mid-word.
      5. Original diacritics are preserved verbatim in the transcript.

      Falls back to improved proportional VAD segmentation if ctc-forced-aligner
      is not installed or alignment fails.

    Install (recommended):
      pip install ctc-forced-aligner
    """
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(raw_dir))
    except Exception as e:
        logger.error(f"Cannot load {subset_key} from {raw_dir}: {e}")
        return

    # Check aligner availability
    try:
        import ctc_forced_aligner   # noqa: F401
        logger.info("  CTC forced aligner available — using word-level alignment")
        use_ctc = True
    except ImportError:
        logger.warning(
            "  ctc-forced-aligner not installed — using VAD fallback. For better alignment: pip install ctc-forced-aligner"
        )
        use_ctc = False

    from leva_tts.data.aligner import align_and_segment, extract_audio as aligner_extract

    logger.info(f"  {subset_key}: {len(ds)} samples, aligning …")

    rows, skip = [], 0
    uid = 0

    for i, sample in enumerate(tqdm(ds, desc=f"OmniLingual/{subset_key}")):
        text = (sample.get("raw_text") or sample.get("prompt") or
                sample.get("sentence") or sample.get("text") or "").strip()
        if not text:
            skip += 1; continue

        wav_full, sr = _get_audio_array(sample)
        if wav_full is None or wav_full.size == 0:
            skip += 1; continue

        # Resample to TARGET_SR before anything else
        wav_full = load_audio_array(wav_full, sr)

        # ── Align and segment ─────────────────────────────────────────────────
        try:
            segments = align_and_segment(
                audio   = wav_full,
                sr      = TARGET_SR,
                raw_text = text,
                device  = align_device if use_ctc else "cpu",
                min_dur = MIN_DUR_S,
                max_dur = MAX_DUR_S,
                pad_sec = 0.10,
            )
        except Exception as exc:
            logger.debug(f"  Sample {i} align failed: {exc}")
            skip += 1; continue

        if not segments:
            skip += 1; continue

        for seg in segments:
            chunk = aligner_extract(wav_full, TARGET_SR, seg)
            if chunk.size == 0:
                continue
            wav_id = f"{subset_key}_{uid:06d}"
            chunk = loudness_normalize(chunk)
            save_wav(chunk, out_dir / "wavs" / f"{wav_id}.wav")
            # Apply partial diacritization (homograph overrides)
            clean_text = normalize_and_diacritize(seg.text, lang="ar")
            if clean_text.strip():
                rows.append((wav_id, clean_text))
                uid += 1

    write_metadata(out_dir, rows)
    logger.info(
        f"{subset_key}: {len(rows)} segments kept  |  {skip} samples skipped → {out_dir}"
    )


def process_common_voice_ar(raw_dir: Path, out_dir: Path):
    """
    Geethuzzz/common_voice_17_0_arabic_New_cleaned
    Columns: audio (dict), sentence (str), speaker_id, etc.
    Cleaned + validated Arabic CV17 subset. Not diacritized.
    """
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(raw_dir))
    except Exception as e:
        logger.error(f"Cannot load CV-AR: {e}")
        return

    # Force audio decoding (same fix as OmniLingual)
    ds = _decode_audio_column(ds)
    logger.info(f"  CV-AR: {len(ds)} samples loaded")

    rows, skip = [], 0
    for i, sample in enumerate(tqdm(ds, desc="CV-AR")):
        text = (sample.get("sentence") or "").strip()
        if not text:
            skip += 1; continue
        wav, sr = _get_audio_array(sample)
        if wav is None or wav.size == 0:
            skip += 1; continue
        wav    = load_audio_array(wav, sr)
        result = process_sample(wav, text, f"cv_{i:07d}", out_dir, lang="ar")
        if result:
            rows.append(result)
        else:
            skip += 1

    write_metadata(out_dir, rows)
    logger.info(f"CV-AR: {len(rows)} kept, {skip} skipped → {out_dir}")


def process_librispeech(
    raw_dir: Path,
    out_dir: Path,
    max_hours: float = 20.0,
):
    """
    openslr/librispeech_asr  train-clean-100
    Multi-speaker, ~100 h total, 16 kHz.  English — no diacritization needed.
    Columns: audio (dict), text, speaker_id, chapter_id, id.

    Args:
        max_hours: Stop after accumulating this many hours of kept audio.
                   Default 20 h — a good balance for ~10-25 h of Arabic data.
                   Set to float("inf") to use the full split.
    """
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(raw_dir))
    except Exception:
        _process_librispeech_direct(raw_dir, out_dir, max_hours=max_hours)
        return

    rows, skip = [], 0
    kept_secs  = 0.0
    max_secs   = max_hours * 3600

    for i, sample in enumerate(tqdm(ds, desc=f"LibriSpeech (cap {max_hours:.0f} h)")):
        if kept_secs >= max_secs:
            logger.info(f"  Reached {max_hours:.0f} h cap — stopping early.")
            break

        audio  = sample.get("audio") or {}
        array  = np.array(audio.get("array", []), dtype=np.float32)
        sr     = int(audio.get("sampling_rate", 16000))
        text   = (sample.get("text") or "").strip().lower()
        spk_id = sample.get("speaker_id", i)
        if not text or array.size == 0:
            skip += 1; continue

        wav    = load_audio_array(array, sr)
        uid    = f"ls_{spk_id}_{i:07d}"
        # English text — skip Arabic normalization entirely; just use raw lowercase
        result = process_sample(wav, text, uid, out_dir, lang="en",
                                skip_snr=False)
        if result:
            rows.append(result)
            kept_secs += len(wav) / TARGET_SR
        else:
            skip += 1

    kept_h = kept_secs / 3600
    write_metadata(out_dir, rows)
    logger.info(
        f"LibriSpeech: {len(rows)} kept ({kept_h:.1f} h), {skip} skipped → {out_dir}"
    )


# Keep old name as alias so any existing call sites still work
process_librispeech_360 = process_librispeech


def _process_librispeech_direct(raw_dir: Path, out_dir: Path, max_hours: float = 20.0):
    """Process LibriSpeech from raw FLAC tree (direct download fallback)."""
    rows, skip   = [], 0
    kept_secs    = 0.0
    max_secs     = max_hours * 3600
    flac_files   = list(raw_dir.rglob("*.flac"))
    for flac in tqdm(flac_files, desc="LibriSpeech (direct)"):
        if kept_secs >= max_secs:
            break
        trans = flac.parent / f"{flac.parent.name}.trans.txt"
        if not trans.exists():
            skip += 1; continue
        try:
            lines = {ln.split()[0]: " ".join(ln.split()[1:])
                     for ln in trans.read_text().strip().splitlines()}
            text = lines.get(flac.stem, "").lower().strip()
            if not text:
                skip += 1; continue
            wav, _ = load_audio(flac)
            result = process_sample(wav, text, flac.stem, out_dir, lang="en")
            if result:
                rows.append(result)
                kept_secs += len(wav) / TARGET_SR
            else:
                skip += 1
        except Exception as e:
            logger.debug(f"{flac.name}: {e}")
            skip += 1
    write_metadata(out_dir, rows)
    logger.info(f"LibriSpeech direct: {len(rows)} kept ({kept_secs/3600:.1f} h), {skip} skipped")


# ── CLI / CONFIGURATION ───────────────────────────────────────────────────────
# ╔═══════════════════════════════ CONFIGURATION ════════════════════════════╗
DATASET   = "asc"                     # which dataset to preprocess
RAW_DIR   = "./data/raw"              # input: raw downloaded data
PROC_DIR  = "./data/processed"        # output: XTTS-ready data
# ╚═════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw  = Path(RAW_DIR)
    proc = Path(PROC_DIR)

    if DATASET == "asc":
        process_asc(raw / "asc_hf", proc / "asc")
    elif DATASET == "omnilingual_apc":
        process_omnilingual_subset(raw / "omnilingual_apc", proc / "omnilingual_apc", "apc_Arab")
    elif DATASET == "omnilingual_ajp":
        process_omnilingual_subset(raw / "omnilingual_ajp", proc / "omnilingual_ajp", "ajp_Arab")
    elif DATASET == "common_voice_ar":
        process_common_voice_ar(raw / "cv_ar_hf", proc / "cv_ar")
    elif DATASET == "librispeech_360":
        process_librispeech_360(raw / "librispeech_hf", proc / "librispeech")
    else:
        logger.error(f"Unknown DATASET: {DATASET}")