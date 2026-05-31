"""
CTC Forced Alignment for Levantine Arabic TTS data preparation.

Uses ctc-forced-aligner (MMS-300M / wav2vec2) to align an existing
diacritized transcript to audio at word level, then segments at sentence
boundaries — preserving the original transcription with all diacritics.

Install
-------
    pip install ctc-forced-aligner

Usage
-----
    from leva_tts.data.aligner import align_and_segment
    segments = align_and_segment(audio_array, sample_rate, raw_text)
    # segments = [{"text": "sentence with diacritics", "start": 0.4, "end": 3.2}, ...]
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Arabic/text utilities ─────────────────────────────────────────────────────
_HARAKAT = re.compile(r"[ً-ٰٟ]")   # diacritic marks
_TATWEEL = re.compile(r"ـ")                   # kashida


def strip_diacritics(text: str) -> str:
    return _HARAKAT.sub("", _TATWEEL.sub("", text))


def split_sentences(text: str, min_chars: int = 6) -> List[str]:
    """
    Split Arabic/English mixed text into sentences at punctuation boundaries.
    Handles both Arabic (؟ ! ،) and Latin (? ! .) sentence endings.
    Consecutive commas/semicolons are NOT split points to avoid fragmentation.
    """
    # Sentence boundaries: period, ?, !, Arabic ؟, plus multi-space
    parts = re.split(r"(?<=[.!?؟])\s+", text.strip())
    sents = []
    buffer = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        buffer = (buffer + " " + part).strip() if buffer else part
        # Emit if the sentence is long enough on its own
        if len(strip_diacritics(buffer)) >= min_chars:
            sents.append(buffer)
            buffer = ""
    if buffer:
        sents.append(buffer)
    return [s for s in sents if len(strip_diacritics(s)) >= min_chars]


def tokenize_words(text: str) -> List[str]:
    """Split text into whitespace-delimited words (preserving diacritics)."""
    return [w for w in text.split() if strip_diacritics(w).strip()]


@dataclass
class WordSpan:
    word: str           # original word (may have diacritics)
    word_clean: str     # stripped version used for alignment
    start: float        # seconds
    end: float          # seconds


@dataclass
class Segment:
    text: str           # original text (with diacritics)
    start: float        # seconds
    end: float          # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


# ── CTC Forced Aligner ────────────────────────────────────────────────────────

# ── Lazy singletons (ONNX-based, auto-download on first use) ────────────────
_ALIGN_SESSION   = None   # ONNX InferenceSession (via AlignmentSingleton)
_ALIGN_TOKENIZER = None   # ctc_forced_aligner.Tokenizer


def _load_ctc_model():
    """
    Load ctc-forced-aligner v1.0.2 ONNX model (singleton).
    Downloads MMS_FA model on first call (~100 MB).
    Returns (alignment_singleton, tokenizer).
    """
    global _ALIGN_SESSION, _ALIGN_TOKENIZER
    if _ALIGN_SESSION is not None:
        return _ALIGN_SESSION, _ALIGN_TOKENIZER

    try:
        from ctc_forced_aligner import AlignmentSingleton
    except ImportError:
        raise ImportError(
            "ctc-forced-aligner not installed. "
            "Install:  pip install ctc-forced-aligner"
        )

    try:
        logger.info("Loading CTC alignment model (MMS_FA, ONNX) — may download ~100 MB …")
        _align = AlignmentSingleton()
        # v1.0.2 attrs: .model = ONNX InferenceSession, .tokenizer = Tokenizer
        _ALIGN_SESSION   = _align.model
        _ALIGN_TOKENIZER = _align.tokenizer
        logger.info("CTC alignment model ready.")
        return _ALIGN_SESSION, _ALIGN_TOKENIZER
    except Exception as exc:
        raise RuntimeError(f"CTC model load failed: {exc}") from exc


def _ctc_align(
    audio: np.ndarray,
    sr: int,
    text_clean: str,   # diacritics stripped — model works better without them
    device: str = "cpu",  # v1.0.2 is ONNX-only (CPU); param kept for compat
) -> List[dict]:
    """
    Run CTC forced alignment using ctc-forced-aligner v1.0.2 (ONNX backend).
    Returns list of {word, start, end} dicts.
    *text_clean* should have diacritics stripped; we map back to original later.

    ctc-forced-aligner v1.0.2 API (ONNX-based):
      AlignmentSingleton()      — loads ONNX model, auto-downloads
      Tokenizer()               — vocabulary tokenizer
      preprocess_text(text, romanize:bool, language:str, ...)
      generate_emissions(session, audio_np_16khz, batch_size=4)
      get_alignments(emissions, tokens, tokenizer)
      get_spans(tokens, segments, blank)
      postprocess_results(text_starred, spans, stride, scores)
    """
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    alignment, tokenizer = _load_ctc_model()

    # Resample to 16 kHz (ONNX model requirement)
    if sr != 16_000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)

    # preprocess_text(text, romanize:bool, language:str, split_size, star_frequency)
    # "arb" = Modern Standard Arabic (handles Levantine Arabic well for alignment)
    tokens_starred, text_starred = preprocess_text(
        text_clean,
        False,       # romanize=False (Arabic script, not romanized)
        "arb",       # ISO 639-3 language code
        split_size   = "word",
        star_frequency = "segment",
    )

    # generate_emissions(onnx_session, audio_numpy_1d, ...)
    # _ALIGN_SESSION is already the ONNX InferenceSession (a.model from AlignmentSingleton)
    emissions, stride = generate_emissions(
        alignment,   # ONNX InferenceSession stored in _ALIGN_SESSION
        audio,       # float32 numpy array, 1-D, 16 kHz
        batch_size = 4,
    )

    # Alignment — uses ctc_aligner native extension (needs libstdcxx-ng on Linux)
    # If this fails with GLIBCXX error, run:
    #   conda install -n leva-tts -c conda-forge libstdcxx-ng
    segments, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
    spans                   = get_spans(tokens_starred, segments, blank)

    # Word-level timestamps [{word, start, end}, ...]
    results = postprocess_results(text_starred, spans, stride, scores)
    return results


def _map_timestamps_to_original(
    word_spans: List[dict],
    original_words: List[str],
) -> List[WordSpan]:
    """
    CTC aligner works on diacritic-stripped text. Map its output back to the
    original diacritized words by index (1:1 correspondence after tokenization).
    """
    out = []
    n = min(len(word_spans), len(original_words))
    for i in range(n):
        ws = word_spans[i]
        out.append(WordSpan(
            word       = original_words[i],
            word_clean = ws.get("word", ""),
            start      = float(ws.get("start", 0.0)),
            end        = float(ws.get("end",   0.0)),
        ))
    return out


def _sentence_spans(
    sentences: List[str],
    word_spans: List[WordSpan],
    pad_sec: float = 0.10,
    min_dur: float = 0.5,
    max_dur: float = 14.0,
) -> List[Segment]:
    """
    Map sentences (by word index) to time spans from the aligned word list.
    Adds *pad_sec* on each side and clips to audio boundaries.
    """
    total_audio_end = word_spans[-1].end if word_spans else 0.0
    segments: List[Segment] = []
    word_ptr = 0

    for sent in sentences:
        sent_words = tokenize_words(sent)
        n = len(sent_words)
        if n == 0 or word_ptr >= len(word_spans):
            continue

        slice_ = word_spans[word_ptr:word_ptr + n]
        word_ptr += n

        if not slice_:
            continue

        t_start = max(0.0, slice_[0].start  - pad_sec)
        t_end   = min(total_audio_end, slice_[-1].end + pad_sec)
        dur = t_end - t_start

        if dur < min_dur or dur > max_dur:
            continue

        segments.append(Segment(text=sent, start=t_start, end=t_end))

    return segments


# ── Public API ────────────────────────────────────────────────────────────────

def align_and_segment(
    audio: np.ndarray,
    sr: int,
    raw_text: str,
    device: str = "cpu",
    min_dur: float = 0.5,
    max_dur: float = 14.0,
    pad_sec: float = 0.10,
) -> List[Segment]:
    """
    Main entry point.

    Given a (potentially long) audio array and its diacritized transcript,
    returns a list of Segment objects — each with the exact sentence text
    (diacritics preserved) and precise audio start/end times from CTC alignment.

    Falls back gracefully: CTC → proportional VAD.

    Parameters
    ----------
    audio    : float32 numpy array, any sample rate
    sr       : sample rate of *audio*
    raw_text : full transcript (may contain Arabic diacritics + English)
    device   : "cpu" or "cuda"
    min_dur  : minimum segment duration in seconds
    max_dur  : maximum segment duration in seconds
    pad_sec  : silence padding added to each segment side

    Returns
    -------
    List[Segment] with .text, .start, .end, .duration
    """
    sentences     = split_sentences(raw_text)
    original_words = tokenize_words(raw_text)

    if not sentences or not original_words:
        return []

    # ── Attempt 1: CTC forced alignment ──────────────────────────────────────
    try:
        text_clean   = strip_diacritics(raw_text)
        word_results = _ctc_align(audio, sr, text_clean, device=device)
        word_spans   = _map_timestamps_to_original(word_results, original_words)
        segments     = _sentence_spans(sentences, word_spans,
                                       pad_sec=pad_sec,
                                       min_dur=min_dur, max_dur=max_dur)
        if segments:
            logger.debug(f"CTC alignment: {len(segments)} segments")
            return segments
        logger.warning("CTC alignment produced 0 segments — falling back to VAD")
    except Exception as exc:
        logger.warning(f"CTC alignment failed ({exc}) — falling back to VAD")

    # ── Fallback: proportional VAD with Silero ────────────────────────────────
    return _vad_proportional_segments(audio, sr, sentences,
                                      min_dur=min_dur, max_dur=max_dur,
                                      pad_sec=pad_sec)


def _vad_proportional_segments(
    audio: np.ndarray,
    sr: int,
    sentences: List[str],
    min_dur: float = 0.5,
    max_dur: float = 14.0,
    pad_sec: float = 0.10,
) -> List[Segment]:
    """
    Fallback: split audio at silence boundaries (VAD), map sentences by
    character-count proportion. Better than fixed time-windows because cuts
    happen at natural silence gaps, not mid-word.
    """
    import librosa

    # Get non-silent intervals
    intervals = librosa.effects.split(audio, top_db=30, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return []

    total_dur    = len(audio) / sr
    n_sents      = len(sentences)
    char_counts  = [len(strip_diacritics(s)) for s in sentences]
    total_chars  = sum(char_counts) or 1

    # Target end-times for each sentence (proportional to character count)
    cum_chars    = 0
    target_times = []
    for cc in char_counts:
        cum_chars += cc
        target_times.append(total_dur * cum_chars / total_chars)

    # Candidate cut-points: start of each silence gap
    cut_candidates = sorted({
        int(start) / sr
        for start, end in intervals
        if (end - start) / sr < max_dur * 0.5  # not a speech boundary itself
    } | {float(end) / sr for _, end in intervals})
    cut_candidates = [t for t in cut_candidates if 0 < t < total_dur]
    cut_candidates.sort()

    def nearest_cut(target: float) -> float:
        if not cut_candidates:
            return target
        return min(cut_candidates, key=lambda t: abs(t - target))

    # Build segments
    segments: List[Segment] = []
    prev = 0.0
    for i, (sent, tgt) in enumerate(zip(sentences, target_times)):
        end_t = nearest_cut(tgt) if i < n_sents - 1 else total_dur
        end_t = max(end_t, prev + min_dur)
        end_t = min(end_t, total_dur)

        t_start = max(0.0, prev - pad_sec)
        t_end   = min(total_dur, end_t + pad_sec)
        dur     = t_end - t_start

        if min_dur <= dur <= max_dur:
            segments.append(Segment(text=sent, start=t_start, end=t_end))

        prev = end_t

    return segments


def extract_audio(audio: np.ndarray, sr: int, segment: Segment) -> np.ndarray:
    """Extract the audio array for a Segment."""
    start = int(segment.start * sr)
    end   = int(segment.end   * sr)
    return audio[start:end]
