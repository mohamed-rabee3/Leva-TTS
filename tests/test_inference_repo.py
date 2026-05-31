"""Inference API via the repo (LevaTTSEngine)."""
import numpy as np
import pytest


def test_engine_loads(engine):
    assert engine is not None
    assert engine.sample_rate == 24_000


@pytest.mark.parametrize("key", ["ar", "cs", "en"])
def test_batch_synthesis(engine, sample_texts, key):
    lang = "en" if key == "en" else "ar"
    wav, sr = engine.synthesize(sample_texts[key], language=lang)
    assert sr == 24_000
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32
    assert len(wav) > sr * 0.3          # at least 0.3 s of audio
    assert np.isfinite(wav).all()
    assert np.abs(wav).max() <= 1.0 + 1e-3


def test_streaming_yields_chunks(engine, sample_texts):
    chunks = list(engine.stream(sample_texts["ar"], language="ar"))
    assert len(chunks) >= 1
    assert all(isinstance(c, np.ndarray) for c in chunks)
    combined = np.concatenate(chunks)
    assert len(combined) > engine.sample_rate * 0.3


def test_streaming_matches_batch_roughly(engine, sample_texts):
    wav_b, _ = engine.synthesize(sample_texts["ar"], language="ar")
    wav_s = np.concatenate(list(engine.stream(sample_texts["ar"], language="ar")))
    # Streaming and batch differ (sampling) but should be the same ballpark length
    ratio = len(wav_s) / max(len(wav_b), 1)
    assert 0.3 < ratio < 3.0


def test_long_text_synthesis(engine):
    long = "هَلَّق أنا عم أشتغل على the project. " * 8
    wav, sr = engine.synthesize(long, language="ar")
    assert len(wav) > sr * 1.0          # long input → long audio


def test_vram_reporting(engine):
    v = engine.peak_vram_gb()
    assert v >= 0.0
