"""
Validate the executable inference snippets shown in the README.

These mirror exactly what a pip-install user runs:
    from leva_tts import LevaTTS, SPEAKERS
    tts = LevaTTS(device=..., preprocess_text=..., verbose=...)
    tts.synthesize / zero_shot_synthesize / stream / zero_shot_stream
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def tts(checkpoint_dir):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from leva_tts import LevaTTS
    # Use the local checkpoint dir (avoids the HF download in CI/tests)
    return LevaTTS(checkpoint=checkpoint_dir, device="cuda",
                   preprocess_text=True, verbose=False)


def test_speakers_constant():
    from leva_tts import SPEAKERS
    assert len(SPEAKERS) == 10
    assert "Badr" in SPEAKERS and "Mona" in SPEAKERS


def test_speakers_property(tts):
    assert set(tts.speakers) == {
        "Badr", "Mohamed", "Saad", "Rami", "Fadi",
        "Amina", "Fatma", "Lamyaa", "Mona", "Haneen",
    }


def test_synthesize_builtin_speaker(tts):
    """README: tts.synthesize(text, speaker='Badr')."""
    import soundfile as sf, tempfile, os
    wav, sr = tts.synthesize("هَلَّق أنا عم أشتغل على the project",
                             speaker="Badr", temperature=0.65)
    assert sr == 24_000
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert len(wav) > sr * 0.2
    # README writes it with soundfile
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, wav, sr)
    assert os.path.getsize(path) > 1000
    os.remove(path)


def test_synthesize_invalid_speaker_raises(tts):
    """README: invalid speaker names raise ValueError."""
    with pytest.raises(ValueError):
        tts.synthesize("test", speaker="NotARealSpeaker")


def test_synthesize_english(tts):
    wav, sr = tts.synthesize("Hello, how are you doing today?",
                             speaker="Mona", language="en")
    assert len(wav) > sr * 0.2


def test_zero_shot_synthesize(tts, reference_wav):
    """README: tts.zero_shot_synthesize(text, 'path/to/ref.wav')."""
    wav, sr = tts.zero_shot_synthesize("مرحبا كيفك اليوم", reference_wav)
    assert sr == 24_000 and len(wav) > sr * 0.2


def test_zero_shot_missing_file_raises(tts):
    with pytest.raises(FileNotFoundError):
        tts.zero_shot_synthesize("test", "/nonexistent/ref.wav")


def test_stream_builtin(tts):
    """README: for chunk in tts.stream(text, speaker='Amina')."""
    chunks = list(tts.stream("بِدِّي أحكيلك عن the new feature هَلَّق", speaker="Amina"))
    assert len(chunks) >= 1
    full = np.concatenate(chunks)
    assert len(full) > 24_000 * 0.2


def test_zero_shot_stream(tts, reference_wav):
    """README: for chunk in tts.zero_shot_stream(text, 'ref.wav')."""
    chunks = list(tts.zero_shot_stream("مرحبا كيفك", reference_wav))
    assert len(chunks) >= 1


def test_generation_params_accepted(tts):
    """README: generation params (temperature, repetition_penalty, top_p, ...)."""
    wav, sr = tts.synthesize(
        "كيفك اليوم؟", speaker="Saad",
        temperature=0.7, repetition_penalty=5.0, top_p=0.85, top_k=50, speed=1.0,
    )
    assert len(wav) > 0
