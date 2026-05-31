"""Shared pytest fixtures for the Leva-TTS test suite."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _first_reference():
    rj = ROOT / "reference_audios" / "references.json"
    if not rj.exists():
        return None
    data = json.loads(rj.read_text(encoding="utf-8"))
    for r in data:
        ap = ROOT / r["audio_path"]
        if ap.exists():
            return str(ap)
    return None


@pytest.fixture(scope="session")
def reference_wav():
    ref = _first_reference()
    if ref is None:
        pytest.skip("No reference audio available")
    return ref


@pytest.fixture(scope="session")
def checkpoint_dir():
    ck = ROOT / "checkpoints"
    if not ck.exists() or not any(ck.rglob("*.pth")):
        pytest.skip("No fine-tune checkpoint available")
    return str(ck)


@pytest.fixture(scope="session")
def engine(checkpoint_dir, reference_wav):
    """A loaded LevaTTSEngine (session-scoped — loads the model once)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from leva_tts.inference.engine import LevaTTSEngine
    eng = LevaTTSEngine.from_checkpoint(
        checkpoint_dir, speaker_wav=reference_wav,
        use_deepspeed=False, device="cuda",
    )
    return eng


_SAMPLE_TEXTS = {
    "ar":  "كيفك اليوم؟ إنت شو عم تعمل هَلَّق؟",
    "cs":  "هَلَّق أنا عم أشتغل على the project اللي حكيتلك عنه.",
    "en":  "Hello, how are you doing today?",
}


@pytest.fixture(scope="session")
def sample_texts():
    return dict(_SAMPLE_TEXTS)
