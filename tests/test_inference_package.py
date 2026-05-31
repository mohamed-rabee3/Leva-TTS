"""Inference API via the installable package (LevaTTS high-level class)."""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def leva(checkpoint_dir):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from leva_tts import LevaTTS
    return LevaTTS(checkpoint=checkpoint_dir, device="cuda",
                   preprocess_text=True, verbose=False)


def test_package_synthesize(leva):
    wav, sr = leva.synthesize("كيفك اليوم؟", speaker="Badr", language="ar")
    assert sr == 24_000
    assert isinstance(wav, np.ndarray)
    assert len(wav) > sr * 0.2


def test_package_synthesize_codeswitch(leva):
    wav, sr = leva.synthesize("هَلَّق عم أشتغل على the project", speaker="Mohamed")
    assert len(wav) > sr * 0.2


def test_package_stream(leva):
    chunks = list(leva.stream("مرحبا كيفك اليوم", speaker="Amina"))
    assert len(chunks) >= 1
    assert all(isinstance(c, np.ndarray) for c in chunks)


def test_package_zero_shot(leva, reference_wav):
    wav, sr = leva.zero_shot_synthesize("مرحبا", reference_wav)
    assert len(wav) > sr * 0.1
