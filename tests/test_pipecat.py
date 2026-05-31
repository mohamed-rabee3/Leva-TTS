"""Pipecat integration: send text, gather audio chunks as frames."""
import numpy as np
import pytest


def _pipecat_available():
    try:
        import leva_tts.pipecat_plugin.leva_tts_service as m
        return m._PIPECAT_AVAILABLE
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _pipecat_available(),
                                reason="pipecat-ai not available")


@pytest.fixture(scope="module")
def svc_params(checkpoint_dir, reference_wav):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return dict(mode="local", checkpoint=checkpoint_dir,
                speaker_wav=reference_wav, language="ar",
                sample_rate=24_000, chunk_size=20)


def _build_and_run(svc_params, text):
    """Build the service AND run run_tts inside one event loop (pipecat needs it)."""
    import asyncio
    from leva_tts.pipecat_plugin import LevaTTSService

    async def _go():
        svc = LevaTTSService(**svc_params)
        frames = []
        async for f in svc.run_tts(text):
            frames.append(f)
        return svc, frames

    return asyncio.new_event_loop().run_until_complete(_go())


def test_service_constructs(svc_params):
    import asyncio
    from leva_tts.pipecat_plugin import LevaTTSService

    async def _go():
        return LevaTTSService(**svc_params)

    svc = asyncio.new_event_loop().run_until_complete(_go())
    assert svc is not None
    assert svc.mode == "local"


def test_run_tts_emits_frames(svc_params):
    from pipecat.frames.frames import TTSAudioRawFrame
    _, frames = _build_and_run(svc_params, "كيفك اليوم؟ شو عم تعمل")

    types = [type(f).__name__ for f in frames]
    assert "TTSStartedFrame" in types
    assert "TTSStoppedFrame" in types

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) >= 1, "expected at least one audio frame"

    total = b"".join(f.audio for f in audio_frames)
    assert len(total) > 0
    pcm = np.frombuffer(total, dtype=np.int16)
    assert len(pcm) > 24_000 * 0.2
    for f in audio_frames:
        assert getattr(f, "sample_rate", 24_000) == 24_000


def test_frame_order(svc_params):
    _, frames = _build_and_run(svc_params, "مرحبا كيفك")
    types = [type(f).__name__ for f in frames]
    assert types.index("TTSStartedFrame") < types.index("TTSStoppedFrame")
