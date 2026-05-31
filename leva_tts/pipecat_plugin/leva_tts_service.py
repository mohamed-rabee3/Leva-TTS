"""
Pipecat TTSService plugin for Leva-TTS.

Follows the Pipecat service pattern:
  - Subclasses pipecat.services.ai_services.TTSService
  - Implements run_tts() as an async generator
  - Emits TTSStartedFrame → TTSAudioRawFrame(s) → TTSStoppedFrame
  - Reports TTFB (time-to-first-byte) via self._metrics
  - text_aggregation_mode = TOKEN for lowest latency

Two integration modes:
  1. LOCAL  – uses LevaTTSEngine directly (GPU must be on same machine)
  2. REMOTE – connects to the FastAPI WebSocket server

Usage (local)::

    from leva_tts.pipecat_plugin import LevaTTSService

    tts = LevaTTSService(
        mode="local",
        checkpoint="./checkpoints/best_model",
        speaker_wav="./data/reference_speaker.wav",
    )
    pipeline = Pipeline([..., tts, ...])

Usage (remote)::

    tts = LevaTTSService(
        mode="remote",
        server_url="ws://localhost:8000/stream",
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from typing import AsyncGenerator, Optional

import numpy as np

logger = logging.getLogger(__name__)

def _noop_decorator(fn):
    return fn

try:
    from pipecat.frames.frames import (
        ErrorFrame, Frame,
        TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame,
    )
    from pipecat.services.ai_services import TTSService
    _PIPECAT_AVAILABLE = True

    # traced_tts moved/was removed across pipecat versions — make it optional
    try:
        from pipecat.utils.tracing.helpers import traced_tts
    except Exception:
        try:
            from pipecat.utils.tracing.service_decorators import traced_tts
        except Exception:
            traced_tts = _noop_decorator
except Exception as _pc_err:
    _PIPECAT_AVAILABLE = False
    logger.warning(f"pipecat-ai not available ({_pc_err}). Install: pip install pipecat-ai openai")
    class TTSService:  # type: ignore
        def __init__(self, *a, **k): pass
    class Frame: pass            # type: ignore
    class ErrorFrame(Frame): pass
    class TTSAudioRawFrame(Frame): pass
    class TTSStartedFrame(Frame): pass
    class TTSStoppedFrame(Frame): pass
    traced_tts = _noop_decorator


class LevaTTSService(TTSService):
    """
    Pipecat TTS service backed by Leva-TTS (local GPU or remote WebSocket).
    """

    def __init__(
        self,
        mode: str = "local",
        # ── local mode ──
        checkpoint: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        device: str = "cuda",
        use_deepspeed: bool = True,
        # ── remote mode ──
        server_url: str = "ws://localhost:8000/stream",
        # ── common ──
        language: str = "ar",
        sample_rate: int = 24_000,
        chunk_size: int = 20,
        **kwargs,
    ):
        if not _PIPECAT_AVAILABLE:
            raise ImportError("pipecat-ai is required: pip install pipecat-ai")

        super().__init__(sample_rate=sample_rate, **kwargs)
        self._sr         = sample_rate
        self.mode        = mode
        self.language    = language
        self.chunk_size  = chunk_size
        self._engine     = None
        self._server_url = server_url

        if mode == "local":
            self._init_local(checkpoint, speaker_wav, device, use_deepspeed)

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_local(self, checkpoint, speaker_wav, device, use_deepspeed):
        from leva_tts.inference.engine import LevaTTSEngine
        if checkpoint:
            from pathlib import Path
            if Path(checkpoint).exists():
                self._engine = LevaTTSEngine.from_checkpoint(
                    checkpoint,
                    speaker_wav=speaker_wav,
                    use_deepspeed=use_deepspeed,
                    device=device,
                )
            else:
                logger.warning(f"Checkpoint not found: {checkpoint}, loading base model")
        if self._engine is None:
            self._engine = LevaTTSEngine.from_pretrained(
                speaker_wav=speaker_wav,
                use_deepspeed=use_deepspeed,
                device=device,
            )
        self._engine.warmup()

    # ── Pipecat interface ─────────────────────────────────────────────────────
    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Core synthesis loop called by Pipecat for each text segment.

        Yields frames in order:
          TTSStartedFrame  →  N × TTSAudioRawFrame  →  TTSStoppedFrame
        """
        logger.debug(f"[LevaTTS] run_tts: {text[:60]!r}")

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        try:
            if self.mode == "local":
                async for chunk in self._local_stream(text):
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=chunk.tobytes(),
                        sample_rate=getattr(self, "sample_rate", None) or self._sr,
                        num_channels=1,
                    )
            else:
                async for chunk in self._remote_stream(text):
                    await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(
                        audio=chunk.tobytes(),
                        sample_rate=getattr(self, "sample_rate", None) or self._sr,
                        num_channels=1,
                    )
        except Exception as e:
            logger.exception("[LevaTTS] Synthesis error")
            yield ErrorFrame(str(e))
        finally:
            yield TTSStoppedFrame()

    # ── Local stream (in-process GPU) ─────────────────────────────────────────
    async def _local_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        loop  = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)

        def _produce():
            try:
                for chunk in self._engine.stream(text, language=self.language,
                                                  chunk_size=self.chunk_size):
                    pcm_s16 = (chunk * 32767).astype(np.int16)
                    loop.call_soon_threadsafe(queue.put_nowait, pcm_s16)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # ── Remote stream (WebSocket) ─────────────────────────────────────────────
    async def _remote_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        try:
            import websockets
        except ImportError:
            raise ImportError("pip install websockets")

        async with websockets.connect(self._server_url) as ws:
            await ws.send(json.dumps({
                "text":       text,
                "language":   self.language,
                "chunk_size": self.chunk_size,
            }))
            async for msg in ws:
                if isinstance(msg, bytes):
                    arr = np.frombuffer(msg, dtype=np.float32)
                    pcm = (arr * 32767).astype(np.int16)
                    yield pcm
                elif isinstance(msg, str):
                    data = json.loads(msg)
                    if data.get("event") == "end":
                        break
                    elif data.get("event") == "error":
                        raise RuntimeError(data.get("detail", "Remote TTS error"))
