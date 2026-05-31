"""
FastAPI streaming TTS server for Leva-TTS.

Edit the SERVER_* variables at the bottom, then run:
    python -m leva_tts.server.app
or:
    uvicorn leva_tts.server.app:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health    → server status + VRAM
  GET  /metrics   → TTFA/RTF statistics
  POST /synthesize → full WAV synthesis
  WS   /stream    → WebSocket streaming PCM chunks
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from leva_tts.inference.engine import LevaTTSEngine
from leva_tts.server.schemas import HealthResponse, MetricsResponse, SynthRequest

logger = logging.getLogger(__name__)

_ENGINE: Optional[LevaTTSEngine] = None
_TTFA_HISTORY: List[float] = []
_RTF_HISTORY:  List[float] = []
_REQUEST_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ENGINE
    checkpoint  = os.environ.get("LEVA_CHECKPOINT", "")
    speaker_wav = os.environ.get("LEVA_SPEAKER_WAV", "")
    device      = os.environ.get("LEVA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    use_ds      = os.environ.get("LEVA_DEEPSPEED", "1") == "1"

    if checkpoint and Path(checkpoint).exists():
        logger.info(f"Loading checkpoint: {checkpoint}")
        _ENGINE = LevaTTSEngine.from_checkpoint(
            checkpoint, speaker_wav=speaker_wav or None,
            use_deepspeed=use_ds, device=device,
        )
    else:
        logger.info("Loading base XTTS-v2")
        _ENGINE = LevaTTSEngine.from_pretrained(
            speaker_wav=speaker_wav or None,
            use_deepspeed=use_ds, device=device,
        )
    _ENGINE.warmup()
    logger.info("Server ready.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Leva-TTS Streaming Server",
    description="Low-latency Levantine Arabic / English code-switching TTS",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return HealthResponse(
        status="ok", model="xtts-v2-levantine",
        device=str(next(_ENGINE.model.parameters()).device) if _ENGINE else "unknown",
        vram_allocated_gb=round(vram, 3),
        sample_rate=_ENGINE.sample_rate if _ENGINE else 0,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    def pct(data, p): return float(np.percentile(data, p)) if data else 0.0
    return MetricsResponse(
        requests_total=_REQUEST_COUNT,
        ttfa_p50_ms=pct(_TTFA_HISTORY, 50), ttfa_p95_ms=pct(_TTFA_HISTORY, 95),
        rtf_p50=pct(_RTF_HISTORY, 50),       rtf_p95=pct(_RTF_HISTORY, 95),
        peak_vram_gb=round(_ENGINE.peak_vram_gb() if _ENGINE else 0.0, 3),
    )


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    global _REQUEST_COUNT
    if _ENGINE is None:
        raise HTTPException(503, "Engine not loaded")
    _REQUEST_COUNT += 1
    try:
        wav, sr = _ENGINE.synthesize(req.text, language=req.language)
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {e}")

    if req.format == "wav":
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    return Response(content=(wav * 32767).astype(np.int16).tobytes(), media_type="audio/pcm")


@app.websocket("/stream")
async def stream_ws(ws: WebSocket):
    global _REQUEST_COUNT
    await ws.accept()
    if _ENGINE is None:
        await ws.send_text(json.dumps({"event": "error", "detail": "Engine not loaded"}))
        await ws.close()
        return

    try:
        msg = json.loads(await ws.receive_text())
    except Exception as e:
        await ws.send_text(json.dumps({"event": "error", "detail": str(e)}))
        await ws.close()
        return

    text       = msg.get("text", "")
    language   = msg.get("language", "ar")
    chunk_size = int(msg.get("chunk_size", 20))

    if not text:
        await ws.send_text(json.dumps({"event": "error", "detail": "Empty text"}))
        await ws.close()
        return

    _REQUEST_COUNT += 1
    t0, first, ttfa_ms, total_dur = time.perf_counter(), True, 0.0, 0.0

    try:
        for chunk in _ENGINE.stream(text, language=language, chunk_size=chunk_size):
            if first:
                ttfa_ms = (time.perf_counter() - t0) * 1000
                _TTFA_HISTORY.append(ttfa_ms)
                first = False
            # Stream int16 PCM (matches POST /synthesize and the client decoders)
            f32 = chunk.astype(np.float32)
            total_dur += len(f32) / _ENGINE.sample_rate
            pcm16 = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16)
            await ws.send_bytes(pcm16.tobytes())

        wall = time.perf_counter() - t0
        rtf  = wall / (total_dur + 1e-9)
        _RTF_HISTORY.append(rtf)
        await ws.send_text(json.dumps({
            "event": "end", "ttfa_ms": round(ttfa_ms, 1),
            "rtf": round(rtf, 4), "duration_s": round(total_dur, 3),
        }))
    except WebSocketDisconnect:
        logger.info("Client disconnected mid-stream")
    except Exception as e:
        logger.exception("Stream error")
        try:
            await ws.send_text(json.dumps({"event": "error", "detail": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ── Run directly ──────────────────────────────────────────────────────────────
# ╔══════════════════════ SERVER CONFIGURATION ══════════════════════════════════╗
_SERVER_HOST       = "0.0.0.0"
_SERVER_PORT       = 8000
_SERVER_WORKERS    = 1
# ╚═════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Set env vars before uvicorn starts so lifespan picks them up
    # os.environ["LEVA_CHECKPOINT"]  = "/path/to/checkpoint"
    # os.environ["LEVA_SPEAKER_WAV"] = "/path/to/speaker.wav"
    # os.environ["LEVA_DEVICE"]      = "cuda"
    uvicorn.run(
        "leva_tts.server.app:app",
        host=_SERVER_HOST,
        port=_SERVER_PORT,
        workers=_SERVER_WORKERS,
        log_level="info",
    )
