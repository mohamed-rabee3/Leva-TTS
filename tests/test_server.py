"""Streaming server: start it, send requests, gather audio."""
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close()
    return p


@pytest.fixture(scope="module")
def server(checkpoint_dir, reference_wav):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    requests = pytest.importorskip("requests")

    port = _free_port()
    env = dict(os.environ)
    env["LEVA_CHECKPOINT"]  = checkpoint_dir
    env["LEVA_SPEAKER_WAV"] = reference_wav
    env["LEVA_DEVICE"]      = "cuda"
    env["LEVA_DEEPSPEED"]   = "0"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "leva_tts.server.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    # wait for the model to load (up to 180 s)
    ready = False
    for _ in range(90):
        time.sleep(2)
        try:
            r = requests.get(base + "/health", timeout=3)
            if r.status_code == 200:
                ready = True; break
        except Exception:
            pass
    if not ready:
        proc.terminate()
        pytest.skip("Server did not become ready in time")

    yield base
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def test_health(server):
    import requests
    r = requests.get(server + "/health", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["sample_rate"] == 24_000


def test_synthesize_wav(server):
    import requests, soundfile as sf
    r = requests.post(server + "/synthesize",
                      json={"text": "كيفك اليوم؟", "language": "ar", "format": "wav"},
                      timeout=120)
    assert r.status_code == 200
    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 24_000
    assert len(wav) > sr * 0.2


def test_synthesize_pcm(server):
    import requests
    r = requests.post(server + "/synthesize",
                      json={"text": "مرحبا", "language": "ar", "format": "pcm"},
                      timeout=120)
    assert r.status_code == 200
    pcm = np.frombuffer(r.content, dtype=np.int16)
    assert len(pcm) > 0


def test_websocket_stream(server):
    websockets = pytest.importorskip("websockets")
    import asyncio

    ws_url = server.replace("http://", "ws://") + "/stream"

    async def _run():
        chunks = []
        meta = None
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"text": "كيفك اليوم", "language": "ar"}))
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    chunks.append(np.frombuffer(msg, dtype=np.float32))
                else:
                    meta = json.loads(msg)
                    if meta.get("event") == "end":
                        break
        return chunks, meta

    chunks, meta = asyncio.get_event_loop().run_until_complete(_run())
    assert len(chunks) >= 1
    assert meta is not None and meta.get("event") == "end"
    combined = np.concatenate(chunks)
    assert len(combined) > 24_000 * 0.2
