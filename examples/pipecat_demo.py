#!/usr/bin/env python3
"""
Leva-TTS × Pipecat — live integration demo (for the demo video).

This drives the **Pipecat `LevaTTSService`** directly: for each sentence it calls
the service's ``run_tts()`` — the exact method Pipecat invokes inside a pipeline —
and consumes the standard Pipecat frame contract it emits:

    TTSStartedFrame → TTSAudioRawFrame × N → TTSStoppedFrame

For each sentence it prints the frames as they arrive, measures time-to-first-audio
(TTFB), and writes one WAV. The model is loaded once and reused, so the three
samples (Levantine, code-switch, English) render back-to-back, fast.

Run:
    python examples/pipecat_demo.py \
        --checkpoint checkpoints \
        --speaker reference_audios/Mohamed.wav

(omit --speaker to use any built-in reference clip in reference_audios/)
"""
import argparse
import asyncio
import glob
import os
import time
import wave

from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from leva_tts.pipecat_plugin import LevaTTSService

SAMPLES = [
    ("Pure Levantine Arabic", "ar", "مرحبا كيفك اليوم؟ إن شاء الله تكون بخير وكل أمورك تمام."),
    ("Code-switching",        "ar", "هَلَّق أنا عم أشتغل على the new project، وبكرا عندي meeting مهم كتير."),
    ("Pure English",          "en", "Hello! This is the Leva-TTS streaming service running inside Pipecat."),
]


async def synth_one(tts: LevaTTSService, label: str, lang: str, text: str, idx: int):
    """Run one sentence through the Pipecat service's run_tts() and save a WAV."""
    tts.language = lang
    print(f"  • [{label}]  {text}")

    pcm = bytearray()
    t0 = time.time()
    ttfb = None
    sr = getattr(tts, "sample_rate", 24_000)

    # run_tts() is the Pipecat TTSService entry point; it yields real Pipecat frames.
    async for frame in tts.run_tts(text):
        if isinstance(frame, TTSStartedFrame):
            print("      ⟶ TTSStartedFrame")
        elif isinstance(frame, TTSAudioRawFrame):
            if ttfb is None:
                ttfb = (time.time() - t0) * 1000
                sr = frame.sample_rate or sr
                print(f"      ⟶ first TTSAudioRawFrame  (TTFB {ttfb:.0f} ms)")
            pcm.extend(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            print("      ⟶ TTSStoppedFrame")

    path = f"pipecat_{idx:02d}_{label.lower().replace(' ', '_').replace('-', '_')}.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)            # int16 PCM
        w.setframerate(sr)
        w.writeframes(bytes(pcm))
    secs = len(pcm) / 2 / sr
    total = (time.time() - t0) * 1000
    print(f"      ✅ saved {path}  ({secs:.2f}s audio · synth {total:.0f} ms)\n")


async def main(checkpoint: str, speaker: str, device: str):
    print("=" * 70)
    print("  Leva-TTS × Pipecat — streaming TTS integration demo")
    print("=" * 70)

    print("\n[1/3] Loading Leva-TTS as a Pipecat TTSService (local GPU mode)…")
    tts = LevaTTSService(
        mode="local",
        checkpoint=checkpoint,
        speaker_wav=speaker,
        device=device,
        use_deepspeed=False,
        language="ar",
    )

    # Drive run_tts() directly (the same coroutine Pipecat calls inside a pipeline).
    # Neutralize the in-pipeline metrics hooks so the service runs stand-alone.
    async def _noop(*a, **k):
        return None
    tts.start_ttfb_metrics = _noop
    tts.stop_ttfb_metrics = _noop

    print(f"      ✅ service ready (speaker: {os.path.basename(speaker)})")
    print("\n[2/3] Streaming 3 sentences through LevaTTSService.run_tts():\n")

    for i, (label, lang, text) in enumerate(SAMPLES, 1):
        await synth_one(tts, label, lang, text, i)

    print("[3/3] Done. WAVs written:")
    for f in sorted(glob.glob("pipecat_*.wav")):
        print("      •", f)
    print("\nLeva-TTS emits the Pipecat TTS frame contract")
    print("(TTSStartedFrame → TTSAudioRawFrame… → TTSStoppedFrame), streaming audio")
    print("chunk-by-chunk for low-latency conversational agents.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Leva-TTS Pipecat integration demo")
    ap.add_argument("--checkpoint", default="checkpoints",
                    help="Path to the fine-tuned checkpoint dir")
    ap.add_argument("--speaker", default=None,
                    help="Reference speaker WAV (defaults to first in reference_audios/)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spk = args.speaker
    if not spk:
        cand = sorted(glob.glob("reference_audios/*.wav")) or sorted(glob.glob("reference_audios/*.mp3"))
        spk = cand[0] if cand else None
        if spk:
            print(f"[demo] using reference speaker: {spk}")
    if not spk or not os.path.exists(spk):
        raise SystemExit("No reference speaker WAV found. Pass --speaker path/to/ref.wav")

    asyncio.run(main(args.checkpoint, spk, args.device))
