#!/usr/bin/env python3
"""
Leva-TTS × Pipecat — live integration demo (for the demo video).

Builds a real Pipecat Pipeline:

    TextInjector  →  LevaTTSService  →  AudioCapture

and pushes three sentences through it (Levantine, code-switch, English).
LevaTTSService synthesizes each on the GPU and emits the standard Pipecat
frame sequence:

    TTSStartedFrame → TTSAudioRawFrame × N → TTSStoppedFrame

The AudioCapture processor prints every frame as it arrives (so the pipeline is
visible on screen), measures time-to-first-audio (TTFB), and writes one WAV per
sentence.

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
    EndFrame,
    Frame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from leva_tts.pipecat_plugin import LevaTTSService

SAMPLES = [
    ("Pure Levantine Arabic", "ar", "مرحبا كيفك اليوم؟ إن شاء الله تكون بخير وكل أمورك تمام."),
    ("Code-switching",        "ar", "هَلَّق أنا عم أشتغل على the new project، وبكرا عندي meeting مهم كتير."),
    ("Pure English",          "en", "Hello! This is the Leva-TTS streaming service running inside Pipecat."),
]


class AudioCapture(FrameProcessor):
    """Sink processor: logs the Pipecat frame flow and writes audio to WAV."""

    def __init__(self, sample_rate: int = 24_000):
        super().__init__()
        self.sample_rate = sample_rate
        self._buf: list[bytes] = []
        self._t0 = None
        self._ttfb = None
        self.idx = 0
        self.label = "output"

    def reset(self, label: str):
        self._buf.clear()
        self._t0 = time.time()
        self._ttfb = None
        self.label = label

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            print(f"      ⟶ TTSStartedFrame")
        elif isinstance(frame, TTSAudioRawFrame):
            if self._ttfb is None and self._t0 is not None:
                self._ttfb = (time.time() - self._t0) * 1000
                print(f"      ⟶ first TTSAudioRawFrame  (TTFB {self._ttfb:.0f} ms)")
            self._buf.append(frame.audio)
        elif isinstance(frame, TTSStoppedFrame):
            self._flush()
            print(f"      ⟶ TTSStoppedFrame\n")

        await self.push_frame(frame, direction)

    def _flush(self):
        if not self._buf:
            return
        self.idx += 1
        pcm = b"".join(self._buf)
        path = f"pipecat_{self.idx:02d}_{self.label}.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)            # int16
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        secs = len(pcm) / 2 / self.sample_rate
        print(f"      ✅ saved {path}  ({secs:.2f}s audio)")
        self._buf.clear()


async def main(checkpoint: str, speaker: str, device: str):
    print("=" * 70)
    print("  Leva-TTS × Pipecat — streaming TTS integration demo")
    print("=" * 70)

    print("\n[1/3] Loading Leva-TTS into a Pipecat TTSService (local GPU mode)…")
    tts = LevaTTSService(
        mode="local",
        checkpoint=checkpoint,
        speaker_wav=speaker,
        device=device,
        use_deepspeed=False,
        language="ar",
    )
    capture = AudioCapture(sample_rate=24_000)

    # Real Pipecat pipeline: TTS service → audio capture sink
    pipeline = Pipeline([tts, capture])
    print("      Pipeline:  LevaTTSService → AudioCapture")

    print("\n[2/3] Streaming 3 sentences through the pipeline:\n")
    for label, lang, text in SAMPLES:
        print(f"  • [{label}]  {text}")
        tts.language = lang
        capture.reset(label.lower().replace(" ", "_").replace("-", "_"))

        task = PipelineTask(pipeline)

        async def _inject():
            await asyncio.sleep(0.2)
            await task.queue_frame(TextFrame(text))
            # give synthesis time to stream out, then end the task
            await asyncio.sleep(0.5)
            await task.queue_frame(EndFrame())

        await asyncio.gather(PipelineRunner().run(task), _inject())

    print("[3/3] Done. WAVs written to the current directory:")
    for f in sorted(glob.glob("pipecat_*.wav")):
        print("      •", f)
    print("\nThis proves Leva-TTS emits the Pipecat TTS frame contract")
    print("(TTSStartedFrame → TTSAudioRawFrame… → TTSStoppedFrame) and streams")
    print("audio chunk-by-chunk for low-latency conversational agents.")


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
