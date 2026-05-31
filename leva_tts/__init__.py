"""
Leva-TTS — Levantine Arabic / English code-switching TTS.

Pip-install inference quick start
---------------------------------
    pip install leva-tts

    from leva_tts import LevaTTS

    tts = LevaTTS(device="cuda", preprocess_text=True, verbose=False)
    # downloads the fine-tuned checkpoint + 10 reference speakers on first use

    # Built-in speaker (must be one of the 10 names)
    wav, sr = tts.synthesize("كيفك اليوم؟ بدي اشتغل على the project",
                             speaker="Badr", temperature=0.65)

    # Zero-shot with your own reference clip
    wav, sr = tts.zero_shot_synthesize("هلق عم نشتغل", "my_voice.wav")

    # Streaming generators
    for chunk in tts.stream("...", speaker="Amina"):
        ...
    for chunk in tts.zero_shot_stream("...", "my_voice.wav"):
        ...
"""
__version__ = "0.1.2"
__author__  = "Mohammed Aly"

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

# ── The 10 built-in Levantine speakers ────────────────────────────────────────
SPEAKERS: List[str] = [
    "Badr", "Mohamed", "Saad", "Rami", "Fadi",        # male
    "Amina", "Fatma", "Lamyaa", "Mona", "Haneen",     # female
]


class LevaTTS:
    """
    High-level Leva-TTS inference interface.

    On first construction it auto-downloads the fine-tuned XTTS-v2 checkpoint
    and the 10 reference speaker clips from HuggingFace (falls back to the
    GitHub release if the HF download fails).

    Parameters
    ----------
    checkpoint : str | None
        Local checkpoint directory. If ``None`` (default), the model is
        downloaded from HuggingFace (``mohammedaly22/leva-tts``).
    device : str | None
        ``"cuda"`` | ``"cpu"``. Auto-detected if ``None``.
    preprocess_text : bool
        Apply the Levantine text front-end (number/date/currency verbalization,
        partial diacritics, lexicon overrides) before synthesis. Default ``True``.
    verbose : bool
        Print the text-processing pipeline stages. Default ``False``.

    Generation parameters (shared defaults for all synthesis calls)
    ----------------------------------------------------------------
    temperature, length_penalty, repetition_penalty, top_k, top_p, speed —
    overridable per-call via keyword args on ``synthesize`` / ``stream`` / etc.
    """

    HF_MODEL_ID      = "mohammedaly22/leva-tts"
    GITHUB_RELEASE   = (
        "https://github.com/MohammedAly22/Leva-TTS/releases/latest/download/leva-tts.zip"
    )

    DEFAULT_GEN = {
        "temperature":        0.65,
        "length_penalty":     1.0,
        "repetition_penalty": 5.0,
        "top_k":              50,
        "top_p":              0.85,
        "speed":              1.0,
    }

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        preprocess_text: bool = True,
        verbose: bool = False,
    ):
        import torch
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess_text = preprocess_text
        self.verbose         = verbose

        self._model_root = Path(checkpoint) if checkpoint else self._resolve_model()
        self._speaker_refs = self._index_speakers(self._model_root)

        from leva_tts.text.processor import TextProcessor
        self._tp = TextProcessor(verbose=verbose)

        from leva_tts.inference.engine import LevaTTSEngine
        self._engine = LevaTTSEngine.from_checkpoint(
            str(self._model_root), speaker_wav=None,
            use_deepspeed=False, device=self.device,
        )

    # ── public synthesis API ──────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        speaker: str,
        language: str = "ar",
        **gen,
    ) -> Tuple["np.ndarray", int]:
        """
        Synthesize *text* with one of the 10 built-in speakers.

        Parameters
        ----------
        text : str
        speaker : str
            One of ``LevaTTS.SPEAKERS`` (Badr, Mohamed, …). Raises ``ValueError``
            otherwise.
        language : str
            ``"ar"`` (Levantine + code-switch) | ``"en"``.
        **gen : generation overrides (temperature, repetition_penalty, top_p,
            top_k, speed, length_penalty).

        Returns
        -------
        (wav: np.ndarray float32 @ 24 kHz, sample_rate: int)
        """
        ref = self._speaker_ref(speaker)
        return self._synth(text, ref, language, gen)

    def zero_shot_synthesize(
        self,
        text: str,
        reference_audio: str,
        language: str = "ar",
        **gen,
    ) -> Tuple["np.ndarray", int]:
        """Synthesize *text* cloning the voice in *reference_audio* (3–10 s WAV/MP3)."""
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        return self._synth(text, reference_audio, language, gen)

    def stream(
        self,
        text: str,
        speaker: str,
        language: str = "ar",
        **gen,
    ) -> Generator["np.ndarray", None, None]:
        """Streaming variant of :meth:`synthesize` — yields float32 audio chunks."""
        ref = self._speaker_ref(speaker)
        yield from self._stream(text, ref, language, gen)

    def zero_shot_stream(
        self,
        text: str,
        reference_audio: str,
        language: str = "ar",
        **gen,
    ) -> Generator["np.ndarray", None, None]:
        """Streaming variant of :meth:`zero_shot_synthesize`."""
        if not Path(reference_audio).exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
        yield from self._stream(text, reference_audio, language, gen)

    # ── helpers ────────────────────────────────────────────────────────────────
    @property
    def speakers(self) -> List[str]:
        """List of available built-in speaker names."""
        return list(self._speaker_refs.keys())

    def _speaker_ref(self, speaker: str) -> str:
        if speaker not in self._speaker_refs:
            raise ValueError(
                f"Unknown speaker '{speaker}'. "
                f"Choose one of: {', '.join(self._speaker_refs.keys())}"
            )
        return self._speaker_refs[speaker]

    def _process(self, text: str) -> str:
        return self._tp.process(text) if self.preprocess_text else text

    def _synth(self, text, ref, language, gen):
        return self._engine.synthesize(
            self._process(text), speaker_wav=ref, language=language,
            **{**self.DEFAULT_GEN, **gen},
        )

    def _stream(self, text, ref, language, gen):
        yield from self._engine.stream(
            self._process(text), speaker_wav=ref, language=language,
            **{**self.DEFAULT_GEN, **gen},
        )

    # ── model + speaker resolution ─────────────────────────────────────────────
    def _resolve_model(self) -> Path:
        cache = Path.home() / ".cache" / "leva_tts"
        # already downloaded?
        if any(cache.rglob("best_model.pth")) or any(cache.rglob("*.pth")):
            return cache
        cache.mkdir(parents=True, exist_ok=True)
        # 1. HuggingFace — fetch ONLY what inference needs (skip the sample_wavs/ demos)
        try:
            from huggingface_hub import snapshot_download
            print(f"[Leva-TTS] Downloading model from {self.HF_MODEL_ID} …")
            return Path(snapshot_download(
                self.HF_MODEL_ID, local_dir=str(cache),
                allow_patterns=["best_model.pth", "config.json", "reference_audios/*"],
            ))
        except Exception as e:
            print(f"[Leva-TTS] HF download failed ({e}); trying GitHub release …")
        # 2. GitHub release fallback
        try:
            import io, urllib.request, zipfile
            with urllib.request.urlopen(self.GITHUB_RELEASE) as resp:
                data = resp.read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(cache)
            return cache
        except Exception as e:
            raise RuntimeError(
                f"Could not download the Leva-TTS model.\n"
                f"  HuggingFace: huggingface-cli download {self.HF_MODEL_ID}\n"
                f"  GitHub:      {self.GITHUB_RELEASE}\n"
                f"  Error: {e}"
            ) from e

    @staticmethod
    def _index_speakers(model_root: Path) -> dict:
        """
        Build {speaker_name: ref_wav_path}. Looks for reference clips bundled with
        the model (reference_audios/) or alongside the repo.
        """
        import json
        candidates = [
            model_root / "reference_audios" / "references.json",
            Path.cwd() / "reference_audios" / "references.json",
            Path(__file__).resolve().parents[1] / "reference_audios" / "references.json",
        ]
        for rj in candidates:
            if rj.exists():
                data = json.loads(rj.read_text(encoding="utf-8"))
                out = {}
                for r in data:
                    ap = rj.parent / Path(r["audio_path"]).name
                    if not ap.exists():
                        ap = (rj.parent.parent / r["audio_path"])
                    if ap.exists():
                        out[Path(r["audio_path"]).stem] = str(ap)
                if out:
                    return out
        raise RuntimeError(
            "No reference speaker audio found. Expected reference_audios/references.json "
            "bundled with the model or in the working directory."
        )


def download_model(dest: Optional[str] = None, include_samples: bool = False) -> str:
    """
    Download the Leva-TTS checkpoint + reference speakers. Returns the local path.

    By default only the inference files are fetched (best_model.pth, config.json,
    reference_audios/). Pass ``include_samples=True`` to also pull the audio
    comparison samples used in the README/model card.
    """
    from huggingface_hub import snapshot_download
    local = dest or str(Path.home() / ".cache" / "leva_tts")
    patterns = None if include_samples else [
        "best_model.pth", "config.json", "reference_audios/*",
    ]
    print(f"Downloading {LevaTTS.HF_MODEL_ID} → {local}")
    snapshot_download(LevaTTS.HF_MODEL_ID, local_dir=local, allow_patterns=patterns)
    print("Done.")
    return local
