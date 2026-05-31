"""
LevaTTSEngine — high-level streaming/batch inference wrapper around XTTS-v2
fine-tuned for Levantine Arabic / English code-switching.

Loading correctly handles:
  - Base XTTS-v2 weights + fine-tuned GPT checkpoint
  - The transformers/torch isin() incompatibility that breaks streaming
  - XTTS .eval() returning None (must not chain .to().eval())
  - Long text splitting (XTTS 400-token limit)
  - TextProcessor normalization (numbers, dates, lexicon, partial diacritics)
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import AsyncGenerator, Generator, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_RATE = 24_000


# ── transformers/torch isin compatibility patch (needed for streaming) ────────
def _patch_torch_isin():
    if getattr(torch.isin, "_leva_patched", False):
        return
    _orig = torch.isin

    def _isin(*args, **kwargs):
        if "elements" in kwargs or "test_elements" in kwargs:
            el = kwargs.pop("elements", args[0] if len(args) > 0 else None)
            te = kwargs.pop("test_elements", args[1] if len(args) > 1 else None)
            if not torch.is_tensor(el) and not torch.is_tensor(te):
                el = torch.tensor(el)
            return _orig(el, te, **kwargs)
        if len(args) == 2 and not torch.is_tensor(args[0]) and not torch.is_tensor(args[1]):
            return _orig(torch.tensor(args[0]), args[1], **kwargs)
        return _orig(*args, **kwargs)

    _isin._leva_patched = True
    torch.isin = _isin


_patch_torch_isin()


def _split_text(text: str, max_chars: int = 180) -> List[str]:
    """Split long text into chunks under the XTTS token limit."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.؟!؛])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            if len(s) > max_chars:
                parts = re.split(r"(?<=[،,])\s+", s)
                cur2 = ""
                for prt in parts:
                    prt = prt.strip()
                    if len(cur2) + len(prt) + 1 <= max_chars:
                        cur2 = (cur2 + " " + prt).strip()
                    else:
                        if cur2:
                            chunks.append(cur2)
                        while len(prt) > max_chars:
                            chunks.append(prt[:max_chars]); prt = prt[max_chars:]
                        cur2 = prt
                cur = cur2
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return [ch for ch in chunks if ch.strip()]


class LevaTTSEngine:
    """Streaming / batch XTTS-v2 inference engine for Leva-TTS."""

    def __init__(
        self,
        model,
        config,
        speaker_wav: Optional[str | List[str]] = None,
        language: str = "ar",
        use_deepspeed: bool = False,
        device: str = "cuda",
        stream_chunk_size: int = 20,
    ):
        self.model           = model
        self.config          = config
        self.speaker_wav     = speaker_wav
        self.language        = language
        self.device          = device
        self.stream_chunk_sz = stream_chunk_size
        self.sample_rate     = _DEFAULT_SAMPLE_RATE
        self._cond_cache: dict = {}
        self._warmed_up = False

        from leva_tts.text.processor import TextProcessor
        self.processor = TextProcessor()

    # ── Factory constructors ──────────────────────────────────────────────────
    @staticmethod
    def _ensure_base_xtts(cache: Path) -> Path:
        """
        Return the base XTTS-v2 model directory, downloading it from Coqui on
        first use. Leva-TTS fine-tunes only the GPT, so the base XTTS-v2
        (config, vocab, speaker encoder, HiFi-GAN) must be present locally.
        """
        model_dir = "tts_models--multilingual--multi-dataset--xtts_v2"
        # Coqui's default user-data dir (where ModelManager() downloads to).
        try:
            from TTS.utils.generic_utils import get_user_data_dir
            coqui_default = Path(get_user_data_dir("tts"))
        except Exception:
            coqui_default = Path.home() / ".local/share/tts"

        # Already downloaded? (check the configured cache and Coqui's default)
        for base in (cache, coqui_default):
            d = base / model_dir
            if (d / "config.json").exists():
                return d

        # Auto-accept the Coqui Public Model License (non-interactive) and download.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        logger.info("Downloading base XTTS-v2 model (one-time, ~1.8 GB) ...")
        from TTS.utils.manage import ModelManager
        out = ModelManager().download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2")
        # download_model returns (model_path, config_path, item); return the dir
        # that actually contains config.json.
        if isinstance(out, (list, tuple)):
            for cand in out:
                if not cand:
                    continue
                cp = Path(cand)
                d = cp if cp.is_dir() else cp.parent
                if (d / "config.json").exists():
                    return d
        for base in (coqui_default, cache):
            d = base / model_dir
            if (d / "config.json").exists():
                return d
        raise FileNotFoundError(
            "Could not download the base XTTS-v2 model from Coqui. "
            "Set COQUI_TOS_AGREED=1 and check your internet connection."
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = "checkpoints",
        speaker_wav: Optional[str] = None,
        use_deepspeed: bool = False,
        device: str = "cuda",
    ) -> "LevaTTSEngine":
        """Load base XTTS-v2 + the fine-tuned GPT checkpoint found under *checkpoint_path*."""
        cache = Path(os.environ.get("COQUI_MODEL_PATH",
                                    Path.home() / ".local/share/tts"))
        xtts = cls._ensure_base_xtts(cache)

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(str(xtts / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(xtts), eval=True)

        # Locate and load the fine-tuned GPT weights
        ck = Path(checkpoint_path)
        pths = sorted(ck.rglob("best_model.pth"), key=lambda f: f.stat().st_mtime)
        if not pths:
            pths = sorted(ck.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
        if pths:
            state = torch.load(str(pths[-1]), map_location="cpu", weights_only=False)
            state = state.get("model", state)
            cleaned = {
                (k[len("xtts.gpt."):] if k.startswith("xtts.gpt.")
                 else k[len("gpt."):] if k.startswith("gpt.") else k): v
                for k, v in state.items()
            }
            model.gpt.load_state_dict(cleaned, strict=False)
            logger.info(f"Loaded fine-tuned checkpoint: {pths[-1]}")
        else:
            logger.warning(f"No checkpoint found in {checkpoint_path}; using base XTTS-v2")

        # IMPORTANT: do NOT chain .to().eval() — XTTS .eval() returns None.
        model.to(device)
        model.eval()
        return cls(model=model, config=config, speaker_wav=speaker_wav,
                   use_deepspeed=use_deepspeed, device=device)

    @classmethod
    def from_pretrained(
        cls,
        speaker_wav: Optional[str] = None,
        use_deepspeed: bool = False,
        device: str = "cuda",
    ) -> "LevaTTSEngine":
        """Load the base (non-fine-tuned) XTTS-v2 model."""
        cache = Path(os.environ.get("COQUI_MODEL_PATH",
                                    Path.home() / ".local/share/tts"))
        xtts = cls._ensure_base_xtts(cache)
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        config = XttsConfig()
        config.load_json(str(xtts / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(xtts), eval=True)
        model.to(device)
        model.eval()
        return cls(model=model, config=config, speaker_wav=speaker_wav,
                   use_deepspeed=use_deepspeed, device=device)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    def warmup(self):
        if self._warmed_up:
            return
        try:
            ref = self._resolve_speaker(None)
            if ref:
                _ = self.synthesize("مرحبا", speaker_wav=ref)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._warmed_up = True
        except Exception as e:
            logger.warning(f"Warm-up skipped: {e}")

    # ── Conditioning cache ─────────────────────────────────────────────────────
    def _get_conditioning(self, ref_wav: str):
        if ref_wav not in self._cond_cache:
            gpt, emb = self.model.get_conditioning_latents(audio_path=[ref_wav])
            self._cond_cache[ref_wav] = (gpt, emb)
        return self._cond_cache[ref_wav]

    def _resolve_speaker(self, speaker_wav):
        spk = speaker_wav or self.speaker_wav
        if isinstance(spk, (list, tuple)):
            spk = spk[0] if spk else None
        return spk

    # ── Batch synthesis ────────────────────────────────────────────────────────
    _GEN_KEYS = ("temperature", "length_penalty", "repetition_penalty",
                 "top_k", "top_p")

    def _clean_gen(self, gen: dict) -> dict:
        g = {k: gen[k] for k in self._GEN_KEYS if k in gen}
        if "repetition_penalty" in g:
            g["repetition_penalty"] = max(float(g["repetition_penalty"]), 1.0)
        if "length_penalty" in g:
            g["length_penalty"] = max(float(g["length_penalty"]), 0.1)
        for k in ("temperature", "top_p"):
            if k in g: g[k] = float(g[k])
        if "top_k" in g: g["top_k"] = int(g["top_k"])
        return g

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str | List[str]] = None,
        language: Optional[str] = None,
        **gen,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize *text* → (float32 audio @ 24 kHz, sample_rate)."""
        ref = self._resolve_speaker(speaker_wav)
        if not ref:
            raise ValueError("speaker_wav is required (no default set)")
        lang      = language or self.language
        processed = self.processor.process(text)
        g = self._clean_gen(gen)
        wavs = []
        for seg in _split_text(processed):
            out = self.model.synthesize(seg, self.config, speaker_wav=[ref],
                                        language=lang, gpt_cond_len=3, **g)
            wavs.append(np.array(out["wav"], dtype=np.float32))
        wav = np.concatenate(wavs) if wavs else np.zeros(1, np.float32)
        return wav, self.sample_rate

    # ── Streaming synthesis ────────────────────────────────────────────────────
    def stream(
        self,
        text: str,
        speaker_wav: Optional[str | List[str]] = None,
        language: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **gen,
    ) -> Generator[np.ndarray, None, None]:
        """Yield float32 audio chunks as they are generated."""
        ref = self._resolve_speaker(speaker_wav)
        if not ref:
            raise ValueError("speaker_wav is required (no default set)")
        lang      = language or self.language
        csz       = chunk_size or self.stream_chunk_sz
        gpt, emb  = self._get_conditioning(ref)
        processed = self.processor.process(text)
        g = self._clean_gen(gen)
        if "speed" in gen:
            g["speed"] = float(gen["speed"])
        for seg in _split_text(processed):
            for chunk in self.model.inference_stream(
                seg, lang, gpt_cond_latent=gpt, speaker_embedding=emb,
                stream_chunk_size=csz, **g,
            ):
                yield chunk.squeeze().cpu().numpy().astype(np.float32)

    async def astream(
        self,
        text: str,
        speaker_wav: Optional[str | List[str]] = None,
        language: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Async streaming wrapper — runs the sync generator in a thread executor."""
        import asyncio
        loop  = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _produce():
            try:
                for c in self.stream(text, speaker_wav, language):
                    loop.call_soon_threadsafe(queue.put_nowait, c)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _produce)
        while True:
            c = await queue.get()
            if c is None:
                break
            yield c

    # ── Metrics ────────────────────────────────────────────────────────────────
    def peak_vram_gb(self) -> float:
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    def reset_vram_stats(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
