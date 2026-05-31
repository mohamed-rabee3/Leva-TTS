"""
TensorBoard callback that logs audio samples during XTTS fine-tuning.

Hooks into the Coqui `Trainer` callback system:
  - on_eval_end: synthesize held-out test sentences and log waveforms
    + spectrograms to TensorBoard so you can *listen* during training.
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class AudioLoggerCallback:
    """
    Trainer-compatible callback that writes sample audio to TensorBoard.

    Attach to the Coqui Trainer via:
        trainer.callbacks = [AudioLoggerCallback(writer, sentences, model, cfg)]
    """

    def __init__(
        self,
        tb_writer: SummaryWriter,
        test_sentences: List[Dict[str, Any]],
        sample_rate: int = 24_000,
        log_every_n_steps: int = 500,
    ):
        self.writer          = tb_writer
        self.test_sentences  = test_sentences
        self.sample_rate     = sample_rate
        self.log_every       = log_every_n_steps
        self._step           = 0

    # ── Coqui Trainer hooks ───────────────────────────────────────────────────
    def on_train_step_start(self, trainer):
        pass

    def on_train_step_end(self, trainer):
        self._step += 1

    def on_eval_end(self, trainer):
        """Called after each evaluation run — log audio samples."""
        if self._step % self.log_every != 0:
            return
        model = trainer.model
        cfg   = trainer.config
        self._log_samples(model, cfg)

    # ── Internal ──────────────────────────────────────────────────────────────
    def _log_samples(self, model, cfg):
        model.eval()
        ref_wav = getattr(cfg, "speaker_reference", None)
        if isinstance(ref_wav, list):
            ref_wav = ref_wav[0]

        for i, sent in enumerate(self.test_sentences[:5]):
            text = sent.get("text", "")
            lang = sent.get("language", "ar")
            try:
                with torch.no_grad():
                    out = model.synthesize(
                        text,
                        cfg,
                        speaker_wav=ref_wav,
                        language=lang,
                        gpt_cond_len=3,
                    )
                wav = np.array(out["wav"], dtype=np.float32)
                if wav.ndim == 1:
                    wav = wav[np.newaxis, :]  # (1, T) for TB
                self.writer.add_audio(
                    f"samples/sent_{i:02d}",
                    wav,
                    global_step=self._step,
                    sample_rate=self.sample_rate,
                )
                # Also log mel spectrogram image
                self._log_spectrogram(wav.squeeze(), i)
                logger.info(f"[TB] logged audio for sentence {i} at step {self._step}")
            except Exception as e:
                logger.warning(f"[TB] audio log failed for sent {i}: {e}")

        model.train()

    def _log_spectrogram(self, wav: np.ndarray, idx: int):
        try:
            import librosa
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            S = librosa.feature.melspectrogram(y=wav, sr=self.sample_rate, n_mels=80)
            S_db = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(10, 3))
            img = librosa.display.specshow(S_db, sr=self.sample_rate, ax=ax,
                                            x_axis="time", y_axis="mel")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set(title=f"Sample {idx}")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=72)
            plt.close(fig)
            buf.seek(0)

            img_arr = np.frombuffer(buf.read(), dtype=np.uint8)
            # Convert PNG bytes to HWC tensor for TB
            import torch
            from torchvision.io import decode_image
            tensor = decode_image(torch.from_numpy(img_arr))  # (C, H, W)
            self.writer.add_image(f"specs/sent_{idx:02d}", tensor, self._step)
        except Exception as e:
            logger.debug(f"Spectrogram logging failed: {e}")
