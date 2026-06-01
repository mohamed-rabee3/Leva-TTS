---
language:
- ar
- en
license: cc-by-nc-4.0
library_name: coqui
pipeline_tag: text-to-speech
tags:
- text-to-speech
- tts
- xtts
- xtts-v2
- arabic
- levantine
- code-switching
- voice-cloning
- streaming
base_model:
- coqui/XTTS-v2
---

<img src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/docs/LevaTTS-Banner-Image.png" alt="Leva-TTS — Levantine Arabic ⇄ English Text-to-Speech">

---

<div align="center">

# 🌿 Leva-TTS — Low-Latency Code-Switching TTS (Levantine Arabic ⇄ English)

*A production-oriented Levantine Text-to-Speech model — a fine-tuned **XTTS-v2** optimized for real-time conversational agents.*

[![Demo](https://img.shields.io/badge/🔊_Live_Demo-Listen-2ea043)](https://mohammedaly22.github.io/Leva-TTS/)
[![GitHub](https://img.shields.io/badge/GitHub-Leva--TTS-181717?logo=github)](https://github.com/MohammedAly22/Leva-TTS)
[![HF Space](https://img.shields.io/badge/🤗_Space-Gradio_Demo-FFD21E)](https://huggingface.co/spaces/mohammedaly22/leva-tts)
[![PyPI](https://img.shields.io/pypi/v/leva-tts?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/leva-tts/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Leva-TTS/blob/main/examples/01_quick_start.ipynb)

| 🎯 KPI | Target | **Measured** | Status |
|---|---|---|---|
| Peak VRAM (inference) | ≤ 3 GB | **2.13 GB** | ✅ |
| Time-to-First-Audio (p50) | < 300 ms | **565 ms** | ⚠️ |
| Real-Time Factor (RTF) | < 0.3 | **0.21** | ✅ |
| Streaming output | required | **chunked PCM + WS** | ✅ |

</div>

---

**Leva-TTS** is a text-to-speech model for **Levantine Arabic / English
code-switching**, built by fine-tuning [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
on **50,000 synthetic utterances** generated with
[Lahgtna-OmniVoice v2](https://huggingface.co/oddadmix/lahgtna-omnivoice-v2).
It handles natural intra-sentence switching between Levantine dialect and English,
supports **10 built-in speakers** and **zero-shot voice cloning**, and offers a
**streaming** generator for low-latency conversational use.

- **Base model:** `coqui/XTTS-v2` (GPT autoregressive backbone + HiFi-GAN decoder)
- **Languages:** Levantine Arabic (`ar`), English (`en`), and code-switch mixes
- **Sample rate:** 24 kHz
- **Speakers:** Badr, Mohamed, Saad, Rami, Fadi (M) · Amina, Fatma, Lamyaa, Mona, Haneen (F)

### ✨ Key Features

| Feature | Details |
|---------|---------|
| 🗣️ **Natural code-switching** | Intra-sentence Arabic ↔ English |
| ⚡ **Streaming output** | First audio chunk < 300 ms |
| 💾 **Low VRAM** | ≤ 3 GB at inference |
| 🌿 **Levantine dialect** | ق→/ʔ/ glottal, ج→/ʒ/, *il-* article, *b-* prefix |
| 🔤 **Smart text front-end** | Partial diacritics on homographs + Levantine lexicon |
| 👥 **10 speakers** | 5 male + 5 female, diverse Levantine accents |
| 📡 **WebSocket streaming** | FastAPI server with real-time chunked PCM |
| 🔌 **Pipecat ready** | Drop-in `TTSService` for voice agents |

---

## 🚀 Quick start (pip)

```bash
conda create -n leva-tts python=3.10 -y && conda activate leva-tts
sudo apt-get install -y espeak-ng ffmpeg libsndfile1

# Install PyTorch first so pip locks a CUDA build matching your GPU driver.
# (torch >= 2.9 ships CUDA-13 wheels that fail on common CUDA-12.x drivers.)
pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip install leva-tts
```

> Leva-TTS uses the maintained **`coqui-tts`** fork (same `TTS`/XTTS modules); the
> unmaintained `TTS` package pins `numpy==1.22.0` and cannot resolve on modern
> Python. A plain `pip install leva-tts` resolves cleanly.

```python
from leva_tts import LevaTTS, SPEAKERS
import soundfile as sf

tts = LevaTTS(device="cuda", preprocess_text=True, verbose=False)
# auto-downloads this checkpoint + the 10 reference speakers on first use

# 1) Built-in speaker  (speaker must be one of SPEAKERS, else ValueError)
wav, sr = tts.synthesize("هَلَّق أنا عم أشتغل على the project",
                         speaker="Badr", temperature=0.65)
sf.write("out.wav", wav, sr)            # sr == 24000

# 2) Zero-shot voice cloning (your own 3–10 s clip)
wav, sr = tts.zero_shot_synthesize("والله the meeting كانت important كتير",
                                   "my_voice.wav")

# 3) Streaming generators
for chunk in tts.stream("بِدِّي أحكيلك عن the new feature", speaker="Amina"):
    ...                                  # play / forward each chunk
for chunk in tts.zero_shot_stream("هلق عم نشتغل", "my_voice.wav"):
    ...
```

**Generation parameters** (optional, per-call on every method):
`temperature`, `length_penalty`, `repetition_penalty`, `top_k`, `top_p`, `speed`.

For the FastAPI streaming server, Pipecat integration, the Gradio demo, evaluation
and fine-tuning, clone the repo:
👉 **https://github.com/MohammedAly22/Leva-TTS**

---

## 📦 Files in this repo

| File | Description |
|------|-------------|
| `best_model.pth` | Fine-tuned XTTS-v2 checkpoint (GPT + decoder) |
| `config.json` | XTTS-v2 config |
| `reference_audios/` | The 10 built-in speaker reference clips + `references.json` |
| `sample_wavs/` | Audio sample comparisons (Base XTTS-v2 vs Lahgtna v2 vs Leva-TTS) |

> Manual download: `huggingface-cli download mohammedaly22/leva-tts`

---

## 🎵 Audio samples — Model comparison

Click a sentence to expand and play the three models. Progression:
**Base XTTS-v2 → Lahgtna v2 → Leva-TTS**.

### 🔀 Code-switching (Levantine + English)

<details open>
<summary>هَلَّق أنا عم أشتغل على the new project — <b>Badr (M)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/badr_cs.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/badr_cs.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/badr_cs.wav"></audio>

</details>

<details>
<summary>والله the weather today كتير حلو — <b>Fatma (F)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/fatma_cs.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/fatma_cs.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/fatma_cs.wav"></audio>

</details>

<details>
<summary>بِدِّي أحكيلك عن the meeting المهم — <b>Mona (F)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/mona_cs.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/mona_cs.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/mona_cs.wav"></audio>

</details>

### Pure Levantine Arabic

<details>
<summary>كيفك اليوم؟ إنت شو عم تعمل هَلَّق؟ — <b>Badr (M)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/badr_ar.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/badr_ar.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/badr_ar.wav"></audio>

</details>

<details>
<summary>هَلَّق رح أروح على البيت وبكرا برجع — <b>Amina (F)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/amina_ar.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/amina_ar.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/amina_ar.wav"></audio>

</details>

<details>
<summary>شو رأيك نطلع نتمشى شوي بعد الشغل؟ — <b>Rami (M)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/rami_ar.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/rami_ar.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/rami_ar.wav"></audio>

</details>

### 🇬🇧 Pure English

<details>
<summary>Hello, how are you doing today? — <b>Lamyaa (F)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/lamyaa_en.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/lamyaa_en.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/lamyaa_en.wav"></audio>

</details>

<details>
<summary>The project deadline is next Friday. — <b>Mohamed (M)</b></summary>

**Base XTTS-v2**

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/xtts_base/mohamed_en.wav"></audio>

**Lahgtna v2** (Levantine fine-tune)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/lahgtna/mohamed_en.wav"></audio>

**🟢 Leva-TTS** (this model)

<audio controls src="https://huggingface.co/mohammedaly22/leva-tts/resolve/main/sample_wavs/leva_tts/mohamed_en.wav"></audio>

</details>

---

## 📊 Evaluation

Speaker Mohamed · NVIDIA H100 · Whisper large-v3 ASR round-trip · UTMOS (reference-free MOS).

| Metric | Value |
|--------|-------|
| Peak VRAM (inference) | 2.13 GB |
| RTF p50 / p95 | 0.36 / 0.53 |
| TTFA p50 / p95 (batch) | 1194 / 1743 ms |
| TTFA streaming (first chunk) | ~565 ms |
| CER (mean) | 0.255 |
| WER (mean) | 0.496 |
| **UTMOS** | **3.13 / 5.0** |

| Category | CER ↓ | WER ↓ | UTMOS ↑ |
|----------|-------|-------|---------|
| Pure English | 0.144 | 0.190 | 3.35 |
| Pure Levantine Arabic | 0.236 | 0.544 | 2.97 |
| Code-Switching | 0.330 | 0.602 | 3.19 |

An optimized inference path (TF32 + `torch.compile` on the GPT) lowers RTF p95 by
~6% and TTFA while slightly improving UTMOS (3.24). See the repo's `scripts/evaluate.py --optimize`.

---

## 🏗️ How it was built

1. **Text collection** — 50K Levantine / code-switching / English sentences.
2. **Synthesis** — audio generated with **Lahgtna-OmniVoice v2** (`apc` language code).
3. **Data prep** — 24 kHz, paired with a Levantine text front-end (number/date/
   currency verbalization, partial diacritics on homographs, dialect lexicon).
4. **Fine-tuning** — XTTS-v2 GPT fine-tuned on the synthetic corpus.

A **text front-end** runs before synthesis (enabled via `preprocess_text=True`):
language-aware normalization of numbers, floats, dates, times, currency,
percentages, URLs, emails, phone numbers and codes, plus partial diacritics and a
Levantine lexicon.

---

## ⚠️ Limitations & intended use

- Optimized for **Levantine** dialect + English code-switching; other Arabic
  dialects (Egyptian, Gulf, MSA) are out of distribution.
- Trained on **synthetic** speech — voices reflect the Lahgtna v2 generator.
- License **CC-BY-NC-4.0** (inherited from XTTS-v2): research / non-commercial use.

## 📜 Citation

```bibtex
@software{leva_tts_2026,
  author = {Mohammed Aly},
  title  = {Leva-TTS: Low-Latency Code-Switching TTS for Levantine Arabic and English},
  year   = {2026},
  url    = {https://github.com/MohammedAly22/Leva-TTS}
}
```

Built on [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) and
[Lahgtna-OmniVoice v2](https://huggingface.co/oddadmix/lahgtna-omnivoice-v2).
