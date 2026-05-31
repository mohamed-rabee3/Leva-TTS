# Leva-TTS — Design Document

**Low-Latency Code-Switching Text-to-Speech for Levantine Arabic ⇄ English**

Author: Mohammed Aly · Version 1.0

---

## 1. Problem & Goals

Real Levantine speakers constantly mix dialectal Arabic with English inside a single
sentence ("هَلَّق عم أشتغل على the new project"). Off-the-shelf TTS systems handle
neither the **Levantine dialect** (they target MSA) nor **intra-sentence
code-switching** (they break at language boundaries). The goal of Leva-TTS is a
**production-oriented, low-latency, streaming** TTS that:

- Speaks natural Levantine Arabic and English, including code-switched sentences.
- Supports **10 built-in speakers** and **zero-shot voice cloning** from a 3–10 s clip.
- Streams audio chunk-by-chunk for conversational agents (Pipecat, WebSocket).

**KPI targets** (set at project start) and outcomes:

| KPI | Target | Measured | Status |
|-----|--------|----------|--------|
| Peak VRAM (inference) | ≤ 3 GB | **2.13 GB** | ✅ |
| Real-Time Factor (RTF) | < 0.3 | 0.36 p50 | ⚠️ (close; see §6) |
| Time-to-First-Audio (streaming) | < 300 ms | ~565 ms | ⚠️ |
| Streaming output | required | chunked PCM + WebSocket | ✅ |
| Naturalness (UTMOS) | qualitative | 3.13 / 5.0 | ✅ |

---

## 2. Data & Phonemization Strategy

### 2.1 Data collection

There is no large, public, **aligned** Levantine + code-switching speech corpus.
Building one by hand-recording is out of scope for this timeframe, so we used a
**synthetic-data** approach:

1. **Text collection (50K sentences).** We assembled ~50,000 sentences spanning
   three buckets — pure Levantine, code-switched (Levantine + English), and pure
   English — covering everyday conversational registers (work, daily life, plans,
   opinions). Code-switch sentences were constructed so English spans fall at
   natural insertion points (nouns, technical terms, fixed phrases).

2. **Audio synthesis with Lahgtna-OmniVoice v2.** Each sentence was synthesized
   with [`oddadmix/lahgtna-omnivoice-v2`](https://huggingface.co/oddadmix/lahgtna-omnivoice-v2)
   (language code `apc`, North-Levantine Arabic), a model that produces convincing
   Levantine prosody. This yields **paired (text, audio)** data at scale without
   manual recording.

3. **Data preparation.** Audio resampled to 24 kHz mono; sentences paired with the
   text front-end output (below); a per-speaker reference clip set was curated for
   the 10 built-in voices.

The trade-off is explicit: synthetic data lets us cover code-switching breadth
quickly, at the cost of inheriting the generator's voice characteristics and any
of its systematic artifacts (see §7).

### 2.2 Text front-end ("phonemization" strategy)

XTTS-v2 uses a learned BPE tokenizer with a built-in multilingual grapheme→acoustic
mapping rather than explicit phonemes, so our front-end focuses on **normalizing
text into the form the model pronounces best**, language-aware:

- **Language detection per span.** An Arabic-letter regex splits each utterance into
  Arabic vs. Latin spans so each is normalized in its own language.
- **Entity verbalization** (numbers, floats, dates, times, currency, percentages,
  URLs, emails, phone numbers, codes). Numbers are spelled out **in the surrounding
  language** — Arabic digits → Levantine words ("3" → "تَلَاتِه"), English digits →
  English words. Long digit runs (IDs, phones) are read digit-by-digit.
- **Partial diacritics on homographs.** Levantine is normally written undiacritized,
  which makes some words ambiguous. We add **targeted** harakat only on known
  homographs (via a curated Levantine lexicon CSV) rather than full diacritization,
  which keeps pronunciation cues where they matter without over-constraining.
- **Dialect lexicon overrides.** A 150-entry Levantine lexicon corrects spellings the
  base model mispronounces and preserves dialectal forms (e.g. keeping
  والله / الله and pronoun suffixes intact).

This front-end is bundled in the package (`leva_tts.text.TextProcessor`) and runs
automatically when `preprocess_text=True`.

---

## 3. Model & Optimization Choices (with justification)

### 3.1 Why XTTS-v2

We fine-tune **XTTS-v2** (Coqui) rather than training from scratch or using a
single-speaker model. Justification:

- **Zero-shot multi-speaker, multilingual backbone.** XTTS-v2 is a GPT-style
  autoregressive model over audio-codec tokens conditioned on a speaker embedding,
  followed by a HiFi-GAN-style decoder. It already speaks 17 languages including
  Arabic and English, so the **code-switching capability is partly native** — we are
  steering an existing bilingual model toward Levantine, not teaching bilingualism
  from zero.
- **Voice cloning for free.** The speaker-conditioning design gives us zero-shot
  cloning and 10 built-in voices without extra modeling.
- **Streaming-friendly.** The autoregressive GPT supports `inference_stream`, which
  is exactly what a low-latency conversational agent needs.
- **Fine-tuning is cheap & data-efficient.** We only need to adapt the GPT to
  Levantine prosody/lexis, not retrain the decoder — small compute, low risk of
  catastrophic forgetting of English.

Alternatives considered: **VITS / single-speaker fine-tunes** (no zero-shot cloning,
weaker code-switching), and **training from scratch** (infeasible without a large
aligned corpus). XTTS-v2 dominated on capability-per-unit-effort.

### 3.2 Optimization choices

- **Fine-tune the GPT only**, load it on top of the frozen base XTTS-v2 decoder /
  speaker encoder. Keeps English quality and speaker fidelity intact.
- **TF32 matmul + `torch.compile` (`reduce-overhead`) on the GPT** — the
  autoregressive loop is the latency bottleneck. This is the "v2 optimized" path.
- **Rejected:** full **fp16 on the HiFi-GAN decoder** broke the fp32 conv filters in
  the speaker encoder (dtype mismatch). **ONNX export** of the AR GPT is non-trivial
  (KV-cache + dynamic loop) and gave no reliable speedup over `torch.compile` for
  streaming. So TF32 + compile is the recommended path.
- **Conditioning latents are cached** per reference speaker, so repeated synthesis
  for the same voice skips the encoder.

---

## 4. Training / Fine-tuning Approach

- **Objective.** Continue XTTS-v2's GPT training (next audio-token prediction) on the
  50K synthetic (text, audio) pairs, with the text front-end applied to inputs so the
  model learns on the same normalized text it will see at inference.
- **What is trained.** Only the GPT weights are updated; the DVAE, speaker encoder,
  and HiFi-GAN decoder are kept from the base model. At load time we merge the
  fine-tuned GPT state dict onto the base XTTS-v2.
- **Checkpoint selection.** `best_model.pth` chosen on validation loss; intermediate
  checkpoints retained during training, only the best is shipped.
- **Reproducibility.** `scripts/prepare_dataset.py` builds the manifest;
  `scripts/train.py` runs the fine-tune from `configs/finetune_xtts.yaml`. Inference
  loads base XTTS-v2 (auto-downloaded) + the fine-tuned GPT.

---

## 5. System & Serving

- **`LevaTTS` package API** — `synthesize`, `zero_shot_synthesize`, `stream`,
  `zero_shot_stream`; auto-downloads the checkpoint + reference speakers from
  HuggingFace.
- **FastAPI streaming server** — `POST /synthesize` (WAV/PCM), `WS /stream`
  (real-time int16 PCM chunks), `/health`, `/metrics`.
- **Pipecat plugin** — `LevaTTSService` (local-GPU or remote-WebSocket) emitting the
  Pipecat contract `TTSStartedFrame → TTSAudioRawFrame… → TTSStoppedFrame` for
  conversational agents.
- **Gradio app** — speaker dropdown, processed-text preview, batch + streaming,
  zero-shot upload, generation-parameter sliders.

---

## 6. Benchmark Results vs Targets

Measured with `scripts/evaluate.py` — speaker **Mohamed**, **NVIDIA H100**, Whisper
large-v3 ASR round-trip for CER/WER, UTMOS for reference-free naturalness.

**Overall**

| Metric | Target | Measured |
|--------|--------|----------|
| Peak VRAM (inference) | ≤ 3 GB | **2.13 GB** ✅ |
| RTF p50 / p95 | < 0.3 | 0.36 / 0.53 |
| TTFA p50 / p95 (batch) | — | 1194 / 1743 ms |
| TTFA streaming (first chunk) | < 300 ms | ~565 ms |
| CER (mean) | — | 0.255 |
| WER (mean) | — | 0.496 |
| **UTMOS** (ref-free MOS) | — | **3.13 / 5.0** |

**Per-category (intelligibility via ASR round-trip)**

| Category | n | CER ↓ | WER ↓ | RTF ↓ | UTMOS ↑ |
|----------|---|-------|-------|-------|---------|
| Pure English | 3 | **0.144** | **0.190** | 0.365 | **3.35** |
| Pure Levantine Arabic | 6 | 0.236 | 0.544 | 0.412 | 2.97 |
| Code-Switching | 6 | 0.330 | 0.602 | 0.358 | 3.19 |

**v2 optimization (TF32 + `torch.compile` on the GPT)**

| Metric | Default | Optimized | Δ |
|--------|---------|-----------|---|
| RTF p50 | 0.362 | **0.355** | −1.9% |
| RTF p95 | 0.528 | **0.494** | **−6.4%** |
| TTFA p50 (ms) | 1194 | **1150** | −44 ms |
| UTMOS ↑ | 3.13 | **3.24** | **+3.5%** |

**Reading the results.** VRAM is comfortably under target. Pure English is near-ASR-
perfect, confirming English quality is retained. Arabic CER/WER are higher in part
because **Whisper large-v3 transcribes Arabic toward MSA orthography** while our
references keep Levantine spelling + partial diacritics — so a fraction of the
"errors" are orthographic, not acoustic. Code-switching is hardest (language
boundaries), as expected. RTF/TTFA are close to but not under the aggressive
latency targets on the full sentence set; the optimized path narrows the gap
(p95 RTF −6.4%) while slightly *improving* UTMOS.

---

## 7. Known Limitations & Future Work

### Known limitations

- **Synthetic training data.** Voices and prosody inherit Lahgtna-OmniVoice v2's
  characteristics; the model has never seen real human Levantine recordings, so it
  can reproduce the generator's systematic quirks.
- **Latency targets not fully met.** RTF p50 (0.36) and streaming TTFA (~565 ms)
  miss the < 0.3 / < 300 ms targets; good for many agents, not yet "instant."
- **Dialect scope.** Optimized for (North) Levantine; Egyptian/Gulf/MSA are out of
  distribution.
- **Evaluation proxy.** CER/WER via ASR round-trip conflates orthographic and
  acoustic errors for Arabic; UTMOS is reference-free and imperfect. No formal human
  MOS study was run.
- **Code-switch boundaries** remain the weakest category (occasional accent bleed at
  the switch point).

### What I'd do with more time

1. **Gather real, representative Levantine data — from podcasts.** Mine Levantine
   podcasts / interviews / shows, run diarization + ASR to segment, and **carefully
   verify text↔audio alignment** (forced alignment + manual spot-checks). Real
   spontaneous speech would fix the "synthetic voice" ceiling and improve prosody and
   code-switch naturalness. Quality of alignment matters more than raw quantity.
2. **Push the optimization further** to actually beat the latency targets: smaller /
   distilled GPT, INT8 or proper fp16 paths that don't break the decoder, KV-cache
   ONNX/TensorRT export of the AR loop, and a streaming decoder warm path to cut
   first-chunk TTFA below 300 ms.
3. **Better evaluation.** A proper **human MOS / preference study** (Levantine native
   listeners), plus a Levantine-aware ASR (or normalized scoring) so CER/WER reflect
   pronunciation rather than spelling conventions.
4. **Explore other architectures.** Benchmark a **VITS / VITS2** end-to-end model
   (single-stage, often lower latency) and modern codec-LM TTS for Levantine, to see
   whether a non-XTTS backbone gives better latency-per-quality. Investigate explicit
   **phonemization** for Arabic (e.g. a Levantine G2P) to reduce orthographic
   ambiguity at the source.
5. **Richer front-end & data balance.** Expand the Levantine lexicon, add more
   diacritization coverage on ambiguous homographs, and balance the training mix
   toward harder code-switch patterns.
6. **Productionization.** Batched/continuous-batching server, autoscaling, and a
   speaker-embedding cache service for multi-tenant low-latency serving.

---

*Artifacts: model weights — https://huggingface.co/mohammedaly22/leva-tts ·
code — https://github.com/MohammedAly22/Leva-TTS · demo — https://mohammedaly22.github.io/Leva-TTS/*
