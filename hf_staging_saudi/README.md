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
- saudi-arabic
- najdi-arabic
- code-switching
- voice-cloning
base_model:
- coqui/XTTS-v2
---

# Saudi XTTS-v2

A fine-tuned [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) model for **Saudi Arabic / English code-switching** text-to-speech, with a single female speaker voice ("Hoda").

## Model Details

| Detail | Value |
|---|---|
| Base model | coqui/XTTS-v2 |
| Fine-tuned on | ~50 000 synthetic Saudi Arabic sentences |
| Speaker | Hoda (female, Saudi) |
| Languages | Arabic (`ar`) + English (`en`) |
| Sample rate | 24 000 Hz |
| Training dialect | Najdi / Saudi colloquial Arabic (`ars`) |
| Precision | fp16 |
| Checkpoint step | 320 190 |

## Training Data

The model was trained on ~50 000 synthetically generated sentences covering:
- **Saudi dialect corpus** segments (TaghreedT/SDC, ~60%)
- **Code-switching templates** mixing Saudi Arabic and English (~24%)
- **Number verbalization** sentences — digits rendered in Saudi colloquial words (~16%)

Audio was synthesized with the `lahgtna-omnivoice-v2` TTS system using the Najdi Arabic (`ars`) voice.

## Files

| File | Description |
|---|---|
| `model.pth` | Fine-tuned GPT weights (upload this as the XTTS checkpoint) |
| `config.json` | Training / inference configuration |
| `vocab.json` | XTTS-v2 tokenizer vocabulary |
| `dvae.pth` | Discrete VAE (from XTTS-v2 base, unchanged) |
| `mel_stats.pth` | Mel spectrogram normalisation stats (from XTTS-v2 base, unchanged) |
| `speakers_xtts.pth` | Speaker embedding library (from XTTS-v2 base) |
| `reference_audios/hoda.wav` | Reference audio for voice cloning |
| `references.json` | Speaker metadata |

## Quick Start

```python
from TTS.api import TTS

tts = TTS(model_path="Rabe3/saudi-xtts-v2", progress_bar=True)

tts.tts_to_file(
    text="كيف الحال؟ وش قاعد تسوي اليوم؟",
    speaker_wav="reference_audios/hoda.wav",
    language="ar",
    file_path="output.wav",
)
```

Or using the XTTS model directly:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("config.json")

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=".",   # directory containing model.pth, dvae.pth, etc.
    use_deepspeed=False,
)
model.cuda()

outputs = model.synthesize(
    text="أبغى أقول لك عن the new project اللي قاعدين نشتغل عليه",
    config=config,
    speaker_wav="reference_audios/hoda.wav",
    language="ar",
)
```

## License

CC BY-NC 4.0 — non-commercial use only, consistent with the XTTS-v2 base model license.
