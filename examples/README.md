# 📓 Examples — Colab Notebooks

Run Leva-TTS end-to-end on a **free Colab T4 GPU** — no local setup. Each notebook
has step-by-step cells you run top to bottom.

| Notebook | What it does | Open |
|----------|--------------|------|
| **01 · Quick Start** | Install, load the model, synthesize built-in speakers, zero-shot cloning, streaming | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Leva-TTS/blob/main/examples/01_quick_start.ipynb) |
| **02 · Inference Server** | Launch the FastAPI streaming server in Colab; send `POST /synthesize` + `WS /stream` requests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Leva-TTS/blob/main/examples/02_inference_server.ipynb) |
| **03 · Evaluation** | Reproduce RTF / TTFA / VRAM / CER / WER / UTMOS on a T4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Leva-TTS/blob/main/examples/03_evaluation.ipynb) |
| **04 · Gradio App** | Launch the full web demo with a public share link | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Leva-TTS/blob/main/examples/04_gradio_app.ipynb) |

> **Tip:** set **Runtime ▸ Change runtime type ▸ T4 GPU** before running.
> The first model load downloads the checkpoint (~2 GB) from HuggingFace.
