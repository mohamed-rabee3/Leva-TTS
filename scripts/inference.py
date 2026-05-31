"""
🎙️  Leva-TTS  ·  Inference Script

Synthesize Levantine Arabic / English code-switching speech.

Features:
  - Batch or streaming synthesis
  - 10 built-in reference speakers (from references.json)
  - Custom reference audio (zero-shot voice cloning)
  - Rich terminal UI with progress and metrics

Usage
-----
  # Default speaker (Badr), batch mode
  python scripts/inference.py --text "كيفك اليوم؟ the weather is great هَلَّق"

  # Streaming mode
  python scripts/inference.py --text "..." --stream

  # Choose speaker
  python scripts/inference.py --text "..." --speaker Amina

  # Zero-shot with your own audio
  python scripts/inference.py --text "..." --ref-audio /path/to/your.wav

  # Save to file
  python scripts/inference.py --text "..." --out output.wav
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
#                           CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
DEFAULT_CHECKPOINT   = "checkpoints"          # auto-detects best_model.pth
DEFAULT_SPEAKER      = "Badr"                 # from references.json
REFERENCES_JSON      = "reference_audios/references.json"
DEFAULT_OUTPUT       = "output.wav"
DEVICE               = "cuda"
LANGUAGE             = "ar"
# ╚══════════════════════════════════════════════════════════════════════════╝

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


# ── Speaker helpers ───────────────────────────────────────────────────────────
def load_references(path: str) -> dict:
    """Load references.json → {speaker_name: ref_dict}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        _extract_name(r["speaker_id"], r["audio_path"]): r
        for r in data
        if Path(r["audio_path"]).exists()
    }


def _extract_name(spk_id: str, audio_path: str) -> str:
    """Extract human name from audio filename (Badr.wav → Badr)."""
    return Path(audio_path).stem


def find_best_checkpoint(ckpt_dir: str) -> str | None:
    p = Path(ckpt_dir)
    # Prefer best_model.pth
    bests = sorted(p.rglob("best_model.pth"), key=lambda f: f.stat().st_mtime)
    if bests:
        return str(bests[-1])
    pths = sorted(p.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
    return str(pths[-1]) if pths else None


def load_engine(checkpoint: str, device: str):
    """Load XTTS fine-tuned engine with progress display."""
    import os
    cache_dir = Path(os.environ.get("COQUI_MODEL_PATH",
                                    Path.home() / ".local/share/tts"))
    xtts_dir  = cache_dir / "tts_models--multilingual--multi-dataset--xtts_v2"

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import torch

    config = XttsConfig()
    config.load_json(str(xtts_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(xtts_dir), eval=True)

    # Load fine-tuned GPT weights
    ckpt_file = find_best_checkpoint(checkpoint)
    if ckpt_file:
        console.print(f"  📂 Checkpoint: [dim]{ckpt_file}[/dim]")
        state = torch.load(ckpt_file, map_location="cpu")
        state = state.get("model", state)
        cleaned = {}
        for k, v in state.items():
            key = k
            for prefix in ("xtts.gpt.", "gpt."):
                if k.startswith(prefix):
                    key = k[len(prefix):]
                    break
            cleaned[key] = v
        missing, unexpected = model.gpt.load_state_dict(cleaned, strict=False)
        if missing:
            console.print(f"  [dim yellow]⚠  {len(missing)} GPT keys not found in checkpoint[/dim yellow]")
    else:
        console.print("  [yellow]⚠  No fine-tune checkpoint found — using base XTTS-v2[/yellow]")

    model = model.to(device)
    model.eval()
    return model, config


# ── Text processing ───────────────────────────────────────────────────────────
def process_text(text: str, verbose: bool = False) -> str:
    from leva_tts.text.processor import TextProcessor
    tp = TextProcessor(verbose=verbose)
    return tp.process(text)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="🎙️  Leva-TTS inference")
    ap.add_argument("--text",       required=True,             help="Text to synthesize")
    ap.add_argument("--speaker",    default=DEFAULT_SPEAKER,   help="Speaker name (e.g. Badr, Amina)")
    ap.add_argument("--ref-audio",  default=None,              help="Custom reference audio (zero-shot)")
    ap.add_argument("--language",   default=LANGUAGE,          help="'ar' or 'en'")
    ap.add_argument("--out",        default=DEFAULT_OUTPUT,    help="Output WAV path")
    ap.add_argument("--stream",     action="store_true",       help="Streaming synthesis")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device",     default=DEVICE)
    ap.add_argument("--verbose-text", action="store_true",     help="Show text pipeline stages")
    args = ap.parse_args()

    # ── Banner ────────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "🎙️  [bold cyan]Leva-TTS[/bold cyan]  ·  Levantine Arabic / English TTS\n"
        "[dim]Powered by XTTS-v2 fine-tuned on Lahgtna-OmniVoice synthetic data[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    # ── Speaker / reference audio ─────────────────────────────────────────────
    refs = load_references(REFERENCES_JSON)
    if args.ref_audio:
        ref_wav  = args.ref_audio
        ref_text = ""
        spk_name = "Custom"
        console.print(f"  🎤 Mode     : [bold]Zero-shot[/bold] (custom reference)")
        console.print(f"  📁 Ref audio: [dim]{ref_wav}[/dim]")
    else:
        if args.speaker not in refs:
            avail = ", ".join(refs.keys())
            console.print(f"  [red]Speaker '{args.speaker}' not found. Available: {avail}[/red]")
            sys.exit(1)
        ref      = refs[args.speaker]
        ref_wav  = ref["audio_path"]
        ref_text = ref.get("reference_text", "")
        spk_name = args.speaker
        console.print(f"  🎤 Speaker  : [bold]{spk_name}[/bold]  ({ref['gender']})")

    # ── Process text ──────────────────────────────────────────────────────────
    console.print(f"\n  📝 Input    : {args.text[:80]}")
    processed = process_text(args.text, verbose=args.verbose_text)
    if processed != args.text:
        console.print(f"  ✨ Processed: [green]{processed[:80]}[/green]")
    console.print(f"  🌐 Language : {args.language}")
    console.print(f"  🎵 Mode     : {'Streaming' if args.stream else 'Batch'}")
    console.print()

    # ── Load model ────────────────────────────────────────────────────────────
    import torch
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  console=console, transient=True) as prog:
        task = prog.add_task("Loading model …", total=None)
        model, config = load_engine(args.checkpoint, args.device)
        prog.update(task, description="Model ready ✅")

    # ── Speaker conditioning ──────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                  console=console, transient=True) as prog:
        task = prog.add_task(f"Computing speaker conditioning …", total=None)
        gpt_cond, spk_emb = model.get_conditioning_latents(audio_path=[ref_wav])
        prog.update(task, description="Speaker conditioning ready ✅")

    # ── Synthesize ────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t_start = time.perf_counter()

    if args.stream:
        console.print("  🎵 [bold]Streaming[/bold] synthesis …")
        chunks    = []
        first     = True
        ttfa_ms   = 0.0
        with Progress(SpinnerColumn(), BarColumn(),
                      TextColumn("[cyan]{task.description}"),
                      console=console) as prog:
            task = prog.add_task("Generating …", total=None)
            try:
                for chunk in model.inference_stream(
                    processed, args.language,
                    gpt_cond_latent=gpt_cond, speaker_embedding=spk_emb,
                    stream_chunk_size=20,
                ):
                    if first:
                        ttfa_ms = (time.perf_counter() - t_start) * 1000
                        first = False
                    arr = chunk.squeeze().cpu().numpy()
                    chunks.append(arr)
                    prog.update(task, description=f"chunk {len(chunks)}")
            except Exception as e:
                console.print(f"  [yellow]Stream fallback to batch ({e})[/yellow]")
                out = model.synthesize(processed, config,
                                       speaker_wav=[ref_wav], language=args.language,
                                       gpt_cond_len=3)
                chunks = [np.array(out["wav"], dtype=np.float32)]
                ttfa_ms = (time.perf_counter() - t_start) * 1000
        wav = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    else:
        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"),
                      console=console, transient=True) as prog:
            task = prog.add_task("Synthesizing …", total=None)
            out = model.synthesize(
                processed, config,
                speaker_wav=[ref_wav], language=args.language, gpt_cond_len=3,
            )
            wav = np.array(out["wav"], dtype=np.float32)
            ttfa_ms = (time.perf_counter() - t_start) * 1000

    wall    = time.perf_counter() - t_start
    dur     = len(wav) / 24000
    rtf     = wall / (dur + 1e-9)
    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # ── Save ──────────────────────────────────────────────────────────────────
    sf.write(str(out_path), wav, 24000, subtype="PCM_16")

    # ── Results table ─────────────────────────────────────────────────────────
    tbl = Table(box=box.ROUNDED, border_style="green", show_header=False, padding=(0,1))
    tbl.add_column("", style="dim")
    tbl.add_column("", style="bold white")
    tbl.add_row("💾 Saved",     str(out_path))
    tbl.add_row("🕐 Duration",  f"{dur:.2f} s")
    tbl.add_row("⏱️  TTFA",      f"{ttfa_ms:.0f} ms")
    tbl.add_row("🎚️  RTF",       f"{rtf:.4f}{'  ✅' if rtf < 0.3 else '  ⚠️'}")
    tbl.add_row("💾 VRAM",      f"{vram_gb:.2f} GB")
    console.print()
    console.print(tbl)
    console.print()
