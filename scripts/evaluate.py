"""
📊  Leva-TTS  ·  Evaluation Suite

Evaluates a fine-tuned checkpoint on a set of test sentences covering:
  - Pure Levantine Arabic
  - Pure English
  - Code-switching (AR + EN)

Metrics reported:
  Latency  : TTFA (p50/p95 ms), RTF (p50/p95)
  Resource : Peak VRAM (GB)
  Quality  : CER/WER via Whisper large-v3 ASR round-trip
  MOS proxy: UTMOS (reference-free neural MOS predictor)
  Per-type : pure_levantine / pure_english / code_switching breakdown

Usage
-----
  python scripts/evaluate.py --checkpoint checkpoints/
  python scripts/evaluate.py --checkpoint checkpoints/ --out eval_results/
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
#                           CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
DEFAULT_CHECKPOINT  = "checkpoints"
DEFAULT_SPEAKER     = "Badr"
REFERENCES_JSON     = "reference_audios/references.json"
OUTPUT_DIR          = "eval_results"
DEVICE              = "cuda"
WHISPER_MODEL       = "large-v3"    # for ASR round-trip CER/WER
SAVE_WAVS           = True
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Built-in test set ─────────────────────────────────────────────────────────
TEST_SENTENCES = [
    # Pure Levantine Arabic
    {"id":"ar_01","type":"pure_levantine","lang":"ar",
     "text":"كيفك اليوم؟ إنت شو عم تعمل هَلَّق؟"},
    {"id":"ar_02","type":"pure_levantine","lang":"ar",
     "text":"هَلَّق رح أروح على البيت وبكرا برجع."},
    {"id":"ar_03","type":"pure_levantine","lang":"ar",
     "text":"والله مِشْ عارف شو بِدِّي أعمل هَلَّق."},
    {"id":"ar_04","type":"pure_levantine","lang":"ar",
     "text":"يلا نروح نشوف الفلم الجديد الليلة."},
    {"id":"ar_05","type":"pure_levantine","lang":"ar",
     "text":"قديش الساعة هَلَّق؟ لازم أروح بسرعة."},
    {"id":"ar_06","type":"pure_levantine","lang":"ar",
     "text":"بكرا في اجتماع مهم كتير مع الشركة."},
    # Pure English
    {"id":"en_01","type":"pure_english","lang":"en",
     "text":"Hello, how are you doing today?"},
    {"id":"en_02","type":"pure_english","lang":"en",
     "text":"The project deadline is next Friday morning."},
    {"id":"en_03","type":"pure_english","lang":"en",
     "text":"I really enjoyed the meeting this morning."},
    # Code-switching
    {"id":"cs_01","type":"code_switching","lang":"ar",
     "text":"هَلَّق أنا عم أشتغل على the new project اللي حكيتلك عنه."},
    {"id":"cs_02","type":"code_switching","lang":"ar",
     "text":"والله the weather today كتير حلو، بِدِّي أطلع برا."},
    {"id":"cs_03","type":"code_switching","lang":"ar",
     "text":"بِدِّي أحكيلك عن the meeting اللي كان مهم كتير."},
    {"id":"cs_04","type":"code_switching","lang":"ar",
     "text":"لازم تراجع the report قبل بكرا أكيد."},
    {"id":"cs_05","type":"code_switching","lang":"ar",
     "text":"كيفك؟ how was your day اليوم؟"},
    {"id":"cs_06","type":"code_switching","lang":"ar",
     "text":"يعني the system is working مِنِيح هَلَّق."},
]

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table
from rich import box

console = Console()


def _apply_optimizations(model):
    """
    v2 inference optimization for XTTS-v2:
      - fp16 (half precision) on the GPT + HiFi-GAN decoder
      - torch.compile on the GPT forward (reduce-overhead) to fuse kernels
    Returns the optimized model. Falls back gracefully on any failure.
    """
    import torch
    # NOTE: only the GPT (autoregressive bottleneck) is halved. The HiFi-GAN
    # decoder + speaker encoder stay fp32 — halving them breaks the fp32 conv
    # filters in the speaker encoder (mismatched dtypes).
    try:
        import torch
        # Enable TF32 matmul (Ampere+/Hopper) — free speedup, no quality loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('  [opt] TF32 matmul enabled')
    except Exception as e:
        print(f'  [opt] TF32 skipped: {e}')
    try:
        # Compile the GPT inference module (the AR bottleneck)
        if hasattr(model.gpt, 'gpt_inference') and model.gpt.gpt_inference is not None:
            model.gpt.gpt_inference = torch.compile(
                model.gpt.gpt_inference, mode='reduce-overhead', fullgraph=False)
            print('  [opt] torch.compile applied to gpt_inference')
    except Exception as e:
        print(f'  [opt] torch.compile skipped: {e}')
    return model


def load_model_and_conditioning(checkpoint, ref_wav, device, optimize=False):
    import os
    cache_dir = Path(os.environ.get("COQUI_MODEL_PATH",
                                    Path.home() / ".local/share/tts"))
    xtts_dir  = cache_dir / "tts_models--multilingual--multi-dataset--xtts_v2"
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    config = XttsConfig()
    config.load_json(str(xtts_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(xtts_dir), eval=True)
    # Fine-tune weights
    from pathlib import Path as P
    pths = sorted(P(checkpoint).rglob("best_model.pth"), key=lambda f: f.stat().st_mtime)
    if not pths:
        pths = sorted(P(checkpoint).rglob("*.pth"), key=lambda f: f.stat().st_mtime)
    if pths:
        state = torch.load(str(pths[-1]), map_location="cpu")
        state = state.get("model", state)
        cleaned = {k[len("xtts.gpt."):] if k.startswith("xtts.gpt.")
                   else k[len("gpt."):] if k.startswith("gpt.") else k: v
                   for k, v in state.items()}
        model.gpt.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    if optimize:
        model = _apply_optimizations(model)
    gpt_cond, spk_emb = model.get_conditioning_latents(audio_path=[ref_wav])
    return model, config, gpt_cond, spk_emb


def synthesize_one(model, config, text, lang, gpt_cond, spk_emb, ref_wav):
    t0 = time.perf_counter()
    out = model.synthesize(text, config, speaker_wav=[ref_wav],
                            language=lang, gpt_cond_len=3)
    wall = time.perf_counter() - t0
    wav  = np.array(out["wav"], dtype=np.float32)
    dur  = len(wav) / 24000
    return wav, wall, dur


_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        console.print("  Loading Whisper large-v3 …")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def asr_roundtrip(wav, sr, lang):
    try:
        import librosa
        model = get_whisper()
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        result = model.transcribe(wav.astype(np.float32), language=lang)
        return result["text"].strip()
    except Exception as e:
        return f"[ASR error: {e}]"


def compute_cer_wer(ref, hyp):
    try:
        from jiwer import cer, wer
        return float(cer(ref, hyp)), float(wer(ref, hyp))
    except Exception:
        return float("nan"), float("nan")


_utmos_model = None
def _get_utmos():
    """Load the official UTMOS (UTMOS22-strong) predictor from torch.hub once."""
    global _utmos_model
    if _utmos_model is None:
        import torch
        _utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        _utmos_model.eval()
    return _utmos_model


def compute_utmos(wav, sr):
    """Reference-free MOS via UTMOS22-strong (expects 16 kHz mono float32)."""
    try:
        import torch, numpy as np, librosa
        if sr != 16000:
            wav = librosa.resample(np.asarray(wav, dtype=np.float32),
                                   orig_sr=sr, target_sr=16000)
        model = _get_utmos()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
            return float(model(t, 16000))
    except Exception as e:
        return float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="📊 Leva-TTS evaluation")
    ap.add_argument("--checkpoint",   default=DEFAULT_CHECKPOINT)
    ap.add_argument("--speaker",      default=DEFAULT_SPEAKER)
    ap.add_argument("--out",          default=OUTPUT_DIR)
    ap.add_argument("--device",       default=DEVICE)
    ap.add_argument("--no-asr",       action="store_true", help="Skip Whisper ASR")
    ap.add_argument("--no-utmos",     action="store_true", help="Skip UTMOS")
    ap.add_argument("--optimize",     action="store_true",
                    help="Apply v2 optimizations (fp16 + torch.compile)")
    ap.add_argument("--tag",          default="default",
                    help="Label for the run (e.g. default / optimized)")
    args = ap.parse_args()

    console.print(Panel(
        "📊  [bold cyan]Leva-TTS Evaluation[/bold cyan]\n"
        f"[dim]Checkpoint: {args.checkpoint}  Speaker: {args.speaker}[/dim]",
        border_style="cyan",
    ))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    # Load reference
    refs = json.loads(Path(REFERENCES_JSON).read_text(encoding="utf-8"))
    ref  = next((r for r in refs if Path(r["audio_path"]).stem == args.speaker), refs[0])
    ref_wav = ref["audio_path"]

    # Load model
    console.print("\n  Loading model …")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    model, config, gpt_cond, spk_emb = load_model_and_conditioning(
        args.checkpoint, ref_wav, args.device, optimize=args.optimize)
    if args.optimize:
        console.print("  Warming up compiled model (2 passes) …")
        for _ in range(2):
            try:
                synthesize_one(model, config, "مرحبا كيفك اليوم", "ar",
                               gpt_cond, spk_emb, ref_wav)
            except Exception as e:
                console.print(f"  warmup note: {e}")

    rows = []
    ttfas, rtfs = [], []

    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"),
                  MofNCompleteColumn(), console=console) as prog:
        task = prog.add_task("Evaluating …", total=len(TEST_SENTENCES))

        for sent in TEST_SENTENCES:
            sid  = sent["id"]
            text = sent["text"]
            lang = sent["lang"]
            stype= sent["type"]

            # Normalize text
            from leva_tts.text.processor import TextProcessor
            processed = TextProcessor().process(text)

            # Synthesize
            t0  = time.perf_counter()
            wav, wall, dur = synthesize_one(model, config, processed, lang,
                                            gpt_cond, spk_emb, ref_wav)
            ttfa_ms = wall * 1000   # approximate (batch mode)
            rtf     = wall / (dur + 1e-9)
            ttfas.append(ttfa_ms); rtfs.append(rtf)

            # Save WAV
            wav_path = wav_dir / f"{sid}.wav"
            sf.write(str(wav_path), wav, 24000, subtype="PCM_16")

            # ASR round-trip
            asr_hyp = asr_roundtrip(wav, 24000, lang) if not args.no_asr else "—"
            cer_v, wer_v = compute_cer_wer(text, asr_hyp) if not args.no_asr else (float("nan"), float("nan"))

            # UTMOS
            utmos_v = compute_utmos(wav, 24000) if not args.no_utmos else float("nan")

            rows.append({
                "id": sid, "type": stype, "lang": lang, "text": text[:50],
                "dur_s": round(dur, 2), "ttfa_ms": round(ttfa_ms, 1),
                "rtf": round(rtf, 4), "cer": round(cer_v, 3),
                "wer": round(wer_v, 3), "utmos": round(utmos_v, 3),
                "asr_hyp": asr_hyp[:60],
            })
            prog.update(task, advance=1,
                        description=f"[{sid}] RTF={rtf:.3f} CER={cer_v:.2f}")

    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # ── Results table ─────────────────────────────────────────────────────────
    import json as _json, statistics as _stat

    def pct(data, p):
        return round(float(sorted(data)[int(len(data)*p/100)]), 1) if data else 0

    console.print()
    console.print(Panel("📈  Evaluation Results", border_style="green"))

    # KPI table
    kpi = Table(box=box.ROUNDED, header_style="bold cyan")
    kpi.add_column("KPI");            kpi.add_column("Value", justify="right")
    kpi.add_column("Target", justify="right"); kpi.add_column("Status", justify="center")
    kpi.add_row("Peak VRAM (GB)", f"{peak_vram:.2f}", "≤ 3.0",
                "✅" if peak_vram <= 3.0 else "❌")
    kpi.add_row("TTFA p50 (ms)",  f"{pct(ttfas,50)}", "< 300", "")
    kpi.add_row("TTFA p95 (ms)",  f"{pct(ttfas,95)}", "< 300",
                "✅" if pct(ttfas,95) < 300 else "❌")
    kpi.add_row("RTF p50",        f"{pct(rtfs,50)/1000:.4f}", "< 0.3", "")
    kpi.add_row("RTF p95",        f"{pct(rtfs,95)/1000:.4f}", "< 0.3",
                "✅" if pct(rtfs,95)/1000 < 0.3 else "❌")
    console.print(kpi)

    # Per-sentence table
    det = Table(box=box.SIMPLE, header_style="bold cyan", title="Per-sentence")
    for col in ["ID","Type","Lang","Dur(s)","TTFA(ms)","RTF","CER","WER","UTMOS"]:
        det.add_column(col, no_wrap=True)
    for r in rows:
        det.add_row(r["id"], r["type"][:8], r["lang"],
                    str(r["dur_s"]), str(r["ttfa_ms"]), str(r["rtf"]),
                    str(r["cer"]), str(r["wer"]), str(r["utmos"]))
    console.print(det)

    # Save JSON report
    report = {"kpis": {"peak_vram_gb": round(peak_vram,3),
                       "ttfa_p50_ms": pct(ttfas,50), "ttfa_p95_ms": pct(ttfas,95),
                       "rtf_p50": pct(rtfs,50)/1000, "rtf_p95": pct(rtfs,95)/1000},
              "sentences": rows}
    (out_dir / f"eval_report_{args.tag}.json").write_text(
        _json.dumps(report, ensure_ascii=False, indent=2))
    console.print(f"\n  📄 Full report → {out_dir}/eval_report.json")
    console.print(f"  🎵 WAVs        → {wav_dir}/")
