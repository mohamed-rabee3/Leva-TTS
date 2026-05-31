"""
Reproducible latency benchmark — TTFA, RTF, peak VRAM.

Edit the variables below, then run:
    python scripts/benchmark.py
"""

# ╔══════════════════════ CONFIGURATION ════════════════════════════════════════╗
CHECKPOINT         = "./checkpoints/best_model"      # fine-tuned checkpoint dir
SPEAKER_WAV        = "./data/reference_speaker.wav"  # reference speaker
N_RUNS             = 5          # runs per sentence (results averaged)
DEVICE             = "cuda"     # "cuda" | "cpu"
USE_DEEPSPEED      = True       # DeepSpeed fp16 inference
OUT_JSON           = "benchmark_results.json"
# ╚═════════════════════════════════════════════════════════════════════════════╝

import json
import logging
import time
from pathlib import Path

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
    from rich.logging import RichHandler
    from rich import box
    console = Console()
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
except ImportError:
    console = None
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

logger = logging.getLogger(__name__)

BENCHMARK_SENTENCES = [
    {"text": "كيفك اليوم؟ إنت شو عم تعمل هلق؟",                                 "lang": "ar", "label": "AR short"},
    {"text": "بدي أحكيلك عن the new project اللي عم نشتغل عليه هلق في الشركة.", "lang": "ar", "label": "AR+EN medium"},
    {"text": "Hello, how are you doing today? I hope everything is going well.",  "lang": "en", "label": "EN medium"},
    {"text": "والله the weather today كتير حلو، بدي أطلع برا وأتمشى شوي.",       "lang": "ar", "label": "AR+EN long"},
    {"text": "هلق رح نبدأ بال training session اللي حكينا عنها مبارح مع ال team.", "lang": "ar", "label": "AR+EN long2"},
]


def _banner():
    if console:
        console.print(Panel(
            "[bold cyan]⚡  Leva-TTS  ·  Latency Benchmark[/bold cyan]\n"
            f"[dim]Checkpoint : {CHECKPOINT}[/dim]\n"
            f"[dim]Runs/sent  : {N_RUNS}   Device: {DEVICE}   DeepSpeed: {USE_DEEPSPEED}[/dim]",
            border_style="cyan", padding=(1, 4),
        ))
    else:
        print("=== Leva-TTS Benchmark ===")


def _kpi_badge(value, target, lower_is_better=True):
    """Return colored PASS/FAIL string."""
    ok = (value <= target) if lower_is_better else (value >= target)
    if console:
        return "[bold green]✅ PASS[/bold green]" if ok else "[bold red]❌ FAIL[/bold red]"
    return "PASS" if ok else "FAIL"


def run():
    _banner()
    import torch
    from leva_tts.inference.engine import LevaTTSEngine

    if console:
        console.print("\n[bold yellow]🔄  Loading engine …[/bold yellow]")

    if Path(CHECKPOINT).exists():
        engine = LevaTTSEngine.from_checkpoint(
            CHECKPOINT,
            speaker_wav=SPEAKER_WAV if Path(SPEAKER_WAV).exists() else None,
            use_deepspeed=USE_DEEPSPEED,
            device=DEVICE,
        )
    else:
        if console:
            console.print(f"[yellow]⚠️  Checkpoint not found ({CHECKPOINT}). Using base XTTS-v2.[/yellow]")
        engine = LevaTTSEngine.from_pretrained(
            speaker_wav=SPEAKER_WAV if Path(SPEAKER_WAV).exists() else None,
            use_deepspeed=USE_DEEPSPEED,
            device=DEVICE,
        )

    if console:
        console.print("[bold yellow]🔥  Warming up …[/bold yellow]")
    engine.warmup()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    all_results = []

    if console:
        prog_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        )
    else:
        prog_ctx = None

    def _run_sentence(sent):
        ttfa_list, rtf_list = [], []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            chunks, first = [], True
            ttfa_ms = 0.0
            for chunk in engine.stream(sent["text"], language=sent["lang"]):
                if first:
                    ttfa_ms = (time.perf_counter() - t0) * 1000
                    first = False
                chunks.append(chunk)
            wav  = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
            dur  = len(wav) / engine.sample_rate
            wall = time.perf_counter() - t0
            ttfa_list.append(ttfa_ms)
            rtf_list.append(wall / (dur + 1e-9))
        return {
            "label":    sent["label"],
            "lang":     sent["lang"],
            "ttfa_p50": float(np.percentile(ttfa_list, 50)),
            "ttfa_p95": float(np.percentile(ttfa_list, 95)),
            "rtf_p50":  float(np.percentile(rtf_list, 50)),
            "rtf_p95":  float(np.percentile(rtf_list, 95)),
            "raw_ttfa": ttfa_list,
            "raw_rtf":  rtf_list,
        }

    if prog_ctx:
        with prog_ctx as prog:
            task = prog.add_task("📊 Benchmarking …", total=len(BENCHMARK_SENTENCES))
            for sent in BENCHMARK_SENTENCES:
                prog.update(task, description=f"📊 {sent['label']} …")
                all_results.append(_run_sentence(sent))
                prog.advance(task)
    else:
        for sent in BENCHMARK_SENTENCES:
            logger.info(f"Benchmarking: {sent['label']}")
            all_results.append(_run_sentence(sent))

    peak_vram = engine.peak_vram_gb()
    all_ttfa  = [v for r in all_results for v in r["raw_ttfa"]]
    all_rtf   = [v for r in all_results for v in r["raw_rtf"]]

    summary = {
        "device":        DEVICE,
        "deepspeed":     USE_DEEPSPEED,
        "n_runs":        N_RUNS,
        "peak_vram_gb":  round(peak_vram, 3),
        "ttfa_p50_ms":   round(float(np.percentile(all_ttfa, 50)), 1),
        "ttfa_p95_ms":   round(float(np.percentile(all_ttfa, 95)), 1),
        "rtf_p50":       round(float(np.percentile(all_rtf, 50)),  4),
        "rtf_p95":       round(float(np.percentile(all_rtf, 95)),  4),
        "kpi_vram_ok":   peak_vram <= 3.0,
        "kpi_ttfa_ok":   float(np.percentile(all_ttfa, 95)) < 300,
        "kpi_rtf_ok":    float(np.percentile(all_rtf,  95)) < 0.3,
        "per_sentence":  all_results,
    }

    # ── Rich results table ────────────────────────────────────────────────────
    if console:
        console.print()
        console.print(Panel("[bold white]📈  Benchmark Results[/bold white]", border_style="cyan"))

        # KPI summary
        kpi_tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        kpi_tbl.add_column("KPI",           style="bold", width=24)
        kpi_tbl.add_column("Value",         justify="right", width=14)
        kpi_tbl.add_column("Target",        justify="right", width=12)
        kpi_tbl.add_column("Status",        justify="center", width=12)

        kpi_tbl.add_row(
            "💾 Peak VRAM (GB)",
            f"{peak_vram:.3f}",
            "≤ 3.0",
            _kpi_badge(peak_vram, 3.0),
        )
        kpi_tbl.add_row(
            "⏱️  TTFA p95 (ms)",
            f"{summary['ttfa_p95_ms']:.0f}",
            "< 300",
            _kpi_badge(summary["ttfa_p95_ms"], 300),
        )
        kpi_tbl.add_row(
            "🎚️  RTF p95",
            f"{summary['rtf_p95']:.4f}",
            "< 0.3",
            _kpi_badge(summary["rtf_p95"], 0.3),
        )
        kpi_tbl.add_row(
            "⏱️  TTFA p50 (ms)",
            f"{summary['ttfa_p50_ms']:.0f}",
            "—",
            "",
        )
        kpi_tbl.add_row(
            "🎚️  RTF p50",
            f"{summary['rtf_p50']:.4f}",
            "—",
            "",
        )
        console.print(kpi_tbl)

        # Per-sentence breakdown
        sent_tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan",
                         title="[bold]Per-sentence breakdown[/bold]")
        sent_tbl.add_column("Sentence",   style="bold", width=18)
        sent_tbl.add_column("Lang",       justify="center", width=6)
        sent_tbl.add_column("TTFA p50",   justify="right", width=10)
        sent_tbl.add_column("TTFA p95",   justify="right", width=10)
        sent_tbl.add_column("RTF p50",    justify="right", width=9)
        sent_tbl.add_column("RTF p95",    justify="right", width=9)
        for r in all_results:
            sent_tbl.add_row(
                r["label"], r["lang"],
                f"{r['ttfa_p50']:.0f} ms", f"{r['ttfa_p95']:.0f} ms",
                f"{r['rtf_p50']:.4f}",     f"{r['rtf_p95']:.4f}",
            )
        console.print(sent_tbl)
    else:
        print(f"\nPeak VRAM: {peak_vram:.3f} GB")
        print(f"TTFA p95:  {summary['ttfa_p95_ms']:.0f} ms")
        print(f"RTF  p95:  {summary['rtf_p95']:.4f}")

    Path(OUT_JSON).write_text(json.dumps(summary, indent=2))
    if console:
        console.print(f"\n[dim]Full results saved to [cyan]{OUT_JSON}[/cyan][/dim]")
    else:
        print(f"Results saved to {OUT_JSON}")


if __name__ == "__main__":
    run()
