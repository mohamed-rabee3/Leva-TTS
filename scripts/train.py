"""Fine-tune XTTS-v2 on Levantine Arabic / English code-switching data."""

# ╔══════════════════════ CONFIGURATION ════════════════════════════════════════╗
CONFIG_FILE        = "configs/finetune_xtts.yaml"   # training config
LOG_LEVEL          = "INFO"                          # DEBUG | INFO | WARNING
# ╚═════════════════════════════════════════════════════════════════════════════╝

import logging
import sys
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.logging import RichHandler
    
    console = Console()
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
except ImportError:
    console = None
    logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                        format="%(asctime)s %(levelname)s %(message)s")


def _banner():
    if console:
        console.print(Panel(
            "[bold cyan]🎙️  Leva-TTS  ·  XTTS-v2 Fine-tuning[/bold cyan]\n"
            "[dim]Levantine Arabic / English Code-Switching TTS[/dim]",
            border_style="cyan", padding=(1, 4),
        ))
        console.print(f"  📄 Config : [yellow]{CONFIG_FILE}[/yellow]")
        console.print(f"  📝 Log    : [yellow]{LOG_LEVEL}[/yellow]")
        console.print()
    else:
        print("=== Leva-TTS Fine-tuning ===")


if __name__ == "__main__":
    _banner()

    cfg = Path(CONFIG_FILE)
    if not cfg.exists():
        msg = f"Config file not found: {CONFIG_FILE}"
        if console:
            console.print(f"[bold red]❌  {msg}[/bold red]")
        else:
            print(f"ERROR: {msg}")
        sys.exit(1)

    if console:
        console.print("[bold green]🚀  Starting fine-tuning …[/bold green]")

    from leva_tts.training.finetune import run_finetuning
    run_finetuning(str(cfg))
