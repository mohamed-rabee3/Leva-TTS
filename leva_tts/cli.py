"""leva-tts CLI entry point."""
import sys


def main():
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    console.print(Panel(
        "[bold cyan]🎙️  Leva-TTS  v0.1.0[/bold cyan]\n"
        "[dim]Levantine Arabic / English Code-Switching TTS[/dim]\n\n"
        "Commands:\n"
        "  [green]python scripts/inference.py[/green]   — synthesize text\n"
        "  [green]python app.py[/green]                 — launch Gradio demo\n"
        "  [green]python scripts/train.py[/green]       — fine-tune\n"
        "  [green]tensorboard --logdir checkpoints/[/green] — monitor training",
        border_style="cyan",
    ))
