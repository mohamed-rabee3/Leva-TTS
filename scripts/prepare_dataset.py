"""
Prepare the 3 training data sources for Leva-TTS fine-tuning.

Sources
-------
  1. LibriSpeech clean-100  (data/raw/librispeech_hf)
       English multi-speaker, capped to MAX_HOURS_EN hours
  2. Lahgtna synthetic      (data/synthetic_data)
       50K high-quality Saudi + CS + number utterances from lahgtna-omnivoice-v2
       (single speaker: hoda). Already at synthesis quality — only
       resampling + metadata needed.

Output (XTTS-compatible metadata.csv per source)
-------
  data/processed/librispeech/        metadata.csv  +  wavs/
  data/processed/synthetic/          metadata.csv  +  wavs/   (symlinked / copied)

Usage
-----
  # Edit SKIP_DOWNLOAD if data is already downloaded
  python scripts/prepare_dataset.py
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
#                            CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
RAW_DIR            = "./data/raw"
PROC_DIR           = "./data/processed"
SYNTHETIC_SRC      = "./data/synthetic_data"   # output of generate_lahgetna_data.py

MAX_HOURS_EN       = 20.0          # LibriSpeech English cap (hours)
SKIP_DOWNLOAD      = True         # True = preprocess only (already downloaded)
# ╚══════════════════════════════════════════════════════════════════════════╝

import logging
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.logging import RichHandler
    console = Console()
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
except ImportError:
    console = None
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

logger = logging.getLogger(__name__)


def step(label, fn):
    if console:
        console.print(f"\n[bold yellow]{label}[/bold yellow]")
    else:
        print(f"\n{label}")
    try:
        fn()
        if console:
            console.print("[green]  ✅  Done[/green]")
        else:
            print("  Done.")
    except Exception as e:
        if console:
            console.print(f"[red]  ❌  Failed: {e}[/red]")
        else:
            print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()


def prepare_synthetic(src_dir: str, proc_dir: Path, delete_source: bool = False):
    """
    Prepare the Lahgtna synthetic data.

    The WAVs are already 24 kHz, high quality. We resample them to 22050 Hz
    (XTTS-v2 native rate), apply loudness normalisation, and copy to proc_dir
    along with metadata.csv.

    delete_source: delete each source WAV after processing to save disk space.
    """
    import csv
    import numpy as np
    import librosa
    import soundfile as sf
    from tqdm import tqdm

    src      = Path(src_dir)
    meta_src = src / "metadata.csv"

    if not meta_src.exists():
        logger.error(f"metadata.csv not found in {src_dir}. "
                     "Run generate_lahgetna_data.py first.")
        return

    proc_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(meta_src, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2:
                rows.append((row[0], row[1]))

    logger.info(f"Synthetic source: {len(rows):,} entries in {meta_src}")
    if delete_source:
        logger.info("--delete-source enabled: source WAVs deleted after processing")

    TARGET_SR = 22_050
    out_rows  = []
    skip      = 0

    meta_out  = proc_dir / "metadata.csv"
    done = set()
    if meta_out.exists():
        with open(meta_out, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if row: done.add(row[0])

    with open(meta_out, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, delimiter="|")

        for wav_rel, text in tqdm(rows, desc="Synthetic (resample+norm)"):
            if wav_rel in done:
                if delete_source:
                    wav_src = src / wav_rel
                    if wav_src.exists():
                        wav_src.unlink()
                continue
            wav_src = src / wav_rel
            if not wav_src.exists():
                skip += 1; continue

            try:
                wav, sr = librosa.load(str(wav_src), sr=TARGET_SR, mono=True)
                rms = np.sqrt(np.mean(wav ** 2))
                if rms > 1e-9:
                    target_rms = 10 ** (-23.0 / 20.0)
                    wav = wav * (target_rms / rms)
                    peak = np.max(np.abs(wav))
                    if peak > 0.99:
                        wav *= 0.99 / peak

                out_wav = proc_dir / wav_rel
                out_wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_wav), wav, TARGET_SR, subtype="PCM_16")

                writer.writerow([wav_rel, text])
                fout.flush()
                out_rows.append((wav_rel, text))

                if delete_source:
                    wav_src.unlink()
            except Exception as e:
                logger.debug(f"Skip {wav_src.name}: {e}")
                skip += 1

    logger.info(f"Synthetic: {len(out_rows):,} processed, {skip} skipped → {proc_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-source", action="store_true",
                        help="Delete each source WAV after processing (saves ~14 GB disk)")
    args = parser.parse_args()

    raw  = Path(RAW_DIR)
    proc = Path(PROC_DIR)

    if console:
        console.print(Panel(
            "[bold cyan]⚙️  Leva-TTS  ·  Data Preparation[/bold cyan]\n"
            f"[dim]Lahgtna synthetic: {SYNTHETIC_SRC}[/dim]"
            + ("\n[bold red]--delete-source: raw WAVs will be deleted as processed[/bold red]"
               if args.delete_source else ""),
            border_style="cyan", padding=(1, 4),
        ))

    # ── Lahgtna synthetic ─────────────────────────────────────────────────────
    step("⚙️  Prepare Lahgtna synthetic → 22 kHz + normalise",
         lambda: prepare_synthetic(SYNTHETIC_SRC, proc / "synthetic",
                                   delete_source=args.delete_source))

    if console:
        console.print("\n[bold green]🎉  All 2 sources prepared.[/bold green]\n")
        console.print(
            "[dim]Add these to configs/finetune_xtts.yaml:\n"
            "  - name: librispeech\n"
            "    path: ./data/processed/librispeech\n"
            "    meta_file: metadata.csv\n"
            "    language: en\n"
            "    formatter: ljspeech\n\n"
            "  - name: synthetic_saudi\n"
            "    path: ./data/processed/synthetic\n"
            "    meta_file: metadata.csv\n"
            "    language: ar\n"
            "    formatter: ljspeech[/dim]"
        )
    else:
        print("\nAll 2 sources prepared. ✅")
