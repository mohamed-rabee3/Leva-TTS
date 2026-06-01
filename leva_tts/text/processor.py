"""
TextProcessor — complete text normalization for Levantine Arabic / English TTS.

Pipeline (no external diacritizer required):
  1. Unicode NFC + control-char removal
  2. Number / currency expansion (Levantine verbalization)
  3. Arabic pre-processing (alef, tatweel)
  4. Levantine lexicon CSV overrides  (partial diacritics on homographs + dialect words)
  5. Final whitespace cleanup

Usage::

    from leva_tts.text.processor import TextProcessor

    tp = TextProcessor()
    text = tp.process("هلق أنا عم أشتغل على the project")

    # Verbose pipeline view
    tp_v = TextProcessor(verbose=True)
    tp_v.process("شو عم تعمل؟ the deadline is بكرا")
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

# Default CSV path
from leva_tts.text.normalizer import normalize_entities, int_to_levantine, float_to_levantine

def _find_lexicon() -> Path:
    """Locate the Levantine lexicon CSV: bundled-in-package first, then repo-root."""
    bundled = Path(__file__).with_name("levantine_lexicon.csv")
    if bundled.exists():
        return bundled
    return Path(__file__).parents[2] / "data" / "levantine_lexicon.csv"


_DEFAULT_LEXICON_CSV = _find_lexicon()

# ── Arabic helpers ────────────────────────────────────────────────────────────
_HARAKAT = re.compile(r"[ً-ٰٟ]")
_TATWEEL = re.compile(r"ـ")


def strip_diac(t: str) -> str:
    return _HARAKAT.sub("", _TATWEEL.sub("", t))


# ── Number verbalization (Levantine) ──────────────────────────────────────────
_ONES = {
    0:"",1:"واحد",2:"اتنين",3:"تلاتة",4:"أربعة",5:"خمسة",
    6:"ستة",7:"سبعة",8:"تمانية",9:"تسعة",10:"عشرة",
    11:"إحدعش",12:"اتناعش",13:"تلتعش",14:"أربعتعش",15:"خمستعش",
    16:"ستعش",17:"سبعتعش",18:"تمنتعش",19:"تسعتعش",
}
_TENS   = {2:"عشرين",3:"تلاتين",4:"أربعين",5:"خمسين",
            6:"ستين",7:"سبعين",8:"تمانين",9:"تسعين"}
_HUNDS  = {1:"مية",2:"مئتين",3:"تلتمية",4:"أربعمية",5:"خمسمية",
            6:"ستمية",7:"سبعمية",8:"تمنمية",9:"تسعمية"}


def int_to_levantine(n: int) -> str:
    if n == 0: return "صفر"
    if n < 0:  return "ناقص " + int_to_levantine(-n)
    parts = []
    if n >= 1_000_000:
        parts.append(int_to_levantine(n // 1_000_000) + " مليون")
        n %= 1_000_000
    if n >= 1_000:
        parts.append(int_to_levantine(n // 1_000) + " ألف")
        n %= 1_000
    if n >= 100:
        parts.append(_HUNDS.get(n // 100, f"{n//100} مية"))
        n %= 100
    if n in _ONES and _ONES[n]:
        parts.append(_ONES[n])
    elif n > 20:
        t, o = n // 10, n % 10
        parts.append((_ONES[o] + " و" + _TENS[t]) if o else _TENS[t])
    return " ".join(parts)


_CURRENCY = {
    "$":"دولار","€":"يورو","£":"جنيه","¥":"ين","₪":"شيكل","%":"بالمية"
}


# ── Rich console helper ───────────────────────────────────────────────────────
def _get_console():
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


# ── TextProcessor ─────────────────────────────────────────────────────────────
class TextProcessor:
    """
    Complete text-processing pipeline for Levantine Arabic / English TTS.

    Parameters
    ----------
    lexicon_csv : str | Path
        Path to ``data/levantine_lexicon.csv`` (Levantine override CSV).
    verbose : bool
        Show pipeline stages in the terminal using rich panels.
    """

    def __init__(
        self,
        lexicon_csv: str | Path = _DEFAULT_LEXICON_CSV,
        verbose: bool = False,
    ):
        self.verbose      = verbose
        self._console     = _get_console()
        self._lexicon_csv = Path(lexicon_csv)
        self._overrides: Optional[dict] = None   # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, text: str) -> str:
        """Return a cleaned, lexicon-corrected string ready for XTTS-v2."""
        if not text.strip():
            return text

        if self.verbose:
            self._print_header(text)

        s = self._stage_unicode(text)
        s = self._stage_numbers(s)

        if self.verbose and s != text:
            self._print_stage("1-3 Normalize + numbers", text, s)

        before_lex = s
        s = self._stage_lexicon(s)

        if self.verbose:
            self._print_lexicon_stage(before_lex, s)

        s = re.sub(r"\s+", " ", s).strip()

        if self.verbose:
            self._print_final(s)

        return s

    def process_batch(self, texts: list[str]) -> list[str]:
        return [self.process(t) for t in texts]

    # ── Pipeline stages ───────────────────────────────────────────────────────
    def _stage_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[​-\u200F\u202A-\u202E⁠-⁯]", "", text)
        # text = re.sub(r"[أإآٱ]", "أ", text)
        text = _TATWEEL.sub("", text)
        text = re.sub(r"([!?.,;:]){2,}", r"\1", text)
        return text

    def _stage_numbers(self, text: str) -> str:
        """Verbalize all numeric entities (numbers, dates, times, URLs, …)."""
        return normalize_entities(text)

    def _stage_lexicon(self, text: str) -> str:
        """Apply Levantine CSV overrides (partial diacritics + dialect corrections)."""
        overrides = self._get_overrides()
        if not overrides:
            return text
        tokens = re.split(r"(\s+|[،؟؛,\.!?\-:]+)", text)
        result = []
        for tok in tokens:
            bare = strip_diac(tok.strip())
            result.append(overrides.get(bare, tok))
        return "".join(result)

    def _get_overrides(self) -> dict:
        if self._overrides is not None:
            return self._overrides
        self._overrides = {}
        if not self._lexicon_csv.exists():
            return self._overrides
        try:
            import csv
            with open(self._lexicon_csv, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row.get("word", "").strip()
                    diac = row.get("diacritized", "").strip()
                    if word and diac:
                        self._overrides[word] = diac
        except Exception:
            pass
        return self._overrides

    # ── Verbose output ────────────────────────────────────────────────────────
    def _print_header(self, text: str):
        if not self._console: return
        try:
            from rich.panel import Panel
            self._console.print()
            self._console.print(Panel(
                f"[bold white]{text}[/bold white]",
                title="[bold cyan]🔤 TextProcessor[/bold cyan]",
                border_style="cyan", padding=(0, 2),
            ))
        except Exception: pass

    def _print_stage(self, name: str, before: str, after: str):
        if not self._console or before == after: return
        try:
            self._console.print(f"   [dim]↓[/dim]  [yellow]{name}[/yellow]")
            self._console.print(f"      [dim]{after[:120]}[/dim]")
        except Exception: pass

    def _print_lexicon_stage(self, before: str, after: str):
        if not self._console: return
        try:
            from rich.table import Table
            from rich import box
            # Find substitutions
            b_tokens = re.split(r"(\s+)", before)
            a_tokens = re.split(r"(\s+)", after)
            subs = [(b, a) for b, a in zip(b_tokens, a_tokens) if b != a and b.strip()]
            label = f"[bold green]🌿 Lexicon overrides[/bold green] ({len(subs)} subs)"
            self._console.print(f"   [dim]↓[/dim]  {label}")
            if subs:
                tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold green")
                tbl.add_column("Original", style="red", no_wrap=True)
                tbl.add_column("→ Override", style="green", no_wrap=True)
                for b, a in subs[:8]:
                    tbl.add_row(b, a)
                self._console.print(tbl)
        except Exception: pass

    def _print_final(self, text: str):
        if not self._console: return
        try:
            from rich.panel import Panel
            self._console.print(Panel(
                f"[bold green]{text}[/bold green]",
                title="[bold green]✅ Final (XTTS-ready)[/bold green]",
                border_style="green", padding=(0, 2),
            ))
        except Exception: pass
