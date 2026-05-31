"""
Levantine lexicon helper – thin wrapper around the CSV file.

The source of truth for all word→diacritic mappings is:
    data/levantine_lexicon.csv

This module provides a backward-compatible apply_lexicon() function that
loads the CSV and applies it.  New code should use CATTDiacritizer directly
(which loads the CSV internally) or TextProcessor.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

# Default CSV path (relative to project root)
_DEFAULT_CSV = Path(__file__).parents[2] / "data" / "levantine_lexicon.csv"

_cache: Dict[str, Tuple[str, str, str]] | None = None


def get_overrides(csv_path: str | Path | None = None) -> Dict[str, Tuple[str, str, str]]:
    """Load and cache the Levantine override dictionary from CSV."""
    global _cache
    if _cache is not None:
        return _cache
    from leva_tts.text.diacritizer import load_levantine_overrides
    _cache = load_levantine_overrides(csv_path or _DEFAULT_CSV)
    return _cache


def apply_lexicon(text: str, csv_path: str | Path | None = None) -> str:
    """Backward-compatible function: apply Levantine CSV overrides to *text*."""
    import re
    overrides = get_overrides(csv_path)
    tokens = re.split(r"(\s+)", text)
    result = []
    for tok in tokens:
        key = re.sub(r"[ً-ٟـ]", "", tok.strip())  # strip diacritics for lookup
        if key in overrides:
            result.append(overrides[key][0])   # diacritized replacement
        else:
            result.append(tok)
    return "".join(result)
