"""
Dataset download helpers for Leva-TTS.

Dataset registry
----------------
English (multi-speaker, high quality)
  librispeech   openslr/librispeech_asr  train-clean-360  ~360 h

Levantine Arabic (primary — diacritized with CATT if raw)
  asc               halabi2016/arabic_speech_corpus          ~3.7 h  48 kHz
  omnilingual_apc   facebook/omnilingual-asr-corpus  apc_Arab  (North Levantine: Syrian/Lebanese)
  omnilingual_ajp   facebook/omnilingual-asr-corpus  ajp_Arab  (South Levantine: Jordanian/Palestinian)

Arabic general (dialectal, used for domain coverage)
  common_voice_ar   Geethuzzz/common_voice_17_0_arabic_New_cleaned

Note on diacritics
  Datasets marked (*) ship without tashkeel.
  The preprocess pipeline feeds undiacritized text through CATTEncoderOnly
  automatically — the model trains on diacritized transcripts in all cases.

Usage
-----
Edit DATASET and OUT_DIR at the bottom then:
    python -m leva_tts.data.download
"""
from __future__ import annotations

import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Dataset registry ──────────────────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    # ── English ──────────────────────────────────────────────────────────────
    "librispeech": {
        "hf_path":  "openslr/librispeech_asr",
        "hf_name":  "clean",
        "hf_split": "train.100",          # train-clean-100 (~100 h); further capped by MAX_HOURS_EN
        "desc":     "LibriSpeech train-clean-100 — English multi-speaker; capped to MAX_HOURS_EN h",
        "lang":     "en",
    },
    # ── South Levantine Arabic (primary, ~3.7 h, 48 kHz) ────────────────────
    "asc": {
        "hf_path":  "halabi2016/arabic_speech_corpus",
        "hf_name":  None,
        "hf_split": "train",
        "desc":     "Arabic Speech Corpus — South Levantine (Damascene/Syrian, Nawar Halabi, ~3.7 h)",
        "lang":     "ar",
        "diacritized": False,   # raw text → CATT diacritizes at preprocess time
        "audio_col":  "audio",
        "text_col":   "text",   # use 'orthographic' as fallback
    },
    # ── OmniLingual — North Levantine (Syrian, Lebanese) — ALL splits ────────
    # ajp_Arab (South Levantine/Jordanian-Palestinian) is NOT available in this
    # dataset.  apc_Arab covers North Levantine (Syrian/Lebanese) and is the
    # only confirmed Levantine config.  We download all three splits to maximise
    # coverage (train + validation + test are concatenated at preprocess time).
    "omnilingual_apc": {
        "hf_path":   "facebook/omnilingual-asr-corpus",
        "hf_name":   "apc_Arab",
        "hf_splits": ["train", "validation", "test"],   # ALL splits
        "desc":      "OmniLingual apc_Arab — North Levantine Arabic (Syrian, Lebanese) — all splits",
        "lang":      "ar",
        "diacritized": False,
        "audio_col":   "audio",
        "text_col":    "sentence",
    },
    # ── Arabic CommonVoice (dialectal, cleaned) ──────────────────────────────
    "common_voice_ar": {
        "hf_path":  "Geethuzzz/common_voice_17_0_arabic_New_cleaned",
        "hf_name":  None,
        "hf_split": "train",
        "desc":     "Common Voice 17 Arabic (cleaned) — mixed dialects",
        "lang":     "ar",
        "diacritized": False,
        "audio_col":  "audio",
        "text_col":   "sentence",
    },
}


# ── HuggingFace download ──────────────────────────────────────────────────────
def download_hf(
    hf_path: str,
    out_dir: Path,
    name:    Optional[str] = None,
    split:   str           = "train",
    trust_remote_code: bool = True,
) -> object:
    """
    Download a HuggingFace dataset and save to disk.

    Falls back to huggingface_hub.snapshot_download() if load_dataset()
    raises a decode error (e.g. gzip-compressed Parquet treated as UTF-8,
    which happens with some audio datasets like halabi2016/arabic_speech_corpus).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("pip install datasets")

    logger.info(f"Downloading HF dataset: {hf_path}  name={name}  split={split}")
    kwargs: dict = {
        "split": split,
        "trust_remote_code": trust_remote_code,
        "verification_mode": "no_checks",    # skip checksum; avoids some decode errors
    }
    if name:
        kwargs["name"] = name

    # ── Attempt 1: standard load_dataset ─────────────────────────────────────
    try:
        ds = load_dataset(hf_path, **kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_dir))
        logger.info(f"  Saved {len(ds)} samples → {out_dir}")
        return ds
    except (UnicodeDecodeError, Exception) as e:
        if "utf-8" in str(e).lower() or "codec" in str(e).lower() or isinstance(e, UnicodeDecodeError):
            logger.warning(
                f"  load_dataset decode error ({e}) → falling back to snapshot_download"
            )
        else:
            raise   # re-raise non-decode errors

    # ── Attempt 2: snapshot_download (downloads raw repo files) ──────────────
    _snapshot_download_fallback(hf_path, out_dir)
    return None   # caller must use load_from_snapshot()


def _snapshot_download_fallback(hf_path: str, out_dir: Path):
    """Download raw repo files via huggingface_hub.snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("pip install huggingface-hub>=0.23")

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  snapshot_download: {hf_path} → {out_dir}")
    snapshot_download(
        repo_id=hf_path,
        repo_type="dataset",
        local_dir=str(out_dir),
        ignore_patterns=["*.gitattributes", ".git/*"],
    )
    logger.info(f"  Snapshot saved → {out_dir}")


# ── Per-dataset helpers ───────────────────────────────────────────────────────
def download_asc(out_dir: Path):
    """
    Download ASC (halabi2016/arabic_speech_corpus).

    Tries load_dataset first; if that fails with a decode error (common with
    this dataset due to gzip-compressed Parquet metadata), falls back to
    snapshot_download.  Both outputs are handled by process_asc().
    """
    cfg     = DATASETS["asc"]
    out     = out_dir / "asc_hf"
    if out.exists() and any(out.iterdir()):
        logger.info(f"ASC already downloaded at {out} — skipping")
        return
    download_hf(cfg["hf_path"], out, split=cfg["hf_split"])


def download_omnilingual_levantine(out_dir: Path):
    """
    Download OmniLingual Levantine Arabic configs.

    Each config downloads ALL available splits (train + validation + test)
    and concatenates them so we get maximum coverage.
    Only apc_Arab (North Levantine) is available in this dataset;
    ajp_Arab (South Levantine/Jordanian-Palestinian) does NOT exist.
    """
    for key in ("omnilingual_apc",):   # ajp_Arab removed — not in dataset
        cfg    = DATASETS[key]
        out    = out_dir / key
        splits = cfg["hf_splits"]

        # Check if the existing download is complete (has all splits).
        # A stale download (e.g. test-only = 112 samples) will be re-fetched.
        existing_n = _count_saved_samples(out)
        if existing_n > 0:
            # Count how many samples each split has on the hub
            expected_n = _count_hub_samples(cfg["hf_path"], cfg["hf_name"], splits)
            if existing_n >= expected_n:
                logger.info(f"  {key}: {existing_n} samples already on disk — skipping")
                continue
            else:
                logger.info(
                    f"  {key}: existing {existing_n} samples < expected {expected_n} "                    f"(need all {splits}) — re-downloading"
                )
                import shutil; shutil.rmtree(out, ignore_errors=True)

        try:
            _download_omnilingual_all_splits(
                hf_path=cfg["hf_path"],
                config=cfg["hf_name"],
                splits=splits,
                out_dir=out,
            )
        except Exception as e:
            logger.warning(f"  {key} download failed: {e} — skipping")


def _count_saved_samples(out_dir: Path) -> int:
    """Return the number of samples in a save_to_disk Arrow dataset (0 if absent)."""
    info = out_dir / "dataset_info.json"
    if not info.exists():
        return 0
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(out_dir))
        return len(ds)
    except Exception:
        return 0


def _count_hub_samples(hf_path: str, config: str, splits: list) -> int:
    """Return the total sample count across all splits on the Hub (best-effort)."""
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(hf_path, config)
        builder.download_and_prepare()
        total = sum(
            builder.info.splits[s].num_examples
            for s in splits
            if s in builder.info.splits
        )
        return total
    except Exception:
        # Fall back to a large number so re-download is never skipped on error
        return 999_999


def _download_omnilingual_all_splits(
    hf_path: str,
    config:  str,
    splits:  list,
    out_dir: Path,
):
    """Download and concatenate all requested splits for one OmniLingual config."""
    try:
        from datasets import load_dataset, concatenate_datasets
    except ImportError:
        raise RuntimeError("pip install datasets")

    collected = []
    for split in splits:
        try:
            logger.info(f"  Downloading {config} / {split} …")
            ds = load_dataset(
                hf_path, config,
                split=split,
                trust_remote_code=True,
                verification_mode="no_checks",
            )
            logger.info(f"    {split}: {len(ds)} samples")
            collected.append(ds)
        except Exception as e:
            logger.warning(f"    {config}/{split} not available: {e}")

    if not collected:
        raise ValueError(f"No splits could be downloaded for {hf_path}/{config}")

    combined = concatenate_datasets(collected) if len(collected) > 1 else collected[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(out_dir))
    logger.info(f"  {config}: {len(combined)} total samples → {out_dir}")


def download_common_voice_ar(out_dir: Path):
    cfg = DATASETS["common_voice_ar"]
    download_hf(cfg["hf_path"], out_dir / "cv_ar_hf")


def download_librispeech(out_dir: Path):
    cfg = DATASETS["librispeech"]
    try:
        download_hf(
            cfg["hf_path"],
            out_dir / "librispeech_hf",
            name=cfg["hf_name"],
            split=cfg["hf_split"],
        )
    except Exception as e:
        logger.warning(f"LibriSpeech HF download failed ({e}); trying direct URL")
        _download_librispeech_direct(out_dir)


def _download_librispeech_direct(out_dir: Path):
    """Fallback: download LibriSpeech train-clean-360 directly from OpenSLR."""
    url  = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    dest = out_dir / "train-clean-100.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        logger.info(f"Downloading {url}  (~23 GB, may take a while) …")
        with tqdm(unit="B", unit_scale=True, desc="train-clean-100") as t:
            def hook(b=1, bsize=1, tsize=None):
                if tsize: t.total = tsize
                t.update(b * bsize - t.n)
            urlretrieve(url, dest, reporthook=hook)
    logger.info(f"Extracting {dest} …")
    with tarfile.open(dest) as tar:
        tar.extractall(out_dir)
    logger.info("LibriSpeech train-clean-100 extracted.")


def download_all(out_dir: Path):
    download_asc(out_dir)
    download_omnilingual_levantine(out_dir)
    download_common_voice_ar(out_dir)
    download_librispeech(out_dir)


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATASET  = "all"         # "asc" | "librispeech" | "common_voice_ar" |
                         # "omnilingual_apc" | "omnilingual_ajp" | "all"
OUT_DIR  = "./data/raw"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = Path(OUT_DIR)
    if DATASET == "all":
        download_all(p)
    elif DATASET == "asc":
        download_asc(p)
    elif DATASET == "librispeech":
        download_librispeech(p)
    elif DATASET == "common_voice_ar":
        download_common_voice_ar(p)
    elif DATASET in ("omnilingual_apc", "omnilingual_ajp"):
        cfg = DATASETS[DATASET]
        download_hf(cfg["hf_path"], p / DATASET,
                    name=cfg["hf_name"], split=cfg["hf_split"])
    else:
        logger.error(f"Unknown dataset: {DATASET}")
