"""
Reorganize local WAV files into 10k-file subdirectories to stay under
HuggingFace's 10,000 files-per-directory limit, update metadata.csv,
then upload everything.

Bucket layout (files already on HF in wavs/hoda/ are untouched):
  wavs/hoda/10k/  → hoda_0010000 – hoda_0019999
  wavs/hoda/20k/  → hoda_0020000 – hoda_0029999
  wavs/hoda/30k/  → hoda_0030000 – hoda_0039999
  wavs/hoda/40k/  → hoda_0040000 – hoda_0049999

Usage:
    python3 scripts/reorganize_and_upload.py [--no-delete]
"""
import argparse
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_DATASET_REPO = "Rabe3/saudi-tts-synthetic"
SYNTH_DIR       = Path("data/synthetic_data")
WAV_ROOT        = SYNTH_DIR / "wavs" / "hoda"
METADATA        = SYNTH_DIR / "metadata.csv"


def bucket_for(stem: str) -> str:
    """Return the subdirectory name for a given hoda_XXXXXXX stem."""
    idx = int(stem.split("_")[1])
    if idx < 10_000:
        return ""           # already on HF in wavs/hoda/
    elif idx < 20_000:
        return "10k"
    elif idx < 30_000:
        return "20k"
    elif idx < 40_000:
        return "30k"
    else:
        return "40k"


def reorganize_local_files():
    wavs = sorted(WAV_ROOT.glob("*.wav"))
    if not wavs:
        print("No flat WAV files to reorganize.")
        return

    print(f"Reorganizing {len(wavs)} flat WAV files into subdirectories…")
    for wav in wavs:
        bucket = bucket_for(wav.stem)
        if not bucket:
            continue   # shouldn't be here, but skip
        dest_dir = WAV_ROOT / bucket
        dest_dir.mkdir(exist_ok=True)
        wav.rename(dest_dir / wav.name)

    print("Reorganization done.")


def update_metadata():
    if not METADATA.exists():
        print(f"ERROR: {METADATA} not found")
        sys.exit(1)

    lines = METADATA.read_text(encoding="utf-8").splitlines()
    updated = []
    changed = 0
    for line in lines:
        if not line.strip():
            updated.append(line)
            continue
        path_part, _, text_part = line.partition("|")
        # path_part looks like: wavs/hoda/hoda_0012345.wav
        fname = Path(path_part).name          # hoda_0012345.wav
        stem  = Path(fname).stem              # hoda_0012345
        bucket = bucket_for(stem)
        if bucket and f"/{bucket}/" not in path_part:
            new_path = f"wavs/hoda/{bucket}/{fname}"
            updated.append(f"{new_path}|{text_part}")
            changed += 1
        else:
            updated.append(line)

    METADATA.write_text("\n".join(updated), encoding="utf-8")
    print(f"metadata.csv updated: {changed} paths rewritten.")


def upload():
    print(f"\nUploading {SYNTH_DIR} → {HF_DATASET_REPO}")
    print("(upload_large_folder skips already-present files)")
    api = HfApi()
    api.upload_large_folder(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        folder_path=str(SYNTH_DIR),
        num_workers=4,
    )
    print("Upload complete.")


def delete_local_wavs():
    total = 0
    for bucket in ("10k", "20k", "30k", "40k"):
        d = WAV_ROOT / bucket
        if d.exists():
            wavs = list(d.glob("*.wav"))
            for w in wavs:
                w.unlink()
            total += len(wavs)
    print(f"Deleted {total} local WAV files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-delete", action="store_true",
                        help="Keep local WAVs after upload")
    args = parser.parse_args()

    reorganize_local_files()
    update_metadata()
    upload()

    if not args.no_delete:
        delete_local_wavs()
    else:
        print("--no-delete: local files kept.")


if __name__ == "__main__":
    main()
