"""
Download ALL synthetic WAV files from Rabe3/saudi-tts-synthetic back to
data/synthetic_data/ (preserving the wavs/hoda/ and wavs/hoda/{10k,20k,30k,40k}/
layout that matches metadata.csv).

Usage:
    python3 scripts/download_hf_wavs.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download

HF_DATASET_REPO = "Rabe3/saudi-tts-synthetic"
LOCAL_DIR       = Path("data/synthetic_data")


def main():
    existing = sum(1 for _ in (LOCAL_DIR / "wavs").rglob("*.wav"))
    if existing >= 49_386:
        print(f"Already have {existing} WAVs in {LOCAL_DIR}/wavs — nothing to do.")
        return

    print(f"Have {existing} WAVs locally. Downloading all from {HF_DATASET_REPO} …")
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
        allow_patterns=["wavs/**", "metadata.csv"],
    )

    total = sum(1 for _ in (LOCAL_DIR / "wavs").rglob("*.wav"))
    print(f"Done. {total} WAV files now in {LOCAL_DIR}/wavs/")


if __name__ == "__main__":
    main()
