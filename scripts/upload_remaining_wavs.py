"""
Upload remaining local WAV files to HF using upload_large_folder,
which handles chunking and retries automatically.

After upload completes, local files are deleted to free disk space.

Usage:
    python3 scripts/upload_remaining_wavs.py [--no-delete]
"""
import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_DATASET_REPO = "Rabe3/saudi-tts-synthetic"
# Upload the entire synthetic_data folder; HF will skip already-present files.
LOCAL_FOLDER    = Path("data/synthetic_data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-delete", action="store_true",
                        help="Skip deleting local WAVs after upload")
    args = parser.parse_args()

    wav_dir = LOCAL_FOLDER / "wavs" / "hoda"
    wav_count = sum(1 for _ in wav_dir.glob("*.wav"))
    if wav_count == 0:
        print(f"No WAV files found in {wav_dir}")
        sys.exit(0)

    print(f"Uploading {wav_count} WAV files → {HF_DATASET_REPO}")
    print("(upload_large_folder handles chunking and skips already-uploaded files)")

    api = HfApi()
    api.upload_large_folder(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        folder_path=str(LOCAL_FOLDER),
        num_workers=4,
    )

    print("\nUpload complete.")

    if not args.no_delete:
        print("Deleting local WAV files…")
        deleted = 0
        for wav in wav_dir.glob("*.wav"):
            wav.unlink()
            deleted += 1
        print(f"Deleted {deleted} local files.")
    else:
        print("--no-delete: local files kept.")


if __name__ == "__main__":
    main()
