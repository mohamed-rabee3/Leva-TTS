"""
Generate synthetic Saudi Arabic audio using Lahgtna-OmniVoice.

Model : oddadmix/lahgtna-omnivoice-v2
Repo  : https://github.com/Oddadmix/Lahgtna-OmniVoice

Single-speaker setup: all 100K sentences are voiced by "hoda"
(reference_audios/hoda.wav) in Najdi/Saudi Arabic (ISO 639-3: ars,
203 h in the OmniVoice training mix — the best-covered Saudi variety).

WAVs are uploaded to HF_DATASET_REPO in batches of UPLOAD_BATCH_SIZE and
deleted locally immediately after upload to keep disk usage near zero.
metadata.csv is kept locally (tiny) and uploaded at the end.

Usage
-----
  python scripts/generate_lahgetna_data.py

Output (on HuggingFace)
-----------------------
  HF_DATASET_REPO/wavs/<spk_id>/<spk_id>_<idx:07d>.wav
  HF_DATASET_REPO/metadata.csv   (wav_rel_path|transcript)
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
#                            CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
SENTENCES_FILE        = "data/saudi_200k.txt"
REFERENCES_JSON       = "reference_audios/references.json"
OUTPUT_DIR            = "data/synthetic_data"

MODEL_ID              = "Rabe3/lahgtna-omnivoice-v2"   # mirror of oddadmix/lahgtna-omnivoice-v2
LANGUAGE              = "ars"          # Najdi (Saudi) Arabic; acw = Hijazi
GPUS                  = "0"            # comma-separated GPU IDs to use in parallel
WORKERS_PER_GPU       = 10             # parallel OmniVoice instances per GPU (throughput ↑)
SENTENCES_PER_SPEAKER = 200_000        # 1 speaker (hoda) × 200 000 = 200 000 total

RESUME                = True            # skip already-generated WAVs

# HuggingFace upload — WAVs are KEPT locally (never deleted) and ALSO pushed to
# HF every UPLOAD_BATCH_SIZE generated WAVs.
HF_DATASET_REPO       = "Rabe3/saudi-tts-synthetic-200k"   # dataset repo (created if absent)
UPLOAD_BATCH_SIZE     = 10_000          # push to HF every N new WAVs (kept locally regardless)

# Qwen3-TTS training format: one JSONL line per utterance:
#   {"audio": "wavs/<spk>/<file>.wav", "text": "<transcript>", "ref_audio": "ref/<spk>.wav"}
# Same ref_audio for all samples (single speaker) — recommended by Qwen3-TTS.
JSONL_NAME            = "train_raw.jsonl"
REF_DIR               = "ref"           # reference audio copied here (local + repo)
# ╚══════════════════════════════════════════════════════════════════════════╝

import json
import shutil
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm
from huggingface_hub import HfApi, CommitOperationAdd


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sentences(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"[ERR] {path} not found. Run: python scripts/gather_saudi_text.py")
        sys.exit(1)
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return lines


def load_references(path: str) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = []
    for ref in data:
        if Path(ref["audio_path"]).exists():
            valid.append(ref)
        else:
            print(f"[WARN] Audio not found: {ref['audio_path']}  — skipping {ref['speaker_id']}")
    if not valid:
        print(f"[ERR] No valid speakers in {path}.")
        sys.exit(1)
    return valid


def chunk_speakers(references: list, n_gpus: int) -> list:
    """
    Split speakers into n_gpus contiguous chunks.
    e.g. 10 speakers / 4 GPUs → [3, 3, 2, 2]
    """
    n_spk  = len(references)
    base   = n_spk // n_gpus
    extra  = n_spk % n_gpus
    chunks, start = [], 0
    for i in range(n_gpus):
        size = base + (1 if i < extra else 0)
        # Each element: (global_speaker_index, ref_dict)
        chunks.append([(start + j, references[start + j]) for j in range(size)])
        start += size
    return chunks


def merge_metadata(out_dir: Path):
    """
    Merge ALL per-worker JSONL parts (metadata_*.jsonl, incl. legacy metadata_gpu*)
    into a single Qwen3-TTS train_raw.jsonl. Existing entries preserved; parts kept.
    """
    meta_path = out_dir / JSONL_NAME
    existing = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line)["audio"])
                except Exception:
                    continue

    new_lines = []
    for part_path in sorted(out_dir.glob("metadata_*.jsonl")):
        with open(part_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    audio = json.loads(line)["audio"]
                except Exception:
                    continue
                if audio not in existing:
                    new_lines.append(line)
                    existing.add(audio)

    if new_lines:
        with open(meta_path, "a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

    total = len(existing)
    print(f"[MERGE] {JSONL_NAME} → {total:,} total entries")
    return total


# ── HuggingFace upload helper ─────────────────────────────────────────────────

def hf_ensure_repo(api: HfApi, repo_id: str):
    # exist_ok=True is race-safe when multiple workers call this concurrently.
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception as exc:
        print(f"[HF] ensure_repo note: {str(exc)[:100]}")


def hf_upload_reference(api: HfApi, repo_id: str, ref_abs: Path, repo_path: str):
    """Upload the speaker reference audio once so ref_audio paths resolve on HF."""
    try:
        api.upload_file(
            path_or_fileobj=str(ref_abs),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add reference audio {repo_path}",
        )
        print(f"[HF] Uploaded reference → {repo_id}/{repo_path}")
    except Exception as exc:
        print(f"[HF] WARN reference upload failed: {str(exc)[:100]}")


def hf_upload_metadata(api: HfApi, repo_id: str, part_path: Path, gpu_id: int):
    """Push this GPU's partial Qwen3-TTS JSONL to HF (intermediate visibility)."""
    if not part_path.exists():
        return
    try:
        api.upload_file(
            path_or_fileobj=str(part_path),
            path_in_repo=part_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update metadata {part_path.name}",
        )
    except Exception as exc:
        tqdm.write(f"[GPU {gpu_id}] WARN metadata upload failed: {str(exc)[:80]}")


def hf_upload(batch: list, api: HfApi, repo_id: str, gpu_id: int):
    """Upload a list of (wav_abs, path_in_repo) tuples to HF. Files are KEPT locally."""
    if not batch:
        return
    operations = [
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(wav_abs))
        for wav_abs, repo_path in batch
    ]
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {len(batch)} WAVs (GPU {gpu_id})",
        )
        tqdm.write(f"[GPU {gpu_id}] ↑ Uploaded {len(batch)} WAVs → {repo_id} (kept locally)")
    except Exception as exc:
        tqdm.write(f"[GPU {gpu_id}] WARN upload failed ({str(exc)[:100]}) — WAVs kept locally, will retry next batch")


# ── Per-GPU worker (runs in its own process) ──────────────────────────────────

def generate_worker(
    worker_id:    int,
    gpu_id:       int,
    shard:        list,   # global sentence indices this worker handles
    sentences:    list,
    ref:          dict,   # single-speaker reference dict
    out_dir_str:  str,
):
    """
    One OmniVoice instance on cuda:{gpu_id} generating its sentence shard.
    Multiple workers may share a GPU. Writes metadata_w{worker_id}.jsonl
    (Qwen3-TTS format). WAVs are kept locally and pushed to HF every
    UPLOAD_BATCH_SIZE; a final folder sync in main guarantees completeness.
    """
    device  = f"cuda:{gpu_id}"
    out_dir = Path(out_dir_str)
    tag     = f"W{worker_id}@gpu{gpu_id}"

    print(f"\n[{tag}] Loading {MODEL_ID} on {device} … ({len(shard):,} sentences)")
    try:
        from omnivoice import OmniVoice
    except ImportError:
        print(f"[{tag}] omnivoice not installed. "
              "Run: pip install git+https://github.com/Oddadmix/Lahgtna-OmniVoice.git")
        return

    model = OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=torch.float16)
    SR    = model.sampling_rate
    print(f"[{tag}] Model ready (SR={SR} Hz)")

    api = HfApi()
    hf_ensure_repo(api, HF_DATASET_REPO)   # idempotent (exist_ok)
    upload_buffer: list = []

    spk_id   = ref["speaker_id"]
    ref_path = ref["audio_path"]
    ref_text = ref.get("reference_text", "")
    wav_dir  = out_dir / "wavs" / spk_id
    wav_dir.mkdir(parents=True, exist_ok=True)

    ref_rel   = f"{REF_DIR}/{spk_id}.wav"
    ref_local = out_dir / REF_DIR / f"{spk_id}.wav"
    if worker_id == 0:   # only one worker manages the shared reference
        ref_local.parent.mkdir(parents=True, exist_ok=True)
        if not ref_local.exists() and Path(ref_path).exists():
            shutil.copyfile(ref_path, ref_local)
            hf_upload_reference(api, HF_DATASET_REPO, ref_local, ref_rel)

    part_meta_path = out_dir / f"metadata_w{worker_id}.jsonl"
    # Resume: skip anything already in the merged meta OR any worker's part file.
    done: set = set()
    for mp in [out_dir / JSONL_NAME, *out_dir.glob("metadata_*.jsonl")]:
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["audio"])
                    except Exception:
                        continue

    fh = open(part_meta_path, "a", encoding="utf-8")

    def write_meta(audio_rel: str, transcript: str):
        fh.write(json.dumps(
            {"audio": audio_rel, "text": transcript, "ref_audio": ref_rel},
            ensure_ascii=False) + "\n")
        fh.flush()

    # Build todo from this worker's shard
    todo = []
    for g_idx in shard:
        text     = sentences[g_idx]
        wav_name = f"{spk_id}_{g_idx:07d}.wav"
        wav_rel  = f"wavs/{spk_id}/{wav_name}"
        wav_abs  = wav_dir / wav_name
        if RESUME and wav_rel in done:
            continue
        if RESUME and wav_abs.exists():
            write_meta(wav_rel, text)
            done.add(wav_rel)
            continue
        todo.append((g_idx, text, wav_rel, wav_abs))

    already = len(shard) - len(todo)
    print(f"[{tag}] {len(todo):,} to generate  ({already:,} already done)")

    gen, err = 0, 0
    t0 = time.perf_counter()
    with tqdm(
        total    = len(todo),
        desc     = f"  {tag}",
        unit     = "utt",
        position = worker_id,
        leave    = True,
        dynamic_ncols = True,
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for g_idx, text, wav_rel, wav_abs in todo:
            try:
                audio = model.generate(
                    text              = text,
                    ref_audio         = ref_path,
                    ref_text          = ref_text if ref_text else None,
                    language          = LANGUAGE,
                    repetition_penalty = 1.2,
                    top_p             = 0.7,
                    temperature       = 0.7,
                )
                sf.write(str(wav_abs), audio[0], SR)
                write_meta(wav_rel, text)
                done.add(wav_rel)
                upload_buffer.append((wav_abs, wav_rel))
                gen += 1

                if len(upload_buffer) >= UPLOAD_BATCH_SIZE:
                    hf_upload(upload_buffer, api, HF_DATASET_REPO, worker_id)
                    hf_upload_metadata(api, HF_DATASET_REPO, part_meta_path, worker_id)
                    upload_buffer.clear()
            except Exception as exc:
                tqdm.write(f"[{tag}] WARN idx={g_idx}  {str(exc)[:80]}")
                err += 1

            pbar.set_postfix(gen=gen, err=err)
            pbar.update(1)

    if upload_buffer:
        hf_upload(upload_buffer, api, HF_DATASET_REPO, worker_id)
        upload_buffer.clear()
    hf_upload_metadata(api, HF_DATASET_REPO, part_meta_path, worker_id)

    fh.close()
    elapsed = time.perf_counter() - t0
    rate = gen / max(elapsed, 1)
    print(f"\n[{tag}] FINISHED — {gen:,} generated | {err} errors | "
          f"{rate:.2f} utt/s | {elapsed/60:.1f} min")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp

    # CUDA requires "spawn" start method to safely fork GPU contexts
    mp.set_start_method("spawn", force=True)

    # 1. Parse GPU list
    gpu_ids = [int(g.strip()) for g in GPUS.split(",") if g.strip()]
    n_gpus  = len(gpu_ids)
    print(f"\n{'='*60}")
    print(f"  Lahgtna-OmniVoice  ·  Multi-Worker Generation")
    print(f"  GPUs        : {gpu_ids}  × {WORKERS_PER_GPU} workers each")
    print(f"  Model       : {MODEL_ID}")
    print(f"  Sentences   : {SENTENCES_FILE}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"{'='*60}")

    # 2. Load shared data (sentences + references)
    sentences  = load_sentences(SENTENCES_FILE)
    references = load_references(REFERENCES_JSON)
    if len(references) > 1:
        print(f"[WARN] {len(references)} speakers in references.json — this multi-worker "
              f"build is single-speaker; using only '{references[0]['speaker_id']}'.")
    ref = references[0]

    N = min(len(sentences), SENTENCES_PER_SPEAKER)
    total_workers = n_gpus * WORKERS_PER_GPU

    # 3. Shard sentence indices across all workers (strided → balanced resume)
    shards = [list(range(k, N, total_workers)) for k in range(total_workers)]
    print(f"\n  {N:,} sentences  ·  speaker '{ref['speaker_id']}'  ·  "
          f"{total_workers} workers ({n_gpus} GPU × {WORKERS_PER_GPU})")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    # 4. Launch one process per worker (workers_per_gpu share each GPU)
    print(f"\n  Launching {total_workers} worker processes …\n")
    processes = []
    for k in range(total_workers):
        gpu_id = gpu_ids[k // WORKERS_PER_GPU]
        p = mp.Process(
            target = generate_worker,
            args   = (k, gpu_id, shards[k], sentences, ref, OUTPUT_DIR),
            name   = f"W{k}",
        )
        p.start()
        processes.append(p)
        print(f"  ▶ Started worker {k} on GPU {gpu_id}  (PID {p.pid}, {len(shards[k]):,} sents)")

    # 5. Wait for all workers to finish
    print(f"\n  Waiting for all {total_workers} workers to complete …")
    for p in processes:
        p.join()
        status = "✅ OK" if p.exitcode == 0 else f"❌ exited {p.exitcode}"
        print(f"  {p.name}: {status}")

    # 6. Merge per-worker JSONL files → train_raw.jsonl (Qwen3-TTS format)
    print()
    out_dir = Path(OUTPUT_DIR)
    total   = merge_metadata(out_dir)

    # 6b. Final folder sync — guarantees every local WAV is on HF even if some
    # mid-run batch commits raced/failed across workers.
    try:
        api = HfApi()
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(repo_id=HF_DATASET_REPO, repo_type="dataset",
                                    folder_path=str(out_dir))
        else:
            api.upload_folder(repo_id=HF_DATASET_REPO, repo_type="dataset",
                              folder_path=str(out_dir))
        print(f"  Final folder sync → {HF_DATASET_REPO}")
    except Exception as exc:
        print(f"  WARN final folder sync failed: {str(exc)[:120]}")

    wall_elapsed = time.perf_counter() - wall_start
    dur_est      = total * 5 / 3600

    # Upload final train_raw.jsonl to HF
    meta_path = out_dir / JSONL_NAME
    if meta_path.exists():
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=str(meta_path),
                path_in_repo=JSONL_NAME,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Final {JSONL_NAME}",
            )
            print(f"  Metadata uploaded → {HF_DATASET_REPO}/{JSONL_NAME}")
        except Exception as exc:
            print(f"  WARN metadata upload failed: {exc}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total utterances : {total:,}")
    print(f"  Wall time        : {wall_elapsed/60:.1f} min")
    print(f"  ~Audio duration  : {dur_est:.1f} h  (est. 5 s avg)")
    print(f"  Metadata (Qwen3) : {out_dir}/{JSONL_NAME}")
    print(f"  Local WAVs       : {out_dir}/wavs/  (kept, not deleted)")
    print(f"  HF dataset       : https://huggingface.co/datasets/{HF_DATASET_REPO}")
    print(f"{'='*60}\n")
