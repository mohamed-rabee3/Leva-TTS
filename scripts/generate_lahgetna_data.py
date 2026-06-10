"""
Generate synthetic Saudi Arabic audio using Lahgtna-OmniVoice.

Model : oddadmix/lahgtna-omnivoice-v2
Repo  : https://github.com/Oddadmix/Lahgtna-OmniVoice

Single-speaker setup: all 50K sentences are voiced by "hoda"
(reference_audios/hoda.wav) in Najdi/Saudi Arabic (ISO 639-3: ars,
203 h in the OmniVoice training mix — the best-covered Saudi variety).

Multi-GPU support is kept (speakers split into contiguous chunks across
GPUs), but with one speaker and one GPU it simply runs a single process.

Usage
-----
  pip install git+https://github.com/Oddadmix/Lahgtna-OmniVoice.git
  python scripts/generate_lahgetna_data.py

Output
------
  data/synthetic_data/wavs/<spk_id>/<spk_id>_<idx:07d>.wav
  data/synthetic_data/metadata.csv   (wav_rel_path|transcript)
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
#                            CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
SENTENCES_FILE        = "data/saudi_50k.txt"
REFERENCES_JSON       = "reference_audios/references.json"
OUTPUT_DIR            = "data/synthetic_data"

MODEL_ID              = "oddadmix/lahgtna-omnivoice-v2"
LANGUAGE              = "ars"          # Najdi (Saudi) Arabic; acw = Hijazi
GPUS                  = "0"            # comma-separated GPU IDs to use in parallel
SENTENCES_PER_SPEAKER = 50_000         # 1 speaker (hoda) × 50 000 = 50 000 total

RESUME                = True            # skip already-generated WAVs
# ╚══════════════════════════════════════════════════════════════════════════╝

import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm


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


def merge_metadata(gpu_ids: list, out_dir: Path):
    """
    Merge per-GPU metadata files into a single metadata.csv.
    Existing entries are preserved (resume support).
    """
    meta_path = out_dir / "metadata.csv"
    # Load existing entries
    existing = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if row:
                    existing.add(row[0])

    new_rows = []
    for gpu_id in gpu_ids:
        part_path = out_dir / f"metadata_gpu{gpu_id}.csv"
        if not part_path.exists():
            continue
        with open(part_path, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) >= 2 and row[0] not in existing:
                    new_rows.append(row)
                    existing.add(row[0])
        part_path.unlink()   # clean up

    if new_rows:
        with open(meta_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="|")
            for row in new_rows:
                w.writerow(row)

    total = len(existing)
    print(f"[MERGE] metadata.csv → {total:,} total entries")
    return total


# ── Per-GPU worker (runs in its own process) ──────────────────────────────────

def generate_for_gpu(
    gpu_id:       int,
    spk_chunk:    list,   # [(global_spk_idx, ref_dict), ...]
    sentences:    list,
    per_spk:      int,
    out_dir_str:  str,
):
    """
    Load the model on cuda:{gpu_id} and generate all speakers in spk_chunk.
    Writes per-GPU metadata to metadata_gpu{gpu_id}.csv.
    """
    device  = f"cuda:{gpu_id}"
    out_dir = Path(out_dir_str)

    print(f"\n[GPU {gpu_id}] Loading {MODEL_ID} on {device} …")
    try:
        from omnivoice import OmniVoice
    except ImportError:
        print(f"[GPU {gpu_id}] omnivoice not installed. "
              "Run: pip install git+https://github.com/Oddadmix/Lahgtna-OmniVoice.git")
        return

    model = OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=torch.float16)
    SR    = model.sampling_rate
    print(f"[GPU {gpu_id}] Model ready (SR={SR} Hz) — "
          f"handling {len(spk_chunk)} speaker(s): "
          f"{[r['speaker_id'] for _, r in spk_chunk]}")

    # Per-GPU metadata file (avoids write conflicts between processes)
    part_meta_path = out_dir / f"metadata_gpu{gpu_id}.csv"
    # Load already-done entries from both the main meta and this GPU's partial meta
    done: set = set()
    main_meta = out_dir / "metadata.csv"
    for mp in [main_meta, part_meta_path]:
        if mp.exists():
            with open(mp, encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="|"):
                    if row:
                        done.add(row[0])

    fh  = open(part_meta_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh, delimiter="|")

    gpu_generated = 0
    gpu_errors    = 0
    gpu_start     = time.perf_counter()

    for spk_idx, (global_spk_idx, ref) in enumerate(spk_chunk):
        spk_id   = ref["speaker_id"]
        ref_path = ref["audio_path"]
        ref_text = ref.get("reference_text", "")
        wav_dir  = out_dir / "wavs" / spk_id
        wav_dir.mkdir(parents=True, exist_ok=True)

        # Sentences for this speaker (same assignment as single-GPU version)
        start = global_spk_idx * per_spk
        end   = start + per_spk
        sents = sentences[start:end]

        # Build todo list
        todo = []
        for i, text in enumerate(sents):
            g_idx   = start + i
            wav_name = f"{spk_id}_{g_idx:07d}.wav"
            wav_rel  = f"wavs/{spk_id}/{wav_name}"
            wav_abs  = wav_dir / wav_name
            if RESUME and wav_rel in done:
                continue
            if RESUME and wav_abs.exists():
                writer.writerow([wav_rel, text])
                fh.flush()
                done.add(wav_rel)
                continue
            todo.append((g_idx, text, wav_rel, wav_abs))

        already = len(sents) - len(todo)
        print(f"[GPU {gpu_id}] Speaker {spk_idx+1}/{len(spk_chunk)} "
              f"[{spk_id}]  {len(todo):,} to generate  ({already:,} done)")

        if not todo:
            print(f"[GPU {gpu_id}] → All done, skipping.")
            continue

        gen, err  = 0, 0
        spk_start = time.perf_counter()

        with tqdm(
            total    = len(todo),
            desc     = f"  GPU{gpu_id} {spk_id}",
            unit     = "utt",
            position = gpu_id,        # separate tqdm line per GPU
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
                    writer.writerow([wav_rel, text])
                    fh.flush()
                    done.add(wav_rel)
                    gen += 1
                except Exception as exc:
                    tqdm.write(f"[GPU {gpu_id}] WARN idx={g_idx}  {str(exc)[:80]}")
                    err += 1

                pbar.set_postfix(gen=gen, err=err)
                pbar.update(1)

        spk_elapsed = time.perf_counter() - spk_start
        rate = gen / max(spk_elapsed, 1)
        print(f"[GPU {gpu_id}] → {spk_id}: {gen:,} generated | "
              f"{err} errors | {rate:.1f} utt/s | {spk_elapsed/60:.1f} min")
        gpu_generated += gen
        gpu_errors    += err

    fh.close()
    gpu_elapsed = time.perf_counter() - gpu_start
    print(f"\n[GPU {gpu_id}] FINISHED — "
          f"{gpu_generated:,} generated | {gpu_errors} errors | "
          f"{gpu_elapsed/60:.1f} min total")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp

    # CUDA requires "spawn" start method to safely fork GPU contexts
    mp.set_start_method("spawn", force=True)

    # 1. Parse GPU list
    gpu_ids = [int(g.strip()) for g in GPUS.split(",") if g.strip()]
    n_gpus  = len(gpu_ids)
    print(f"\n{'='*60}")
    print(f"  Lahgtna-OmniVoice  ·  Multi-GPU Generation")
    print(f"  GPUs        : {gpu_ids}")
    print(f"  Model       : {MODEL_ID}")
    print(f"  Sentences   : {SENTENCES_FILE}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"{'='*60}")

    # 2. Load shared data (sentences + references)
    sentences  = load_sentences(SENTENCES_FILE)
    references = load_references(REFERENCES_JSON)
    n_spk      = len(references)

    per_spk = SENTENCES_PER_SPEAKER
    if len(sentences) < n_spk * per_spk:
        per_spk = len(sentences) // n_spk
        print(f"[WARN] Sentences short — reduced to {per_spk:,}/speaker")

    print(f"\n  {n_spk} speakers  ÷  {n_gpus} GPUs")

    # 3. Chunk speakers across GPUs
    chunks = chunk_speakers(references, n_gpus)
    for i, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        spk_names = [r["speaker_id"] for _, r in chunk]
        print(f"  GPU {gid}: {len(chunk)} speaker(s) → {spk_names}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    # 4. Launch one process per GPU (all run simultaneously)
    print(f"\n  Launching {n_gpus} parallel processes …\n")
    processes = []
    for gpu_id, spk_chunk in zip(gpu_ids, chunks):
        p = mp.Process(
            target = generate_for_gpu,
            args   = (gpu_id, spk_chunk, sentences, per_spk, OUTPUT_DIR),
            name   = f"GPU-{gpu_id}",
        )
        p.start()
        processes.append(p)
        print(f"  ▶ Started process for GPU {gpu_id}  (PID {p.pid})")

    # 5. Wait for all GPUs to finish
    print(f"\n  Waiting for all GPUs to complete …")
    for p in processes:
        p.join()
        status = "✅ OK" if p.exitcode == 0 else f"❌ exited {p.exitcode}"
        print(f"  {p.name}: {status}")

    # 6. Merge per-GPU metadata files → metadata.csv
    print()
    out_dir = Path(OUTPUT_DIR)
    total   = merge_metadata(gpu_ids, out_dir)

    wall_elapsed = time.perf_counter() - wall_start
    dur_est      = total * 5 / 3600

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total utterances : {total:,}")
    print(f"  Wall time        : {wall_elapsed/60:.1f} min")
    print(f"  ~Audio duration  : {dur_est:.1f} h  (est. 5 s avg)")
    print(f"  Metadata         : {out_dir}/metadata.csv")
    print(f"{'='*60}\n")
