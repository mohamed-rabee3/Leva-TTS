"""
Finish the Saudi-Arabic corpus using the Gemini **Batch API** (~50% cheaper
than synchronous calls — no latency requirement for offline bulk generation).

Resumes data/saudi_200k.txt: continues the pure pool to 140k and generates the
60k code-switching set. Reuses every filter / dedup / normalization rule from
gather_saudi_text_gemini.py so output is identical in form to the sync run.

How it works
------------
  1. Reconstruct progress + dedup state from the existing output file.
  2. Each round: build many prompts (40 sentences each) for whatever is still
     missing, submit them as inline Batch jobs (chunked), poll until done.
  3. Parse every response through the shared try_add() (dialect/length/English
     gates + global dedup + TTS normalization), append to the same file.
  4. Repeat until both targets are met (bounded by MAX_ROUNDS).

Usage
-----
  export GEMINI_API_KEY=...
  python scripts/gather_saudi_text_gemini_batch.py --selftest   # tiny end-to-end check
  python scripts/gather_saudi_text_gemini_batch.py              # finish the corpus
"""
import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# ── Reuse the sync pipeline (filters, prompts, Corpus, normalizer) ────────────
_spec = importlib.util.spec_from_file_location(
    "gbase", Path(__file__).with_name("gather_saudi_text_gemini.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

# ── Batch config ──────────────────────────────────────────────────────────────
CHUNK        = 250      # inline requests per batch job
POLL_SECS    = 30       # seconds between job-status polls
MAX_ROUNDS   = 5        # safety cap on generate→dedup rounds
SAFETY       = 1.15     # over-provision prompts by 15% to reduce extra rounds
# Assumed keep-rate (post dedup/filter) used to size each round. Pure is far
# along (heavy dedup) so it's pessimistic; CS is fresh so it's higher.
KEEP_PURE_START = 0.35
KEEP_CS_START   = 0.55
MIN_KEEP        = 0.12  # floor so we never under-provision catastrophically


def gen_config():
    return types.GenerateContentConfig(
        temperature=G.TEMPERATURE, top_p=0.95, max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def build_requests(kind: str, n_prompts: int, seeds: list) -> list:
    reqs = []
    for _ in range(n_prompts):
        prompt = (G.build_cs_prompt(G.SENT_PER_CALL) if kind == "cs"
                  else G.build_pure_prompt(seeds, G.SENT_PER_CALL))
        reqs.append(types.InlinedRequest(model=G.MODEL, contents=prompt,
                                         config=gen_config()))
    return reqs


def submit_chunked(client, kind: str, reqs: list) -> list:
    jobs = []
    for i in range(0, len(reqs), CHUNK):
        chunk = reqs[i:i + CHUNK]
        job = client.batches.create(
            model=G.MODEL, src=chunk,
            config=types.CreateBatchJobConfig(display_name=f"saudi-{kind}-{i//CHUNK}"),
        )
        jobs.append((kind, job.name))
        print(f"  [batch] submitted {kind} job {job.name}  ({len(chunk)} reqs)", flush=True)
    return jobs


_TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY")

def wait_jobs(client, jobs: list) -> list:
    pending = {name: kind for kind, name in jobs}
    finished = []
    t0 = time.time()
    while pending:
        for name in list(pending):
            try:
                j = client.batches.get(name=name)
            except Exception as e:
                print(f"  [batch] poll error {name}: {str(e)[:80]}", flush=True)
                continue
            st = str(j.state)
            if any(k in st for k in _TERMINAL):
                finished.append((pending.pop(name), j))
                print(f"  [batch] {name} → {st}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        if pending:
            time.sleep(POLL_SECS)
    return finished


def process_job(corpus, kind: str, job, normalize) -> int:
    dest = getattr(job, "dest", None)
    responses = getattr(dest, "inlined_responses", None) if dest else None
    if not responses:
        return 0
    is_cs = (kind == "cs")
    added = 0
    for r in responses:
        if getattr(r, "error", None) or not getattr(r, "response", None):
            continue
        try:
            text = r.response.text or ""
        except Exception:
            continue
        for ln in text.splitlines():
            if ln.strip() and corpus.try_add(ln, is_cs, normalize):
                added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure", type=int, default=G.TARGET_PURE)
    ap.add_argument("--cs",   type=int, default=G.TARGET_CS)
    ap.add_argument("--out",  type=str, default=G.OUTPUT_FILE)
    ap.add_argument("--selftest", action="store_true",
                    help="submit a 2-request batch to validate the flow, then exit")
    args = ap.parse_args()

    if not G.API_KEY:
        print("[ERR] Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        sys.exit(1)
    client = genai.Client(api_key=G.API_KEY)

    lex = G.load_lexicon(G.LEXICON_CSV)
    def normalize(s: str) -> str:
        return G.clean(G.apply_lexicon(G._NORM.normalize_entities(s), lex))

    seeds = G.load_sdc_seeds()

    # ── Self-test: tiny end-to-end batch (validates create/poll/read) ──
    if args.selftest:
        print("[selftest] submitting 2-request batch …", flush=True)
        reqs = build_requests("pure", 1, seeds) + build_requests("cs", 1, seeds)
        jobs = submit_chunked(client, "pure", reqs)   # kind label irrelevant here
        done = wait_jobs(client, jobs)
        for _, j in done:
            resp = getattr(j.dest, "inlined_responses", []) or []
            print(f"[selftest] job returned {len(resp)} responses")
            for r in resp[:2]:
                try:
                    print("  sample:", (r.response.text or "").splitlines()[:2])
                except Exception as e:
                    print("  (no text)", str(e)[:60])
        print("[selftest] done.")
        return

    out_path = Path(args.out)
    corpus = G.Corpus(Path(G.SEEN_FILE), out_path, lex)
    p, c = corpus.counts()
    print(f"\n{'='*64}\n  BATCH finish · model={G.MODEL}\n"
          f"  start: {p:,} pure / {c:,} cs   targets: {args.pure:,} / {args.cs:,}\n{'='*64}", flush=True)

    keep_pure, keep_cs = KEEP_PURE_START, KEEP_CS_START
    for rnd in range(1, MAX_ROUNDS + 1):
        p, c = corpus.counts()
        pure_need = max(0, args.pure - p)
        cs_need   = max(0, args.cs - c)
        if pure_need == 0 and cs_need == 0:
            print("  ✅ targets met.", flush=True)
            break

        n_pure = math.ceil(pure_need / max(keep_pure, MIN_KEEP) / G.SENT_PER_CALL * SAFETY) if pure_need else 0
        n_cs   = math.ceil(cs_need   / max(keep_cs,   MIN_KEEP) / G.SENT_PER_CALL * SAFETY) if cs_need   else 0
        print(f"\n── Round {rnd}: need {pure_need:,} pure (+{n_pure} prompts) | "
              f"{cs_need:,} cs (+{n_cs} prompts) ──", flush=True)

        jobs = []
        if n_pure:
            jobs += submit_chunked(client, "pure", build_requests("pure", n_pure, seeds))
        if n_cs:
            jobs += submit_chunked(client, "cs",   build_requests("cs",   n_cs,   seeds))

        done = wait_jobs(client, jobs)
        added_p = added_c = 0
        for kind, j in done:
            a = process_job(corpus, kind, j, normalize)
            if kind == "cs": added_c += a
            else:            added_p += a
        np_, nc_ = corpus.counts()
        print(f"  round {rnd} added: {added_p:,} pure, {added_c:,} cs  →  now {np_:,}/{args.pure:,} pure, {nc_:,}/{args.cs:,} cs", flush=True)

        # adapt keep-rate estimates from observed yield (avoids endless rounds)
        if n_pure:
            keep_pure = max(MIN_KEEP, added_p / (n_pure * G.SENT_PER_CALL))
        if n_cs:
            keep_cs = max(MIN_KEEP, added_c / (n_cs * G.SENT_PER_CALL))

        if added_p == 0 and added_c == 0:
            print("  ⚠️  no new uniques this round (diversity saturated) — stopping.", flush=True)
            break

    corpus.close()

    # Final exact dedup + shuffle (same as the sync pipeline's finish step).
    import random
    random.seed(G.SEED)
    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    final, seen2 = [], set()
    for l in lines:
        k = G.norm_key(l)
        if k and k not in seen2:
            seen2.add(k); final.append(l)
    random.shuffle(final)
    out_path.write_text("\n".join(final) + "\n", encoding="utf-8")

    pure = sum(1 for l in final if not G._LATIN.search(l))
    cs   = len(final) - pure
    print(f"\n{'='*64}\n  DONE → {out_path}\n"
          f"  Pure Saudi     : {pure:,}\n  Code-switching : {cs:,}\n"
          f"  Total unique   : {len(final):,}\n{'='*64}", flush=True)


if __name__ == "__main__":
    main()
