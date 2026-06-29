import glob
import os

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_DIR   = "data/synthetic_data"          # folder containing your .jsonl files
OUTPUT_FILE = "merged.jsonl"
# ────────────────────────────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))

if not files:
    print("No .jsonl files found in:", os.path.abspath(INPUT_DIR))
    exit(1)

total = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]   # skip blank lines
            out.writelines(lines)
            # ensure last line ends with newline before next file
            if lines and not lines[-1].endswith("\n"):
                out.write("\n")
            total += len(lines)
        print(f"  ✓ {os.path.basename(path)}  ({len(lines)} lines)")

print(f"\nDone → {OUTPUT_FILE}  ({total} total lines)")