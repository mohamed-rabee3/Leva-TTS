"""
Generate a large, UNIQUE Saudi-Arabic text corpus for OmniVoice TTS training,
seeded by the real SDC corpus and scaled with Google Gemini.

Target (configurable below):
  • 140,000 PURE Saudi-dialect sentences   (raw Arabic, no English)
  •  60,000 CODE-SWITCHING sentences        (Saudi Arabic + inline English)
  ──────────────────────────────────────────────────────────────────────
    200,000 unique sentences  →  data/saudi_200k.txt

Methodology (synthetic-data best practices)
--------------------------------------------
  1. SEED ANCHORING — the ~13k genuine unique SDC segments are injected
     directly into the pure pool AND rotated as few-shot examples in every
     Gemini prompt, so generations stay phonetically/lexically authentic
     instead of drifting toward MSA.  Seeds are *rotated* per batch to avoid
     the known few-shot vocabulary-collapse problem.
  2. PERSONA + TOPIC + LENGTH + TONE diversity — each batch samples a random
     (region, topic, sentence-type, length, tone) cell so the corpus covers
     the breadth a model needs to learn the dialect "from scratch", and to
     fight mode collapse / lexical repetition.
  3. AGGRESSIVE DEDUP — every candidate is normalized (diacritics + whitespace
     stripped) and checked against a global seen-set (seeded with SDC + all
     prior generations).  Guarantees 100% unique output and is fully resumable.
  4. QUALITY GATES — Arabic-ratio + length bounds (reused from the original
     pipeline), a dialect-marker gate that rejects pure-MSA lines, and an
     English-presence gate (pure = zero Latin, CS = must contain Latin).
  5. TTS NORMALIZATION — digits are verbalized with the SAME front-end
     normalizer used at inference, then partial diacritics from the Saudi
     lexicon are applied, so transcript == what the model is asked to speak.

Refs: arxiv 2602.15675 (LLM-to-Speech dialectal TTS pipeline),
      arxiv 2505.17390 (persona prompting for lexical diversity).

Usage
-----
  pip install google-genai
  export GEMINI_API_KEY=...            # or GOOGLE_API_KEY
  python scripts/gather_saudi_text_gemini.py            # full run
  python scripts/gather_saudi_text_gemini.py --pure 2000 --cs 1000   # smoke test

Output
------
  data/saudi_200k.txt          one sentence per line (shuffled, normalized)
  data/.saudi_200k.seen        normalized-key cache (resume support; safe to delete)
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
#                              CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════╝
import os

TARGET_PURE   = 140_000
TARGET_CS     =  60_000
OUTPUT_FILE   = "data/saudi_200k.txt"
SEEN_FILE     = "data/.saudi_200k.seen"

SDC_LOCAL     = "data/SDC.txt"
LEXICON_CSV   = "data/levantine_lexicon.csv"      # Saudi lexicon (kept filename)

MODEL         = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY       = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

SENT_PER_CALL = 40       # sentences requested per Gemini call
SEEDS_PER_CALL = 5       # real SDC examples shown as few-shot anchors per call
CONCURRENCY   = 16       # parallel Gemini requests
TEMPERATURE   = 1.15     # high → lexical diversity (we dedup hard afterwards)
MAX_RETRIES   = 4
CHECKPOINT_EVERY = 100   # flush output + seen cache every N new sentences
SEED          = 42
# ╚══════════════════════════════════════════════════════════════════════════╝

import argparse
import csv
import importlib.util
import random
import re
import sys
import threading
import time
import unicodedata
from pathlib import Path

random.seed(SEED)

try:
    from rich.console import Console
    _c = Console()
    def info(m): _c.print(f"  [cyan]{m}[/cyan]")
    def good(m): _c.print(f"  [green]✅  {m}[/green]")
    def warn(m): _c.print(f"  [yellow]⚠️   {m}[/yellow]")
except ImportError:
    def info(m): print(f"  INFO: {m}")
    def good(m): print(f"  OK: {m}")
    def warn(m): print(f"  WARN: {m}")


# ── Shared text utilities (same gates as gather_saudi_text.py) ────────────────
_AR      = re.compile(r"[ء-غف-يٱ-ۓ]")
_HAR     = re.compile(r"[ً-ٰٟ]")
_TAT     = re.compile(r"ـ")
_URL     = re.compile(r"https?://\S+")
_MENTION = re.compile(r"[@#]\w+")
_LATIN   = re.compile(r"[A-Za-z]")
_EMOJI   = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]"
)

def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = _URL.sub("", t); t = _MENTION.sub("", t)
    t = _TAT.sub("", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    # strip leading list / enumeration markers Gemini sometimes adds:
    #   "1.", "1)", "- ", "•", "S4:", "Sentence 3:", a leading quote
    t = re.sub(r'^\s*(?:S\d+\s*[:.\)]|\d+[\.\)]|[-•*])\s*', "", t)
    t = t.strip(' "\'“”')
    # strip inline word-count annotations the model occasionally injects: "(17)"
    t = re.sub(r"\(\s*\d+\s*\)", "", t)
    # collapse runaway character elongation (helloooooo → max 2)
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# lines that are clearly meta/annotation rather than content → discard outright
_BAD_LINE = re.compile(r"->|words?\.|^\s*S\d+\s*:|كلمة\)|عدد الكلمات", re.IGNORECASE)

def ok(t: str, min_chars=15, max_chars=220, min_ar_ratio=0.35) -> bool:
    bare = _HAR.sub("", t).strip()
    if not (min_chars <= len(bare) <= max_chars):
        return False
    ar    = len(_AR.findall(bare))
    total = max(len(bare.replace(" ", "")), 1)
    return ar / total >= min_ar_ratio

def strip_diac(t: str) -> str:
    return _HAR.sub("", t)

def norm_key(t: str) -> str:
    """Aggressive normalized key for dedup: no diacritics, no punctuation,
    alef/ya/ta-marbuta folded, single-spaced."""
    k = strip_diac(t).lower()
    k = re.sub(r"[^\w\s]", " ", k)          # drop punctuation
    k = k.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    k = k.replace("ى", "ي").replace("ة", "ه")
    k = re.sub(r"\s+", " ", k).strip()
    return k

# Saudi/Gulf dialect markers — at least one must appear for a line to count as
# dialectal (rejects clean MSA that Gemini may slip in for the "pure" set).
_DIALECT_MARKERS = [
    "وش", "ايش", "إيش", "اش", "ليه", "كيف", "وين", "متى", "كم", "شلون",
    "ابغى", "أبغى", "ابي", "أبي", "تبغى", "يبغى", "ودي", "بغيت",
    "الحين", "دحين", "توه", "عقب", "بعدين", "هسه",
    "مرة", "مره", "واجد", "وايد", "شوي", "زين", "كذا", "كذه", "جذي",
    "مو", "مب", "ماهو", "خلاص", "يلا", "طيب", "عاد", "عشان", "علشان",
    "هالـ", "هذي", "هاذي", "ذولا", "ذي", "ترى", "تره", "يعني", "بس",
    "والله", "يا ليت", "ياليت", "قاعد", "قاعده", "صاير", "يصير", "وش رايك",
    "ماقدر", "ما اقدر", "تكفى", "تكفون", "محد", "احد", "فيه", "ماكو", "اكو",
    "عندي", "ودك", "وشو", "كيفك", "شخبارك", "وشلونك", "هرجة", "يهرج",
]
def has_dialect(t: str) -> bool:
    bare = strip_diac(t)
    return any(m in bare for m in _DIALECT_MARKERS)


# ── Saudi number verbalizer + lexicon (shared with inference front-end) ───────
def _load_normalizer():
    path = Path(__file__).parents[1] / "leva_tts" / "text" / "normalizer.py"
    spec = importlib.util.spec_from_file_location("saudi_normalizer", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
_NORM = _load_normalizer()

def load_lexicon(path: str) -> dict:
    try:
        overrides = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                word = (row.get("word") or "").strip()
                diac = (row.get("diacritized") or "").strip()
                if word and diac:
                    overrides[word] = diac
        return overrides
    except Exception as e:
        warn(f"Lexicon not loaded: {e}")
        return {}

def apply_lexicon(text: str, lex: dict) -> str:
    if not lex:
        return text
    tokens = re.split(r"(\s+|[،؟؛,\.!?\-:]+)", text)
    return "".join(lex.get(strip_diac(tok), tok) for tok in tokens)


# ── SDC seeds (real authentic Saudi segments) ─────────────────────────────────
def load_sdc_seeds() -> list:
    p = Path(SDC_LOCAL)
    if not p.exists():
        warn(f"{SDC_LOCAL} not found — proceeding without real seeds")
        return []
    raw = p.read_text(encoding="utf-8-sig", errors="ignore")
    segs, seen = [], set()
    for line in raw.splitlines():
        for part in re.split(r"\s{2,}", line.strip()):
            for s in re.split(r"(?<=[.!؟?؛])\s+", part):
                s = clean(s)
                if not s or not ok(s, min_ar_ratio=0.5):
                    continue
                k = norm_key(s)
                if k and k not in seen:
                    seen.add(k); segs.append(s)
    good(f"SDC: {len(segs):,} real unique seed segments")
    return segs


# ── Diversity dimensions (persona / topic / form / tone) ──────────────────────
# Najdi & general weighted heavier — the OmniVoice reference voice (hoda) is Najdi.
REGIONS = (
    ["نجدي (الرياض ووسط نجد)"] * 4 +
    ["لهجة سعودية عامة مفهومة لكل السعوديين"] * 4 +
    ["حجازي (مكة وجدة والمدينة)"] * 2 +
    ["شرقاوي (الدمام والخبر والأحساء)"] * 1 +
    ["جنوبي (أبها وجازان ونجران)"] * 1 +
    ["شمالي (حائل والجوف وتبوك)"] * 1
)

TOPICS = [
    "العائلة والأهل", "الطبخ والأكل والمطاعم", "الطقس والجو", "الدوام والشغل",
    "المدرسة والجامعة والدراسة", "كرة القدم والأندية والمنتخب", "السوشال ميديا والجوال",
    "التسوق والأسواق والمولات", "السفر والسياحة والطيران", "الصحة والمستشفى والدكتور",
    "رمضان والعيد والمناسبات", "الزواج والملكة والعرس", "السيارات والقيادة والبنزين",
    "العلاقات والصداقة والمشاعر", "أعمال البيت والترتيب والتنظيف", "النصايح والحكم",
    "الشكوى والتذمر", "النكت والمزح والضحك", "ردة فعل على خبر أو موقف",
    "الفلوس والأسعار والرواتب", "القهوة والشاي والمجالس", "الأطفال وتربيتهم",
    "الجيران والحي", "الإجازة والفراغ", "المواعيد والتأخير", "الرياضة والصالة والمشي",
    "الإنترنت والألعاب والترفيه", "العمل التطوعي والخير", "الحظ والقدر والدعاء",
    "موقف محرج أو طريف صار", "تخطيط لشي جاي", "ذكرى من الماضي",
]

SENT_TYPES = [
    "جُمل خبرية عادية", "أسئلة", "طلبات وأوامر قصيرة", "تعجب وانفعال",
    "ردود قصيرة في محادثة", "سرد موقف من جملتين أو ثلاث", "نصيحة أو رأي",
    "دعوة أو تمني", "شكوى", "مزحة أو تعليق ساخر",
]

LENGTHS = [
    ("قصيرة جداً (كلمتين أو ثلاث كلمات بس)", 3),
    ("قصيرة", 4),
    ("متوسطة الطول", 3),
    ("طويلة شوي مثل تغريدة كاملة", 2),
]
_LEN_POOL = [l for l, w in LENGTHS for _ in range(w)]

TONES = [
    "محايدة", "فرحانة ومبسوطة", "زعلانة أو منفعلة", "متحمسة", "متضايقة ومتذمرة",
    "ساخرة وفيها مزح", "حنونة وعاطفية", "تعبانة ومرهقة", "مستغربة", "واثقة",
]

CS_TOPICS = [
    "الشغل والاجتماعات والإيميلات", "التقنية والبرمجة والسيرفرات", "الجامعة والمواد والمشاريع",
    "الجيم والرياضة والدايت", "الكافيهات والمطاعم والطلبات أونلاين", "السوشال ميديا والمحتوى",
    "السفر والحجوزات والمطار", "التسوق الإلكتروني والأونلاين شوبينق", "الميتنقات والـ deadlines",
    "الستارت أب والبزنس", "الستديو والمونتاج والتصوير", "الألعاب والقيمنق",
]


# ── Prompt builders ───────────────────────────────────────────────────────────
def build_pure_prompt(seeds: list, n: int) -> str:
    region   = random.choice(REGIONS)
    topic    = random.choice(TOPICS)
    stype    = random.choice(SENT_TYPES)
    length   = random.choice(_LEN_POOL)
    tone     = random.choice(TONES)
    examples = "\n".join(f"- {s}" for s in random.sample(seeds, min(SEEDS_PER_CALL, len(seeds)))) if seeds else ""
    seed_block = (
        f"\nاستوحِ الأسلوب واللهجة من هذه الأمثلة الحقيقية (لا تنسخها ولا تعيد صياغتها، بس خذ منها روح اللهجة):\n{examples}\n"
        if examples else ""
    )
    return f"""أنت كاتب محتوى سعودي محترف وخبير في اللهجة السعودية المحكية.

اكتب لي {n} جملة باللهجة السعودية {region}.
الموضوع: {topic}.
نوع الجمل: {stype}.
الطول المطلوب: {length}.
الإحساس/النبرة: {tone}.
{seed_block}
شروط صارمة:
- لهجة سعودية محكية ١٠٠٪ (مثل ما يتكلمون ويكتبون في تويتر والواتساب)، ممنوع الفصحى الرسمية.
- كل جملة مختلفة تماماً عن الثانية بالمعنى والكلمات — تنوّع كبير، لا تكرار.
- عربي فقط، بدون أي كلمة إنجليزية وبدون رموز/إيموجي وبدون هاشتاقات.
- اكتب الأرقام كلمات (مثال: "ثلاثة" مو "3").
- لا تشكّل الحروف (بدون تشكيل).
- جملة واحدة في كل سطر، بدون ترقيم وبدون شرطات وبدون علامات اقتباس.
- لا تكتب أي مقدمة أو تعليق، فقط الجمل."""


def build_cs_prompt(n: int) -> str:
    topic  = random.choice(CS_TOPICS)
    length = random.choice(_LEN_POOL)
    tone   = random.choice(TONES)
    return f"""أنت شاب/بنت سعودي يكتب في تويتر والواتساب وعادته يخلط عربي وإنجليزي (code-switching) مثل أغلب السعوديين المتعلمين.

اكتب لي {n} جملة باللهجة السعودية المحكية مع خلط طبيعي للإنجليزي داخل الجملة.
الموضوع: {topic}.
الطول: {length}.
النبرة: {tone}.

شروط صارمة:
- الأساس عربي سعودي محكي، والكلمات الإنجليزية تجي طبيعية وسط الجملة (مثل: meeting, deadline, project, gym, online, update...).
- الكلمات الإنجليزية بالحروف اللاتينية (مو معرّبة).
- كل جملة فيها كلمة إنجليزية وحدة على الأقل، والأساس يبقى عربي (٦٠٪+ عربي).
- كل جملة مختلفة تماماً، تنوّع كبير ولا تكرار.
- اكتب الأرقام العربية كلمات عربية، والإنجليزي عادي.
- بدون إيموجي وبدون هاشتاقات وبدون تشكيل.
- جملة في كل سطر، بدون ترقيم وبدون علامات اقتباس ولا أي مقدمة."""


# ── Gemini client ─────────────────────────────────────────────────────────────
def make_client():
    try:
        from google import genai
    except ImportError:
        print("[ERR] google-genai not installed. Run: pip install google-genai")
        sys.exit(1)
    if not API_KEY:
        print("[ERR] Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
        sys.exit(1)
    return genai.Client(api_key=API_KEY)


def gemini_call(client, prompt: str) -> list:
    """Return parsed candidate lines from one Gemini generation (with retries)."""
    from google.genai import types
    # gemini-3.x flash is a thinking model; disable reasoning for bulk generation
    # (otherwise it burns the token budget on thoughts and truncates the list).
    cfg = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=0.95,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=cfg
            )
            text = resp.text or ""
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        except Exception as exc:
            wait = 2 ** attempt
            if attempt == MAX_RETRIES - 1:
                return []
            time.sleep(wait)
    return []


# ── Generation loop (concurrent, resumable) ───────────────────────────────────
class Corpus:
    """Thread-safe accumulator with dedup + incremental checkpointing."""
    def __init__(self, seen_path: Path, out_path: Path, lex: dict):
        self.lock = threading.Lock()
        self.seen = set()
        self.lex = lex
        self.seen_path = seen_path
        self.out_path = out_path
        self.pure: list = []
        self.cs: list = []
        self._buf_lines: list = []      # raw (pre-shuffle) lines pending flush
        self._seen_buf: list = []
        self._since_ckpt = 0
        self.resumed = False
        # ── Resume: rebuild dedup set + phase counts from a prior run ──
        # The output file is authoritative for counts; the seen file adds the
        # pre-normalization keys (so number-bearing lines also dedup correctly).
        if seen_path.exists():
            for ln in seen_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    self.seen.add(ln.strip())
        if out_path.exists():
            for ln in out_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                self.seen.add(norm_key(ln))
                (self.cs if _LATIN.search(ln) else self.pure).append(ln)
            if self.pure or self.cs:
                self.resumed = True
                info(f"Resume: {len(self.pure):,} pure + {len(self.cs):,} CS on disk "
                     f"({len(self.seen):,} dedup keys)")
        self._out_fh = open(out_path, "a", encoding="utf-8")
        self._seen_fh = open(seen_path, "a", encoding="utf-8")

    def add_seed_sentences(self, seeds: list, normalize):
        """Write real SDC sentences straight into the pure output (count toward 140k).
        Skips any seed already present (idempotent / resume-safe)."""
        added = 0
        for s in seeds:
            k = norm_key(s)
            if k in self.seen:
                continue
            self.seen.add(k)
            line = normalize(s)
            self.seen.add(norm_key(line))
            self.pure.append(line)
            self._out_fh.write(line + "\n")
            added += 1
        self._out_fh.flush()
        return added

    def try_add(self, raw: str, is_cs: bool, normalize) -> bool:
        if _BAD_LINE.search(raw):
            return False
        cleaned = clean(raw)
        if not cleaned or not ok(cleaned):
            return False
        has_latin = bool(_LATIN.search(cleaned))
        if is_cs:
            if not has_latin:              # CS must contain English
                return False
        else:
            if has_latin:                  # pure must be Latin-free
                return False
            if not has_dialect(cleaned):   # reject pure MSA
                return False
        key = norm_key(cleaned)
        if not key:
            return False
        with self.lock:
            if key in self.seen:
                return False
            self.seen.add(key)
            line = normalize(cleaned)
            (self.cs if is_cs else self.pure).append(cleaned)
            self._out_fh.write(line + "\n")
            self._seen_fh.write(key + "\n")
            self._since_ckpt += 1
            if self._since_ckpt >= CHECKPOINT_EVERY:
                self._out_fh.flush(); self._seen_fh.flush()
                self._since_ckpt = 0
        return True

    def counts(self):
        with self.lock:
            return len(self.pure), len(self.cs)

    def close(self):
        self._out_fh.flush(); self._seen_fh.flush()
        self._out_fh.close(); self._seen_fh.close()


def run_phase(client, corpus, seeds, normalize, target, is_cs, label):
    from concurrent.futures import ThreadPoolExecutor

    def worker():
        prompt = (build_cs_prompt(SENT_PER_CALL) if is_cs
                  else build_pure_prompt(seeds, SENT_PER_CALL))
        return [(c, is_cs) for c in gemini_call(client, prompt)]

    info(f"Generating {label} → target {target:,} unique …")
    t0 = time.perf_counter()
    last_report = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = set()
        def fill():
            while len(futures) < CONCURRENCY * 2:
                futures.add(ex.submit(worker))
        fill()
        while True:
            done = next((f for f in list(futures) if f.done()), None)
            if done is None:
                time.sleep(0.05); continue
            futures.discard(done)
            for cand, cs_flag in done.result():
                corpus.try_add(cand, cs_flag, normalize)
            p, c = corpus.counts()
            cur = c if is_cs else p
            if cur - last_report >= 1000:
                last_report = cur
                rate = cur / max(time.perf_counter() - t0, 1)
                info(f"  {label}: {cur:,}/{target:,}  ({rate:.0f}/s, {len(corpus.seen):,} seen)")
            if cur >= target:
                break
            fill()
    good(f"{label} done: reached target in {(time.perf_counter()-t0)/60:.1f} min")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure", type=int, default=TARGET_PURE)
    ap.add_argument("--cs",   type=int, default=TARGET_CS)
    ap.add_argument("--out",  type=str, default=OUTPUT_FILE)
    args = ap.parse_args()

    print(f"\n{'='*64}")
    print(f"  Saudi-TTS · Gemini synthetic text  ·  model={MODEL}")
    print(f"  Target: {args.pure:,} pure + {args.cs:,} code-switching")
    print(f"  Output: {args.out}")
    print(f"{'='*64}\n")

    lex = load_lexicon(LEXICON_CSV)
    info(f"Loaded {len(lex)} Saudi lexicon rules")

    def normalize(s: str) -> str:
        """digits → Saudi words, then partial diacritics (== inference front-end)."""
        s = _NORM.normalize_entities(s)
        s = apply_lexicon(s, lex)
        return clean(s)

    seeds = load_sdc_seeds()
    random.shuffle(seeds)

    out_path  = Path(args.out)
    seen_path = Path(SEEN_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(seen_path, out_path, lex)

    # Inject real SDC seeds into the pure pool first (authentic backbone),
    # capped so they never exceed the pure target. Skipped on resume.
    seed_quota = min(len(seeds), args.pure)
    if not corpus.resumed:
        added = corpus.add_seed_sentences(seeds[:seed_quota], normalize)
        good(f"Seeded pure pool with {added:,} real SDC sentences")
    else:
        p, c = corpus.counts()
        good(f"Resuming — {p:,} pure / {c:,} CS already done")

    client = make_client()

    run_phase(client, corpus, seeds, normalize, args.pure, is_cs=False, label="PURE")
    run_phase(client, corpus, seeds, normalize, args.cs,   is_cs=True,  label="CODE-SWITCH")

    corpus.close()

    # Final shuffle of the on-disk file (order-independent for training).
    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # de-dup once more on the final normalized form, just in case
    final, seen2 = [], set()
    for l in lines:
        k = norm_key(l)
        if k and k not in seen2:
            seen2.add(k); final.append(l)
    random.shuffle(final)
    out_path.write_text("\n".join(final) + "\n", encoding="utf-8")

    p, c = corpus.counts()
    print(f"\n{'='*64}")
    print(f"  DONE → {out_path}")
    print(f"  Pure Saudi      : {p:,}")
    print(f"  Code-switching  : {c:,}")
    print(f"  Total unique    : {len(final):,}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
