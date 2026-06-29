"""
Gather 100,000 Saudi Arabic sentences for OmniVoice synthesis.

Sources
-------
  1. Saudi Dialect Corpus (SDC) — TaghreedT/SDC (~210K words, mixed Saudi
       dialects collected from Twitter/Facebook):
       https://github.com/TaghreedT/SDC
     Long lines hold several tweets; they are segmented, cleaned, deduped,
     and short real segments are paired to extend the pure pool.

  2. Synthetic code-switching sentences (Saudi Arabic + English)
       Template library covering work/tech, daily life, education

  3. Number sentences (NEW) — digits (1, 354, 5645), prices, percentages,
     times and dates, verbalized into Saudi Arabic words with the same
     normalizer used by the inference front-end, so the model learns to
     speak numbers exactly the way the runtime text processor renders them.

Distribution: ~60% pure Saudi (SDC), ~28% code-switching, ~12% numbers

Output
------
  data/saudi_100k.txt  — one sentence per line, partial diacritics from the
                         Saudi lexicon CSV (same overrides as inference)

Usage
-----
  python scripts/gather_saudi_text.py

Requirements: urllib (built-in), rich (optional)
"""

# ╔═══════════════════════════════════════════════════════════════════╗
#                          CONFIGURATION
# ╚═══════════════════════════════════════════════════════════════════╝
TARGET       = 100_000
PURE_RATIO   = 0.60          # fraction from SDC corpus
CS_RATIO     = 0.28          # code-switching templates
NUM_RATIO    = 0.12          # number-bearing sentences
OUTPUT_FILE  = "data/saudi_100k.txt"
SDC_LOCAL    = "data/SDC.txt"
SDC_URL      = "https://raw.githubusercontent.com/TaghreedT/SDC/main/SDC.txt"
LEXICON_CSV  = "data/levantine_lexicon.csv"   # Saudi lexicon (kept filename)
SEED         = 42
# ╚═══════════════════════════════════════════════════════════════════╝

import csv
import importlib.util
import random
import re
import unicodedata
from pathlib import Path
from urllib.request import urlopen

random.seed(SEED)

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    def log(m):  console.print(f"  [dim]{m}[/dim]")
    def info(m): console.print(f"  [cyan]{m}[/cyan]")
    def good(m): console.print(f"  [green]✅  {m}[/green]")
    def warn(m): console.print(f"  [yellow]⚠️   {m}[/yellow]")
except ImportError:
    def log(m): print(f"  {m}")
    def info(m): print(f"  INFO: {m}")
    def good(m): print(f"  OK: {m}")
    def warn(m): print(f"  WARN: {m}")


# ── Saudi number verbalizer (shared with the inference front-end) ─────────────
def _load_normalizer():
    path = Path(__file__).parents[1] / "leva_tts" / "text" / "normalizer.py"
    spec = importlib.util.spec_from_file_location("saudi_normalizer", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_NORM = _load_normalizer()


# ── Text utilities ────────────────────────────────────────────────────────────
_AR      = re.compile(r"[ء-غف-يٱ-ۓ]")
_HAR     = re.compile(r"[ً-ٰٟ]")
_TAT     = re.compile(r"ـ")
_URL     = re.compile(r"https?://\S+")
_MENTION = re.compile(r"[@#]\w+")

def clean(t):
    t = unicodedata.normalize("NFC", t)
    t = _URL.sub("", t); t = _MENTION.sub("", t)
    t = _TAT.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def ok(t, min_chars=20, max_chars=220, min_ar_ratio=0.35):
    bare = _HAR.sub("", t).strip()
    if not (min_chars <= len(bare) <= max_chars): return False
    ar    = len(_AR.findall(bare))
    total = max(len(bare.replace(" ", "")), 1)
    return ar / total >= min_ar_ratio

def strip_diac(t):
    return _HAR.sub("", t)


# ── SDC corpus ────────────────────────────────────────────────────────────────
def load_sdc():
    """Load SDC.txt (local file preferred, else download) → unique segments."""
    local = Path(SDC_LOCAL)
    if local.exists():
        info(f"Using local SDC corpus: {SDC_LOCAL}")
        raw = local.read_text(encoding="utf-8-sig", errors="ignore")
    else:
        info(f"Downloading SDC corpus …")
        with urlopen(SDC_URL, timeout=60) as r:
            raw = r.read().decode("utf-8-sig", errors="ignore")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(raw, encoding="utf-8")

    segs, seen = [], set()
    for line in raw.splitlines():
        # Lines hold several tweets: split on 2+ spaces, then on punctuation
        for part in re.split(r"\s{2,}", line.strip()):
            for s in re.split(r"(?<=[.!؟?؛])\s+", part):
                s = clean(s)
                if not s:
                    continue
                # Stricter Arabic ratio for the real corpus (quality gate)
                if not ok(s, min_ar_ratio=0.5):
                    continue
                k = strip_diac(s)
                if k not in seen:
                    seen.add(k); segs.append(s)
    good(f"SDC: {len(segs):,} clean unique segments")
    return segs, seen


def extend_with_pairs(segs: list, seen: set, need: int) -> list:
    """
    Extend the pure pool by joining two short real SDC segments with a comma.
    Keeps the text 100% authentic Saudi while adding prosodic variety.
    """
    extra = []
    shorts = [s for s in segs if len(s) <= 90]
    tries = 0
    while len(extra) < need and tries < need * 20:
        tries += 1
        a, b = random.choice(shorts), random.choice(shorts)
        if a == b:
            continue
        s = f"{a}، {b}"
        if not ok(s, min_ar_ratio=0.5):
            continue
        k = strip_diac(s)
        if k in seen:
            continue
        seen.add(k); extra.append(s)
    good(f"Paired-segment top-up: {len(extra):,} sentences")
    return extra


# ── Code-switching template library (Saudi Arabic + English) ──────────────────
EN_TECH   = ["the project","the system","the model","the code","the server",
             "the database","the API","the app","the bug","the feature",
             "the report","the meeting","the deadline","the pipeline",
             "the config","the output","the test","the deployment",
             "the update","the review","the script","the data","the file",
             "the team","the PR","the branch","the build","the error",
             "the log","the module","the function","the service","the cluster",
             "the dataset","the training","the experiment","the baseline",
             "the evaluation","the benchmark","the architecture"]
EN_DOMAIN = ["the project","the presentation","the assignment","the exam",
             "the interview","the course","the lecture","the seminar",
             "the workshop","the training","the task","the schedule",
             "the plan","the strategy","the design","the prototype",
             "the proposal","the budget","the timeline","the scope"]
EN_BARE   = ["AI","ML","cloud","backend","frontend","DevOps","Docker",
             "Python","GPU","training","inference","fine-tuning","LLM",
             "TTS","ASR","NLP","API","CI/CD","Kubernetes","PyTorch",
             "Linux","bash","model","dataset","pipeline","notebook"]
EN_ACT    = ["working on","fixing","reviewing","testing","debugging",
             "updating","checking","training","deploying","running",
             "preparing","downloading","uploading","optimizing","building",
             "reading","writing","refactoring","analyzing","evaluating"]
EN_ADJ    = ["important","complex","ready","done","broken","slow","fast",
             "critical","amazing","useful","difficult","simple","heavy",
             "interesting","clean","messy","confusing","clear","stable"]
EN_TIME   = ["today","tomorrow","tonight","this week","next week","soon","now",
             "this morning","later","by Friday","in an hour","this weekend",
             "by end of day","before lunch","after the meeting"]
EN_PLACE  = ["online","on GitHub","in production","locally","on the server",
             "in the cloud","on Slack","in the repo","in the logs","remotely"]

AR_ADJ    = ["مرة زين","مرة مهم","مو شغال","مو جاهز","مرة ثقيل",
             "يبيله وقت","جاهز","خلص","مو واضح","مهم مرة","مرة صعب",
             "ما ضبط","ضابط","مرة خايس","لازم نصلحه"]
AR_TIME   = ["الحين","بكرة","اليوم","بعدين","قريب","هالأسبوع",
             "الأسبوع الجاي","عقب بكرة","بعد شوي","هالشهر","أمس"]

CS_TEMPLATES = [
    # Work + Tech
    "الحين قاعد {act} {tech}",
    "أبغى {act} {tech} قبل {time}",
    "والله {tech} {ar_adj}",
    "مادري ليه {tech} مو شغال",
    "لازم نراجع {tech} قبل بكرة",
    "بنبدأ {act} {tech} {time}",
    "فيه مشكلة بـ{tech}، لازم نصلحها",
    "يلا نكمل {act} {tech} مع بعض",
    "قعدت {act} {tech} لوقت متأخر",
    "نبغى نخلص {tech} قبل {time}",
    "وش تبغى من {tech}؟ قل لي",
    "أنا {act} {tech} من الصبح",
    "كنت {act} {tech} أمس",
    "وش صار على {tech}؟ خلص؟",
    "بكرة فيه meeting عن {tech}",
    "{tech} {ar_adj}، بس لازم نكمل",
    "وش رايك بـ{tech}؟",
    "قاعدين نشتغل على {tech} هالأسبوع",
    "قلت لك عن {tech} قبل؟ هو {en_adj}",
    "جا {tech} {time}، أخيراً!",
    "{tech} جاهز {place}",
    "الـ{bare} مو شغال {place}، تدري ليه؟",
    "الحين قاعدين نسوي review على {tech}",
    "فيه issue بالـ{bare}، لازم نشوف",
    "خلصنا الـ{bare} أمس",
    "بكرة بنسوي demo لـ{tech}",
    "المشكلة كلها من {tech}",
    "نسيت أعدل {tech}، ذي المشكلة",
    "الـ{bare} crashed {time}",
    "قاعدين نسوي debug للـ{bare} الحين",
    "كتبت script لـ{tech}، وش رايك؟",
    "الـ{bare} شغال fine {place}",
    "جربت {tech}؟ وش رايك؟",
    "فيه update جديد لـ{tech}",
    "الـ{bare} version الجديدة {en_adj} مرة",
    "ليه الـ{bare} مو شغال {place}؟",
    "وش رايك بالـ{bare}؟ جربته؟",
    "كيفك؟ وش فيه جديد مع {tech}؟",
    "ياخي {tech} {en_adj} مرة! وش تبغى بعد؟",
    "والله! {tech} خلص أخيراً!",
    "وش هذا؟ الـ{bare} سوى update؟",
    "تعرف كيف تحل مشكلة الـ{bare}؟",
    "فيه مشكلة بالـ{bare}، وش تقترح؟",
    "الـ{bare} طلع {en_adj}، ما توقعت",
    "وش الفرق بين {tech} و{tech2}؟",
    "أنا {act} على {tech} من أمس",
    "ما قدرت أفهم ليه الـ{bare} كذا",
    "الـ{bare} يبيله configuration جديد",
    # Daily life
    "اليوم كان productive مرة، سويت شغل كثير",
    "أبغى آخذ break شوي، وبعدين نكمل",
    "الـworkshop كان {en_adj} مرة، تعلمنا كثير",
    "بكرة فيه deadline مهم، لازم نجهز",
    "الـpresentation حقتنا جاهزة، {ar_adj}",
    "سوينا meeting اليوم وقررنا أشياء كثيرة",
    "الـteam قاعد يشتغل بشكل {en_adj}",
    "نبغى نسوي plan للأسبوع الجاي",
    "الـfeedback كان positive، الحمد لله",
    "الـprogress {en_adj}، لازم نكمل كذا",
    "عندي interview بكرة، قاعد أتجهز",
    "الـresult {en_adj}، بس لازم نتأكد",
    "سوينا testing وكل شي {en_adj}",
    "هالـcourse {en_adj}، تنصح فيه؟",
    "الـsupport حقهم {en_adj}، ساعدونا",
    "مو قادر أفهم هالـconcept، فيه شرح؟",
    "أرسلت له email بس ما رد لين الحين",
    "الـproject manager قال {tech} {ar_adj}",
    "قعدت على call ساعتين عن {tech}",
    "قلت له عن {tech} بس ما فهم",
    "اشتغلنا على {dom} كثير اليوم",
    "الـ{dom} {ar_adj}، يبيله وقت",
    "نبغى نراجع {dom} قبل {time}",
    "{dom} راح يكون جاهز {time}",
    "فيه تحديث جديد لـ{dom}، شفته؟",
]

def generate_cs(target: int, seen: set) -> list:
    results, tries = [], 0
    while len(results) < target and tries < target * 30:
        tries += 1
        tmpl = random.choice(CS_TEMPLATES)
        try:
            tech1 = random.choice(EN_TECH)
            tech2 = random.choice(EN_TECH)
            while tech2 == tech1:
                tech2 = random.choice(EN_TECH)
            sent = tmpl.format(
                tech=tech1, tech2=tech2,
                act=random.choice(EN_ACT),
                en_adj=random.choice(EN_ADJ),
                ar_adj=random.choice(AR_ADJ),
                time=random.choice(EN_TIME),
                place=random.choice(EN_PLACE),
                bare=random.choice(EN_BARE),
                dom=random.choice(EN_DOMAIN),
            )
        except KeyError:
            continue
        sent = clean(sent)
        key  = strip_diac(sent)
        if key not in seen and ok(sent):
            seen.add(key)
            results.append(sent)
    return results


# ── Number sentence library (digits → Saudi words via the normalizer) ─────────
AR_ITEM   = ["ريال", "ريال بس", "ريال تقريباً", "طلب", "رسالة", "ملف",
             "مشترك", "زائر", "موظف", "طالب", "كيلو", "كيلومتر", "دقيقة",
             "ساعة", "يوم", "أسبوع", "سنة", "مرة", "نقطة", "سؤال"]

NUM_TEMPLATES = [
    "عندي {n} {item}",
    "وصلنا {n} {item} اليوم",
    "السعر {n} ريال",
    "اشتريت الجوال بـ {n} ريال",
    "دفعت {price} ريال على الطلب",
    "الفاتورة طلعت {price} ريال",
    "باقي لنا {n} {item} بس",
    "احتاج {n} {item} زيادة",
    "صار لي {n} {item} ما شفتك",
    "العدد وصل {n} {item}",
    "الخصم {pct}% على كل شي",
    "النسبة طلعت {pct}% بس",
    "زادت المبيعات {pct}% هالشهر",
    "الموعد الساعة {time}",
    "الدوام يبدأ الساعة {time}",
    "نتقابل بكرة الساعة {time}",
    "الرحلة تقلع الساعة {time}",
    "الاجتماع يوم {date}",
    "آخر موعد للتسليم {date}",
    "ولدت سنة {year}",
    "تخرجت من الجامعة سنة {year}",
    "المشروع بدأ سنة {year} وما زال شغال",
    "المسافة {n} كيلومتر تقريباً",
    "وزنه {flt} كيلو",
    "درجة الحرارة اليوم {n}",
    "عمره {n} سنة وما زال نشيط",
    "عددهم {big} شخص في الفعالية",
    "الفرق بينهم {n} نقاط بس",
    "حولت لك {price} ريال على الحساب",
    "الراتب {big} ريال بالشهر",
    "رقم الطلب {code} لا تنساه",
    "المقاس {n} وزي ما تبغى",
    "خذيت {n} {item} والباقي بكرة",
    "السيارة مشت {big} كيلومتر",
    "التطبيق حمله {big} شخص",
    "صار عندنا {n} فروع في الرياض",
    "التذكرة بـ {n} ريال للشخص الواحد",
    "نسبة الإنجاز وصلت {pct}%",
    # numbers inside code-switching context
    "الـmeeting بكرة الساعة {time}",
    "الـdeadline بعد {n} أيام",
    "الـserver فيه {n} GPU",
    "الـdataset فيه {big} sample",
    "الـmodel خذ {n} ساعات training",
    "حملنا {pct}% من الـdata لين الحين",
]

def generate_numbers(target: int, seen: set) -> list:
    """Generate digit-bearing sentences, then verbalize the digits into
    Saudi Arabic words with the inference normalizer (audio == transcript)."""
    results, tries = [], 0
    while len(results) < target and tries < target * 40:
        tries += 1
        tmpl = random.choice(NUM_TEMPLATES)
        n    = random.choice([random.randint(1, 9), random.randint(10, 99),
                              random.randint(100, 999), random.randint(1000, 9999)])
        big  = random.choice([random.randint(1_000, 99_999),
                              random.randint(100_000, 999_999)])
        price= random.choice([random.randint(10, 999), random.randint(1000, 49_999)])
        pct  = random.choice([random.randint(1, 99),
                              round(random.uniform(0.5, 99.5), 1)])
        hh   = random.randint(1, 12)
        mm   = random.choice([0, 15, 30, 45, random.randint(1, 59)])
        time_s = f"{hh}:{mm:02d}"
        date_s = (f"{random.randint(1,28)}/{random.randint(1,12)}"
                  f"/{random.randint(2020, 2030)}")
        year = random.randint(1380, 2026)
        flt  = round(random.uniform(0.5, 99.9), 1)
        code = f"{random.choice(['AB','RJ','SV','XR'])}{random.randint(100, 9999)}"
        try:
            sent = tmpl.format(n=n, big=big, price=price, pct=pct,
                               time=time_s, date=date_s, year=year,
                               flt=flt, item=random.choice(AR_ITEM), code=code)
        except KeyError:
            continue
        # Verbalize digits → Saudi words (same path as runtime front-end)
        sent = clean(_NORM.normalize_entities(sent))
        key  = strip_diac(sent)
        if key not in seen and ok(sent):
            seen.add(key)
            results.append(sent)
    return results


# ── Partial diacritization from the Saudi lexicon CSV ─────────────────────────
def load_lexicon(path):
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
    if not lex: return text
    tokens = re.split(r"(\s+|[،؟؛,\.!?\-:]+)", text)
    return "".join(lex.get(strip_diac(tok), tok) for tok in tokens)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        console.print(Panel(
            "[bold cyan]📝  Saudi-TTS  ·  Saudi Arabic Text Gathering[/bold cyan]\n"
            f"[dim]Sources : SDC corpus + CS templates + number sentences[/dim]\n"
            f"[dim]Target  : {TARGET:,} sentences "
            f"({int(PURE_RATIO*100)}% pure + {int(CS_RATIO*100)}% CS + {int(NUM_RATIO*100)}% numbers)[/dim]\n"
            f"[dim]Output  : {OUTPUT_FILE}[/dim]",
            border_style="cyan", padding=(1, 4),
        ))
    except Exception:
        print(f"=== Gather {TARGET:,} Saudi sentences ===")

    lexicon = load_lexicon(LEXICON_CSV)
    info(f"Loaded {len(lexicon)} Saudi lexicon rules")

    # 1. SDC corpus
    sdc, seen = load_sdc()
    random.shuffle(sdc)

    need_pure = int(TARGET * PURE_RATIO)
    need_cs   = int(TARGET * CS_RATIO)
    need_num  = TARGET - need_pure - need_cs

    pure = sdc[:need_pure]
    if len(pure) < need_pure:
        pure += extend_with_pairs(sdc, seen, need_pure - len(pure))

    # 2. Code-switching
    info(f"Generating {need_cs:,} code-switching sentences …")
    cs = generate_cs(need_cs, seen)
    good(f"Got {len(cs):,} unique CS sentences")

    # 3. Numbers
    info(f"Generating {need_num:,} number sentences …")
    nums = generate_numbers(need_num, seen)
    good(f"Got {len(nums):,} unique number sentences")

    # Top up any shortage (CS templates saturate ~10K unique) with more
    # number sentences — random values make them effectively unlimited
    short = TARGET - len(pure) - len(cs) - len(nums)
    if short > 0:
        info(f"Topping up {short:,} extra number sentences …")
        nums += generate_numbers(short, seen)

    combined = pure + cs + nums
    random.shuffle(combined)
    combined = combined[:TARGET]

    # Apply partial diacritization (same overrides as the inference front-end)
    info("Applying Saudi lexicon diacritization …")
    combined = [apply_lexicon(s, lexicon) for s in combined]
    combined = [s for s in combined if ok(s)]

    # Save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FILE).write_text("\n".join(combined) + "\n", encoding="utf-8")

    pure_count = sum(1 for s in combined
                     if not any(c.isascii() and c.isalpha() for c in s))
    cs_count = len(combined) - pure_count

    try:
        console.print(Panel(
            f"[bold green]✅  Saved {len(combined):,} sentences → {OUTPUT_FILE}[/bold green]\n"
            f"[dim]Pure Arabic (incl. numbers) : {pure_count:,}\n"
            f"With English tokens         : {cs_count:,}\n"
            f"Lexicon rules               : {len(lexicon)}[/dim]",
            border_style="green",
        ))
    except Exception:
        print(f"\nSaved {len(combined):,} sentences → {OUTPUT_FILE}")
        print(f"  Pure: {pure_count:,}  CS: {cs_count:,}")
