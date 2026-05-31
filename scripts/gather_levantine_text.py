"""
Gather 50,000 Levantine Arabic sentences for OmniVoice synthesis.

Sources
-------
  1. GU-CLASP Shami Corpus (4 dialect files, ~60K sentences total):
       Syrian, Lebanese, Palestinian, Jordanian
       https://github.com/GU-CLASP/shami-corpus/tree/master/Data

  2. Synthetic code-switching sentences (Levantine Arabic + English)
       Template library covering work/tech, daily life, education

Distribution: ~70% pure Levantine (from Shami), ~30% code-switching

Output
------
  data/levantine_50k.txt   — one sentence per line, partial diacritics on homographs

Usage
-----
  python scripts/gather_levantine_text.py

Requirements: requests (or urllib — built-in), rich (optional)
"""

# ╔═══════════════════════════════════════════════════════════════════╗
#                          CONFIGURATION
# ╚═══════════════════════════════════════════════════════════════════╝
TARGET         = 50_000
PURE_RATIO     = 0.70          # fraction from Shami corpus
OUTPUT_FILE    = "data/levantine_50k.txt"
HOMOGRAPHS_FILE = "data/homographs.json"
SEED           = 42
# ╚═══════════════════════════════════════════════════════════════════╝

import json
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


# ── Shami corpus files ────────────────────────────────────────────────────────
SHAMI_FILES = {
    "syrian":      "https://raw.githubusercontent.com/GU-CLASP/shami-corpus/master/Data/syrian.txt",
    "lebanese":    "https://raw.githubusercontent.com/GU-CLASP/shami-corpus/master/Data/Lebanees.txt",
    "palestinian": "https://raw.githubusercontent.com/GU-CLASP/shami-corpus/master/Data/Palestinian.txt",
    "jordanian":   "https://raw.githubusercontent.com/GU-CLASP/shami-corpus/master/Data/jordinian.txt",
}

def download_shami():
    all_sents = []
    for dialect, url in SHAMI_FILES.items():
        try:
            info(f"Downloading {dialect} …")
            with urlopen(url, timeout=30) as r:
                lines = r.read().decode("utf-8", errors="ignore").splitlines()
            valid = [clean(l) for l in lines if clean(l) and ok(clean(l))]
            good(f"{dialect}: {len(valid):,} clean sentences")
            all_sents.extend(valid)
        except Exception as e:
            warn(f"{dialect} failed: {e}")
    # Deduplicate
    seen, dedup = set(), []
    for s in all_sents:
        k = strip_diac(s)
        if k not in seen:
            seen.add(k); dedup.append(s)
    good(f"Shami total after dedup: {len(dedup):,}")
    return dedup, seen


# ── Code-switching template library ──────────────────────────────────────────
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

AR_ADJ    = ["كتير منيح","كتير مهم","مش شغّال","مش جاهز","كتير تقيل",
             "بدو وقت","جاهز","خلص","مش واضح","مهم كتير","كتير صعب",
             "ما زبط","زابط","كتير زبالة","لازم نصلّحو"]
AR_TIME   = ["هلق","بكرا","اليوم","بعدين","قريب","هالأسبوع",
             "الأسبوع الجاي","قبل بكرا","بعد شوي","هالشهر","امبارح"]

CS_TEMPLATES = [
    # Work + Tech
    "هَلَّق عم {act} {tech}",
    "بِدِّي {act} {tech} قبل {time}",
    "والله {tech} {ar_adj}",
    "مِش عارف ليش {tech} مش شغّال",
    "لازم نراجع {tech} قبل بكرا",
    "رح نبلش {act} {tech} {time}",
    "في مشكلة بـ{tech}، لازم نصلّحها",
    "يلا نكمّل {act} {tech} مع بعض",
    "ضَلّيت {act} {tech} لوقت متأخر",
    "بِدْنَا ننهي {tech} قبل {time}",
    "شو بِدَّك من {tech}؟ قلّي",
    "أنا {act} {tech} من الصبح",
    "عملت {act} {tech} امبارح",
    "شو صار مع {tech}؟ خلص؟",
    "بكرا في meeting عن {tech}",
    "{tech} {ar_adj}، بس لازم نكمّل",
    "شو رأيك بـ{tech}؟",
    "عم نشتغل على {tech} هالأسبوع",
    "حكيتلك عن {tech} قبل؟ هو {en_adj}",
    "اجا {tech} {time}، أخيراً!",
    "{tech} جاهز {place}",
    "الـ{bare} مش شغّال {place}، بتعرف ليش؟",
    "هَلَّق عم نعمل review على {tech}",
    "في issue بالـ{bare}، لازم نشوف",
    "خلّصنا الـ{bare} امبارح",
    "بكرا رح نعمل demo لـ{tech}",
    "المشكلة كلها من {tech}",
    "نسيت أعدّل {tech}، ها المشكلة",
    "الـ{bare} crashed {time}",
    "عم ن-debug الـ{bare} هَلَّق",
    "كتبت script لـ{tech}، شو رأيك؟",
    "الـ{bare} عامل fine {place}",
    "جرّبت {tech}؟ شو رأيك؟",
    "في update جديد لـ{tech}",
    "الـ{bare} version الجديدة {en_adj} كتير",
    "ليش الـ{bare} مش شغّال {place}؟",
    "شو رأيك بالـ{bare}؟ جربته؟",
    "كيفك؟ شو في جديد مع {tech}؟",
    "يا زلمة {tech} {en_adj} كتير! شو بِدَّك كَمَان؟",
    "والله! {tech} خلص أخيراً!",
    "شو هاد؟ الـ{bare} عمل update؟",
    "بتعرف كيف تحل مشكلة الـ{bare}؟",
    "في مشكلة بالـ{bare}، شو بتقترح؟",
    "الـ{bare} طلع {en_adj}، ما توقعت",
    "شو الفرق بين {tech} و{tech}؟",
    "أنا {act} على {tech} من امبارح",
    "ما قدرت أفهم ليش الـ{bare} هيك",
    "الـ{bare} بدو configuration جديد",
    # Daily life
    "اليوم كان productive كتير، عملت كتير شغل",
    "بِدِّي آخد break شوي، من بعدين نكمّل",
    "الـworkshop كان {en_adj} كتير، تعلمنا كتير",
    "بكرا في deadline مهم، لازم نجهّز",
    "الـpresentation تبعتنا جاهزة، {ar_adj}",
    "عملنا meeting اليوم وقررنا كتير أمور",
    "الـteam عم يشتغل بشكل {en_adj}",
    "بِدْنَا نعمل plan للأسبوع الجاي",
    "الـfeedback كان positive، الحمد لله",
    "الـprogress {en_adj}، لازم نكمّل هيك",
    "عندي interview بكرا، عم أتحضّر",
    "الـresult {en_adj}، بس لازم نتحقق",
    "عملنا testing وكلشي {en_adj}",
    "هالـcourse {en_adj}، بتنصح فيه؟",
    "الـsupport تبعهم {en_adj}، ساعدونا",
    "مش قادر أفهم هالـconcept، في شرح؟",
    "بعتلو email بس ما رد لهلق",
    "الـproject manager قال {tech} {ar_adj}",
    "ضَلّيت على call لساعتين عن {tech}",
    "حكيتلو عن {tech} بس ما فهم",
    "اشتغلنا على {dom} كتير اليوم",
    "الـ{dom} {ar_adj}، محتاج وقت",
    "بِدْنَا نراجع {dom} قبل {time}",
    "{dom} رح يكون جاهز {time}",
    "في تحديث جديد لـ{dom}، شفته؟",
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


# ── Partial diacritization (homographs) ──────────────────────────────────────
def load_homographs(path):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {k: v["diacritized"] for k, v in raw.items() if not k.startswith("_")}
    except Exception as e:
        warn(f"Homographs not loaded: {e}")
        return {}

def apply_homographs(text: str, hg: dict) -> str:
    if not hg: return text
    tokens = re.split(r"(\s+|[،؟؛,\.!?\-:]+)", text)
    return "".join(hg.get(strip_diac(tok), tok) for tok in tokens)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        console.print(Panel(
            "[bold cyan]📝  Leva-TTS  ·  Levantine Text Gathering[/bold cyan]\n"
            f"[dim]Sources : GU-CLASP Shami corpus (4 dialects) + CS templates[/dim]\n"
            f"[dim]Target  : {TARGET:,} sentences ({int(PURE_RATIO*100)}% pure + {int((1-PURE_RATIO)*100)}% code-switched)[/dim]\n"
            f"[dim]Output  : {OUTPUT_FILE}[/dim]",
            border_style="cyan", padding=(1, 4),
        ))
    except Exception:
        print(f"=== Gather {TARGET:,} Levantine sentences ===")

    homographs = load_homographs(HOMOGRAPHS_FILE)
    info(f"Loaded {len(homographs)} homograph rules")

    # Download Shami
    shami, seen = download_shami()
    random.shuffle(shami)

    # Determine split
    need_pure = min(len(shami), int(TARGET * PURE_RATIO))
    need_cs   = TARGET - need_pure

    pure = shami[:need_pure]

    info(f"Generating {need_cs:,} code-switching sentences …")
    cs = generate_cs(need_cs, seen)
    good(f"Got {len(cs):,} unique CS sentences")

    # Top up from Shami if CS is short
    if len(cs) < need_cs and len(shami) > need_pure:
        extra = shami[need_pure:need_pure + (need_cs - len(cs))]
        pure += extra
        info(f"Topped up with {len(extra):,} extra Shami sentences")

    combined = pure + cs
    random.shuffle(combined)
    combined = combined[:TARGET]

    # Apply homograph partial diacritization
    info(f"Applying homograph diacritization …")
    combined = [apply_homographs(s, homographs) for s in combined]
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
            f"[dim]Pure Levantine  : {pure_count:,}\n"
            f"Code-switching  : {cs_count:,}\n"
            f"Homograph rules : {len(homographs)}[/dim]",
            border_style="green",
        ))
    except Exception:
        print(f"\nSaved {len(combined):,} sentences → {OUTPUT_FILE}")
        print(f"  Pure: {pure_count:,}  CS: {cs_count:,}")
