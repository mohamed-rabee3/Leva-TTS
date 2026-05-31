"""
Comprehensive text normalization for Levantine Arabic TTS.

Verbalizes (in priority order to avoid pattern conflicts):
  1. URLs            https://... www....        → "رابط"
  2. Emails          user@gmail.com              → "user آت gmail نقطة com"
  3. Dates           31/01/2026, 15-3-2026       → "واحد وتلاتين كانون الثاني ..."
  4. Times           7:35, 10:15, 6:45           → "الساعة سبعة وخمسة وتلاتين دقيقة"
  5. Phone numbers   0790001234, +962...         → digit-by-digit
  6. Alphanumeric    RJ402, AB1234               → "RJ أربعمية واتنين"
  7. Currency syms   $245.75, 50€                → "مئتين ... دولار"
  8. Percentages     18.5%, 25%                  → "تمنتعش فاصلة خمسة بالمية"
  9. Grouped numbers 12,450.90                   → strips commas → float/int
 10. Floats          2.5, 245.75                 → "اتنين فاصلة خمسة"
 11. Integers        150, 2026, 900              → "مية وخمسين", "ألفين وستة وعشرين"

All numbers are rendered in Levantine Arabic words.

Public API:
    normalize_entities(text) -> str    # the full pipeline
    int_to_levantine(n) -> str
    float_to_levantine("2.5") -> str
"""
from __future__ import annotations

import re

# ── Digit / number maps ───────────────────────────────────────────────────────
_DIGIT = {
    "0": "صفر", "1": "واحد", "2": "اتنين", "3": "تلاتة", "4": "أربعة",
    "5": "خمسة", "6": "ستة", "7": "سبعة", "8": "تمانية", "9": "تسعة",
}
_ONES = {
    0: "", 1: "واحد", 2: "اتنين", 3: "تلاتة", 4: "أربعة", 5: "خمسة",
    6: "ستة", 7: "سبعة", 8: "تمانية", 9: "تسعة", 10: "عشرة",
    11: "إحدعش", 12: "اتناعش", 13: "تلتعش", 14: "أربعتعش", 15: "خمستعش",
    16: "ستعش", 17: "سبعتعش", 18: "تمنتعش", 19: "تسعتعش",
}
_TENS = {
    2: "عشرين", 3: "تلاتين", 4: "أربعين", 5: "خمسين",
    6: "ستين", 7: "سبعين", 8: "تمانين", 9: "تسعين",
}
_HUNDREDS = {
    1: "مية", 2: "مئتين", 3: "تلتمية", 4: "أربعمية", 5: "خمسمية",
    6: "ستمية", 7: "سبعمية", 8: "تمنمية", 9: "تسعمية",
}
_MONTHS = {
    1: "كانون الثاني", 2: "شباط", 3: "آذار", 4: "نيسان", 5: "أيار",
    6: "حزيران", 7: "تموز", 8: "آب", 9: "أيلول",
    10: "تشرين الأول", 11: "تشرين الثاني", 12: "كانون الأول",
}
_CUR_SYM = {
    "$": "دولار", "€": "يورو", "£": "جنيه", "₪": "شيكل", "﷼": "ريال",
}


# ── Integer → Levantine words ─────────────────────────────────────────────────
def int_to_levantine(n: int) -> str:
    if n == 0:
        return "صفر"
    if n < 0:
        return "ناقص " + int_to_levantine(-n)

    parts: list[str] = []

    # millions
    if n >= 1_000_000:
        m = n // 1_000_000
        if m == 1:
            parts.append("مليون")
        elif m == 2:
            parts.append("مليونين")
        else:
            parts.append(int_to_levantine(m) + " مليون")
        n %= 1_000_000

    # thousands
    if n >= 1_000:
        k = n // 1_000
        if k == 1:
            parts.append("ألف")
        elif k == 2:
            parts.append("ألفين")
        elif 3 <= k <= 10:
            parts.append(int_to_levantine(k) + " آلاف")
        else:
            parts.append(int_to_levantine(k) + " ألف")
        n %= 1_000

    # hundreds
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100

    # tens / ones
    if n > 0:
        if n < 20:
            parts.append(_ONES[n])
        else:
            t, o = n // 10, n % 10
            parts.append((_ONES[o] + " و" + _TENS[t]) if o else _TENS[t])

    return " و".join(p for p in parts if p)


# ── Float → Levantine words ───────────────────────────────────────────────────
def float_to_levantine(s: str) -> str:
    s = s.replace(",", "")
    if "." not in s:
        return int_to_levantine(int(s or "0"))
    ip, dp = s.split(".", 1)
    res = int_to_levantine(int(ip or "0")) + " فاصلة "
    # Leading-zero decimals (e.g. 1.05) read digit-by-digit; else as a number
    if dp.startswith("0") and len(dp) > 1:
        res += " ".join(_DIGIT[d] for d in dp if d in _DIGIT)
    else:
        res += int_to_levantine(int(dp))
    return res


def _num_to_words(s: str) -> str:
    s = s.replace(",", "")
    return float_to_levantine(s) if "." in s else int_to_levantine(int(s))


def _read_digits(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_DIGIT[ch])
        elif ch == "+":
            out.append("زائد")
    return " ".join(out)


# ── Date / time ───────────────────────────────────────────────────────────────
def _date_to_levantine(d: str, mo: str, y: str) -> str:
    d, mo, y = int(d), int(mo), int(y)
    # Assume DD/MM; swap if clearly MM/DD (first ≤12, second >12)
    if d <= 12 and mo > 12:
        d, mo = mo, d
    if y < 100:
        y += 2000 if y < 50 else 1900
    day   = int_to_levantine(d)
    month = _MONTHS.get(mo, int_to_levantine(mo))
    year  = int_to_levantine(y)
    return f"{day} {month} {year}"


def _time_to_levantine(h: str, m: str, with_prefix: bool = True) -> str:
    h, m = int(h), int(m)
    hour = int_to_levantine(h)
    pre  = "الساعة " if with_prefix else ""
    if m == 0:
        return pre + hour
    if m == 15:
        return pre + hour + " وربع"
    if m == 30:
        return pre + hour + " ونص"
    if m == 45:
        return pre + hour + " إلا ربع"
    return pre + hour + " و" + int_to_levantine(m) + " دقيقة"


# ── Abbreviations (safe, unambiguous only) ────────────────────────────────────
_ABBREV = {
    "كغ":   "كيلوغرام",
    "كم/س": "كيلومتر بالساعة",
    "ص.ب":  "صندوق بريد",
    "د.أ":  "دينار أردني",
    "ر.س":  "ريال سعودي",
    "ل.س":  "ليرة سورية",
    "ل.ل":  "ليرة لبنانية",
}


# ── Compiled patterns ─────────────────────────────────────────────────────────
_RE_URL    = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_RE_EMAIL  = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")
_RE_DATE   = re.compile(r"(?<!\d)(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})(?!\d)")
_RE_TIME   = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_RE_PHONE  = re.compile(r"(?<!\d)(\+\d[\d \-]{6,16}\d|0\d[\d \-]{6,14}\d)(?!\d)")
_RE_CODE   = re.compile(r"\b([A-Za-z]{2,4})(\d{2,6})\b")
_RE_CURSYM1= re.compile(r"([$€£₪﷼])\s*([\d,]+(?:\.\d+)?)")
_RE_CURSYM2= re.compile(r"([\d,]+(?:\.\d+)?)\s*([$€£₪﷼])")
_RE_PCT    = re.compile(r"([\d,]+(?:\.\d+)?)\s*%")
_RE_GROUP  = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_RE_FLOAT  = re.compile(r"(?<!\d)\d+\.\d+(?!\d)")
_RE_INT    = re.compile(r"(?<!\d)\d+(?!\d)")



# ══════════════════════════════════════════════════════════════════════════════
#                         ENGLISH NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
_E_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_E_TENS = ["", "", "twenty", "thirty", "forty", "fifty",
           "sixty", "seventy", "eighty", "ninety"]
_E_CUR  = {"$": "dollars", "€": "euros", "£": "pounds", "₪": "shekels", "﷼": "riyals"}


def int_to_english(n: int) -> str:
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + int_to_english(-n)
    if n < 20:
        return _E_ONES[n]
    if n < 100:
        return _E_TENS[n // 10] + ("-" + _E_ONES[n % 10] if n % 10 else "")
    if n < 1_000:
        return _E_ONES[n // 100] + " hundred" + (
            " " + int_to_english(n % 100) if n % 100 else "")
    if n < 1_000_000:
        return int_to_english(n // 1_000) + " thousand" + (
            " " + int_to_english(n % 1_000) if n % 1_000 else "")
    if n < 1_000_000_000:
        return int_to_english(n // 1_000_000) + " million" + (
            " " + int_to_english(n % 1_000_000) if n % 1_000_000 else "")
    return " ".join(_E_ONES[int(d)] for d in str(n))


def float_to_english(s: str) -> str:
    s = s.replace(",", "")
    if "." not in s:
        return int_to_english(int(s or "0"))
    ip, dp = s.split(".", 1)
    res = int_to_english(int(ip or "0")) + " point"
    for d in dp:
        res += " " + _E_ONES[int(d)]
    return res


def _e_num(s: str) -> str:
    s = s.replace(",", "")
    return float_to_english(s) if "." in s else int_to_english(int(s))


def _e_digits(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_E_ONES[int(ch)])
        elif ch == "+":
            out.append("plus")
    return " ".join(out)


def _normalize_english(text: str) -> str:
    """Normalize a pure-English string (numbers, currency, dates, etc.)."""
    # URLs
    text = _RE_URL.sub(" link ", text)
    # Emails
    text = _RE_EMAIL.sub(
        lambda m: " " + re.sub(r"\d+", lambda d: " " + _e_digits(d.group(0)) + " ",
                               m.group(0).replace("@", " at ").replace(".", " dot ")) + " ",
        text,
    )
    # Phone numbers (+ / 0 prefixed, long runs)
    text = _RE_PHONE.sub(lambda m: " " + _e_digits(m.group(0)) + " ", text)
    # Times  H:MM  → "seven thirty five"
    def _e_time(m):
        h, mm = int(m.group(1)), int(m.group(2))
        if mm == 0:
            return " " + int_to_english(h) + " o'clock "
        return " " + int_to_english(h) + " " + int_to_english(mm) + " "
    text = _RE_TIME.sub(_e_time, text)
    # Alphanumeric codes  RJ402 → "RJ four zero two"
    text = _RE_CODE.sub(lambda m: m.group(1) + " " + _e_digits(m.group(2)), text)
    # Currency symbols
    text = _RE_CURSYM1.sub(lambda m: _e_num(m.group(2)) + " " + _E_CUR[m.group(1)], text)
    text = _RE_CURSYM2.sub(lambda m: _e_num(m.group(1)) + " " + _E_CUR[m.group(2)], text)
    # Percentages
    text = _RE_PCT.sub(lambda m: _e_num(m.group(1)) + " percent", text)
    # Comma-grouped numbers
    text = _RE_GROUP.sub(lambda m: _e_num(m.group(0)), text)
    # Floats
    text = _RE_FLOAT.sub(lambda m: float_to_english(m.group(0)), text)
    # Integers — long runs (≥5 digits, no decimal) read digit-by-digit (IDs/phones)
    def _e_int(m):
        s = m.group(0)
        return _e_digits(s) if len(s) >= 5 else int_to_english(int(s))
    text = _RE_INT.sub(_e_int, text)
    return re.sub(r"\s+", " ", text).strip()


# Arabic-letter detector → decides which normalizer to use
_AR_LETTER = re.compile(r"[\u0600-\u06FF]")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def _normalize_arabic(text: str) -> str:
    """Levantine-Arabic entity normalization (URLs → integers)."""

    # 0. Abbreviations
    for abbr, full in _ABBREV.items():
        text = text.replace(abbr, full)

    # 1. URLs
    text = _RE_URL.sub(" رابط ", text)

    # 2. Emails
    def _email_repl(m):
        s = m.group(0).replace("@", " آت ").replace(".", " نقطة ")
        # read any digit runs digit-by-digit
        s = re.sub(r"\d+", lambda d: " " + " ".join(_DIGIT[c] for c in d.group(0)) + " ", s)
        return " " + s + " "
    text = _RE_EMAIL.sub(_email_repl, text)

    # 3. Dates
    text = _RE_DATE.sub(lambda m: " " + _date_to_levantine(*m.groups()) + " ", text)

    # 4. Times — avoid double "الساعة" when the word already precedes the time
    def _time_repl(m):
        start = m.start()
        before = text[max(0, start - 8):start]
        has_prefix = "الساعة" in before or "ساعة" in before
        return " " + _time_to_levantine(m.group(1), m.group(2), with_prefix=not has_prefix) + " "
    text = _RE_TIME.sub(_time_repl, text)

    # 5. Phone numbers
    text = _RE_PHONE.sub(lambda m: " " + _read_digits(m.group(0)) + " ", text)

    # 6. Alphanumeric codes (keep letters, verbalize trailing digits)
    text = _RE_CODE.sub(lambda m: m.group(1) + " " + int_to_levantine(int(m.group(2))), text)

    # 7. Currency symbols
    text = _RE_CURSYM1.sub(lambda m: _num_to_words(m.group(2)) + " " + _CUR_SYM[m.group(1)], text)
    text = _RE_CURSYM2.sub(lambda m: _num_to_words(m.group(1)) + " " + _CUR_SYM[m.group(2)], text)

    # 8. Percentages
    text = _RE_PCT.sub(lambda m: _num_to_words(m.group(1)) + " بالمية", text)

    # 9. Comma-grouped numbers
    text = _RE_GROUP.sub(lambda m: _num_to_words(m.group(0)), text)

    # 10. Floats
    text = _RE_FLOAT.sub(lambda m: float_to_levantine(m.group(0)), text)

    # 11. Integers
    text = _RE_INT.sub(lambda m: int_to_levantine(int(m.group(0))), text)

    # Collapse extra whitespace introduced by replacements
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Public dispatcher ─────────────────────────────────────────────────────────
def normalize_entities(text: str) -> str:
    """
    Normalize numeric / structured entities, choosing the language automatically:
      - Text containing Arabic letters → Levantine Arabic verbalization
      - Pure-English text             → English verbalization
    """
    if _AR_LETTER.search(text):
        return _normalize_arabic(text)
    return _normalize_english(text)
