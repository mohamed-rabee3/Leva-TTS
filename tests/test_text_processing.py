"""Text-processing / normalization tests (CPU-only, no model)."""
import pytest

from leva_tts.text.processor import TextProcessor
from leva_tts.text.normalizer import (
    normalize_entities, int_to_levantine, int_to_english, float_to_english,
)


@pytest.fixture(scope="module")
def tp():
    return TextProcessor()


def test_processor_basic(tp):
    out = tp.process("كيفك اليوم؟")
    assert isinstance(out, str) and out.strip()


def test_processor_idempotent_type(tp):
    assert isinstance(tp.process("hello world"), str)


def test_arabic_number_verbalization():
    assert int_to_levantine(0) == "صفر"
    assert int_to_levantine(3) == "تلاتة"
    assert int_to_levantine(100) == "مية"
    assert int_to_levantine(2026) == "ألفين وستة وعشرين"


def test_english_number_verbalization():
    assert int_to_english(0) == "zero"
    assert int_to_english(21) == "twenty-one"
    assert int_to_english(100) == "one hundred"
    assert float_to_english("24.5") == "twenty-four point five"


def test_arabic_entities():
    out = normalize_entities("الساعة 7:35 دفعت 245.75 دولار وزيادة 18.5%")
    assert "7:35" not in out          # time verbalized
    assert "245.75" not in out        # currency verbalized
    assert "18.5" not in out          # percent verbalized
    assert "بالمية" in out


def test_english_entities_language_switch():
    # pure-English input → English verbalization
    out = normalize_entities("call me on 905484 and your credit is 24.5$")
    assert "nine zero five four eight four" in out
    assert "twenty-four point five dollars" in out


def test_no_double_saa3a():
    out = normalize_entities("الساعة 10:15 صار عندي meeting")
    assert "الساعة الساعة" not in out


def test_long_text_does_not_crash(tp):
    long = "اليوم بتاريخ 15 كانون الثاني 2026 صحيت الساعة 7:35. " * 10
    out = tp.process(long)
    assert isinstance(out, str) and len(out) > 0
