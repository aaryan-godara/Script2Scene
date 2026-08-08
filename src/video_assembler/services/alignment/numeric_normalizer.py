"""Numeric-aware normalization for ASR <-> script comparison.

Raw Whisper evidence is NEVER rewritten. Numeric normalization exists only for
comparison/alignment: it resolves many written/spoken spellings of the same
value to one canonical semantic value so equivalent numbers are not mistakenly
penalized for formatting, while genuinely different values are never treated as
equal.

Supported forms:
    * currency: $283,000 / 283,000 / 283,000 dollars / 283 thousand dollars
                 / two hundred eighty-three thousand dollars
    * millions: $1,560,000 / one million five hundred sixty thousand dollars
                 / 1.56 million dollars
    * millions (decimal): $2.5 million / two point five million dollars
    * percentages: 15% / fifteen percent / 15 percent
    * years: 2025 / twenty twenty-five / two thousand twenty-five
    * Indian numbering: 2 lakh 83 thousand / 2,83,000 / 1 crore

Comparison rule (NumericValue):
    * currency and bare numbers compare by raw numeric value (unit "number"),
      so "$283,000" == "283,000" == "two hundred eighty-three thousand".
    * percentages compare only against percentages.
    * values that differ (e.g. $1,560,000 vs $10,000) never match.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional

try:
    from num2words import num2words  # noqa: F401  (used indirectly/tests)
except ImportError:
    num2words = None


NUMERIC_VALIDATION_VERSION = "1.0"


class NumericValue(NamedTuple):
    value: float
    unit: str  # "number" | "percent"
    source: Optional[str] = None

    @property
    def canonical(self) -> str:
        if self.unit == "percent":
            return f"{_clean_num(self.value)} percent"
        return _clean_num(self.value)


def _clean_num(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return ("%.2f" % value).rstrip("0").rstrip(".")


# ----------------------------------------------------------------- word maps
_SMALL = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "lakh": 100_000,
    "crore": 10_000_000,
}
_MULTIPLIER_SCALES = {"thousand", "million", "billion", "trillion", "lakh", "crore"}

_WORD_TOKEN = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|trillion|lakh|crore|zero|oh|point|and|an|a"
)
# Hyphenated spoken numbers ("eighty-two", "twenty-five") are fused into a
# single textual token by Whisper/transcription but represent two word numbers.
# Only pairs where BOTH sides are number words are split ("well-run" stays put).
_HYPHPERS_RE = re.compile(
    r"\b(?:(?P<a>one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|trillion|lakh|crore|zero))-"
    r"(?P<b>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"million|billion|trillion|lakh|crore|zero)\b",
    re.IGNORECASE,
)
_WORD_SEQUENCE_RE = re.compile(
    r"\b(?:" + _WORD_TOKEN + r")(?:\s+(?:" + _WORD_TOKEN + r"))*\b", re.IGNORECASE)
_WORD_PERCENT_RE = re.compile(
    r"\b(?:(?:" + _WORD_TOKEN + r")\s*)+percent\b", re.IGNORECASE)
_WORD_TOKENS = set(_SMALL) | set(_SCALES) | {"point", "oh", "a", "an", "and"}

# digit forms: optional $/currency, comma-grouped digits, optional unit suffix.
# Commas are stripped before parsing, so US ("283,000") and Indian ("2,83,000")
# grouping conventions resolve to the same semantic value.
_RAW_NUMBER_RE = re.compile(
    r"\$?\s*(\d[\d,]*(?:\.\d+)?)"
    r"\s*(percent|million|billion|trillion|thousand|lakh|crore|%|dollars?|usd)?",
    re.IGNORECASE,
)
# Indian compound: "2 lakh 83 thousand" / "1 crore 20 lakh"
_COMPOUND_NUMBER_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(crore|lakh|thousand)\s+"
    r"(\d[\d,]*(?:\.\d+)?)\s*(crore|lakh|thousand)",
    re.IGNORECASE,
)
# Currency decimal spoken as a two-part whole: "1 dollar and 15 cents" /
# "8 dollars and 28 cents". Resolves to the same semantic amount as "$1 .15"
# / "$8 .28", so an otherwise-exact match is not flagged as a numeric mismatch.
_DOLLARS_AND_CENTS_RE = re.compile(
    r"(\d[\d,]*)\s*dollars?\s+and\s+(\d[\d,]*)\s*cents?",
    re.IGNORECASE,
)
_BARE_INT_RE = re.compile(r"(?<![.\d])(\d{4,})(?![.\d])")

# Whisper frequently emits decimals and thousand-groups as whitespace-split
# tokens ("1 .3 million", "$320 ,000", "8 .28"). Collapse the surrounding
# whitespace so those fragments parse as a single numeric value instead of
# unrelated values ("1" and "3 million" / "320" and "0"), which would
# otherwise flag a false numeric mismatch on a correct match.
_SPACED_DECIMAL_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_SPACED_THOUSANDS_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d{3}\b)")


def _collapse_spaced_numbers(text: str) -> str:
    """Normalize Whisper's whitespace-split number tokens to compact forms.

    Applied to a text span (not raw ASR) before numeric extraction:
    ``1 .3`` -> ``1.3``, ``$320 ,000`` -> ``$320,000``, ``8 .28`` -> ``8.28``.
    Reconstruction is bounded to digits directly surrounding a ``.`` or a
    thousand-grouping ``,`` so it never joins unrelated words.
    """
    text = _SPACED_DECIMAL_RE.sub(".", text)
    text = _SPACED_THOUSANDS_RE.sub(",", text)
    return text


def _split_hyphenated_number_words(text: str) -> str:
    """Splits ``twenty-five`` -> ``twenty five`` and ``eighty-two`` -> ``eighty
    two`` so composite spoken numbers parse as one value instead of fragments.

    Only number-word pairs joined by ``-`` are split; ordinary hyphenated words
    (``well-run``, ``low-traffic``) are left untouched.
    """
    return _HYPHPERS_RE.sub(lambda m: f"{m.group('a')} {m.group('b')}", text)


def _rest_has_scale(tokens: List[str], start: int) -> bool:
    """True when any token from ``start`` onward (up to next "and") is a scale
    word, i.e. the text after "and" carries its own magnitude ("eight hundred")
    rather than a low-order remainder ("eighty")."""
    for t in tokens[start:]:
        t = t.lower()
        if t == "and":
            break
        if t in _SCALES:
            return True
    return False


def _parse_word_number(tokens: List[str]) -> List[float]:
    """Parses a spoken-number token sequence into plain magnitudes.

    Returns a list because one sequence can carry multiple values when it reads
    as independent items: "between two hundred and eight hundred" is a range
    {200, 800}, while "one hundred and eighty dollars" is one value {180}. The
    deciding rule at "and" looks at the magnitude already built and the tokens
    that follow:
        * the value before "and" is a scaled magnitude AND what follows is a
          low-order remainder without its own scale ("one hundred AND eighty"
          -> 180, "one thousand six hundred AND twenty" -> 1620): "and"
          continues the same number;
        * the value before "and" is not scaled, OR what follows carries its own
          scale word ("five AND twenty-five" -> {5, 25}; "two hundred AND eight
          hundred" -> {200, 800}): "and" separates values.
    """
    results: List[float] = []
    total = 0.0
    current = 0.0
    frac_div = 10.0
    after_point = False
    prev: Optional[str] = None
    n = len(tokens)
    for i, tok in enumerate(tokens):
        tok = tok.lower()
        # Whisper stutters doubled words ("one one", "a a"); a repeated word is
        # a transcription artifact, not a second value.
        if tok == prev:
            continue
        if tok == "and":
            if prev in _SCALES and not _rest_has_scale(tokens, i + 1):
                continue
            total += current
            if total:
                results.append(total)
            total = 0.0
            current = 0.0
            frac_div = 10.0
            after_point = False
            prev = None
            continue
        if tok == "point":
            after_point = True
            frac_div = 10.0
            prev = tok
            continue
        if tok in ("a", "an"):
            # "a hundred" -> "one hundred"; otherwise "a"/"an" is a bare
            # article ("a two dollar bag") and must not add a phantom 1.
            nxt = tokens[i + 1].lower() if i + 1 < n else None
            if nxt in _SCALES:
                current += 1
            prev = tok
            continue
        if tok == "hundred":
            current *= 100
            prev = tok
            continue
        if tok in _SCALES:
            factor = _SCALES[tok]
            current = current or 1.0
            total += current * factor
            current = 0.0
            after_point = False
            prev = tok
            continue
        if tok in _SMALL:
            if after_point:
                current += _SMALL[tok] / frac_div
                frac_div *= 10
            else:
                current += _SMALL[tok]
            prev = tok
            continue
    total += current
    if total:
        results.append(total)
    return results


def _year_from_words(tokens: List[str]) -> Optional[float]:
    low = [t.lower() for t in tokens]
    # twenty twenty five -> 2025 ; twenty twenty zero -> 2020
    if len(low) == 3 and low[0] == "twenty" and low[1] == "twenty":
        if low[2] == "zero":
            return 2020.0
        if low[2] in _SMALL:
            return 2000 + 20 + _SMALL[low[2]]
    # twenty oh five -> 2005
    if len(low) == 3 and low[0] == "twenty" and low[1] in ("oh", "zero") and low[2] in _SMALL:
        return 2000 + _SMALL[low[2]]
    # nineteen eighty -> 1980
    if len(low) == 2 and low[0] in _SMALL and low[1] in _SMALL:
        hi, lo = _SMALL[low[0]], _SMALL[low[1]]
        if hi in (18, 19, 20):
            return hi * 100 + lo
    return None


def _parse_digit_number(raw_digits: str, suffix: str) -> NumericValue:
    digits = raw_digits.replace(",", "").replace(" ", "")
    val = float(digits)
    suffix_l = (suffix or "").lower().strip()
    if suffix_l in ("percent", "%"):
        return NumericValue(value=val, unit="percent")
    if suffix_l in _MULTIPLIER_SCALES:
        val *= _SCALES[suffix_l]
    return NumericValue(value=val, unit="number")


def _span_overlaps(start: int, end: int, spans: List[tuple]) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


class NumericNormalizer:
    """Resolves written/spoken numeric phrases to canonical semantic values."""

    def __init__(self, support_currency: bool = True, support_percentages: bool = True,
                 support_scale_words: bool = True, support_indian_numbering: bool = True):
        self.support_currency = support_currency
        self.support_percentages = support_percentages
        self.support_scale_words = support_scale_words
        self.support_indian_numbering = support_indian_numbering

    # ------------------------------------------------------------------- API
    def extract(self, text: str) -> List[NumericValue]:
        """Extracts all numeric values from a text span, in order."""
        text = text or ""
        text = _collapse_spaced_numbers(text)
        text = _split_hyphenated_number_words(text)
        values: List[NumericValue] = []
        spans: List[tuple] = []

        # Indian compounds first (consume the whole "2 lakh 83 thousand").
        if self.support_indian_numbering:
            for m in _COMPOUND_NUMBER_RE.finditer(text):
                a, sa, b, sb = m.group(1), m.group(2), m.group(3), m.group(4)
                val = _parse_digit_number(a, sa).value + _parse_digit_number(b, sb).value
                values.append(NumericValue(value=val, unit="number"))
                spans.append((m.start(), m.end()))

        # Dollars-and-cents spoken as a two-part whole: "1 dollar and 15
        # cents" / "8 dollars and 28 cents". Consumed as one value so it is
        # comparable to the "$1 .15" / "$8 .28" decimal form Whisper emits.
        for m in _DOLLARS_AND_CENTS_RE.finditer(text):
            val = float(m.group(1).replace(",", "")) + float(m.group(2).replace(",", "")) / 100.0
            values.append(NumericValue(value=val, unit="number"))
            spans.append((m.start(), m.end()))

        # Percent word forms: "fifteen percent". A spoken percent span can
        # carry multiple values ("five and twenty-five percent" -> {5, 25}).
        if self.support_percentages:
            for m in _WORD_PERCENT_RE.finditer(text):
                if _span_overlaps(m.start(), m.end(), spans):
                    continue
                toks = re.findall(_WORD_TOKEN, m.group(0).lower())
                for value in _parse_word_number(toks):
                    values.append(NumericValue(value=value, unit="percent"))
                spans.append((m.start(), m.end()))

        # Digit forms with optional suffix ($, %, scale words).
        if self.support_currency or self.support_scale_words:
            for m in _RAW_NUMBER_RE.finditer(text):
                if _span_overlaps(m.start(), m.end(), spans):
                    continue
                raw, suffix = m.group(1), m.group(2)
                suffix_l = (suffix or "").lower().strip()
                if suffix_l in ("percent", "%"):
                    if not self.support_percentages:
                        continue
                    unit = "percent"
                else:
                    unit = "number"
                try:
                    v = _parse_digit_number(raw, suffix)
                    # Re-apply unit for percent suffix (% suffix unsupported by
                    # _parse_digit_number scale branch handles it already).
                    if suffix_l in ("percent", "%"):
                        v = NumericValue(value=v.value, unit="percent")
                    values.append(v)
                except ValueError:
                    continue
                spans.append((m.start(), m.end()))

        # Bare 4+ digit integers not consumed above (years, large ints).
        if self.support_scale_words:
            for m in _BARE_INT_RE.finditer(text):
                if _span_overlaps(m.start(), m.end(), spans):
                    continue
                try:
                    values.append(NumericValue(value=float(m.group(1)), unit="number"))
                except ValueError:
                    continue
                spans.append((m.start(), m.end()))

        # Word-number forms (excluding those consumed by percent/year rules).
        for m in _WORD_SEQUENCE_RE.finditer(text):
            if _span_overlaps(m.start(), m.end(), spans):
                continue
            toks = m.group(0).split()
            if not all(t.lower() in _WORD_TOKENS for t in toks):
                continue
            year = _year_from_words(toks)
            if year is not None:
                values.append(NumericValue(value=year, unit="number"))
            else:
                for value in _parse_word_number(toks):
                    values.append(NumericValue(value=value, unit="number"))
            spans.append((m.start(), m.end()))

        return values

    def values_match(self, a: NumericValue, b: NumericValue, tol: float = 1e-3) -> bool:
        if a.unit != b.unit:
            return False
        return abs(float(a.value) - float(b.value)) <= tol

    def text_numeric_consistency(self, expected: str, asr: str,
                                 tol: float = 1e-3) -> Optional[bool]:
        """Compare numbers in two text spans.

        Returns True when both sides contain numbers and every ASR numeric
        value is present among the canonical script values; False when both
        sides contain numbers but the ASR introduces a value the script never
        states (e.g. $10,000 vs $1,560,000); None when either side has no
        numeric evidence.

        Comparison is by magnitude only, intentionally ignoring the unit
        (percent vs bare number). ASR and script express the same amounts with
        inconsistent formatting ("8 to 10%" vs "8% to 10%", "25 to 30%" vs
        "25% to 30%"), and only a real magnitude difference is a mismatch the
        gate must catch.
        """
        exp = self.extract(expected)
        asr = self.extract(asr)
        if not exp or not asr:
            return None
        exp_set = self._expanded_set(exp)
        asr_set = self._expanded_set(asr)
        return asr_set.issubset(exp_set)

    @classmethod
    def _expanded_set(cls, values: List[NumericValue]) -> set:
        """Dollar-and-cents aware signature set.

        Treat a decimal amount like ``2.99`` as equivalent to the plain pair
        ``{2, 99}`` so the "$2 .99" ASR form matches the "2 dollars and 99
        cents" script form (and vice versa). Expansion only splits a value into
        a whole-dollar and cents remainder; integers and large magnitudes are
        otherwise left intact.
        """
        out = set()
        for v in values:
            val = float(v.value)
            out.add(round(val, 6))
            if v.unit == "number" and 0 < val < 1_000_000 and not val.is_integer():
                whole = int(val)
                cents = round((val - whole) * 100)
                if 0 <= cents < 100:
                    out.add(float(whole))
                    out.add(float(cents))
        return out