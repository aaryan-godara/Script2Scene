"""Faithful isolated reproduction of pyVideoTrans's char-level transcript-matching
alignment (videotrans/component/textmatching.py, "Force Alignment Text").

Ports the exact algorithm:
  1. Expand every Whisper word into characters, linearly distributing the word's
     [start, end] across its characters.
  2. Build a character list from the known script text, dropping punctuation/space.
  3. One GLOBAL difflib.SequenceMatcher over (script chars, whisper chars).
  4. 'equal' opcodes assign Whisper char timestamps to script chars.
  5. Unmatched script chars are interpolated linearly between the previous matched
     char end and the next matched char start.

No normalizer, no case folding, no audio-energy logic, no timestamp correction --
this is the literal pyVideoTrans algorithm. Scene boundaries are derived by taking
each scene's text span and reading the first / last non-punctuation char time.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

PUNCTUATIONS = {"，", "。", "？", "！", "；", "：", ",", ".", "?", "!", ";", ":"}


@dataclass
class SceneBoundary:
    scene_id: int
    speech_start: float
    speech_end: float
    matched_chars: int
    total_chars: int


def expand_words_to_chars(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """pyVideoTrans textmatching.py lines 171-181."""
    chars: List[Dict[str, Any]] = []
    for word in words:
        w_text = word.get("word", "").strip()
        if not w_text:
            continue
        start = float(word["start"])
        end = float(word["end"])
        duration = end - start
        char_duration = duration / len(w_text)
        for i, char in enumerate(w_text):
            chars.append(
                {
                    "char": char,
                    "start": start + i * char_duration,
                    "end": start + (i + 1) * char_duration,
                }
            )
    return chars


def _normalized_chars(text: str, normalizer) -> List[str]:
    """Normalized token stream used for Variant B. Mirrors the baseline's TextNormalizer."""
    return list(" ".join(normalizer.normalize(text)))


def normalize_words(words: List[Dict[str, Any]], normalizer) -> List[Dict[str, Any]]:
    """Whisper words with the same normalization applied on the word text."""
    out = []
    for word in words:
        w_text = word.get("word", "").strip()
        if not w_text:
            continue
        normalized_tokens = normalizer.normalize(w_text)
        text = " ".join(normalized_tokens)
        if not text:
            continue
        out.append({"word": text, "start": float(word["start"]), "end": float(word["end"])})
    return out


def align_text_whisper(
    script_chars: List[str],
    whisper_chars: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Core char alignment. Returns (mapped_chars, matched_count, total_non_punc)."""
    target_chars_map: List[Dict[str, Any]] = []
    comparison_target: List[str] = []
    for char in script_chars:
        is_punc = char in PUNCTUATIONS or char.strip() == ""
        target_chars_map.append(
            {"original_char": char, "is_punc": is_punc, "start": None, "end": None}
        )
        if not is_punc:
            comparison_target.append(char)

    comparison_whisper = [x["char"] for x in whisper_chars]
    matcher = difflib.SequenceMatcher(None, comparison_target, comparison_whisper)

    comp_to_orig_map = [
        idx for idx, item in enumerate(target_chars_map) if not item["is_punc"]
    ]
    matched_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            count = i2 - i1
            for k in range(count):
                orig_idx = comp_to_orig_map[i1 + k]
                whisper_idx = j1 + k
                target_chars_map[orig_idx]["start"] = whisper_chars[whisper_idx]["start"]
                target_chars_map[orig_idx]["end"] = whisper_chars[whisper_idx]["end"]
                target_chars_map[orig_idx]["matched_by_equal"] = True
                matched_count += 1

    non_punc_indices = [i for i, x in enumerate(target_chars_map) if not x["is_punc"]]
    for i in range(len(non_punc_indices)):
        curr_real_idx = non_punc_indices[i]
        curr_item = target_chars_map[curr_real_idx]
        if curr_item["start"] is not None:
            continue
        prev_time = 0.0
        if i > 0:
            prev_item = target_chars_map[non_punc_indices[i - 1]]
            if prev_item["end"] is not None:
                prev_time = prev_item["end"]
        next_time = None
        dist = 0
        for j in range(i + 1, len(non_punc_indices)):
            next_real_idx = non_punc_indices[j]
            if target_chars_map[next_real_idx]["start"] is not None:
                next_time = target_chars_map[next_real_idx]["start"]
                dist = j - i
                break
        if next_time is not None:
            duration_per_char = (next_time - prev_time) / (dist + 1)
            curr_item["start"] = prev_time
            curr_item["end"] = prev_time + duration_per_char
        else:
            curr_item["start"] = prev_time
            curr_item["end"] = prev_time + 0.2

    return target_chars_map, matched_count, len(non_punc_indices)


def align(text_content: str, whisper_chars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Variant A: literal pyVideoTrans (raw chars, case-sensitive)."""
    mapped, _, _ = align_text_whisper(list(text_content), whisper_chars)
    return mapped


def align_scenes(
    scenes: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
    normalizer=None,
) -> Tuple[List[SceneBoundary], Dict[str, Any]]:
    whisper_chars = expand_words_to_chars(words)
    normalize_mode = normalizer is not None

    if normalize_mode:
        whisper_chars = expand_words_to_chars(normalize_words(words, normalizer))
        script_chars = []
        for scene in scenes:
            script_chars.extend(_normalized_chars(scene["script_text"], normalizer))
    else:
        script_chars = list("".join(scene["script_text"] for scene in scenes))

    mapped, matched_count, total_non_punc = align_text_whisper(script_chars, whisper_chars)

    boundaries: List[SceneBoundary] = []
    offset = 0
    for scene in scenes:
        script = scene["script_text"]
        n = len(script)
        if normalize_mode:
            norm_len = len(_normalized_chars(script, normalizer))
            span = mapped[offset : offset + norm_len]
            offset += norm_len
        else:
            span = mapped[offset : offset + n]
            offset += n
        non_punc = [c for c in span if not c["is_punc"]]
        first = non_punc[0] if non_punc else None
        last = non_punc[-1] if non_punc else None
        boundaries.append(
            SceneBoundary(
                scene_id=int(scene["scene_id"]),
                speech_start=first["start"] if first is not None else 0.0,
                speech_end=last["end"] if last is not None else 0.0,
                matched_chars=sum(1 for c in non_punc if c.get("matched_by_equal")),
                total_chars=len(non_punc),
            )
        )

    meta = {
        "algorithm": "pyvideotrans_char_difflib",
        "source": "videotrans/component/textmatching.py",
        "normalize_mode": normalize_mode,
        "total_non_punc_chars": total_non_punc,
        "matched_chars": matched_count,
        "match_coverage_pct": round(100.0 * matched_count / total_non_punc, 2) if total_non_punc else 0.0,
    }
    return boundaries, meta
