"""Shared word-merge / dedup utilities for transcription.

Generalizes the deterministic overlap reconciliation used by chunked
transcription and by failed-region recovery, so both paths deduplicate the same
way: matched pairs (same normalized word, sequential) are kept once, preferring
the copy farther from a chunk boundary; legitimate repeated words survive.
"""

from __future__ import annotations

from typing import List, Tuple

from .provider_base import TranscribedWord
from .text_normalizer import TextNormalizer


def match_sequences(prev_list: List[TranscribedWord],
                    cur_list: List[TranscribedWord]
                    ) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Sequential token match over an overlap window.

    Walks both lists in order; a pair matches when normalized text is equal.
    Returns (matched (prev,cur) index pairs, unmatched cur indices).
    """
    normalizer = TextNormalizer()
    i = 0
    j = 0
    matches: List[Tuple[int, int]] = []
    unmatched_cur: List[int] = []
    while j < len(cur_list):
        if i < len(prev_list) and normalizer.normalize(prev_list[i].word) \
                == normalizer.normalize(cur_list[j].word) \
                and prev_list[i].word.strip():
            matches.append((i, j))
            i += 1
            j += 1
        elif i < len(prev_list) and prev_list[i].end < cur_list[j].start:
            # prev captured a word the current chunk missed -> keep prev
            i += 1
        else:
            # current chunk has a word the previous chunk missed -> keep cur
            unmatched_cur.append(j)
            j += 1
    return matches, unmatched_cur


def merge_chunk(merged: List[TranscribedWord], words: List[TranscribedWord],
                g_start: float, prev_start: float, prev_end: float
                ) -> Tuple[List[TranscribedWord], int]:
    """Reconciles one new word set against an existing merged transcript.

    ``words`` may re-capture content ``merged`` already contains (overlap).
    Matched pairs are kept once, preferring the transcription whose word sits
    farther from a chunk boundary. No merged word is dropped unless a strictly
    better duplicate replaces it, so legitimate repeated words survive.
    """
    overlap_end = prev_end

    # only the tail of the merged transcript can be re-captured by this source
    prev_candidates = merged[-len(words):]

    matches, _ = match_sequences(prev_candidates, words)

    dup_cur_ids = set()
    for pi, cj in matches:
        dup_cur_ids.add(cj)
        pw = prev_candidates[pi]
        cw = words[cj]
        pw_dist = min(pw.start - prev_start, prev_end - pw.end)
        cw_dist = min(cw.start - g_start, overlap_end - cw.end)
        if cw_dist > pw_dist:
            for idx, w in enumerate(merged):
                if w is pw:
                    merged[idx] = cw
                    break

    # keep non-duplicate words from this source
    to_add = [w for j, w in enumerate(words) if j not in dup_cur_ids]
    return merged + to_add, len(matches)


def normalize_window_words(words: List[TranscribedWord], offset: float,
                           audio_duration: float) -> List[TranscribedWord]:
    """Shifts local word timestamps into global time and filters invalid words.

    Validation per safety invariant #8 (recovery cannot create invalid or
    backwards timestamps):
        * timestamps must be finite
        * start >= 0
        * end > start (zero-width artifacts filtered)
        * end <= audio_duration (within tolerance)
    """
    out: List[TranscribedWord] = []
    for w in words:
        start = w.start + offset
        end = w.end + offset
        if not _finite(start) or not _finite(end):
            continue
        if start < 0.0 or end - start < 1e-6:
            continue
        if end > audio_duration + 1e-6:
            continue
        out.append(TranscribedWord(
            word=w.word,
            start=round(start, 6),
            end=round(end, 6),
            confidence=w.confidence))
    return out


def _finite(v: float) -> bool:
    import math
    return math.isfinite(v)