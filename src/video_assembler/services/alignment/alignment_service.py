from typing import List, Tuple, Dict, Any, Optional
from video_assembler.models import Scene
from .provider_base import TranscriptionResult, TranscribedWord
from .text_normalizer import TextNormalizer
from .numeric_normalizer import NumericNormalizer
import difflib
import re

# A token that is a continuation of a number Whisper split apart: a thousands
# group (",000"), a fraction (".28", ".8"), or a bare digits token directly
# following a "$256" "000". Used to extend the numeric-analysis window so
# "$256" + ",000." compares against "$256,000".
_NUMBER_FRAG_RE = re.compile(r"^[.,]\d*(?:[.,]\d*)*$|^\d+$")


def _rough_text(word: str) -> str:
    """Lowercase, punctuation-stripped form used for stutter comparison."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _intervals_overlap(a, b) -> bool:
    """True when two word intervals overlap, i.e. they cover the same audio."""
    return max(0.0, min(a.end, b.end) - max(a.start, b.start)) > 1e-6

# Spoken contractions the TextNormalizer folds when it normalizes a whole
# string ("want to" -> "wanna", "do not" -> "dont"). Keys sorted longest-first
# so multi-word contractions match before their shorter subsets.
def _phrase_keys() -> List[str]:
    return sorted(TextNormalizer.PHRASE_REPLACEMENTS.keys(),
                  key=lambda p: -len(p.split()))

class AlignmentDiagnostics:
    def __init__(self):
        self.diagnostics = {}

    def add(self, scene_id: int, data: Dict[str, Any]):
        self.diagnostics[scene_id] = data

    def get(self, scene_id) -> Optional[Dict[str, Any]]:
        return self.diagnostics.get(scene_id)


class AlignmentService:
    def __init__(self, review_threshold: float = 0.75,
                 numeric_normalizer: Optional[NumericNormalizer] = None):
        self.normalizer = TextNormalizer()
        self.review_threshold = review_threshold
        self.numeric_normalizer = numeric_normalizer or NumericNormalizer()

    @staticmethod
    def collapse_stutter(words: List[TranscribedWord],
                         window: Optional[Tuple[float, float]] = None) -> List[TranscribedWord]:
        """Conservative ASR stutter collapse for ALIGNMENT COMPARISON ONLY.

        Whisper occasionally emits a duplicated word stream over a region where
        every word appears twice with near-identical (overlapping) timestamps
        ("a a low low traffic traffic"). Such wholly-overlapping intervals are
        the acoustic fingerprint of a stutter artifact: a real repeated word is
        spoken twice and has clearly separated timestamps. This method drops the
        second copy of any adjacent pair whose normalized text is identical AND
        whose intervals overlap AND which lies inside ``window`` (default: the
        whole stream).

        It is intended to be applied to a COPY used for scoring/matching, never
        to mutate the persisted raw transcription. Rhetorical repetition
        ("very, very important") never overlaps in time and is left untouched.
        """
        if not words:
            return []
        lo, hi = (window if window is not None else (float("-inf"), float("inf")))
        collapsed: List[TranscribedWord] = [words[0]]
        for cur in words[1:]:
            prv = collapsed[-1]
            same = (prv.word.strip() and cur.word.strip() and
                    _rough_text(prv.word) == _rough_text(cur.word))
            in_win = (prv.start >= lo and cur.start <= hi)
            drop = same and _intervals_overlap(prv, cur) and in_win and lo is not None
            if drop:
                # Merge timestamps of the duplicate copies so scene window
                # timestamps still map correctly (the two copies overlap).
                prv = TranscribedWord(word=prv.word, start=prv.start, end=cur.end,
                                      confidence=max(prv.confidence, cur.confidence))
                collapsed[-1] = prv
                continue
            collapsed.append(cur)
        return collapsed

    def _calculate_match_score(self, expected_tokens: List[str], transcribed_tokens: List[str]) -> float:
        """
        Calculates a similarity score between two token sequences using SequenceMatcher.
        """
        if not expected_tokens or not transcribed_tokens:
            return 0.0
            
        sm = difflib.SequenceMatcher(None, expected_tokens, transcribed_tokens)
        return sm.ratio()

    def align_scenes(self, scenes: List[Scene], transcription: TranscriptionResult,
                     collapse_region: Optional[Tuple[float, float]] = None) -> Tuple[List[Scene], AlignmentDiagnostics]:
        """
        Sequentially aligns known scenes to transcription words.

        ``collapse_region`` (time window, optional, default None) bounds the
        conservative ASR stutter collapse to a single damaged region. It is an
        alignment-only representation: the raw persisted words are never
        mutated. When omitted, matching uses the raw word stream unchanged.
        """
        diagnostics = AlignmentDiagnostics()
        
        # Normalize all transcribed words, merging split number fragments first.
        # Whisper emits "$256" ,000 and "$1" .15 as separate words; normalizing
        # each in isolation would yield "two hundred and fifty six dollars" +
        # "" for ",000" and never token-match the script's canonical form. We
        # pre-join a number head (digits / "$N") with a following fragment (a
        # thousands ",000" or a fractional ".NN") into one normalization unit,
        # mapping all resulting tokens back to the lead word index.
        if collapse_region is not None:
            t_words = self.collapse_stutter(transcription.words, window=collapse_region)
        else:
            t_words = transcription.words
        t_tokens: List[str] = []
        t_token_word_idx: List[int] = []
        _frag = _NUMBER_FRAG_RE
        i = 0
        n = len(t_words)
        phrase_keys = _phrase_keys()
        while i < n:
            head = t_words[i].word.strip()
            j = i
            # Merge spoken contractions the normalizer expands on whole-text
            # normalization ("want to" -> "wanna", "do not" -> "dont"). The
            # script goes through normalize() as one string and produces the
            # contracted token; the ASR stream must produce the same token or
            # the phrase never matches. Longest match first.
            head_l = head.lower()
            merged_phrase = None
            for phrase in phrase_keys:
                if phrase[0] == head_l and j + 1 < n:
                    follow = [t_words[k].word.strip().lower() for k in range(i + 1, n)]
                    if " ".join([head_l] + follow[:len(phrase) - 1]) == phrase:
                        j = i + len(phrase) - 1
                        merged_phrase = phrase
                        break
            if merged_phrase:
                unit = self.normalizer.PHRASE_REPLACEMENTS[merged_phrase]
            # Merge trailing continuation fragments into this number head
            # (e.g. "$256" + ",000." ; "$1" + ".15"). Only merge when the head
            # already looks like a plain number/currency token.
            elif re.match(r"^\$?\d\d*(?:,\d{3})*(?:\.\d+)?$", head):
                merged = False
                while j + 1 < n and _frag.match(t_words[j + 1].word.strip()):
                    j += 1
                    merged = True
                # "$2 .59 million" is the scalar 2.59 million; absorb the scale
                # word so the whole figure normalizes on the scalar path (not
                # as a dollars-and-cents amount).
                if merged and j + 1 < n and re.match(
                        r"^[a-z]+[.,]*$", t_words[j + 1].word.strip(),
                        re.IGNORECASE) and t_words[j + 1].word.strip().rstrip(".,").lower() in (
                            "million", "billion", "trillion", "thousand", "lakh", "crore"):
                    j += 1
                if merged:
                    # Re-join for a single normalization of the full figure.
                    frags = [t_words[k].word.strip() for k in range(i + 1, j + 1)]
                    unit = head + "".join(frags)
                else:
                    unit = head
            else:
                unit = head
            wtoks = self.normalizer.normalize(unit)
            if not wtoks:
                wtoks = [""]
            t_tokens.extend(wtoks)
            t_token_word_idx.extend([i] * len(wtoks))
            i = j + 1

        # Sequential cursor
        cursor = 0
        window_size = 100 # Search forward
        
        for scene in scenes:
            if not scene.script_text.strip():
                continue
                
            expected_tokens = self.normalizer.normalize(scene.script_text)
            if not expected_tokens:
                continue
                
            expected_len = len(expected_tokens)
            best_score = 0.0
            best_start_idx = -1
            best_end_idx = -1
            
            # Search window: forward from cursor, plus some backtracking for recovery
            # (bounded by the flat token stream length).
            search_start = max(0, cursor - int(expected_len * 0.5))
            search_end = min(len(t_tokens), cursor + window_size + expected_len)
            
            # Slide over the search window
            for i in range(search_start, search_end - expected_len + 1):
                # Try windows of length from expected_len-2 to expected_len+2 to account for insertions/deletions
                for length_offset in range(-2, 3):
                    cur_end = i + expected_len + length_offset
                    if cur_end > search_end or cur_end <= i:
                        continue
                        
                    window_tokens = t_tokens[i:cur_end]
                    score = self._calculate_match_score(expected_tokens, window_tokens)
                    
                    if score > best_score:
                        best_score = score
                        best_start_idx = i
                        best_end_idx = cur_end - 1
            
            # If the best score is very low, we might need a wider search window (recovery)
            if best_score < self.review_threshold:
                # Expanded search
                expanded_start = max(0, search_start - window_size)
                expanded_end = min(len(t_tokens), search_end + window_size)
                for i in range(expanded_start, expanded_end - expected_len + 1):
                    for length_offset in range(-2, 3):
                        cur_end = i + expected_len + length_offset
                        if cur_end > expanded_end or cur_end <= i:
                            continue
                        
                        window_tokens = t_tokens[i:cur_end]
                        score = self._calculate_match_score(expected_tokens, window_tokens)
                        
                        if score > best_score:
                            best_score = score
                            best_start_idx = i
                            best_end_idx = cur_end - 1
            
            # Trim leading/trailing bleed so the matched ASR window does not
            # include fragments of adjacent scenes. Bleed both depresses the
            # token-similarity below the REVIEW threshold and injects foreign
            # numbers into the numeric check. We keep only the innermost span
            # of ASR tokens that actually correspond to script tokens, then
            # recompute the confidence on the trimmed span.
            if best_start_idx != -1 and best_end_idx != -1:
                trimmed_start, trimmed_end = self._trim_window(
                    expected_tokens, t_tokens, best_start_idx, best_end_idx)
                if trimmed_start != -1:
                    trimmed_score = self._calculate_match_score(
                        expected_tokens, t_tokens[trimmed_start:trimmed_end + 1])
                    if trimmed_score >= best_score:
                        best_start_idx, best_end_idx = trimmed_start, trimmed_end
                        best_score = trimmed_score

            # Convert token indices back to source word indices for timestamps.
            if best_start_idx != -1 and best_end_idx != -1:
                word_start = t_token_word_idx[best_start_idx]
                word_end = t_token_word_idx[best_end_idx]
                # Assign timestamps
                scene.speech_start = t_words[word_start].start
                scene.speech_end = t_words[word_end].end
                scene.match_confidence = best_score

                # Numeric-aware evidence: separate token similarity from numeric
                # consistency so a strong lexical match that contains a numeric
                # MISMATCH is never promoted to HIGH.
                window_text = " ".join(w.word for w in t_words[word_start:word_end + 1])
                num = self._numeric_diagnostic(scene.script_text, window_text)

                # Whisper frequently splits a number's thousand-group or
                # fractional digits into separate tokens ("$256" ,000. , "8"
                # ".28"). The matched window stops at the first token and would
                # otherwise compare "$256" against "$256,000" as a mismatch, so
                # extend the numeric window over trailing pure-number-fragment
                # tokens ("," "000" "." "28") and re-run the check.
                frag_end = word_end + 1
                _num_frag_re = _NUMBER_FRAG_RE
                while frag_end < len(t_words) and _num_frag_re.match(t_words[frag_end].word):
                    frag_end += 1
                # Also absorb a trailing scale word so "$2 .59 million"
                # (split as "$2" ".59" "million") is checked as the scalar
                # 2.59 million rather than a "$2.59" dollars amount.
                if frag_end > word_end + 1 and frag_end < len(t_words):
                    nxt = t_words[frag_end].word.strip().rstrip(".,").lower()
                    if nxt in ("million", "billion", "trillion", "thousand", "lakh", "crore"):
                        frag_end += 1
                if frag_end > word_end + 1:
                    frag_text = " ".join(w.word for w in t_words[word_start:frag_end])
                    frag_num = self._numeric_diagnostic(scene.script_text, frag_text)
                    if frag_num.get("numeric_mismatch") is False:
                        num = frag_num

                status = "HIGH"
                if best_score < 0.5:
                    status = "FAILED"
                elif best_score < self.review_threshold:
                    status = "REVIEW"

                # Downgrade a would-be HIGH scene when the on-screen number the
                # narration actually speaks disagrees with the canonical script.
                if status == "HIGH" and num.get("numeric_mismatch"):
                    status = "REVIEW"

                diag = {
                    "speech_start": scene.speech_start,
                    "speech_end": scene.speech_end,
                    "confidence": best_score,
                    "status": status,
                    "matched_word_start_index": word_start,
                    "matched_word_end_index": word_end,
                    "expected_token_count": len(expected_tokens),
                    "matched_token_count": best_end_idx - best_start_idx + 1,
                    "token_similarity": best_score,
                    "asr_text": window_text,
                    "expected_text": scene.script_text,
                    "search_window_start": search_start,
                    "search_window_end": search_end,
                }
                diag.update(num)
                diagnostics.add(scene.scene_id, diag)

                # Advance cursor only if confident, otherwise advance minimally
                if status == "HIGH":
                    cursor = best_end_idx + 1
                elif status == "REVIEW":
                    cursor = best_end_idx + 1
                else:
                    # Failed match, don't advance cursor too far
                    cursor += expected_len
            else:
                scene.match_confidence = 0.0
                diagnostics.add(scene.scene_id, {
                    "status": "FAILED",
                    "reason": "NO_MATCH_FOUND"
                })
                
        return scenes, diagnostics

    @staticmethod
    def _trim_window(expected_tokens, t_tokens, start, end) -> Tuple[int, int]:
        """Trim leading/trailing bleed from a matched ASR window.

        Uses SequenceMatcher matching blocks to find the innermost span of ASR
        tokens that correspond to script tokens, discarding leading/trailing
        fragments from adjacent scenes. Returns (-1, -1) when nothing matches.
        """
        window = t_tokens[start:end + 1]
        sm = difflib.SequenceMatcher(None, expected_tokens, window)
        blocks = [b for b in sm.get_matching_blocks() if b[2] > 0]
        if not blocks:
            return -1, -1
        # Innermost span covering all real matches (window coordinates).
        ws = min(b[1] for b in blocks)
        we = max(b[1] + b[2] for b in blocks) - 1
        return start + ws, start + we

    def _numeric_diagnostic(self, expected_text: str, window_text: str) -> Dict[str, Any]:
        """Numeric consistency evidence for a matched scene window."""
        consistency = self.numeric_normalizer.text_numeric_consistency(
            expected_text, window_text)
        if consistency is None:
            return {"numeric_consistency": None, "numeric_match": None,
                    "numeric_mismatch": False}
        canonical = [v.canonical for v in self.numeric_normalizer.extract(expected_text)]
        asr = [v.canonical for v in self.numeric_normalizer.extract(window_text)]
        if consistency:
            return {"numeric_consistency": True, "numeric_match": True,
                    "numeric_mismatch": False,
                    "canonical_numeric_values": canonical,
                    "asr_numeric_values": asr}
        return {
            "numeric_consistency": False,
            "numeric_match": False,
            "numeric_mismatch": True,
            "warning_type": "NUMERIC_VALUE_MISMATCH",
            "canonical_numeric_values": canonical,
            "asr_numeric_values": asr,
        }
