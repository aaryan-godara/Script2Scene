from typing import List, Tuple, Dict, Any
from video_assembler.models import Scene
from .provider_base import TranscriptionResult, TranscribedWord
from .text_normalizer import TextNormalizer
import difflib

class AlignmentDiagnostics:
    def __init__(self):
        self.diagnostics = {}

    def add(self, scene_id: int, data: Dict[str, Any]):
        self.diagnostics[scene_id] = data


class AlignmentService:
    def __init__(self, review_threshold: float = 0.75):
        self.normalizer = TextNormalizer()
        self.review_threshold = review_threshold

    def _calculate_match_score(self, expected_tokens: List[str], transcribed_tokens: List[str]) -> float:
        """
        Calculates a similarity score between two token sequences using SequenceMatcher.
        """
        if not expected_tokens or not transcribed_tokens:
            return 0.0
            
        sm = difflib.SequenceMatcher(None, expected_tokens, transcribed_tokens)
        return sm.ratio()

    def align_scenes(self, scenes: List[Scene], transcription: TranscriptionResult) -> Tuple[List[Scene], AlignmentDiagnostics]:
        """
        Sequentially aligns known scenes to transcription words.
        """
        diagnostics = AlignmentDiagnostics()
        
        # Normalize all transcribed words
        t_words = transcription.words
        t_tokens = [self.normalizer.normalize(w.word)[0] if self.normalizer.normalize(w.word) else "" for w in t_words]
        
        # Sequential cursor
        cursor = 0
        window_size = 100 # Search window forward
        
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
            search_start = max(0, cursor - int(expected_len * 0.5))
            search_end = min(len(t_words), cursor + window_size + expected_len)
            
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
                expanded_end = min(len(t_words), search_end + window_size)
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
            
            if best_start_idx != -1 and best_end_idx != -1:
                # Assign timestamps
                scene.speech_start = t_words[best_start_idx].start
                scene.speech_end = t_words[best_end_idx].end
                scene.match_confidence = best_score
                
                status = "HIGH"
                if best_score < 0.5:
                    status = "FAILED"
                elif best_score < self.review_threshold:
                    status = "REVIEW"
                    
                diagnostics.add(scene.scene_id, {
                    "speech_start": scene.speech_start,
                    "speech_end": scene.speech_end,
                    "confidence": best_score,
                    "status": status,
                    "matched_word_start_index": best_start_idx,
                    "matched_word_end_index": best_end_idx,
                    "expected_token_count": len(expected_tokens),
                    "matched_token_count": best_end_idx - best_start_idx + 1,
                    "token_similarity": best_score,
                    "search_window_start": search_start,
                    "search_window_end": search_end
                })
                
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
