import re
import unicodedata

try:
    from num2words import num2words
except ImportError:
    num2words = None

# Whisper emits decimals and thousand-groups split by whitespace tokens
# ("1 .3 million", "$320 ,000", "8 .28"). Collapse the surrounding whitespace
# BEFORE number-to-word conversion so the normalized tokens match the compact
# written form ("one point three million" == "1.3 million") and do not depress
# the sequence-similarity score of an otherwise exact match.
_SPACED_DECIMAL_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
_SPACED_THOUSANDS_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d{3}\b)")

class TextNormalizer:
    # Map common written forms to spoken forms for easier token matching
    # Or map spoken forms to written forms. We'll map written to a common set, and spoken to the same set.
    # Actually, replacing phrases is easier before tokenization.
    PHRASE_REPLACEMENTS = {
        r"\bwant to\b": "wanna",
        r"\bgoing to\b": "gonna",
        r"\bdo not\b": "dont",
        r"\bcannot\b": "cant",
        r"\bwill not\b": "wont",
        r"\bi am\b": "im",
        r"\byou are\b": "youre",
        r"\bthey are\b": "theyre",
        r"\bwe are\b": "were",
        r"\bit is\b": "its",
        r"\bis not\b": "isnt",
        r"\bare not\b": "arent",
        r"\bokay\b": "ok",
        r"\bok\b": "ok"
    }

    def __init__(self):
        pass

    def normalize(self, text: str) -> list[str]:
        # 1. Lowercase
        text = text.lower()
        
        # 2. Unicode normalization
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        # 3. Remove all punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # 4. Remove apostrophes (don't -> dont, it's -> its)
        text = text.replace("'", "")
        
        # 5. Handle dollars e.g., $5 -> five dollars (already stripped $, so this regex needs adjusting)
        # Wait, step 3 removes $. Let's do $ and numbers before step 3.
        pass

    def normalize(self, text: str) -> list[str]:
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

        # Collapse Whisper's whitespace-split number fragments ("1 .3", "8 .28").
        text = _SPACED_DECIMAL_RE.sub('.', text)
        text = _SPACED_THOUSANDS_RE.sub(',', text)

        # Handle dollars: "$8.28" -> "8 dollars and 28 cents" so it token-matches
        # the script form "8 dollars and 28 cents" (rather than the num2words
        # "eight point two eight dollars" for 8.28). A scale word directly
        # following ("$2 .59 million") is a scalar, not a dollars-and-cents
        # amount, so leave those to the generic number-to-words path.
        text = re.sub(
            r'\$(\d+)\.(\d{2})(?!\s*(?:million|billion|trillion|thousand|lakh|crore)\b)',
            r'\1 dollars and \2 cents', text)
        text = re.sub(r'\$(\d+(?:\.\d+)?)', r'\1 dollars', text)
        
        # Handle numbers to words
        if num2words:
            def replace_num(match):
                num_str = match.group(0)
                try:
                    val = float(num_str.replace(',', ''))
                    if val.is_integer():
                        return num2words(int(val)).replace('-', ' ')
                    return num2words(val).replace('-', ' ')
                except:
                    return num_str
            text = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', replace_num, text)

        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove apostrophes
        text = text.replace("'", "")
        
        # Replace phrases
        for pattern, replacement in self.PHRASE_REPLACEMENTS.items():
            text = re.sub(pattern, replacement, text)
            
        # Tokenize
        return text.split()
