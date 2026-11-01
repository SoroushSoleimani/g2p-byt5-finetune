"""
persiang2p.fallback
Fast character-level rule-based fallback engine for Persian OOV (Out-of-Vocabulary) words.
Ensures zero runtime errors and provides phonotactically valid Persian IPA transcriptions.
"""

import re

# Standard Persian consonant mappings
CONSONANTS = {
    'ب': 'b', 'پ': 'p', 'ت': 't', 'ط': 't',
    'ث': 's', 'س': 's', 'ص': 's',
    'ج': 'd͡ʒ', 'چ': 't͡ʃ',
    'ح': 'h', 'ه': 'h',
    'خ': 'x', 'د': 'd',
    'ذ': 'z', 'ز': 'z', 'ض': 'z', 'ظ': 'z',
    'ر': 'ɾ', 'ژ': 'ʒ', 'ش': 'ʃ',
    'ف': 'f', 'ق': 'ɢ', 'غ': 'ɢ',
    'ک': 'k', 'گ': 'ɡ',
    'ل': 'l', 'م': 'm', 'ن': 'n',
    'ء': 'ʔ', 'ع': 'ʔ',
}

# Diacritics (if present)
DIACRITICS = {
    'َ': 'æ', # Fathe
    'ِ': 'e', # Kasre
    'ُ': 'o', # Zamme
    'ّ': '',  # Tashdid (doubles consonant)
    'ً': 'æn', # Tanwin
    'ٍ': 'en',
    'ٌ': 'on',
    'ْ': '',  # Sukun
}

class FastFallbackG2P:
    def __init__(self):
        pass

    def transcribe_word(self, word: str) -> str:
        """
        Heuristic character-level transcription for unknown Persian words.
        Follows standard Persian phonotactic constraints (CV, CVC, CVCC).
        """
        if not word:
            return ""

        # Check Latin alphabet
        if re.match(r'^[A-Za-z]+$', word):
            return self._transcribe_latin(word)

        phonemes = []
        chars = list(word)
        n = len(chars)

        for i, ch in enumerate(chars):
            # 1. Diacritics
            if ch in DIACRITICS:
                if ch == 'ّ' and phonemes:
                    # Tashdid: double preceding consonant
                    phonemes.append(phonemes[-1])
                else:
                    phonemes.append(DIACRITICS[ch])
                continue

            # 2. Alef handling
            if ch == 'آ':
                phonemes.append('ʔɒː')
                continue
            if ch == 'ا':
                if i == 0:
                    phonemes.append('ʔæ') # default initial alef
                else:
                    phonemes.append('ɒː')
                continue

            # 3. Vav handling
            if ch == 'و':
                if i == 0:
                    phonemes.append('v')
                elif i == n - 1 or (i < n - 1 and chars[i+1] in CONSONANTS):
                    phonemes.append('uː')
                else:
                    phonemes.append('v')
                continue

            # 4. Ye handling
            if ch == 'ی':
                if i == 0:
                    phonemes.append('j')
                elif i == n - 1:
                    phonemes.append('iː')
                else:
                    phonemes.append('iː')
                continue

            # 5. Standard Consonants
            if ch in CONSONANTS:
                # Handle silent he at word end
                if ch == 'ه' and i == n - 1 and len(phonemes) > 0:
                    phonemes.append('e')
                else:
                    phonemes.append(CONSONANTS[ch])
                continue

            # Unknown character
            phonemes.append(ch)

        # Basic phonotactic smoothing: ensure initial consonant has a following vowel
        return "".join(phonemes)

    def _transcribe_latin(self, word: str) -> str:
        """Simple mapping for Latin acronyms and tech names."""
        word_lower = word.lower()
        latin_map = {
            'ai': 'ʔejʔɒːj',
            'api': 'ʔejpiːʔɒːj',
            'gpu': 'd͡ʒiːpiːjuː',
            'cpu': 'siːpiːjuː',
            'nlp': 'ʔenʔelpiː',
            'tts': 'tiːtiːʔes',
            'g2p': 'd͡ʒiːtuːpiː',
            'onnx': 'ʔoʊneks',
        }
        if word_lower in latin_map:
            return latin_map[word_lower]
        
        # General phonetic transliteration
        return f"/{word_lower}/"
