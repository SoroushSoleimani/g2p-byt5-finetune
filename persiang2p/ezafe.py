"""
persiang2p.ezafe
Context-aware Ezafe (-e / -ye) resolution and phonetic attachment for Persian sentences.
Handles the syntactic-prosodic bridge between consecutive words in Persian speech.
"""

from typing import List, Tuple

# Prepositions and relational words that almost always carry Ezafe
ALWAYS_EZAFE = {
    'روی', 'زیر', 'پشت', 'جلوی', 'جلو', 'میان', 'کنار', 'برای',
    'درباره', 'مانند', 'مثل', 'بدون', 'داخل', 'سمت', 'طرف', 'برابر',
    'شبیه', 'همراه', 'برحسب', 'به‌منظور', 'بمنظور'
}

# Words that NEVER carry Ezafe (conjunctions, prepositions, pronouns, particles)
NEVER_EZAFE = {
    'و', 'یا', 'اما', 'ولی', 'که', 'تا', 'از', 'به', 'در', 'با',
    'چون', 'اگر', 'چرا', 'چه', 'کی', 'کجا', 'چگونه', 'هم', 'نیز',
    'فقط', 'تنها', 'خیلی', 'بسیار', 'هر', 'هیچ', 'چند', 'همه',
    'من', 'تو', 'او', 'ما', 'شما', 'آنها', 'اینها', 'این', 'آن',
    'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده'
}

# Common Persian verb endings (past & present 3rd person and full verbs)
VERB_ENDINGS = ('کرد', 'کردند', 'شد', 'شدند', 'بود', 'بودند', 'گفت', 'است', 'هست', 'نیست', 'آمد', 'رفت', 'داد', 'دارد', 'داشت', 'کنند', 'کند', 'شود', 'شوند')

VOWEL_PHONEMES = {'ɒː', 'uː', 'iː', 'e', 'o', 'æ'}

class EzafeEngine:
    def __init__(self):
        pass

    def apply_ezafe_to_tokens(self, words: List[str], ipas: List[str]) -> List[str]:
        """
        Applies Persian Ezafe (-e / -ye) to an aligned sequence of words and IPAs.
        Returns the modified list of IPA transcriptions.
        """
        n = len(words)
        if n <= 1:
            return ipas

        result_ipas = list(ipas)

        for i in range(n - 1):
            curr_w = words[i]
            next_w = words[i + 1]
            curr_ipa = result_ipas[i]

            # If current word has no IPA, skip
            if not curr_ipa:
                continue

            # Check if Ezafe is blocked
            if self._should_block_ezafe(curr_w, next_w, i, n):
                continue

            # Check if Ezafe should be triggered
            if self._should_apply_ezafe(curr_w, next_w):
                result_ipas[i] = self._attach_ezafe(curr_w, curr_ipa)

        return result_ipas

    def _should_block_ezafe(self, curr_w: str, next_w: str, index: int, total: int) -> bool:
        # Punctuation
        if curr_w in {'.', '،', '؛', '!', '؟', ':', '-', '»', '«'}:
            return True
        if next_w in {'.', '،', '؛', '!', '؟', ':', '-', '»', '«'}:
            return True

        # Never-Ezafe set
        if curr_w in NEVER_EZAFE:
            return True

        # If current word ends with indefinite 'ی' preceded by standard noun, usually no ezafe
        if curr_w.endswith('ی') and len(curr_w) > 3 and not curr_w in ALWAYS_EZAFE:
            # Often indefinite e.g. کتابی خوب
            pass

        # If current word looks like a finite verb
        if any(curr_w.endswith(v) for v in VERB_ENDINGS):
            return True

        return False

    def _should_apply_ezafe(self, curr_w: str, next_w: str) -> bool:
        # Rule 1: Always-ezafe prepositions
        if curr_w in ALWAYS_EZAFE:
            return True

        # Rule 2: Next word is a conjunction or preposition -> NO ezafe
        if next_w in NEVER_EZAFE or next_w in {'از', 'به', 'در', 'با', 'بر', 'تا', 'برای'}:
            return False

        # Rule 3: Next word is a verb -> NO ezafe
        if any(next_w.endswith(v) for v in VERB_ENDINGS):
            return False

        # Rule 4: Consecutive content words (Noun + Noun, Noun + Adj) in Persian almost universally carry Ezafe
        # (e.g. مدل ترنسفورمر، شبکه عصبی، هوش مصنوعی، یادگیری ماشین)
        return True

    def _attach_ezafe(self, word: str, ipa: str) -> str:
        """
        Attaches the phonological realization of Ezafe:
        - If word ends in /e/ (silent 'ه'): attaches 'je' (/ʃæbæke/ -> /ʃæbækeje/)
        - If word ends in long vowel /ɒː/, /uː/, /iː/: attaches 'je'
        - If word ends in consonant: attaches 'e' (/model/ -> /modele/)
        """
        # Avoid double Ezafe
        if ipa.endswith("e") and not (word.endswith("ه") or word.endswith("ه‌ی")):
            return ipa

        # If Persian spelling ends in 'ه' (silent 'he' representing /e/ vowel)
        if word.endswith("ه") or word.endswith("ه‌ی"):
            return f"{ipa}je"
        
        # If IPA ends in long vowels /ɒː/, /uː/, /iː/
        if ipa.endswith("ɒː") or ipa.endswith("uː") or ipa.endswith("iː"):
            return f"{ipa}je"
        
        # Consonant ending
        return f"{ipa}e"
