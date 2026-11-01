"""
persiang2p.morphology
Rule-based Persian morphological decomposition for G2P.
Decomposes inflected words into stems and affixes, looks up stems in the lexicon,
and reconstructs accurate IPA phonetics.
"""

from typing import Optional, Tuple

# Ordered by length descending so longer matching affixes match first
SUFFIX_RULES = [
    # Plural with possessives
    ("هایمان", "hɒːjemɒːn"),
    ("هایتان", "hɒːjetɒːn"),
    ("هایشان", "hɒːjeʃɒːn"),
    ("هایم", "hɒːjæm"),
    ("هایت", "hɒːjæt"),
    ("هایش", "hɒːjæʃ"),
    ("های", "hɒːje"),
    ("ها", "hɒː"),
    
    # Superlative & Comparative
    ("ترین", "tæɾiːn"),
    ("تر", "tæɾ"),
    
    # Suffixes
    ("ستان", "estɒːn"),
    ("مند", "mænd"),
    ("وار", "vɒːɾ"),
    ("گاه", "ɡɒːh"),
    ("بان", "bɒːn"),
    ("گار", "ɡɒːɾ"),
    ("گر", "ɡæɾ"),
    ("جات", "d͡ʒɒːt"),
    ("ات", "ɒːt"),
    
    # Plural -an
    ("گان", "ɡɒːn"),
    ("ان", "ɒːn"),
    
    # Enclitics
    ("مان", "emɒːn"),
    ("تان", "etɒːn"),
    ("شان", "eʃɒːn"),
    ("ام", "æm"),
    ("ات", "æt"),
    ("اش", "æʃ"),
    ("ی", "iː"),
]

PREFIX_RULES = [
    ("نمی‌", "nemiː"),
    ("نمی", "nemiː"),
    ("می‌", "miː"),
    ("می", "miː"),
    ("بی‌", "biː"),
    ("بی", "biː"),
    ("هم‌", "hæm"),
    ("هم", "hæm"),
    ("نا", "nɒː"),
    ("با", "bɒː"),
]

VOWELS_IPA = {'ɒː', 'uː', 'iː', 'æ', 'e', 'o'}

class PersianMorphology:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def decompose_and_transcribe(self, word: str) -> Optional[str]:
        """
        Attempts to decompose an unseen word into a known lexicon stem + affixes.
        Returns the reconstructed IPA or None if no decomposition is found.
        """
        # Clean ZWNJ at boundaries
        word_clean = word.strip('\u200c ')
        
        # 1. Direct hit
        ipa = self.lexicon.get(word_clean)
        if ipa:
            return ipa

        # 2. Try suffix stripping
        for suf_fa, suf_ipa in SUFFIX_RULES:
            # Check with and without ZWNJ
            match = False
            stem = None
            if word_clean.endswith(f"\u200c{suf_fa}"):
                stem = word_clean[:-len(suf_fa)-1]
                match = True
            elif word_clean.endswith(suf_fa) and len(word_clean) > len(suf_fa) + 1:
                stem = word_clean[:-len(suf_fa)]
                match = True

            if match and stem:
                stem_ipa = self.lexicon.get(stem)
                if stem_ipa:
                    return self._combine_stem_suffix(stem, stem_ipa, suf_fa, suf_ipa)

        # 3. Try prefix stripping
        for pre_fa, pre_ipa in PREFIX_RULES:
            match = False
            stem = None
            if word_clean.startswith(f"{pre_fa}\u200c"):
                stem = word_clean[len(pre_fa)+1:]
                match = True
            elif word_clean.startswith(pre_fa) and len(word_clean) > len(pre_fa) + 1:
                stem = word_clean[len(pre_fa):]
                match = True

            if match and stem:
                stem_ipa = self.lexicon.get(stem)
                if stem_ipa:
                    return f"{pre_ipa}{stem_ipa}"

        # 4. Try prefix + suffix stripping
        for pre_fa, pre_ipa in PREFIX_RULES:
            if word_clean.startswith(pre_fa):
                mid = word_clean[len(pre_fa):].lstrip('\u200c ')
                for suf_fa, suf_ipa in SUFFIX_RULES:
                    if mid.endswith(suf_fa) and len(mid) > len(suf_fa) + 1:
                        stem = mid[:-len(suf_fa)].rstrip('\u200c ')
                        stem_ipa = self.lexicon.get(stem)
                        if stem_ipa:
                            combined = self._combine_stem_suffix(stem, stem_ipa, suf_fa, suf_ipa)
                            return f"{pre_ipa}{combined}"

        return None

    def _combine_stem_suffix(self, stem: str, stem_ipa: str, suf_fa: str, suf_ipa: str) -> str:
        """Handles phonetic linking rules between Persian stems and suffixes."""
        # If stem ends in vowel and suffix starts with vowel, insert glide /j/
        last_ipa_char = stem_ipa[-2:] if len(stem_ipa) >= 2 and stem_ipa[-2:] in VOWELS_IPA else stem_ipa[-1:]
        
        # Suffixes starting with vowel: ان (ɒːn), ام (æm), ات (æt), اش (æʃ), ی (iː)
        if suf_fa in {"ان", "ام", "ات", "اش", "مان", "تان", "شان", "ی"}:
            # If stem ends in /e/ (like خانه)
            if stem_ipa.endswith("e"):
                if suf_fa == "ان":
                    return f"{stem_ipa[:-1]}eɡɒːn" # خانه -> خانگان
                elif suf_fa == "ی":
                    return f"{stem_ipa}ʔiː"        # خانه‌ای
                else:
                    return f"{stem_ipa}je{suf_ipa}" # خانه‌ام -> xɒːnejeæm
            # If stem ends in long vowels /ɒː/ or /uː/
            elif stem_ipa.endswith("ɒː") or stem_ipa.endswith("uː") or stem_ipa.endswith("iː"):
                if suf_fa in {"ام", "ات", "اش"}:
                    return f"{stem_ipa}j{suf_ipa[1:]}" # دانایم -> dɒːnɒːjæm
                elif suf_fa in {"مان", "تان", "شان"}:
                    return f"{stem_ipa}je{suf_ipa}"

        # Default clean concatenation
        return f"{stem_ipa}{suf_ipa}"
