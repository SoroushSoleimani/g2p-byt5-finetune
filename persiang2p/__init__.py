"""
persiang2p
Fast, High-Accuracy Persian G2P (Grapheme-to-Phoneme) Engine.
Combines:
1. Persian Normalizer (character unification & number-to-words expansion)
2. KaamelDict Lexicon (116K+ high-accuracy human-verified word-IPA pairs)
3. Persian Morphology Engine (prefix/suffix decomposition & phonotactic recomposition)
4. Context & Ezafe Layer (syntactic-prosodic bridge for fluent continuous speech)
5. Fast Fallback Engine (character-level phonotactics for OOV terms)
"""

import re
from typing import List, Union
from .normalizer import PersianNormalizer
from .lexicon import Lexicon
from .morphology import PersianMorphology
from .ezafe import EzafeEngine
from .fallback import FastFallbackG2P

class PersianG2P:
    def __init__(self, data_path: str = None, use_cache: bool = True):
        self.normalizer = PersianNormalizer()
        self.lexicon = Lexicon(data_path=data_path, cache=use_cache)
        self.morphology = PersianMorphology(self.lexicon)
        self.ezafe = EzafeEngine()
        self.fallback = FastFallbackG2P()
        
        # Word tokenization regex (preserves internal ZWNJ)
        self.token_pattern = re.compile(r'([^\s،؛؟!.\(\)\[\]«»\-:]+|[،؛؟!.\(\)\[\]«»\-:])')

    def transcribe_word(self, word: str) -> str:
        """
        Transcribes a single Persian word to IPA with 3-tier cascade:
        Tier 1: Lexicon O(1) lookup
        Tier 2: Morphological decomposition
        Tier 3: Phonotactic fallback
        """
        norm_w = self.normalizer.normalize(word, expand_numbers=False)
        if not norm_w:
            return ""

        # Tier 1: Lexicon
        ipa = self.lexicon.get(norm_w)
        if ipa:
            return ipa

        # Tier 2: Morphology
        ipa = self.morphology.decompose_and_transcribe(norm_w)
        if ipa:
            return ipa

        # Tier 3: Fallback
        return self.fallback.transcribe_word(norm_w)

    def transcribe(self, text: str, apply_ezafe: bool = True) -> str:
        """
        Transcribes a Persian word or sentence to an IPA phoneme sequence.
        Normalizes digits, cleans punctuation, resolves Ezafe, and links words.
        """
        if not text:
            return ""

        # Normalize text and expand numbers to spoken words
        norm_text = self.normalizer.normalize(text, expand_numbers=True)

        # Tokenize
        tokens = self.token_pattern.findall(norm_text)
        if not tokens:
            return ""

        # Transcribe tokens
        words = []
        raw_ipas = []

        for tok in tokens:
            # Check if token is punctuation
            if re.match(r'^[،؛؟!.\(\)\[\]«»\-:]+$', tok):
                words.append(tok)
                raw_ipas.append(tok)
            else:
                words.append(tok)
                raw_ipas.append(self.transcribe_word(tok))

        # Apply Ezafe across consecutive tokens
        if apply_ezafe and len(words) > 1:
            final_ipas = self.ezafe.apply_ezafe_to_tokens(words, raw_ipas)
        else:
            final_ipas = raw_ipas

        # Join tokens cleanly
        return " ".join(final_ipas)

    def __call__(self, text: str) -> str:
        return self.transcribe(text)
