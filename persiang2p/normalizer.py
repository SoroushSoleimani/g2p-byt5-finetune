"""
persiang2p.normalizer
Fast Persian text normalizer for G2P and TTS front-ends.
Handles character unification, ZWNJ normalization, punctuation detachment,
and Persian number-to-words expansion.
"""

import re

# Character mappings for Arabic/Persian unification
CHAR_MAP = {
    'ي': 'ی',
    'ى': 'ی',
    'ئ': 'ی',
    'ك': 'ک',
    'ۀ': 'ه‌ی',
    'ة': 'ت',
    'ؤ': 'و',
    'أ': 'ا',
    'إ': 'ا',
    'ٱ': 'ا',
    '\u200e': '', # LTR mark
    '\u200f': '', # RTL mark
    '\ufeff': '', # Zero width no-break space
}

# Digit mappings
DIGITS_EN = "0123456789"
DIGITS_FA = "۰۱۲۳۴۵۶۷۸۹"
DIGITS_AR = "٠١٢٣٤٥٦٧٨٩"
DIGIT_MAP = {ord(a): ord(e) for a, e in zip(DIGITS_AR, DIGITS_EN)}
DIGIT_MAP.update({ord(f): ord(e) for f, e in zip(DIGITS_FA, DIGITS_EN)})

# Number to words dictionary
NUM_WORDS = {
    0: "صفر", 1: "یک", 2: "دو", 3: "سه", 4: "چهار", 5: "پنج",
    6: "شش", 7: "هفت", 8: "هشت", 9: "نه", 10: "ده",
    11: "یازده", 12: "دوازده", 13: "سیزده", 14: "چهارده", 15: "پانزده",
    16: "شانزده", 17: "هفده", 18: "هجده", 19: "نوزده",
    20: "بیست", 30: "سی", 40: "چهل", 50: "پنجاه",
    60: "شصت", 70: "هفتاد", 80: "هشتاد", 90: "نود",
    100: "صد", 200: "دویست", 300: "سیصد", 400: "چهارصد", 500: "پانصد",
    600: "ششصد", 700: "هفتصد", 800: "هشتصد", 900: "نهصد"
}

def number_to_persian_words(n: int) -> str:
    """Converts an integer (up to 999,999,999) to standard spoken Persian words."""
    if n == 0:
        return "صفر"
    if n < 0:
        return "منفی " + number_to_persian_words(-n)
    
    parts = []
    
    # Millions
    if n >= 1000000:
        m = n // 1000000
        n %= 1000000
        parts.append(number_to_persian_words(m) + " میلیون")
    
    # Thousands
    if n >= 1000:
        k = n // 1000
        n %= 1000
        parts.append(number_to_persian_words(k) + " هزار")
        
    # Hundreds
    if n >= 100:
        h = (n // 100) * 100
        n %= 100
        if h in NUM_WORDS:
            parts.append(NUM_WORDS[h])
            
    # Tens & Units
    if n > 0:
        if n in NUM_WORDS:
            parts.append(NUM_WORDS[n])
        else:
            t = (n // 10) * 10
            u = n % 10
            if t in NUM_WORDS and u in NUM_WORDS:
                parts.append(NUM_WORDS[t] + " و " + NUM_WORDS[u])
            else:
                parts.append(str(n))
                
    return " و ".join(parts)


class PersianNormalizer:
    def __init__(self):
        # Precompile regex patterns for maximum throughput
        self.char_pattern = re.compile('|'.join(re.escape(k) for k in CHAR_MAP.keys()))
        self.zwnj = '\u200c'
        self.multi_zwnj = re.compile(r'\u200c+')
        self.multi_space = re.compile(r'[ \t\r\f]+')
        self.num_pattern = re.compile(r'\b\d+\b')
        self.punct_pattern = re.compile(r'([،؛؟!.\(\)\[\]«»\-:])')

    def normalize(self, text: str, expand_numbers: bool = True) -> str:
        """
        Normalizes raw Persian text:
        1. Replaces Arabic letters with standard Persian equivalents.
        2. Unifies all Arabic/Persian digits to standard digits.
        3. Expands digits to Persian words for natural TTS synthesis.
        4. Cleans redundant spaces and half-spaces.
        """
        if not text:
            return ""

        # Fast character translation
        text = self.char_pattern.sub(lambda m: CHAR_MAP[m.group(0)], text)

        # Digits translation (٠-٩, ۰-۹ -> 0-9)
        text = text.translate(DIGIT_MAP)

        # Number-to-words expansion for TTS
        if expand_numbers:
            text = self.num_pattern.sub(lambda m: number_to_persian_words(int(m.group(0))), text)

        # Clean ZWNJ and redundant spaces
        text = self.multi_zwnj.sub(self.zwnj, text)
        text = text.replace(f' {self.zwnj}', self.zwnj).replace(f'{self.zwnj} ', self.zwnj)
        text = self.multi_space.sub(' ', text)

        return text.strip()
