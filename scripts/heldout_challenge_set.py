"""
Held-out Persian G2P challenge set (60 words, 15 per linguistic category).

Every entry is (persian_word, reference_ipa, latin_transliteration). None of these words
appear in the CharsiuG2P Persian Wiktionary train/dev/test splits or in the curated training
lexicon; scripts/build_expanded_dataset.py asserts this before writing any data file.

The reference IPA follows the conventions of the CharsiuG2P Persian Wiktionary lexicon:
/æ e o ɒː iː uː/ vowels, /ɢ/ for both qaf and ghayn, /ɾ/ for the tap, no word-initial glottal
stop for alef-initial words, /ʔ/ for ayn/hamza elsewhere, doubled consonants for gemination,
and Ezafe realised as /-e/ after consonants and /-je/ after vowels. References were annotated
manually and have not been verified by a second annotator.
"""

HELDOUT_CHALLENGE_SET = {
    "AI & Tech Loanwords": [
        ("پروسسور", "pɾosesoɾ", "Processor"),
        ("اسکنر", "eskæneɾ", "Scanner"),
        ("روتر", "ɾuːteɾ", "Router"),
        ("دیتابیس", "dejtɒːbejs", "Database"),
        ("اپلیکیشن", "æpliːkejʃen", "Application"),
        ("بلاکچین", "blɒːkt͡ʃejn", "Blockchain"),
        ("آنتی‌ویروس", "ɒːntiːviːɾuːs", "Antivirus"),
        ("وب‌سایت", "vebsɒːjt", "Website"),
        ("لپ‌تاپ", "læptɒːp", "Laptop"),
        ("پیکسل", "piːksel", "Pixel"),
        ("رباتیک", "ɾobɒːtiːk", "Robotik"),
        ("پلتفرم", "plætfoɾm", "Platform"),
        ("مانیتور", "mɒːniːtoɾ", "Monitor"),
        ("اینترفیس", "iːnteɾfejs", "Interface"),
        ("اسکریپت", "eskɾiːpt", "Script"),
    ],
    "Short-Vowel Ambiguity": [
        ("کاربرد", "kɒːɾbæɾd", "Karbord"),
        ("پیامد", "pæjɒːmæd", "Payamad"),
        ("چکیده", "t͡ʃekiːde", "Chekide"),
        ("ویژگی", "viːʒeɡiː", "Vizhegi"),
        ("پژوهشگر", "pæʒuːheʃɡæɾ", "Pazhuheshgar"),
        ("بسامد", "bæsɒːmæd", "Basamad"),
        ("پراکندگی", "pæɾɒːkændeɡiː", "Parakandegi"),
        ("نگاشت", "neɡɒːʃt", "Negasht"),
        ("گسسته", "ɡosæste", "Gosaste"),
        ("بازخورد", "bɒːzxoɾd", "Bazkhord"),
        ("پیکره", "pejkæɾe", "Peykare"),
        ("همخوان", "hæmxɒːn", "Hamkhan"),
        ("نوشتار", "neveʃtɒːɾ", "Neveshtar"),
        ("گویشور", "ɡuːjeʃvæɾ", "Guyeshvar"),
        ("گزینش", "ɡoziːneʃ", "Gozinesh"),
    ],
    "Compound & Ezafe": [
        ("مدل آماری", "modeleɒːmɒːɾiː", "Model-e Amari"),
        ("شبکه عمیق", "ʃæbækejeʔæmiːɢ", "Shabake-ye Amigh"),
        ("حافظه کوتاه‌مدت", "hɒːfezejekuːtɒːhmodæt", "Hafeze-ye Kutahmodat"),
        ("موتور جستجو", "motoɾed͡ʒostod͡ʒuː", "Motor-e Jostoju"),
        ("نرخ یادگیری", "næɾxejɒːdɡiːɾiː", "Nerkh-e Yadgiri"),
        ("لایه پنهان", "lɒːjejepenhɒːn", "Laye-ye Penhan"),
        ("گراف دانش", "ɡɾɒːfedɒːneʃ", "Graph-e Danesh"),
        ("شبکه بازگشتی", "ʃæbækejebɒːzɡæʃtiː", "Shabake-ye Bazgashti"),
        ("کلان داده", "kælɒːndɒːde", "Kalan Dade"),
        ("بردار ویژگی", "bæɾdɒːɾeviːʒeɡiː", "Bordar-e Vizhegi"),
        ("داده آموزشی", "dɒːdejeɒːmuːzeʃiː", "Dade-ye Amuzeshi"),
        ("دسته‌بندی متن", "dæstebændiːjemætn", "Dastebandi-ye Matn"),
        ("تحلیل احساسات", "tæhliːleehsɒːsɒːt", "Tahlil-e Ehsasat"),
        ("رایانش ابری", "ɾɒːjɒːneʃeæbɾiː", "Rayanesh-e Abri"),
        ("تحلیل داده", "tæhliːledɒːde", "Tahlil-e Dade"),
    ],
    "Gemination & Glottal": [
        ("دقت", "deɢɢæt", "Deghghat"),
        ("تعامل", "tæʔɒːmol", "Taamol"),
        ("تجزیه", "tæd͡ʒziːje", "Tajziye"),
        ("متوسط", "motevæsset", "Motavasset"),
        ("مدرس", "modæɾɾes", "Modarres"),
        ("مقرر", "moɢæɾɾæɾ", "Mogharrar"),
        ("مکرر", "mokæɾɾæɾ", "Mokarrar"),
        ("معطل", "moʔættæl", "Moattal"),
        ("مصمم", "mosæmmæm", "Mosammam"),
        ("متعهد", "motæʔæhhed", "Motaahhed"),
        ("مسلط", "mosællæt", "Mosallat"),
        ("تأکید", "tæʔkiːd", "Takid"),
        ("تأمل", "tæʔæmmol", "Taammol"),
        ("مدون", "modævvæn", "Modavvan"),
        ("دائم", "dɒːʔem", "Daem"),
    ],
}
