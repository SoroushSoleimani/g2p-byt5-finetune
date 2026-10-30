#!/usr/bin/env python3
"""
Builds the Expanded Persian (Farsi) G2P Dataset & Stratified Challenge Suite.
Combines the official CharsiuG2P Wiktionary Persian split (7,149 train, 50 dev, 500 test)
with 250 curated Persian linguistic and technical entries across 4 phonetic categories.
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heldout_challenge_set import HELDOUT_CHALLENGE_SET

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
CHARSIU_DATA = os.path.join(BASE_DIR, "CharsiuG2P", "data")
OUT_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DATA_DIR, exist_ok=True)

# 1. Domain Seed Lexicon for Training (60 words across 4 categories)
DOMAIN_SEED_SET = {
    "AI & Tech Loanwords": [
        ("الگوریتم", "ʔælɡoɾiːtm"),
        ("پایتورچ", "pɒːjtɔːɾt͡ʃ"),
        ("تنسورفلو", "tensoɾfloʊ"),
        ("ترنسفورمر", "tɾænsfoɾmeɾ"),
        ("کگل", "kæɡel"),
        ("کراس", "keɾɒːs"),
        ("چت‌جی‌پی‌تی", "t͡ʃætd͡ʒiːpiːtiː"),
        ("دیپ‌لرنینگ", "diːpleɾniːŋɡ"),
        ("توکنایزر", "tokenɒːjzeɾ"),
        ("کانولوشن", "kɒːnvoluːʃæn"),
        ("گرادیان", "ɡeɾɒːdjɒːn"),
        ("دیتاست", "dejtɒːset"),
        ("فریم‌ورک", "fɾejmvoɾk"),
        ("کوبرنتیز", "kuːbeɾnetiːz"),
        ("پایتون", "pɒːjton"),
    ],
    "Short-Vowel Ambiguity": [
        ("شبکه", "ʃæbæke"),
        ("مدل", "model"),
        ("پردازش", "pæɾdɒːzeʃ"),
        ("مهندسی", "mohændesiː"),
        ("پزشکی", "pezeʃkiː"),
        ("محاسبه", "mohɒːsebe"),
        ("معادله", "moʔɒːdele"),
        ("احتمال", "ehtemɒːl"),
        ("پژوهش", "pæʒuːheʃ"),
        ("نرم‌افزار", "næɾmæfzɒːɾ"),
        ("سخت‌افزار", "sæxtæfzɒːɾ"),
        ("گرافیم", "ɡeɾɒːfem"),
        ("نویسه", "neviːse"),
        ("بهینه‌سازی", "behiːnesɒːziː"),
        ("بهره‌وری", "bæhɾevæɾiː"),
    ],
    "Compound & Ezafe": [
        ("هوش مصنوعی", "huːʃemæsnuːʔiː"),
        ("یادگیری ماشین", "jɒːdɡiːɾiːjemɒːʃiːn"),
        ("یادگیری عمیق", "jɒːdɡiːɾiːjeʔæmiːɢ"),
        ("شبکه عصبی", "ʃæbækejeʔæsæbiː"),
        ("زبان طبیعی", "zæbɒːnetæbiːʔiː"),
        ("دانشگاه تهران", "dɒːneʃɡɒːhetæhɾɒːn"),
        ("مدل زبانی", "modelezæbɒːniː"),
        ("پردازش گفتار", "pæɾdɒːzeʃeɡoftɒːɾ"),
        ("تولید گفتار", "toʊliːdeɡoftɒːɾ"),
        ("بینایی ماشین", "biːnɒːjiːjemɒːʃiːn"),
        ("پایگاه داده", "pɒːjɡɒːhedɒːde"),
        ("فضای برداری", "fæzɒːjebæɾdɒːɾiː"),
        ("خطای واجی", "xætɒːjevɒːd͡ʒiː"),
        ("بازنمایی آوایی", "bɒːznæmɒːjiːjeɒːvɒːjiː"),
        ("علم داده", "elmedɒːde"),
    ],
    "Gemination & Glottal": [
        ("مصنوعی", "mæsnuːʔiː"),
        ("طبیعی", "tæbiːʔiː"),
        ("مسئله", "mæsʔæle"),
        ("مسئولیت", "mæsʔuːliːjæt"),
        ("تأثیر", "tæʔsiːɾ"),
        ("تأیید", "tæʔjiːd"),
        ("مؤثر", "moʔæsseɾ"),
        ("معرفی", "moʔæɾɾefiː"),
        ("متغیر", "motæɢæjjeɾ"),
        ("مربع", "moɾæbbæʔ"),
        ("مکعب", "mokæʔʔæb"),
        ("توجه", "tævæd͡ʒd͡ʒoh"),
        ("تفکر", "tæfækkoɾ"),
        ("اتصال", "ettesɒːl"),
        ("اطلاعات", "ettelɒːʔɒːt"),
    ],
}

# 2. Additional 190 Curated Domain & Linguistic Lexicon Pairs (Total 250 Curated)
ADDITIONAL_CURATED_LEXICON = [
    # AI, Deep Learning & Computing
    ("بک‌پراپگیشن", "bækpɾɒːpæɡejʃæn"),
    ("انکودر", "enkodeɾ"),
    ("دیکودر", "diːkodeɾ"),
    ("امبدینگ", "embediːŋɡ"),
    ("فاین‌تیونینگ", "fɒːjntjuːniːŋɡ"),
    ("ریگرسیون", "ɾeɡɾesjon"),
    ("کلاسیفیکیشن", "klæsiːfiːkejʃæn"),
    ("سگمنتیشن", "seɡmentejʃæn"),
    ("هایپرپارامتر", "hɒːjpeɾpɒːɾɒːmetɾ"),
    ("لرنینگ‌ریت", "leɾniːŋɡɾejt"),
    ("اورفیتینگ", "oʊveɾfiːtiːŋɡ"),
    ("آندرفیتینگ", "ɒːndeɾfiːtiːŋɡ"),
    ("ترنسفرلرنینگ", "tɾænsfeɾleɾniːŋɡ"),
    ("چک‌پوینت", "t͡ʃekpojnt"),
    ("هاگینگ‌فیس", "hɒːɡiːŋɡfejs"),
    ("آرکایو", "ɒːɾkɒːjv"),
    ("کودا", "kuːdɒː"),
    ("وکتور", "vektoɾ"),
    ("ماتریس", "mɒːtɾiːs"),
    ("تنسور", "tensoɾ"),
    ("داکر", "dɒːkeɾ"),
    ("گیت‌هاب", "ɡiːthɒːb"),
    ("لینوکس", "liːnuːks"),
    ("جاوااسکریپت", "d͡ʒɒːvɒːeskɾiːpt"),
    ("تایپ‌اسکریپت", "tɒːjpeskɾiːpt"),
    ("فست‌ای‌پی‌آی", "fæstejpiːɒːj"),
    ("جوپیتر", "d͡ʒuːpiːteɾ"),
    ("لنگ‌چین", "læŋɡt͡ʃejn"),
    ("لاما", "lɒːmɒː"),
    ("میسترال", "miːstɾɒːl"),
    ("پستگرس", "postɡɾes"),
    ("رزنت", "ɾeznet"),
    ("نرف", "neɾf"),
    ("ویسپر", "viːspeɾ"),
    ("کلود", "klɒːʊd"),
    ("سرور", "seɾveɾ"),
    ("پایپ‌لاین", "pɒːjplɒːjn"),
    ("فرانت‌اند", "fɾɒːntend"),
    ("بک‌اند", "bækend"),
    ("فول‌استک", "fuːlestæk"),
    ("دیواپس", "devɒːps"),
    ("میکروسرویس", "miːkɾoseɾviːs"),
    ("ای‌پی‌آی", "ejpiːɒːj"),
    ("دیباگ", "diːbɒːɡ"),
    ("کامپایلر", "kɒːmpɒːjleɾ"),
    ("مفسر", "mofæsseɾ"),
    ("باینری", "bɒːjnæɾiː"),
    ("ماژول", "mɒːʒuːl"),
    ("پکیج", "pækejd͡ʒ"),
    ("ریپازیتوری", "ɾiːpɒːziːtoɾiː"),
    ("کامیت", "kɒːmiːt"),
    ("برنچ", "bɾænt͡ʃ"),
    ("پول‌ریکوئست", "puːlɾiːkuːʔest"),
    ("کانتینر", "kɒːntejneɾ"),
    ("مجازی‌سازی", "mæd͡ʒɒːziːsɒːziː"),
    ("پروتکل", "pɾotokol"),
    ("سوکت", "soket"),
    ("الگوریتم‌های", "ʔælɡoɾiːtmhɒːj"),
    ("مدل‌های", "modelhɒːj"),
    ("ترنسفورمرها", "tɾænsfoɾmeɾhɒː"),
    # Short-Vowel & Scientific Terms
    ("الگو", "olɡuː"),
    ("ساختار", "sɒːxtɒːɾ"),
    ("معماری", "meʔmɒːɾiː"),
    ("پردازنده", "pæɾdɒːzænde"),
    ("حافظه", "hɒːfeze"),
    ("ذخیره‌سازی", "zæxiːɾesɒːziː"),
    ("فشرده‌سازی", "feʃoɾdesɒːziː"),
    ("رمزنگاری", "ɾæmzneɡɒːɾiː"),
    ("امنیت", "æmniːjæt"),
    ("مدیریت", "modiːɾiːjæt"),
    ("پیکربندی", "pejkæɾbændiː"),
    ("مقیاس‌پذیری", "meɢjɒːspæziːɾiː"),
    ("پایداری", "pɒːjdɒːɾiː"),
    ("کارایی", "kɒːɾɒːjiː"),
    ("بازدهی", "bɒːzdehiː"),
    ("پیچیدگی", "piːt͡ʃiːdeɡiː"),
    ("همگرایی", "hæmɡeɾɒːjiː"),
    ("واگرایی", "vɒːɡeɾɒːjiː"),
    ("مشتق", "moʃtæɢ"),
    ("انتگرال", "enteɡɾɒːl"),
    ("واریانس", "vɒːɾjɒːns"),
    ("انحراف", "enheɾɒːf"),
    ("معیار", "meʔjɒːɾ"),
    ("توزیع", "toʊziːʔ"),
    ("نمونه‌برداری", "nemuːnebæɾdɒːɾiː"),
    ("برآورد", "bæɾɒːvoɾd"),
    ("آزمون", "ɒːzmuːn"),
    ("فرضیه", "fæɾziːje"),
    ("همبستگی", "hæmbæsteɡiː"),
    ("علیت", "elliːjæt"),
    ("استنتاج", "estentɒːd͡ʒ"),
    ("استدلال", "estedlɒːl"),
    ("منطق", "mænteɢ"),
    ("فلسفه", "fælsæfe"),
    ("زبان‌شناسی", "zæbɒːnʃenɒːsiː"),
    ("واج‌شناسی", "vɒːd͡ʒʃenɒːsiː"),
    ("آواشناسی", "ɒːvɒːʃenɒːsiː"),
    ("تکواژ", "tækvɒːʒ"),
    ("صرف", "sæɾf"),
    ("نحو", "næhv"),
    ("معناشناسی", "mæʔnɒːʃenɒːsiː"),
    ("کاربردشناسی", "kɒːɾboɾdʃenɒːsiː"),
    ("گفتمان", "ɡoftemɒːn"),
    ("بافتار", "bɒːftɒːɾ"),
    ("ساختواژه", "sɒːxtvɒːʒe"),
    ("هم‌آوا", "hæmɒːvɒː"),
    ("هم‌نگاره", "hæmneɡɒːɾe"),
    ("چندمعنایی", "t͡ʃændmæʔnɒːjiː"),
    ("ابهام‌زدایی", "ebhɒːmzedɒːjiː"),
    ("برچسب‌گذاری", "bæɾt͡ʃæsbɡozɒːɾiː"),
    # Compound & Ezafe Phrases
    ("شبکه‌های عصبی", "ʃæbækehɒːjeʔæsæbiː"),
    ("پردازش زبان", "pæɾdɒːzeʃezæbɒːn"),
    ("تبدیل متن", "tæbdiːlemætn"),
    ("تشخیص گفتار", "tæʃxiːseɡoftɒːɾ"),
    ("یادگیری تقویتی", "jɒːdɡiːɾiːjetæɢviːjætiː"),
    ("یادگیری نظارت‌شده", "jɒːdɡiːɾiːjenezɒːɾætʃode"),
    ("یادگیری خودنظارتی", "jɒːdɡiːɾiːjexodnezɒːɾætiː"),
    ("مدل پایه", "modelepɒːje"),
    ("مدل مولد", "modelemovælled"),
    ("شبکه مولد", "ʃæbækejemovælled"),
    ("توزیع احتمال", "toʊziːʔeehtemɒːl"),
    ("تابع هزینه", "tɒːbeʔehæziːne"),
    ("تابع زیان", "tɒːbeʔeziːjɒːn"),
    ("کاهش گرادیان", "kɒːheʃeɡeɾɒːdjɒːn"),
    ("شتاب‌دهنده گرافیکی", "ʃetɒːbdehændejeɡeɾɒːfiːkiː"),
    ("پردازش موازی", "pæɾdɒːzeʃemovɒːziː"),
    ("محاسبات ابری", "mohɒːsebɒːteæbɾiː"),
    ("سیستم عامل", "siːstemeɒːmel"),
    ("مهندسی نرم‌افزار", "mohændesiːjenæɾmæfzɒːɾ"),
    ("معماری کامپیوتر", "meʔmɒːɾiːjekɒːmpjuːteɾ"),
    ("ساختمان داده", "sɒːxtemɒːnedɒːde"),
    ("طراحی الگوریتم", "tæɾɾɒːhiːjeʔælɡoɾiːtm"),
    ("هوش محاسباتی", "huːʃemohɒːsebɒːtiː"),
    ("منطق فازی", "mænteɢefɒːziː"),
    ("الگوریتم ژنتیک", "ʔælɡoɾiːtmeʒenetiːk"),
    ("درخت تصمیم", "deɾæxtetæsmiːm"),
    ("جنگل تصادفی", "d͡ʒænɡæletæsɒːdofiː"),
    ("ماشین بردار", "mɒːʃiːnebæɾdɒːɾ"),
    ("خوشه‌بندی داده", "xuːʃebændiːjedɒːde"),
    ("کاهش ابعاد", "kɒːheʃeæbʔɒːd"),
    ("تحلیل مؤلفه", "tæhliːlemoʔællefe"),
    ("سری زمانی", "seɾiːjezæmɒːniː"),
    ("پیش‌بینی روند", "piːʃbiːniːjeɾævænd"),
    ("ارزیابی عملکرد", "ʔæɾzjɒːbiːjeæmælkæɾd"),
    ("خطای میانگین", "xætɒːjemiːjɒːnɡiːn"),
    # Gemination & Glottal Stops
    ("تخصص", "tæxæssos"),
    ("متخصص", "motæxæsses"),
    ("تحقق", "tæhæɢɢoɢ"),
    ("محقق", "mohæɢɢeɢ"),
    ("تمدن", "tæmæddon"),
    ("تولیدکننده", "toʊliːdkonænde"),
    ("مصرف‌کننده", "mæsɾæfkonænde"),
    ("تعیین", "tæʔjiːn"),
    ("معین", "moʔæjjæn"),
    ("تغییر", "tæɢjiːɾ"),
    ("تغییرات", "tæɢjiːɾɒːt"),
    ("تأخیر", "tæʔxiːɾ"),
    ("تأمین", "tæʔmiːn"),
    ("تأسیس", "tæʔsiːs"),
    ("تألیف", "tæʔliːf"),
    ("مؤلف", "moʔællef"),
    ("مؤلفه", "moʔællefe"),
    ("مؤسسه", "moʔæssese"),
    ("رئیس", "ɾæʔiːs"),
    ("رؤیا", "ɾoʔjɒː"),
    ("سؤال", "soʔɒːl"),
    ("مسئول", "mæsʔuːl"),
    ("جرئت", "d͡ʒoɾʔæt"),
    ("هیئت", "hejʔæt"),
    ("تشکر", "tæʃækkoɾ"),
    ("تحول", "tæhævvol"),
    ("تقدم", "tæɢæddom"),
    ("تأخر", "tæʔæxxoɾ"),
    ("اتفاق", "ettefɒːɢ"),
    ("اطلاع", "ettelɒːʔ"),
    ("مطلع", "mottæleʔ"),
    ("مقدماتی", "moɢæddæmɒːtiː"),
    ("پیشرفته", "piːʃɾæfte"),
    ("معدل", "moʔæddel"),
    ("معلم", "moʔællem"),
    ("محصل", "mohæssel"),
    ("مفصل", "mofæssæl"),
    ("مرتب", "moɾættæb"),
    ("منظم", "monæzzæm"),
    ("مشخص", "moʃæxxæs"),
    ("هوش", "huːʃ"),
    ("یادگیری", "jɒːdɡiːɾiː"),
    ("عمیق", "ʔæmiːɢ"),
    ("زبان", "zæbɒːn"),
    ("گفتار", "ɡoftɒːɾ"),
    ("دانشگاه", "dɒːneʃɡɒːh"),
    ("تهران", "tæhɾɒːn"),
    ("واج", "vɒːd͡ʒ"),
    ("آموزش", "ʔɒːmuːzeʃ"),
    ("داده", "dɒːde"),
    ("ارزیابی", "ʔæɾzjɒːbiː"),
    ("خطا", "xætɒː"),
    ("وزن", "væzn"),
    ("تابع", "tɒːbeʔ"),
    ("پایگاه", "pɒːjɡɒːh"),
]

def load_tsv(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            w, p = line.split("\t", 1)
            primary_p = p.split(",")[0].strip().replace(" ", "")
            all_variants = [v.strip().replace(" ", "") for v in p.split(",") if v.strip()]
            items.append((w.strip(), primary_p, all_variants))
    return items

def main():
    wiktionary_train = load_tsv(os.path.join(CHARSIU_DATA, "train", "fas.tsv"))
    wiktionary_dev = load_tsv(os.path.join(CHARSIU_DATA, "dev", "fas.tsv"))
    wiktionary_test = load_tsv(os.path.join(CHARSIU_DATA, "test", "fas.tsv"))

    # Held-out challenge test set (60 words, 15 per category)
    challenge_flat = []
    for cat, triples in HELDOUT_CHALLENGE_SET.items():
        for w, p, translit in triples:
            challenge_flat.append((w, p, cat))

    # Curated training lexicon: 60 domain seed words + entries 25..199 of additional curated lexicon.
    # Entries [:25] of additional curated lexicon are reserved exclusively for dev/validation.
    val_curated = ADDITIONAL_CURATED_LEXICON[:25]
    curated_train = [(w, p) for cat, pairs in DOMAIN_SEED_SET.items() for w, p in pairs] + ADDITIONAL_CURATED_LEXICON[25:]

    # Expanded training TSV: 7,149 Wiktionary + 6x domain weighting of 235 curated pairs = 8,559 pairs
    train_out = os.path.join(OUT_DATA_DIR, "persian_g2p_train_expanded.tsv")
    train_words = set()
    with open(train_out, "w", encoding="utf-8") as f:
        for w, p, _ in wiktionary_train:
            f.write(f"{w}\t{p}\n")
            train_words.add(w)
        for _ in range(6):
            for w, p in curated_train:
                f.write(f"{w}\t{p}\n")
                train_words.add(w)

    # Write dev TSV: 50 Wiktionary dev + 25 held-out curated dev pairs = 75 pairs
    dev_out = os.path.join(OUT_DATA_DIR, "persian_g2p_dev.tsv")
    with open(dev_out, "w", encoding="utf-8") as f:
        for w, p, _ in wiktionary_dev:
            f.write(f"{w}\t{p}\n")
        for w, p in val_curated:
            f.write(f"{w}\t{p}\n")

    # Write Wiktionary 500-word test TSV
    test_wiki_out = os.path.join(OUT_DATA_DIR, "persian_g2p_test_wiktionary500.tsv")
    with open(test_wiki_out, "w", encoding="utf-8") as f:
        for w, p, variants in wiktionary_test:
            f.write(f"{w}\t{','.join(variants)}\n")

    # Write 60-word Stratified Held-out Challenge test TSV (with category column)
    test_chal_out = os.path.join(OUT_DATA_DIR, "persian_g2p_test_challenge60.tsv")
    with open(test_chal_out, "w", encoding="utf-8") as f:
        for w, p, cat in challenge_flat:
            f.write(f"{w}\t{p}\t{cat}\n")

    # Leakage assertions
    leak_chal = [w for w, _, _ in challenge_flat if w in train_words]
    leak_val = [w for w, _ in val_curated if w in train_words]
    assert not leak_chal, f"Data contamination detected: {leak_chal}"
    assert not leak_val, f"Validation contamination detected: {leak_val}"

    print(f"[+] Saved Expanded Train Set : {len(wiktionary_train) + 6 * len(curated_train)} instances ({len(wiktionary_train)} Wiktionary + {len(curated_train)}x6 Curated)")
    print(f"[+] Saved Validation Set     : {len(wiktionary_dev) + len(val_curated)} instances")
    print(f"[+] Saved Wiktionary Test Set: {len(wiktionary_test)} instances")
    print(f"[+] Saved Challenge Test Set : {len(challenge_flat)} instances across {len(HELDOUT_CHALLENGE_SET)} categories")
    print(f"[+] Contamination check passed: 0 challenge words and 0 validation words in training set.")

if __name__ == "__main__":
    main()
