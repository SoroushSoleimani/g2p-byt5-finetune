#!/usr/bin/env python3
"""
Builds the Large-Scale Persian G2P Training Dataset using MahtaFetrat/KaamelDict (116K+ words)
augmented with curated modern AI/tech domain terminology, ensuring zero data leakage
against both the 500-word Wiktionary test benchmark and the 60-word held-out challenge suite.
"""

import os
import sys
import ast
import random
import pyarrow.ipc as ipc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heldout_challenge_set import HELDOUT_CHALLENGE_SET
from build_expanded_dataset import DOMAIN_SEED_SET, ADDITIONAL_CURATED_LEXICON

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SAMPA_TO_IPA = {
    "a": "æ",
    "e": "e",
    "o": "o",
    "A": "ɒː",
    "i": "iː",
    "u": "uː",
    "S": "ʃ",
    "Z": "ʒ",
    "C": "t͡ʃ",
    "j": "d͡ʒ",
    "y": "j",
    "q": "ɢ",
    "?": "ʔ",
    "x": "x",
    "b": "b",
    "d": "d",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "t": "t",
    "v": "v",
    "z": "z",
}

def sampa_tuple_to_ipa(grapheme, sampa_tokens):
    tokens = [t for t in sampa_tokens if t and t.strip()]
    # Strip initial glottal stop if Persian word starts with alef
    if grapheme and grapheme[0] in ("ا", "آ") and tokens and tokens[0] == "?":
        tokens = tokens[1:]
    return "".join(SAMPA_TO_IPA.get(t, t) for t in tokens)

def main():
    print("Loading held-out test benchmarks to enforce zero leakage...")
    challenge_words = set()
    for cat, pairs in HELDOUT_CHALLENGE_SET.items():
        for w, ipa, rom in pairs:
            challenge_words.add(w.strip())

    wikt_words = set()
    wikt_path = os.path.join(DATA_DIR, "persian_g2p_test_wiktionary500.tsv")
    with open(wikt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                wikt_words.add(line.split("\t")[0].strip())

    print(f"Loaded {len(challenge_words)} challenge words and {len(wikt_words)} Wiktionary test words.")

    arrow_file = "/home/soroush/.cache/huggingface/datasets/MahtaFetrat___kaamel_dict/default/0.0.0/ecde21ffaaf2968c616b96ded9451e002f0d9c1c/kaamel_dict-train.arrow"
    print(f"Reading KaamelDict directly from Arrow cache: {arrow_file}")
    with open(arrow_file, "rb") as f:
        table = ipc.open_stream(f).read_all()

    raw_graphemes = table["grapheme"].to_pylist()
    raw_phonemes = table["phoneme"].to_pylist()
    print(f"Total raw rows in KaamelDict: {len(raw_graphemes)}")

    seen_words = set()
    kaamel_pairs = []
    skipped_challenge = 0
    skipped_wikt = 0
    parse_errors = 0

    for g_raw, p_raw in zip(raw_graphemes, raw_phonemes):
        g = g_raw.strip()
        if not g:
            continue
        if g in challenge_words:
            skipped_challenge += 1
            continue
        if g in wikt_words:
            skipped_wikt += 1
            continue
        if g in seen_words:
            continue

        try:
            p_list = ast.literal_eval(p_raw)
            if not p_list or not isinstance(p_list, list):
                continue
            primary_tokens = p_list[0]
            ipa = sampa_tuple_to_ipa(g, primary_tokens)
            if not ipa:
                continue
            kaamel_pairs.append((g, ipa))
            seen_words.add(g)
        except Exception:
            parse_errors += 1

    print(f"KaamelDict valid unique pairs: {len(kaamel_pairs)}")
    print(f"Excluded challenge suite words: {skipped_challenge}")
    print(f"Excluded Wiktionary test words: {skipped_wikt}")
    print(f"Parse errors: {parse_errors}")

    # Add Curated AI/Tech and Linguistic Lexicon
    curated_added = 0
    for cat, pairs in DOMAIN_SEED_SET.items():
        for w, ipa in pairs:
            w_clean = w.strip()
            if w_clean not in challenge_words and w_clean not in wikt_words and w_clean not in seen_words:
                kaamel_pairs.append((w_clean, ipa.strip()))
                seen_words.add(w_clean)
                curated_added += 1

    for w, ipa in ADDITIONAL_CURATED_LEXICON:
        w_clean = w.strip()
        if w_clean not in challenge_words and w_clean not in wikt_words and w_clean not in seen_words:
            kaamel_pairs.append((w_clean, ipa.strip()))
            seen_words.add(w_clean)
            curated_added += 1

    print(f"Curated domain loanword pairs added: {curated_added}")
    print(f"Total unified corpus size: {len(kaamel_pairs)}")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(kaamel_pairs)

    # Split: 1,000 dev, rest train
    dev_pairs = kaamel_pairs[:1000]
    train_pairs = kaamel_pairs[1000:]

    train_words = {p[0] for p in train_pairs}
    dev_words = {p[0] for p in dev_pairs}

    # Strict Zero-Leakage Verification
    assert len(train_words.intersection(challenge_words)) == 0, "DATA LEAK: Challenge word found in train!"
    assert len(train_words.intersection(wikt_words)) == 0, "DATA LEAK: Wiktionary word found in train!"
    assert len(dev_words.intersection(challenge_words)) == 0, "DATA LEAK: Challenge word found in dev!"
    assert len(dev_words.intersection(wikt_words)) == 0, "DATA LEAK: Wiktionary word found in dev!"
    assert len(train_words.intersection(dev_words)) == 0, "DATA LEAK: Dev word found in train!"
    print("PASSED: 100% Zero-Leakage assertions verified successfully.")

    train_out = os.path.join(DATA_DIR, "persian_g2p_train_kaamel.tsv")
    dev_out = os.path.join(DATA_DIR, "persian_g2p_dev_kaamel.tsv")

    with open(train_out, "w", encoding="utf-8") as f:
        for g, p in train_pairs:
            f.write(f"{g}\t{p}\n")

    with open(dev_out, "w", encoding="utf-8") as f:
        for g, p in dev_pairs:
            f.write(f"{g}\t{p}\n")

    print(f"Saved Kaamel train dataset to {train_out} ({len(train_pairs):,} entries)")
    print(f"Saved Kaamel dev dataset to {dev_out} ({len(dev_pairs):,} entries)")

if __name__ == "__main__":
    main()
