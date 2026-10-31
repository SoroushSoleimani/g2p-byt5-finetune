#!/usr/bin/env python3
"""
Generates the Complete 19-Cell Persian G2P Large-Scale Fine-Tuning Jupyter Notebook
with MahtaFetrat/KaamelDict (116K+ words), CharsiuG2P, ByT5, 9 Evaluation Metrics,
and 9 Publication-Grade Figures at 300 DPI.
"""

import os
import re
import sys
import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_expanded_dataset import DOMAIN_SEED_SET as _SEED, ADDITIONAL_CURATED_LEXICON as _CUR
from heldout_challenge_set import HELDOUT_CHALLENGE_SET as _HELD

def _fmt_dict(name, d, comment):
    lines = [f"# {comment}", f"{name} = {{"]
    for k, items in d.items():
        lines.append(f"    {k!r}: [")
        lines += [f"        {tuple(it)!r}," for it in items]
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)

DATA_LISTS_SRC = "\n\n".join([
    _fmt_dict("DOMAIN_SEED_SET", _SEED, "1. Curated AI & Technical Domain Seed Lexicon (60 words across 4 categories)"),
    "# 2. Additional Curated AI & Technical Loanword Lexicon (190 words)\nADDITIONAL_CURATED_LEXICON = [\n" + "\n".join(f"    {tuple(it)!r}," for it in _CUR) + "\n]",
    _fmt_dict("HELDOUT_CHALLENGE_SET", _HELD, "3. Held-Out Stratified Challenge Suite (60 words across 4 categories, 100% unseen)"),
])

def add_md(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    source = source.replace("__DATA_LISTS__", DATA_LISTS_SRC)
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

# --- CELL 1: Title & Overview ---
add_md("""# Large-Scale Persian (Farsi) Grapheme-to-Phoneme (G2P) Fine-Tuning on KaamelDict (116K+ Words) & Comprehensive Benchmark Suite
### Built on [CharsiuG2P (Interspeech 2022)](https://github.com/lingjzhu/CharsiuG2P), [KaamelDict (Fetrat et al.)](https://huggingface.co/datasets/MahtaFetrat/KaamelDict), and Byte-Level Transformer Seq2Seq (`ByT5`)

---

### Executive Summary & Architectural Scope
**Grapheme-to-Phoneme (G2P)** conversion maps written orthographic sequences into International Phonetic Alphabet (IPA) representations. For the **Persian (Farsi)** language (`<fas>`), neural G2P conversion addresses four core linguistic challenges:
1. **Unvocalized Short Vowels (Diacritic Omission)**: Standard Persian orthography omits short vowels (`/æ/` Fathe, `/e/` Kasre, `/o/` Zamme), causing pervasive homographic ambiguity in words like *شبکه* (`ʃæbæke`), *مدل* (`model`), *پژوهش* (`pæʒuːheʃ`).
2. **Compound Words & Ezafe Morphophonemics**: Unwritten enclitic vowels (`-e` after consonants, `-je` after vowels) link noun-modifier constructions, e.g., *یادگیری عمیق* (`jɒːdɡiːɾiːjeʔæmiːɢ`), *مدل آماری* (`modeleɒːmɒːɾiː`).
3. **Gemination (Tashdid) & Glottal Stops (Hamza / Ayn)**: Arabic-derived loanwords and morphological templates (`تَفَعُّل`, `مُفَعَّل`, `اِفتِعال`) feature doubled consonants and glottal stops (`/ʔ/`), e.g., *دقت* (`deɢɢæt`), *تعامل* (`tæʔɒːmol`), *تأکید* (`tæʔkiːd`).
4. **Modern AI, Computing & Technical Loanwords**: Rapidly evolving deep learning terminology (*Algorithm, PyTorch, TensorFlow, Transformer, Kaggle, Keras, Tokenizer, Convolution, Processor, Database*) requires specialized phonetic adaptation.

### Key Highlights of This Large-Scale System
* **Authoritative Large-Scale Training Corpus ($N = 115,221$ pairs)**: Constructed from the complete `MahtaFetrat/KaamelDict` corpus (116,629 entries compiled from 10 major Persian phonology lexicons including the Academy of Persian Language and Literature, Sharif GE2PE, Tihu, and Wiktionary) augmented with curated modern AI and computing loanwords.
* **Dual Zero-Leakage Evaluation Benchmarks ($N = 560$ test words)**:
  1. **Official Wiktionary Unseen Test Split ($N = 500$ words)**: Benchmark from CharsiuG2P evaluating broad vocabulary generalization.
  2. **Held-Out Stratified Challenge Suite ($N = 60$ words across 4 categories, 100% unseen during training)**: High-precision evaluation across *AI & Tech Loanwords*, *Short-Vowel Ambiguity*, *Compound & Ezafe Constructions*, and *Gemination & Glottal Stops*.
* **9 Phonetic & Sequence Evaluation Metrics**:
  * Phoneme Error Rate (**PER %**), Word Error Rate (**WER %**), Exact Word Accuracy (**WAcc %**), Normalized Edit Similarity (**Sim %**), Phonetic Character N-Gram F1 (**chrF %**), and Levenshtein Edit Operation Decomposition (**Substitution %**, **Deletion %**, **Insertion %**), plus **Vowel Confusion Matrices** and **Length-Stratified Error Analysis**.
* **9 High-Resolution (300 DPI) Visualizations**: Exported automatically to `./figures/` for publication and technical reporting.""")

# --- CELL 2: Environment & Hardware Verification ---
add_code("""# Step 1: Hardware Acceleration & Environment Setup
import os
import re
import sys
import time
import json
import math
import inspect
from collections import Counter
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

print(f"Python Version  : {sys.version.split()[0]}")
print(f"PyTorch Version : {torch.__version__}")
gpu_available = torch.cuda.is_available()
print(f"GPU Available   : {gpu_available}")

if gpu_available:
    device = torch.device("cuda")
    print(f"Active GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    device = torch.device("cpu")
    print("WARNING: GPU not detected. Running on CPU fallback.")""")

# --- CELL 3: Package Installations & Library Imports ---
add_code("""# Step 2: Install and Import Specialized Libraries
!pip install -q accelerate evaluate jiwer tabulate matplotlib seaborn datasets
import seaborn as sns
from tabulate import tabulate
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)
print("Libraries imported successfully. Output directories initialized.")""")

# --- CELL 4: Data Loading & Lexicon Construction ---
add_md("""## Section 1: Loading KaamelDict (116K+ Entries) & Zero-Leakage Benchmark Auditing
We load the authoritative `MahtaFetrat/KaamelDict` corpus (~116,629 word-phoneme entries), convert its SAMPA phonetic encoding into standard IPA, and clone the official [CharsiuG2P repository](https://github.com/lingjzhu/CharsiuG2P) for the 500-word unseen Wiktionary test split.
A separate 60-word held-out challenge suite across 4 linguistic categories is evaluated. Strict automated assertions verify that zero challenge or test words leak into training or validation splits.""")

# --- CELL 5: Code Data Preparation ---
add_code("""# Step 3: Clone CharsiuG2P & Construct Unified Corpus
if not os.path.exists("CharsiuG2P"):
    !git clone --depth 1 https://github.com/lingjzhu/CharsiuG2P.git
    print("CharsiuG2P repository cloned successfully.")
else:
    print("CharsiuG2P repository already present.")

__DATA_LISTS__

# Load Wiktionary test split (official 500-word benchmark)
def load_charsiu_tsv(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\\t" not in line:
                continue
            w, p = line.split("\\t", 1)
            variants = [v.strip().replace(" ", "") for v in p.split(",") if v.strip()]
            entries.append((w.strip(), variants[0], variants))
    return entries

wiktionary_test = load_charsiu_tsv("CharsiuG2P/data/test/fas.tsv")
wiktionary_test_words = {w for w, _, _ in wiktionary_test}

# Held-out challenge set (60 words across 4 categories)
challenge_items = []
for cat, triples in HELDOUT_CHALLENGE_SET.items():
    for w, p, _ in triples:
        challenge_items.append((w.strip(), p.strip(), [p.strip()], cat))
challenge_words = {w for w, _, _, _ in challenge_items}
challenge_translit = [t for triples in HELDOUT_CHALLENGE_SET.values() for _, _, t in triples]

# SAMPA to Standard IPA Bijective Mapping
SAMPA_TO_IPA = {
    "a": "æ", "e": "e", "o": "o", "A": "ɒː", "i": "iː", "u": "uː",
    "S": "ʃ", "Z": "ʒ", "C": "t͡ʃ", "j": "d͡ʒ", "y": "j", "q": "ɢ",
    "?": "ʔ", "x": "x", "b": "b", "d": "d", "f": "f", "g": "ɡ",
    "h": "h", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p",
    "r": "ɾ", "s": "s", "t": "t", "v": "v", "z": "z"
}

def sampa_tuple_to_ipa(grapheme, sampa_tokens):
    tokens = [t for t in sampa_tokens if t and t.strip()]
    if grapheme and grapheme[0] in ("ا", "آ") and tokens and tokens[0] == "?":
        tokens = tokens[1:]
    return "".join(SAMPA_TO_IPA.get(t, t) for t in tokens)

print("Loading MahtaFetrat/KaamelDict from Hugging Face...")
import ast
kaamel_ds = load_dataset("MahtaFetrat/KaamelDict")["train"]
print(f"Loaded KaamelDict raw rows: {len(kaamel_ds):,}")

seen_words = set()
kaamel_clean_pairs = []
skipped_chal = 0
skipped_wikt = 0

for row in kaamel_ds:
    g = row["grapheme"].strip()
    if not g:
        continue
    if g in challenge_words:
        skipped_chal += 1
        continue
    if g in wiktionary_test_words:
        skipped_wikt += 1
        continue
    if g in seen_words:
        continue
    try:
        p_list = ast.literal_eval(row["phoneme"])
        if not p_list or not isinstance(p_list, list):
            continue
        ipa = sampa_tuple_to_ipa(g, p_list[0])
        if ipa:
            kaamel_clean_pairs.append((g, ipa))
            seen_words.add(g)
    except Exception:
        pass

print(f"Clean unique KaamelDict pairs: {len(kaamel_clean_pairs):,}")
print(f"Excluded challenge suite words: {skipped_chal} (enforcing zero leakage)")
print(f"Excluded Wiktionary test words: {skipped_wikt} (enforcing zero leakage)")

# Augment with curated domain and linguistic lexicon
curated_added = 0
for cat, pairs in DOMAIN_SEED_SET.items():
    for w, ipa in pairs:
        w_c = w.strip()
        if w_c not in challenge_words and w_c not in wiktionary_test_words and w_c not in seen_words:
            kaamel_clean_pairs.append((w_c, ipa.strip()))
            seen_words.add(w_c)
            curated_added += 1

for w, ipa in ADDITIONAL_CURATED_LEXICON:
    w_c = w.strip()
    if w_c not in challenge_words and w_c not in wiktionary_test_words and w_c not in seen_words:
        kaamel_clean_pairs.append((w_c, ipa.strip()))
        seen_words.add(w_c)
        curated_added += 1

print(f"Added {curated_added} curated domain loanwords and linguistic seed pairs.")
print(f"Total unified corpus: {len(kaamel_clean_pairs):,} pairs.")

# Shuffle and split into 1,000 Dev and rest Train
rng_split = np.random.default_rng(42)
shuffled_idx = rng_split.permutation(len(kaamel_clean_pairs))
val_pairs = [kaamel_clean_pairs[i] for i in shuffled_idx[:1000]]
train_pairs = [kaamel_clean_pairs[i] for i in shuffled_idx[1000:]]

train_words = {w for w, _ in train_pairs}
val_words = {w for w, _ in val_pairs}

# Strict Zero-Leakage Assertions
assert len(train_words.intersection(challenge_words)) == 0, "DATA LEAK: Challenge word found in train!"
assert len(train_words.intersection(wiktionary_test_words)) == 0, "DATA LEAK: Wiktionary word found in train!"
assert len(val_words.intersection(challenge_words)) == 0, "DATA LEAK: Challenge word found in dev!"
assert len(val_words.intersection(wiktionary_test_words)) == 0, "DATA LEAK: Wiktionary word found in dev!"
assert len(train_words.intersection(val_words)) == 0, "DATA LEAK: Validation word found in train!"

print("\\n[+] Zero-Leakage Audit PASSED: 100% verified.")
print(f" - Final Training Corpus   : {len(train_pairs):,} word-phoneme pairs")
print(f" - Validation Split (Dev)  : {len(val_pairs):,} pairs")
print(f" - Wiktionary Test Split   : {len(wiktionary_test):,} pairs")
print(f" - Challenge Suite         : {len(challenge_items):,} pairs across {len(HELDOUT_CHALLENGE_SET)} categories")""")

# --- CELL 6: Markdown Section 2 Corpus EDA ---
add_md("""## Section 2: Corpus Exploratory Data Analysis (EDA) & Phoneme Distribution
We analyze the structural properties of our 116,000-instance Persian G2P corpus:
1. **Grapheme vs. Phoneme Length Correlation**: Persian omits short vowels (`/æ/`, `/e/`, `/o/`) in writing, which causes phonetic sequences to be consistently longer than orthographic sequences.
2. **Empirical Phoneme Distribution**: Tokenizing all IPA transcriptions into atomic phoneme units reveals the most frequent vowels and consonants in standard Persian.""")

# --- CELL 7: Code Figure 1 Corpus EDA ---
add_code("""# Figure 1: Corpus Exploratory Data Analysis & Phoneme Distribution
IPA_TOKEN_REGEX = re.compile(r"t͡ʃ|d͡ʒ|ɒː|iː|uː|ɔː|oʊ|eɪ|.")

def tokenize_ipa(ipa_str):
    cleaned = ipa_str.replace(" ", "").replace("ˈ", "").replace("ˌ", "").replace(".", "")
    return [t for t in IPA_TOKEN_REGEX.findall(cleaned) if t.strip()]

# Sample 10,000 words for length regression visualization
sample_subset = [train_pairs[i] for i in rng_split.choice(len(train_pairs), size=min(10000, len(train_pairs)), replace=False)]
grapheme_lengths = np.array([len(w) for w, _ in sample_subset])
phoneme_tokens_list = [tokenize_ipa(p) for _, p in sample_subset]
phoneme_lengths = np.array([len(toks) for toks in phoneme_tokens_list])

phoneme_counter = Counter()
for _, p in train_pairs[:20000]:
    phoneme_counter.update(tokenize_ipa(p))

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

# Panel A: Grapheme vs Phoneme Length with Linear Regression
mask = (grapheme_lengths <= 20) & (phoneme_lengths <= 25)
x_g = grapheme_lengths[mask]
y_p = phoneme_lengths[mask]

jitter_x = x_g + rng_split.normal(0, 0.18, size=len(x_g))
jitter_y = y_p + rng_split.normal(0, 0.18, size=len(y_p))

ax1.scatter(jitter_x, jitter_y, alpha=0.15, s=12, color='#2563EB', edgecolors='none', label='Corpus Sample (N=10,000)')
slope, intercept = np.polyfit(x_g, y_p, 1)
r_corr = np.corrcoef(x_g, y_p)[0, 1]
x_line = np.linspace(1, 18, 100)
ax1.plot(x_line, slope * x_line + intercept, color='#DC2626', linewidth=2.5,
         label=f'Fit: y = {slope:.2f}x + {intercept:.2f} (R = {r_corr:.2f})')
ax1.plot(x_line, x_line, color='#64748B', linestyle='--', linewidth=1.5, label='1:1 Line (y = x)')

ax1.set_title("(A) Persian Grapheme Length vs. IPA Phoneme Length", fontsize=12.5, fontweight='bold', pad=10)
ax1.set_xlabel("Orthographic Word Length (Characters)", fontsize=11)
ax1.set_ylabel("Phonetic Length (IPA Phonemes)", fontsize=11)
ax1.set_xlim(1, 18)
ax1.set_ylim(1, 24)
ax1.legend(fontsize=9.5, loc='upper left', frameon=True)

# Panel B: Top 20 Persian IPA Phonemes
top20 = phoneme_counter.most_common(20)
ph_labels = [f"/{k}/" for k, _ in top20]
ph_counts = [v for _, v in top20]
vowels_set = {'æ', 'e', 'o', 'ɒː', 'iː', 'uː', 'oʊ'}
bar_colors = ['#10B981' if k in vowels_set else '#3B82F6' for k, _ in top20]

ax2.barh(range(len(top20)), ph_counts, color=bar_colors, edgecolor='#1E293B', alpha=0.9, height=0.7)
ax2.set_yticks(range(len(top20)))
ax2.set_yticklabels(ph_labels, fontsize=10, fontweight='bold')
ax2.invert_yaxis()
ax2.set_title("(B) Top 20 Persian Phonemes in KaamelDict (Vowels Green, Consonants Blue)", fontsize=12.5, fontweight='bold', pad=10)
ax2.set_xlabel("Occurrences in Sample", fontsize=11)

for i, c in enumerate(ph_counts):
    ax2.text(c + 100, i, f"{c:,}", va='center', fontsize=8.5, fontweight='bold', color='#1E293B')
ax2.set_xlim(0, max(ph_counts) * 1.15)

plt.tight_layout()
fig.savefig("figures/dataset_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"[+] Saved 'figures/dataset_distribution.png' | Mean Grapheme: {grapheme_lengths.mean():.2f} | Mean Phoneme: {phoneme_lengths.mean():.2f}")""")

# --- CELL 8: Markdown Section 3 Metrics Suite ---
add_md("""## Section 3: Comprehensive Phonetic Evaluation Metrics Suite
To rigorously benchmark model quality before and after fine-tuning, we implement **9 complementary metrics**:
1. **Phoneme Error Rate (PER %)**: $\\text{PER} = \\frac{S + D + I}{N} \\times 100$, where $S$, $D$, $I$ are substitutions, deletions, and insertions, and $N$ is the reference length.
2. **Word Error Rate (WER %)**: Fraction of words with at least one phonetic mismatch.
3. **Word Accuracy (WAcc %)**: $100 - \\text{WER}$, percentage of entirely error-free phonetic transcriptions.
4. **Normalized Sequence Similarity (Sim %)**: $(1 - \\frac{\\text{dist}}{\\max(|P|, |R|)}) \\times 100$.
5. **Character N-Gram F1 (chrF %)**: Geometric mean of n-gram precision and recall ($n=1..4$) over IPA characters.
6. **Substitution Rate (Sub %)**: Fraction of reference phonemes erroneously substituted.
7. **Deletion Rate (Del %)**: Fraction of reference phonemes omitted by the model.
8. **Insertion Rate (Ins %)**: Spurious phonemes inserted relative to reference length.
9. **Average Levenshtein Distance**: Mean phonetic edit distance per word.""")

# --- CELL 9: Code Metrics Implementation ---
add_code("""# Step 4: Metric Functions and Levenshtein Backtracking Alignment
def levenshtein_alignment(pred_seq, ref_seq):
    \"\"\"Dynamic programming table computing edit distance and optimal alignment path.\"\"\"
    m, n = len(pred_seq), len(ref_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_seq[i - 1] == ref_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    sub, dele, ins = 0, 0, 0
    aligned_pairs = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred_seq[i - 1] == ref_seq[j - 1]:
            aligned_pairs.append((ref_seq[j - 1], pred_seq[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            aligned_pairs.append((ref_seq[j - 1], pred_seq[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ins += 1
            aligned_pairs.append(("∅", pred_seq[i - 1]))
            i -= 1
        else:
            dele += 1
            aligned_pairs.append((ref_seq[j - 1], "∅"))
            j -= 1
    return dp[m][n], sub, dele, ins, aligned_pairs[::-1]

def compute_chrf_single(pred, ref, max_n=4):
    \"\"\"Computes character n-gram F1 (chrF) score.\"\"\"
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    f1_scores = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter([pred[i:i+n] for i in range(max(1, len(pred) - n + 1))])
        ref_ngrams = Counter([ref[i:i+n] for i in range(max(1, len(ref) - n + 1))])
        overlap = sum((pred_ngrams & ref_ngrams).values())
        prec = overlap / max(sum(pred_ngrams.values()), 1)
        rec = overlap / max(sum(ref_ngrams.values()), 1)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores))

def evaluate_predictions_comprehensive(predictions, references_list, words_list=None):
    \"\"\"Evaluates predictions against reference variants across all 9 phonetic metrics.\"\"\"
    total_dist = total_sub = total_del = total_ins = total_ref_len = 0
    word_errors = 0
    sim_scores = []
    chrf_scores = []
    per_word_records = []
    all_aligned_token_pairs = []

    for idx, (pred, ref_variants) in enumerate(zip(predictions, references_list)):
        if isinstance(ref_variants, str):
            ref_variants = [ref_variants]
        pred_clean = pred.strip().replace(" ", "")
        best_dist = None
        best_ref = ref_variants[0]
        best_s = best_d = best_i = 0
        for rv in ref_variants:
            rv_clean = rv.strip().replace(" ", "")
            d, s, dl, ins, _ = levenshtein_alignment(pred_clean, rv_clean)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_ref = rv_clean
                best_s, best_d, best_i = s, dl, ins

        pred_toks = tokenize_ipa(pred_clean)
        ref_toks = tokenize_ipa(best_ref)
        _, _, _, _, tok_pairs = levenshtein_alignment(pred_toks, ref_toks)
        all_aligned_token_pairs.extend(tok_pairs)

        ref_len = max(len(best_ref), 1)
        total_dist += best_dist
        total_sub += best_s
        total_del += best_d
        total_ins += best_i
        total_ref_len += ref_len

        is_err = 1 if pred_clean != best_ref else 0
        word_errors += is_err

        max_l = max(len(pred_clean), len(best_ref), 1)
        sim = (1.0 - (best_dist / max_l)) * 100.0
        chrf = compute_chrf_single(pred_clean, best_ref) * 100.0
        sim_scores.append(sim)
        chrf_scores.append(chrf)

        word_str = words_list[idx] if words_list is not None else str(idx)
        per_word_records.append({
            "word": word_str,
            "word_len": len(word_str),
            "ref": best_ref,
            "pred": pred_clean,
            "edit_dist": best_dist,
            "sub": best_s,
            "del": best_d,
            "ins": best_i,
            "per": (best_dist / ref_len) * 100.0,
            "wer": is_err * 100.0,
            "sim": sim,
            "chrf": chrf
        })

    n_words = max(len(predictions), 1)
    return {
        "PER": (total_dist / max(total_ref_len, 1)) * 100.0,
        "WER": (word_errors / n_words) * 100.0,
        "WAcc": (1.0 - (word_errors / n_words)) * 100.0,
        "Sim": float(np.mean(sim_scores)),
        "chrF": float(np.mean(chrf_scores)),
        "Sub_Rate": (total_sub / max(total_ref_len, 1)) * 100.0,
        "Del_Rate": (total_del / max(total_ref_len, 1)) * 100.0,
        "Ins_Rate": (total_ins / max(total_ref_len, 1)) * 100.0,
        "Avg_Edit_Dist": total_dist / n_words,
        "records": per_word_records,
        "token_pairs": all_aligned_token_pairs
    }
print("Phonetic evaluation suite ready.")""")

# --- CELL 10: Markdown Section 4 Baseline Evaluation ---
add_md("""## Section 4: Pre-Fine-Tuning Baseline Evaluation
We load the pretrained `charsiu/g2p_multilingual_byT5_tiny_16_layers_100` model with `<fas>: ` language prefix and evaluate its zero-shot performance across both benchmarks before fine-tuning:
1. **Unseen Wiktionary Test Benchmark ($N = 500$ words)**
2. **Held-Out Stratified Challenge Suite ($N = 60$ words across 4 categories)**""")

# --- CELL 11: Code Baseline Evaluation ---
add_code("""# Step 5: Pretrained Model Loading & Baseline Inference
MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
TOKENIZER_ID = "google/byt5-small"

print(f"Loading pretrained base model: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,} | Trainable Parameters: {trainable_params:,}")

def batch_predict_g2p(words, batch_size=64):
    all_preds = []
    model.eval()
    for i in range(0, len(words), batch_size):
        batch_w = words[i:i + batch_size]
        prefixed = [f"<fas>: {w}" for w in batch_w]
        encoded = tokenizer(
            prefixed,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            gen_ids = model.generate(**encoded, num_beams=1, max_length=64)
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_preds.extend([d.strip().replace(" ", "") for d in decoded])
    return all_preds

wiki_test_words = [w for w, _, _ in wiktionary_test]
wiki_test_refs = [variants for _, _, variants in wiktionary_test]

chal_test_words = [w for w, _, _, _ in challenge_items]
chal_test_refs = [variants for _, _, variants, _ in challenge_items]
chal_test_cats = [cat for _, _, _, cat in challenge_items]

t0 = time.time()
baseline_wiki_preds = batch_predict_g2p(wiki_test_words, batch_size=64)
baseline_chal_preds = batch_predict_g2p(chal_test_words, batch_size=64)
baseline_infer_time = time.time() - t0

baseline_wiki_metrics = evaluate_predictions_comprehensive(baseline_wiki_preds, wiki_test_refs, wiki_test_words)
baseline_chal_metrics = evaluate_predictions_comprehensive(baseline_chal_preds, chal_test_refs, chal_test_words)

print("\\n================ BASELINE EVALUATION (BEFORE FINE-TUNING) ================")
summary_rows = [
    ["Wiktionary Test (N=500)", f"{baseline_wiki_metrics['PER']:.2f}%", f"{baseline_wiki_metrics['WER']:.2f}%",
     f"{baseline_wiki_metrics['WAcc']:.2f}%", f"{baseline_wiki_metrics['Sim']:.2f}%", f"{baseline_wiki_metrics['chrF']:.2f}%",
     f"{baseline_wiki_metrics['Sub_Rate']:.2f}% / {baseline_wiki_metrics['Del_Rate']:.2f}% / {baseline_wiki_metrics['Ins_Rate']:.2f}%"],
    ["Challenge Suite (N=60)", f"{baseline_chal_metrics['PER']:.2f}%", f"{baseline_chal_metrics['WER']:.2f}%",
     f"{baseline_chal_metrics['WAcc']:.2f}%", f"{baseline_chal_metrics['Sim']:.2f}%", f"{baseline_chal_metrics['chrF']:.2f}%",
     f"{baseline_chal_metrics['Sub_Rate']:.2f}% / {baseline_chal_metrics['Del_Rate']:.2f}% / {baseline_chal_metrics['Ins_Rate']:.2f}%"]
]
print(tabulate(summary_rows, headers=["Benchmark", "PER", "WER", "WAcc", "Sim", "chrF", "Sub / Del / Ins"], tablefmt="grid"))""")

# --- CELL 12: Markdown Section 5 Large-Scale Fine-Tuning ---
add_md("""## Section 5: Large-Scale GPU Fine-Tuning on KaamelDict (115,221 Examples)
We fine-tune the byte-level `ByT5` sequence-to-sequence model on our authoritative **115,221-instance Persian phonology corpus**:
* **Training Corpus**: `115,221` verified Persian word-phoneme pairs
* **Validation Corpus**: `1,000` pairs evaluated periodically during training
* **Batch Size**: `64` (`per_device_train_batch_size=64`)
* **Learning Rate**: $3 \\times 10^{-4}$ with Cosine Annealing and 300 warmup steps
* **Epochs**: `3` full passes over the corpus (~5,400 optimization steps)
* **Precision**: Mixed Precision (`FP16`) on GPU""")

# --- CELL 13: Code Fine-Tuning Execution ---
add_code("""# Step 6: Dataset Formatting & GPU Fine-Tuning
rng_shuf = np.random.default_rng(42)
shuffled_indices = rng_shuf.permutation(len(train_pairs))
shuffled_train_pairs = [train_pairs[i] for i in shuffled_indices]

raw_train_ds = Dataset.from_dict({
    "input_text": [f"<fas>: {w}" for w, _ in shuffled_train_pairs],
    "target_text": [p for _, p in shuffled_train_pairs]
})
raw_val_ds = Dataset.from_dict({
    "input_text": [f"<fas>: {w}" for w, _ in val_pairs],
    "target_text": [p for _, p in val_pairs]
})

def preprocess_batch(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=64, truncation=True, padding=False)
    labels = tokenizer(text_target=examples["target_text"], max_length=64, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = raw_train_ds.map(preprocess_batch, batched=True, remove_columns=["input_text", "target_text"])
val_ds = raw_val_ds.map(preprocess_batch, batched=True, remove_columns=["input_text", "target_text"])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints_persian_g2p",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    warmup_steps=300,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=600,
    save_strategy="no",
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    report_to="none",
    seed=42
)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_ds,
    "eval_dataset": val_ds,
    "data_collator": data_collator
}

sig = inspect.signature(Seq2SeqTrainer.__init__)
if "processing_class" in sig.parameters:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in sig.parameters:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = Seq2SeqTrainer(**trainer_kwargs)

print(f"Starting Large-Scale Persian G2P Fine-Tuning on {device} ({len(train_ds):,} train samples, 3 epochs)...")
train_start = time.time()
train_result = trainer.train()
train_duration = time.time() - train_start

print(f"\\nFine-Tuning Completed in {train_duration:.2f} seconds ({train_duration/60:.2f} minutes)!")
print(f"Final Training Loss: {train_result.metrics.get('train_loss', 0.0):.4f}")""")

# --- CELL 14: Code Figure 2 Loss Curve ---
add_code("""# Figure 2: Dual-Axis Training & Validation Convergence + Cosine Learning Rate Schedule
log_history = trainer.state.log_history

train_steps = [entry["step"] for entry in log_history if "loss" in entry]
train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
learning_rates = [entry["learning_rate"] * 1e4 for entry in log_history if "loss" in entry and "learning_rate" in entry]

eval_steps = [entry["step"] for entry in log_history if "eval_loss" in entry]
eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

fig, ax1 = plt.subplots(figsize=(10.5, 5.5), dpi=300)

line1 = ax1.plot(train_steps, train_losses, color='#2563EB', linewidth=2.2, alpha=0.9, label='Training Cross-Entropy Loss')
if eval_steps:
    line2 = ax1.plot(eval_steps, eval_losses, color='#DC2626', linewidth=2.5, marker='s', markersize=6,
                     linestyle='--', label='Validation Loss (Dev Split N=1,000)')

ax1.set_title("Large-Scale Persian G2P Fine-Tuning Convergence on KaamelDict (115K Pairs)", fontsize=13.5, fontweight='bold', pad=14)
ax1.set_xlabel("Optimization Steps (Batch Size = 64, 3 Epochs)", fontsize=11.5)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=11.5, color='#1E293B')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
line3 = ax2.plot(train_steps[:len(learning_rates)], learning_rates, color='#10B981', linewidth=1.8,
                 linestyle=':', label='Learning Rate (x10^-4)')
ax2.set_ylabel("Learning Rate (x 10^-4)", fontsize=11, color='#059669')
ax2.grid(False)

lines = line1 + (line2 if eval_steps else []) + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
fig.savefig("figures/loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/loss_curve.png'")""")

# --- CELL 15: Markdown Section 6 Post-Fine-Tuning Evaluation ---
add_md("""## Section 6: Comprehensive Post-Fine-Tuning Evaluation & Multi-Benchmark Comparison
We now evaluate our fine-tuned Persian G2P model across both benchmarks (`N = 500` Unseen Wiktionary Test Set and `N = 60` Held-Out Stratified Challenge Suite) and compute exact empirical error reductions across all 9 metrics.""")

# --- CELL 16: Code Post-Fine-Tuning Evaluation & Tables ---
add_code("""# Step 7: Post-Fine-Tuning Evaluation on Wiktionary Test and Challenge Suite
t0 = time.time()
finetuned_wiki_preds = batch_predict_g2p(wiki_test_words, batch_size=64)
finetuned_chal_preds = batch_predict_g2p(chal_test_words, batch_size=64)
finetuned_infer_time = time.time() - t0

finetuned_wiki_metrics = evaluate_predictions_comprehensive(finetuned_wiki_preds, wiki_test_refs, wiki_test_words)
finetuned_chal_metrics = evaluate_predictions_comprehensive(finetuned_chal_preds, chal_test_refs, chal_test_words)

print("================ COMPREHENSIVE BENCHMARK COMPARISON (BEFORE VS. AFTER) ================")
comp_table = [
    ["Wiktionary Test (N=500) - PER (%)", f"{baseline_wiki_metrics['PER']:.2f}%", f"{finetuned_wiki_metrics['PER']:.2f}%",
     f"{((finetuned_wiki_metrics['PER'] - baseline_wiki_metrics['PER']) / max(baseline_wiki_metrics['PER'], 1e-6))*100:+.2f}%"],
    ["Wiktionary Test (N=500) - WER (%)", f"{baseline_wiki_metrics['WER']:.2f}%", f"{finetuned_wiki_metrics['WER']:.2f}%",
     f"{((finetuned_wiki_metrics['WER'] - baseline_wiki_metrics['WER']) / max(baseline_wiki_metrics['WER'], 1e-6))*100:+.2f}%"],
    ["Wiktionary Test (N=500) - Word Acc (%)", f"{baseline_wiki_metrics['WAcc']:.2f}%", f"{finetuned_wiki_metrics['WAcc']:.2f}%",
     f"{finetuned_wiki_metrics['WAcc'] - baseline_wiki_metrics['WAcc']:+.2f} pts"],
    ["Wiktionary Test (N=500) - Seq Sim (%)", f"{baseline_wiki_metrics['Sim']:.2f}%", f"{finetuned_wiki_metrics['Sim']:.2f}%",
     f"{finetuned_wiki_metrics['Sim'] - baseline_wiki_metrics['Sim']:+.2f} pts"],
    ["Wiktionary Test (N=500) - chrF (%)", f"{baseline_wiki_metrics['chrF']:.2f}%", f"{finetuned_wiki_metrics['chrF']:.2f}%",
     f"{finetuned_wiki_metrics['chrF'] - baseline_wiki_metrics['chrF']:+.2f} pts"],
    ["-"*36, "-"*18, "-"*18, "-"*18],
    ["Challenge Suite (N=60) - PER (%)", f"{baseline_chal_metrics['PER']:.2f}%", f"{finetuned_chal_metrics['PER']:.2f}%",
     f"{((finetuned_chal_metrics['PER'] - baseline_chal_metrics['PER']) / max(baseline_chal_metrics['PER'], 1e-6))*100:+.2f}%"],
    ["Challenge Suite (N=60) - WER (%)", f"{baseline_chal_metrics['WER']:.2f}%", f"{finetuned_chal_metrics['WER']:.2f}%",
     f"{((finetuned_chal_metrics['WER'] - baseline_chal_metrics['WER']) / max(baseline_chal_metrics['WER'], 1e-6))*100:+.2f}%"],
    ["Challenge Suite (N=60) - Word Acc (%)", f"{baseline_chal_metrics['WAcc']:.2f}%", f"{finetuned_chal_metrics['WAcc']:.2f}%",
     f"{finetuned_chal_metrics['WAcc'] - baseline_chal_metrics['WAcc']:+.2f} pts"],
    ["Challenge Suite (N=60) - Seq Sim (%)", f"{baseline_chal_metrics['Sim']:.2f}%", f"{finetuned_chal_metrics['Sim']:.2f}%",
     f"{finetuned_chal_metrics['Sim'] - baseline_chal_metrics['Sim']:+.2f} pts"],
    ["Challenge Suite (N=60) - chrF (%)", f"{baseline_chal_metrics['chrF']:.2f}%", f"{finetuned_chal_metrics['chrF']:.2f}%",
     f"{finetuned_chal_metrics['chrF'] - baseline_chal_metrics['chrF']:+.2f} pts"],
]
print(tabulate(comp_table, headers=["Benchmark & Metric", "Baseline (Before)", "Fine-Tuned (After)", "Change / Improvement"], tablefmt="grid"))

# Stratified Category Breakdown Table
cat_names = list(HELDOUT_CHALLENGE_SET.keys())
cat_breakdown = {}
cat_rows = []
for cat in cat_names:
    idxs = [i for i, c in enumerate(chal_test_cats) if c == cat]
    b_sub_preds = [baseline_chal_preds[i] for i in idxs]
    f_sub_preds = [finetuned_chal_preds[i] for i in idxs]
    sub_refs = [chal_test_refs[i] for i in idxs]
    sub_words = [chal_test_words[i] for i in idxs]
    bm = evaluate_predictions_comprehensive(b_sub_preds, sub_refs, sub_words)
    fm = evaluate_predictions_comprehensive(f_sub_preds, sub_refs, sub_words)
    cat_breakdown[cat] = {"baseline": bm, "finetuned": fm}
    cat_rows.append([
        cat,
        f"{bm['PER']:.2f}%", f"{fm['PER']:.2f}%",
        f"{bm['WER']:.2f}%", f"{fm['WER']:.2f}%",
        f"{bm['chrF']:.2f}%", f"{fm['chrF']:.2f}%"
    ])

print("\\n================ STRATIFIED LINGUISTIC CATEGORY BREAKDOWN (N=60) ================")
print(tabulate(cat_rows, headers=["Linguistic Category", "Base PER", "FT PER", "Base WER", "FT WER", "Base chrF", "FT chrF"], tablefmt="grid"))""")

# --- CELL 17: Code Figures 3, 4, 5 ---
add_code("""# Figures 3, 4, and 5: Multi-Metric Benchmarks, Category Breakdown, and Edit Operations

# --- Figure 3: Multi-Metric Benchmark Comparison (2 Panels) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

metric_labels = ['PER (%)\\n[Lower Better]', 'WER (%)\\n[Lower Better]', 'Word Acc (%)\\n[Higher Better]', 'Seq Sim (%)\\n[Higher Better]', 'chrF (%)\\n[Higher Better]']
x = np.arange(len(metric_labels))
width = 0.35

# Panel A: Challenge Suite (N=60)
b_chal_vals = [baseline_chal_metrics['PER'], baseline_chal_metrics['WER'], baseline_chal_metrics['WAcc'], baseline_chal_metrics['Sim'], baseline_chal_metrics['chrF']]
f_chal_vals = [finetuned_chal_metrics['PER'], finetuned_chal_metrics['WER'], finetuned_chal_metrics['WAcc'], finetuned_chal_metrics['Sim'], finetuned_chal_metrics['chrF']]

r1 = ax1.bar(x - width/2, b_chal_vals, width, label='Pretrained Baseline (Before)', color='#94A3B8', edgecolor='#475569')
r2 = ax1.bar(x + width/2, f_chal_vals, width, label='Fine-Tuned ByT5 (After)', color='#10B981', edgecolor='#059669')
ax1.set_title("(A) Held-Out Challenge Suite (N=60)", fontsize=12.5, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels, fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, 115)
ax1.set_ylabel("Percentage (%)", fontsize=11)
ax1.legend(fontsize=9.5, loc='upper left')
for rect in r1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#334155')
for rect in r2:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#065F46')

# Panel B: Unseen Wiktionary Benchmark (N=500)
b_wiki_vals = [baseline_wiki_metrics['PER'], baseline_wiki_metrics['WER'], baseline_wiki_metrics['WAcc'], baseline_wiki_metrics['Sim'], baseline_wiki_metrics['chrF']]
f_wiki_vals = [finetuned_wiki_metrics['PER'], finetuned_wiki_metrics['WER'], finetuned_wiki_metrics['WAcc'], finetuned_wiki_metrics['Sim'], finetuned_wiki_metrics['chrF']]

r3 = ax2.bar(x - width/2, b_wiki_vals, width, label='Pretrained Baseline (Before)', color='#94A3B8', edgecolor='#475569')
r4 = ax2.bar(x + width/2, f_wiki_vals, width, label='Fine-Tuned ByT5 (After)', color='#2563EB', edgecolor='#1D4ED8')
ax2.set_title("(B) Official Unseen Wiktionary Test Benchmark (N=500)", fontsize=12.5, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(metric_labels, fontsize=9.5, fontweight='bold')
ax2.set_ylim(0, 115)
ax2.set_ylabel("Percentage (%)", fontsize=11)
ax2.legend(fontsize=9.5, loc='upper left')
for rect in r3:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#334155')
for rect in r4:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#1E3A8A')

plt.tight_layout()
fig.savefig("figures/per_wer_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/per_wer_metrics_comparison.png'")

# --- Figure 4: Stratified Linguistic Category Breakdown ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
xc = np.arange(len(cat_names))
b_cat_per = [cat_breakdown[c]['baseline']['PER'] for c in cat_names]
f_cat_per = [cat_breakdown[c]['finetuned']['PER'] for c in cat_names]
b_cat_wer = [cat_breakdown[c]['baseline']['WER'] for c in cat_names]
f_cat_wer = [cat_breakdown[c]['finetuned']['WER'] for c in cat_names]

ax1.bar(xc - width/2, b_cat_per, width, label='Baseline PER (%)', color='#EF4444', alpha=0.85, edgecolor='#991B1B')
ax1.bar(xc + width/2, f_cat_per, width, label='Fine-Tuned PER (%)', color='#10B981', alpha=0.95, edgecolor='#065F46')
ax1.set_title("(A) Phoneme Error Rate (PER) by Linguistic Category", fontsize=12.5, fontweight='bold', pad=10)
ax1.set_xticks(xc)
ax1.set_xticklabels(cat_names, fontsize=9.5, fontweight='bold', rotation=12)
ax1.set_ylabel("PER (%)", fontsize=11)
ax1.set_ylim(0, max(b_cat_per) * 1.25 + 5)
ax1.legend(fontsize=10)
for i, (bv, fv) in enumerate(zip(b_cat_per, f_cat_per)):
    ax1.text(i - width/2, bv + 0.8, f"{bv:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#991B1B')
    ax1.text(i + width/2, fv + 0.8, f"{fv:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#065F46')

ax2.bar(xc - width/2, b_cat_wer, width, label='Baseline WER (%)', color='#F97316', alpha=0.85, edgecolor='#9A3412')
ax2.bar(xc + width/2, f_cat_wer, width, label='Fine-Tuned WER (%)', color='#3B82F6', alpha=0.95, edgecolor='#1E40AF')
ax2.set_title("(B) Word Error Rate (WER) by Linguistic Category", fontsize=12.5, fontweight='bold', pad=10)
ax2.set_xticks(xc)
ax2.set_xticklabels(cat_names, fontsize=9.5, fontweight='bold', rotation=12)
ax2.set_ylabel("WER (%)", fontsize=11)
ax2.set_ylim(0, 115)
ax2.legend(fontsize=10)
for i, (bv, fv) in enumerate(zip(b_cat_wer, f_cat_wer)):
    ax2.text(i - width/2, bv + 1.5, f"{bv:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#9A3412')
    ax2.text(i + width/2, fv + 1.5, f"{fv:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#1E40AF')

plt.tight_layout()
fig.savefig("figures/category_breakdown.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/category_breakdown.png'")

# --- Figure 5: Levenshtein Edit Operations Decomposition ---
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
op_groups = [
    'Challenge Suite\\nBaseline (Before)',
    'Challenge Suite\\nFine-Tuned (After)',
    'Wiktionary (N=500)\\nBaseline (Before)',
    'Wiktionary (N=500)\\nFine-Tuned (After)'
]
sub_vals = [baseline_chal_metrics['Sub_Rate'], finetuned_chal_metrics['Sub_Rate'], baseline_wiki_metrics['Sub_Rate'], finetuned_wiki_metrics['Sub_Rate']]
del_vals = [baseline_chal_metrics['Del_Rate'], finetuned_chal_metrics['Del_Rate'], baseline_wiki_metrics['Del_Rate'], finetuned_wiki_metrics['Del_Rate']]
ins_vals = [baseline_chal_metrics['Ins_Rate'], finetuned_chal_metrics['Ins_Rate'], baseline_wiki_metrics['Ins_Rate'], finetuned_wiki_metrics['Ins_Rate']]

xg = np.arange(len(op_groups))
w_op = 0.24
ax.bar(xg - w_op, sub_vals, w_op, label='Substitutions (Vowel/Consonant Mismatch)', color='#EF4444', edgecolor='#991B1B')
ax.bar(xg, del_vals, w_op, label='Deletions (Omitted Vowels / Ezafe / Geminates)', color='#F59E0B', edgecolor='#92400E')
ax.bar(xg + w_op, ins_vals, w_op, label='Insertions (Spurious Epenthetic Phonemes)', color='#8B5CF6', edgecolor='#5B21B6')

ax.set_title("Decomposition of Phonetic Edit Operations: Substitutions, Deletions & Insertions", fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(xg)
ax.set_xticklabels(op_groups, fontsize=10, fontweight='bold')
ax.set_ylabel("Error Rate (% of Reference Phonemes)", fontsize=11)
ax.set_ylim(0, max(sub_vals + del_vals) * 1.28 + 2)
ax.legend(fontsize=10, loc='upper right')

for i in range(len(xg)):
    ax.text(xg[i] - w_op, sub_vals[i] + 0.3, f"{sub_vals[i]:.2f}%", ha='center', fontsize=8.5, fontweight='bold')
    ax.text(xg[i], del_vals[i] + 0.3, f"{del_vals[i]:.2f}%", ha='center', fontsize=8.5, fontweight='bold')
    ax.text(xg[i] + w_op, ins_vals[i] + 0.3, f"{ins_vals[i]:.2f}%", ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()
fig.savefig("figures/edit_operations_breakdown.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/edit_operations_breakdown.png'")""")

# --- CELL 18: Code Figures 6, 7, 8 ---
add_code("""# Figures 6, 7, and 8: Word Length Analysis, Persian Vowel Confusion Matrix, and Edit Distance Reduction

# --- Figure 6: Error Rate by Orthographic Word Length Buckets ---
combined_base_records = baseline_wiki_metrics['records'] + baseline_chal_metrics['records']
combined_ft_records = finetuned_wiki_metrics['records'] + finetuned_chal_metrics['records']

buckets = [
    ("Short\\n(1-4 chars)", lambda l: 1 <= l <= 4),
    ("Medium\\n(5-7 chars)", lambda l: 5 <= l <= 7),
    ("Long\\n(8-10 chars)", lambda l: 8 <= l <= 10),
    ("Compound\\n(11+ chars)", lambda l: l >= 11),
]

bucket_labels = [b[0] for b in buckets]
b_bucket_per, f_bucket_per = [], []
b_bucket_wer, f_bucket_wer = [], []

for label, cond in buckets:
    b_sub = [r for r in combined_base_records if cond(r['word_len'])]
    f_sub = [r for r in combined_ft_records if cond(r['word_len'])]
    b_bucket_per.append(np.mean([r['per'] for r in b_sub]) if b_sub else 0.0)
    f_bucket_per.append(np.mean([r['per'] for r in f_sub]) if f_sub else 0.0)
    b_bucket_wer.append(np.mean([r['wer'] for r in b_sub]) if b_sub else 0.0)
    f_bucket_wer.append(np.mean([r['wer'] for r in f_sub]) if f_sub else 0.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), dpi=300)
xb = np.arange(len(bucket_labels))

ax1.plot(xb, b_bucket_per, marker='o', linewidth=2.5, markersize=8, color='#EF4444', label='Pretrained Baseline PER (%)')
ax1.plot(xb, f_bucket_per, marker='s', linewidth=2.5, markersize=8, color='#10B981', label='Fine-Tuned Model PER (%)')
ax1.fill_between(xb, f_bucket_per, b_bucket_per, color='#10B981', alpha=0.12, label='PER Reduction Area')
ax1.set_title("(A) Phoneme Error Rate Across Word Length Buckets", fontsize=12, fontweight='bold', pad=10)
ax1.set_xticks(xb)
ax1.set_xticklabels(bucket_labels, fontsize=10, fontweight='bold')
ax1.set_ylabel("Mean PER (%)", fontsize=11)
ax1.legend(fontsize=9.5)

ax2.bar(xb - 0.18, b_bucket_wer, 0.35, color='#94A3B8', edgecolor='#475569', label='Pretrained Baseline WER (%)')
ax2.bar(xb + 0.18, f_bucket_wer, 0.35, color='#2563EB', edgecolor='#1E3A8A', label='Fine-Tuned Model WER (%)')
ax2.set_title("(B) Word Error Rate Across Word Length Buckets", fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(xb)
ax2.set_xticklabels(bucket_labels, fontsize=10, fontweight='bold')
ax2.set_ylabel("Mean WER (%)", fontsize=11)
ax2.set_ylim(0, 110)
ax2.legend(fontsize=9.5)

plt.tight_layout()
fig.savefig("figures/length_bucket_analysis.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/length_bucket_analysis.png'")

# --- Figure 7: Persian Vowel Confusion Matrix (Before vs. After) ---
vowel_classes = ['æ', 'e', 'o', 'ɒː', 'iː', 'uː', '∅']
vowel_idx = {v: i for i, v in enumerate(vowel_classes)}

def build_vowel_matrix(token_pairs):
    mat = np.zeros((len(vowel_classes), len(vowel_classes)), dtype=float)
    for ref_t, pred_t in token_pairs:
        if ref_t in vowel_idx and ref_t != '∅':
            r_i = vowel_idx[ref_t]
            p_i = vowel_idx[pred_t] if pred_t in vowel_idx else vowel_idx['∅']
            mat[r_i, p_i] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (mat / row_sums) * 100.0

base_all_pairs = baseline_wiki_metrics['token_pairs'] + baseline_chal_metrics['token_pairs']
ft_all_pairs = finetuned_wiki_metrics['token_pairs'] + finetuned_chal_metrics['token_pairs']

mat_before = build_vowel_matrix(base_all_pairs)
mat_after = build_vowel_matrix(ft_all_pairs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
disp_labels = [f"/{v}/" for v in vowel_classes[:6]] + ['DEL']

sns.heatmap(mat_before[:6, :], annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=disp_labels,
            yticklabels=disp_labels[:6], cbar=False, ax=ax1, vmin=0, vmax=100)
ax1.set_title("(A) Baseline Persian Vowel Confusion (% Row-Normalized)", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Predicted Vowel / Deletion", fontsize=11)
ax1.set_ylabel("Ground-Truth Reference Vowel", fontsize=11)

sns.heatmap(mat_after[:6, :], annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=disp_labels,
            yticklabels=disp_labels[:6], cbar=False, ax=ax2, vmin=0, vmax=100)
ax2.set_title("(B) Fine-Tuned Persian Vowel Confusion (% Row-Normalized)", fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel("Predicted Vowel / Deletion", fontsize=11)
ax2.set_ylabel("Ground-Truth Reference Vowel", fontsize=11)

plt.tight_layout()
fig.savefig("figures/vowel_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/vowel_confusion_matrix.png'")

# --- Figure 8: Word-Level Edit Distance Reduction on 20 Challenge Words ---
top20_indices = [i for start in range(0, 60, 15) for i in range(start, start + 5)]
labels_clean = [challenge_translit[i] for i in top20_indices]
b_d20 = [baseline_chal_metrics['records'][i]['edit_dist'] for i in top20_indices]
f_d20 = [finetuned_chal_metrics['records'][i]['edit_dist'] for i in top20_indices]

fig, ax = plt.subplots(figsize=(10.5, 7), dpi=300)
y20 = np.arange(len(labels_clean))
h20 = 0.36
ax.barh(y20 + h20/2, b_d20, h20, label='Pretrained Baseline Edit Distance (Errors)', color='#EF4444', alpha=0.88)
ax.barh(y20 - h20/2, f_d20, h20, label='Fine-Tuned Model Edit Distance', color='#2563EB', alpha=0.95)
ax.set_yticks(y20)
ax.set_yticklabels(labels_clean, fontsize=10, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel("Levenshtein Edit Distance (Phoneme Errors per Word)", fontsize=11.5)
ax.set_title("Word-Level Phonetic Levenshtein Distance Reduction (20 Held-Out Challenge Words)", fontsize=13, fontweight='bold', pad=12)
ax.legend(fontsize=10.5, loc='lower right')

for i, (bv, fv) in enumerate(zip(b_d20, f_d20)):
    ax.text(bv + 0.12, i + h20/2, f"{bv}", va='center', fontsize=9, fontweight='bold', color='#991B1B')
    ax.text(fv + 0.12, i - h20/2, f"{fv}", va='center', fontsize=9, fontweight='bold', color='#1E3A8A')

plt.tight_layout()
fig.savefig("figures/word_edit_distance_reduction.png", dpi=300, bbox_inches='tight')
plt.show()
print("[+] Saved 'figures/word_edit_distance_reduction.png'")""")

# --- CELL 19: Architecture Figure & Final Checkpoint Export ---
add_code("""# Figure 9: ByT5 Architecture Diagram & Checkpoint Export
fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=300)
ax.axis('off')
boxes = [
    ("Persian Input (Graphemes)\\n'<fas>: Algoritm / Shabake'", 0.10, 0.5, '#E0F2FE', '#0284C7'),
    ("Raw UTF-8 Byte Stream\\n[60, 102, 97, 115, ...]", 0.34, 0.5, '#FEF3C7', '#D97706'),
    ("Fine-Tuned ByT5 Seq2Seq\\n(KaamelDict 115K Pairs)", 0.60, 0.5, '#EDE9FE', '#7C3AED'),
    ("Persian IPA Output\\n/ʔælɡoɾiːtm/ | /ʃæbæke/", 0.86, 0.5, '#DCFCE7', '#16A34A')
]
for text, x_c, y_c, bg, border in boxes:
    ax.text(x_c, y_c, text, ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.7", fc=bg, ec=border, lw=2))
for x1, x2 in [(0.20, 0.24), (0.44, 0.49), (0.71, 0.76)]:
    ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="->", color='#334155', lw=2.5))
ax.set_title("Large-Scale Persian Grapheme-to-Phoneme (G2P) Architecture with ByT5", fontsize=13.5, fontweight='bold', pad=10)
plt.tight_layout()
fig.savefig("figures/g2p_byt5_architecture.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("[+] Saved 'figures/g2p_byt5_architecture.png'")

# Qualitative Comparison Table on 20 Challenge Words
qual_rows = []
for idx in top20_indices:
    b_rec = baseline_chal_metrics['records'][idx]
    f_rec = finetuned_chal_metrics['records'][idx]
    qual_rows.append([
        b_rec['word'],
        chal_test_cats[idx],
        f"/{b_rec['ref']}/",
        f"/{b_rec['pred']}/ (err={b_rec['edit_dist']})",
        f"/{f_rec['pred']}/ (err={f_rec['edit_dist']})"
    ])
print("\\n================ QUALITATIVE WORD-LEVEL COMPARISON (20 CHALLENGE SAMPLES) ================")
print(tabulate(qual_rows, headers=["Persian Word", "Category", "Ground Truth IPA", "Baseline (Before)", "Fine-Tuned (After)"], tablefmt="grid"))

# Export comprehensive JSON summary for reproducibility
export_summary = {
    "train_samples_total": len(train_pairs),
    "kaamel_dict_total": len(kaamel_ds),
    "val_samples": len(val_pairs),
    "wiktionary_test_samples": len(wiki_test_words),
    "challenge_test_samples": len(chal_test_words),
    "training_runtime_sec": train_duration,
    "train_steps": train_steps,
    "train_losses": train_losses,
    "eval_steps": eval_steps,
    "eval_losses": eval_losses,
    "learning_rates": learning_rates,
    "wiktionary_500_baseline": {k: v for k, v in baseline_wiki_metrics.items() if k not in ("records", "token_pairs")},
    "wiktionary_500_finetuned": {k: v for k, v in finetuned_wiki_metrics.items() if k not in ("records", "token_pairs")},
    "challenge_60_baseline": {k: v for k, v in baseline_chal_metrics.items() if k not in ("records", "token_pairs")},
    "challenge_60_finetuned": {k: v for k, v in finetuned_chal_metrics.items() if k not in ("records", "token_pairs")},
    "category_breakdown": {
        c: {
            "baseline_per": cat_breakdown[c]["baseline"]["PER"],
            "finetuned_per": cat_breakdown[c]["finetuned"]["PER"],
            "baseline_wer": cat_breakdown[c]["baseline"]["WER"],
            "finetuned_wer": cat_breakdown[c]["finetuned"]["WER"],
            "baseline_chrf": cat_breakdown[c]["baseline"]["chrF"],
            "finetuned_chrf": cat_breakdown[c]["finetuned"]["chrF"],
        } for c in cat_names
    },
    "length_buckets": {
        "labels": [b.replace("\\n", " ") for b in bucket_labels],
        "baseline_per": [float(x) for x in b_bucket_per],
        "finetuned_per": [float(x) for x in f_bucket_per],
        "baseline_wer": [float(x) for x in b_bucket_wer],
        "finetuned_wer": [float(x) for x in f_bucket_wer],
    },
    "vowel_confusion_before": mat_before[:6, :].tolist(),
    "vowel_confusion_after": mat_after[:6, :].tolist(),
    "challenge_records_before": baseline_chal_metrics["records"],
    "challenge_records_after": finetuned_chal_metrics["records"],
}

with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(export_summary, f, indent=2, ensure_ascii=False)
print("[+] Saved 'evaluation_metrics.json'")

# Save Fine-Tuned Model Checkpoint
final_model_path = "./finetuned_byt5_g2p_model"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"[+] Saved fine-tuned Persian G2P model to '{final_model_path}'")
!ls -lh figures/""")

out_path = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune/g2p_implementation_and_finetune.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[+] Successfully generated '{out_path}' with {len(notebook['cells'])} cells.")
