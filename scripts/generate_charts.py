#!/usr/bin/env python3
"""
Generate All 9 Publication-Quality (300 DPI) Figures for the Large-Scale Persian G2P Benchmark.
Reads exact empirical results from 'evaluation_metrics.json' exported by the Kaggle GPU training run,
and saves figures to both 'figures/' and the conversation artifact directory.
"""

import json
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
ARTIFACT_DIR = "/home/soroush/.gemini/antigravity/brain/cc374521-ca11-4d6e-9e2b-4a6513ad3b45"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

IPA_TOKEN_REGEX = re.compile(r"t͡ʃ|d͡ʒ|ɒː|iː|uː|ɔː|oʊ|eɪ|.")
def tokenize_ipa(ipa_str):
    cleaned = ipa_str.replace(" ", "").replace("ˈ", "").replace("ˌ", "").replace(".", "")
    return [t for t in IPA_TOKEN_REGEX.findall(cleaned) if t.strip()]

def save_both(fig, filename):
    p1 = os.path.join(FIGURES_DIR, filename)
    p2 = os.path.join(ARTIFACT_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved 300 DPI figure: {filename}")

def main():
    metrics_path = os.path.join(BASE_DIR, "evaluation_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # 1. Figure 1: Dataset Distribution (EDA)
    train_path = os.path.join(BASE_DIR, "data", "persian_g2p_train_expanded.tsv")
    unique_pairs = {}
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                w, p = line.strip().split("\t")[:2]
                unique_pairs[w] = p
    unique_list = list(unique_pairs.items())
    grapheme_lengths = np.array([len(w) for w, _ in unique_list])
    phoneme_tokens_list = [tokenize_ipa(p) for _, p in unique_list]
    phoneme_lengths = np.array([len(toks) for toks in phoneme_tokens_list])

    phoneme_counter = Counter()
    for toks in phoneme_tokens_list:
        phoneme_counter.update(toks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    mask = (grapheme_lengths <= 20) & (phoneme_lengths <= 25)
    x_g = grapheme_lengths[mask]
    y_p = phoneme_lengths[mask]
    rng = np.random.default_rng(42)
    ax1.scatter(x_g + rng.normal(0, 0.18, len(x_g)), y_p + rng.normal(0, 0.18, len(y_p)),
                alpha=0.15, s=12, color='#2563EB', edgecolors='none', label=f'Unique Persian Words (N={len(unique_list):,})')
    slope, intercept = np.polyfit(x_g, y_p, 1)
    r_corr = np.corrcoef(x_g, y_p)[0, 1]
    x_line = np.linspace(1, 18, 100)
    ax1.plot(x_line, slope * x_line + intercept, color='#DC2626', linewidth=2.5,
             label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f} (R = {r_corr:.2f})')
    ax1.plot(x_line, x_line, color='#64748B', linestyle='--', linewidth=1.5, label='1:1 Identity Line (y = x)')
    ax1.set_title("(A) Persian Grapheme Length vs. IPA Phoneme Length", fontsize=12.5, fontweight='bold', pad=10)
    ax1.set_xlabel("Orthographic Word Length (Characters)", fontsize=11)
    ax1.set_ylabel("Phonetic Transcription Length (IPA Phonemes)", fontsize=11)
    ax1.set_xlim(1, 18)
    ax1.set_ylim(1, 24)
    ax1.legend(fontsize=9.5, loc='upper left', frameon=True)

    top20 = phoneme_counter.most_common(20)
    ph_labels = [f"/{k}/" for k, _ in top20]
    ph_counts = [v for _, v in top20]
    vowels_set = {'æ', 'e', 'o', 'ɒː', 'iː', 'uː', 'oʊ'}
    bar_colors = ['#10B981' if k in vowels_set else '#3B82F6' for k, _ in top20]
    y_pos = np.arange(len(top20))
    ax2.barh(y_pos, ph_counts, color=bar_colors, edgecolor='#1E293B', linewidth=0.6, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ph_labels, fontsize=10.5, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_title("(B) Top 20 Persian IPA Phonemes in Corpus (Green = Vowels)", fontsize=12.5, fontweight='bold', pad=10)
    ax2.set_xlabel("Occurrences in Unique Vocabulary", fontsize=11)
    for i, c in enumerate(ph_counts):
        ax2.text(c + 80, i, f"{c:,}", va='center', fontsize=9, fontweight='bold', color='#1E293B')
    ax2.set_xlim(0, max(ph_counts) * 1.15)
    plt.tight_layout()
    save_both(fig, "dataset_distribution.png")

    # 2. Figure 2: Training & Validation Loss Curve
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    train_steps = m["train_steps"]
    train_losses = m["train_losses"]
    eval_steps = m.get("eval_steps", [])
    eval_losses = m.get("eval_losses", [])
    learning_rates = m.get("learning_rates", [])

    line1 = ax1.plot(train_steps, train_losses, color='#2563EB', linewidth=2.2, marker='o', markersize=3.5,
                     alpha=0.9, label='Training Cross-Entropy Loss')
    if eval_steps:
        line2 = ax1.plot(eval_steps, eval_losses, color='#DC2626', linewidth=2.5, marker='s', markersize=6,
                         linestyle='--', label='Validation Loss (Dev Split)')
    ax1.set_title("Large-Scale Persian G2P Fine-Tuning Convergence & LR Schedule (ByT5)", fontsize=13.5, fontweight='bold', pad=14)
    ax1.set_xlabel("Optimization Steps (Batch Size = 32, 8 Epochs over 8,709 Samples)", fontsize=11.5)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11.5, color='#1E293B')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    line3 = ax2.plot(train_steps[:len(learning_rates)], learning_rates, color='#10B981', linewidth=1.8,
                     linestyle=':', label='Learning Rate (x10^-4)')
    ax2.set_ylabel("Learning Rate (x 10^-4)", fontsize=11, color='#059669')
    ax2.grid(False)
    lines = line1 + (line2 if eval_steps else []) + line3
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    save_both(fig, "loss_curve.png")

    # 3. Figure 3: Multi-Metric Benchmark Comparison
    bw = m["wiktionary_500_baseline"]
    fw = m["wiktionary_500_finetuned"]
    bc = m["challenge_60_baseline"]
    fc = m["challenge_60_finetuned"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    metric_labels = ['PER (%)\n[Lower Better]', 'WER (%)\n[Lower Better]', 'Word Acc (%)\n[Higher Better]', 'Seq Sim (%)\n[Higher Better]', 'chrF (%)\n[Higher Better]']
    x = np.arange(len(metric_labels))
    width = 0.35

    b_chal_vals = [bc['PER'], bc['WER'], bc['WAcc'], bc['Sim'], bc['chrF']]
    f_chal_vals = [fc['PER'], fc['WER'], fc['WAcc'], fc['Sim'], fc['chrF']]
    r1 = ax1.bar(x - width/2, b_chal_vals, width, label='Pretrained Baseline (Before)', color='#94A3B8', edgecolor='#475569')
    r2 = ax1.bar(x + width/2, f_chal_vals, width, label='Fine-Tuned ByT5 (After)', color='#10B981', edgecolor='#059669')
    ax1.set_title("(A) Linguistic & Tech Challenge Suite (N=60)", fontsize=12.5, fontweight='bold', pad=10)
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

    b_wiki_vals = [bw['PER'], bw['WER'], bw['WAcc'], bw['Sim'], bw['chrF']]
    f_wiki_vals = [fw['PER'], fw['WER'], fw['WAcc'], fw['Sim'], fw['chrF']]
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
    save_both(fig, "per_wer_metrics_comparison.png")

    # 4. Figure 4: Stratified Linguistic Category Breakdown
    cb = m["category_breakdown"]
    cat_names = list(cb.keys())
    xc = np.arange(len(cat_names))
    b_cat_per = [cb[c]["baseline_per"] for c in cat_names]
    f_cat_per = [cb[c]["finetuned_per"] for c in cat_names]
    b_cat_wer = [cb[c]["baseline_wer"] for c in cat_names]
    f_cat_wer = [cb[c]["finetuned_wer"] for c in cat_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
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
    save_both(fig, "category_breakdown.png")

    # 5. Figure 5: Levenshtein Edit Operations Decomposition
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    op_groups = [
        'Challenge Suite\nBaseline (Before)',
        'Challenge Suite\nFine-Tuned (After)',
        'Wiktionary (N=500)\nBaseline (Before)',
        'Wiktionary (N=500)\nFine-Tuned (After)'
    ]
    sub_vals = [bc['Sub_Rate'], fc['Sub_Rate'], bw['Sub_Rate'], fw['Sub_Rate']]
    del_vals = [bc['Del_Rate'], fc['Del_Rate'], bw['Del_Rate'], fw['Del_Rate']]
    ins_vals = [bc['Ins_Rate'], fc['Ins_Rate'], bw['Ins_Rate'], fw['Ins_Rate']]
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
    save_both(fig, "edit_operations_breakdown.png")

    # 6. Figure 6: Length Bucket Analysis
    lb = m["length_buckets"]
    bucket_labels = [x.replace(" (", "\n(") for x in lb["labels"]]
    xb = np.arange(len(bucket_labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), dpi=300)
    ax1.plot(xb, lb["baseline_per"], marker='o', linewidth=2.5, markersize=8, color='#EF4444', label='Pretrained Baseline PER (%)')
    ax1.plot(xb, lb["finetuned_per"], marker='s', linewidth=2.5, markersize=8, color='#10B981', label='Fine-Tuned Model PER (%)')
    ax1.fill_between(xb, lb["finetuned_per"], lb["baseline_per"], color='#10B981', alpha=0.12, label='PER Reduction Area')
    ax1.set_title("(A) Phoneme Error Rate Across Word Length Buckets", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(bucket_labels, fontsize=10, fontweight='bold')
    ax1.set_ylabel("Mean PER (%)", fontsize=11)
    ax1.legend(fontsize=9.5)

    ax2.bar(xb - 0.18, lb["baseline_wer"], 0.35, color='#94A3B8', edgecolor='#475569', label='Pretrained Baseline WER (%)')
    ax2.bar(xb + 0.18, lb["finetuned_wer"], 0.35, color='#2563EB', edgecolor='#1E3A8A', label='Fine-Tuned Model WER (%)')
    ax2.set_title("(B) Word Error Rate Across Word Length Buckets", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(bucket_labels, fontsize=10, fontweight='bold')
    ax2.set_ylabel("Mean WER (%)", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9.5)
    plt.tight_layout()
    save_both(fig, "length_bucket_analysis.png")

    # 7. Figure 7: Vowel Confusion Matrix
    vowel_classes = ['æ', 'e', 'o', 'ɒː', 'iː', 'uː']
    disp_labels = [f"/{v}/" for v in vowel_classes] + ['DEL']
    mat_before = np.array(m["vowel_confusion_before"])
    mat_after = np.array(m["vowel_confusion_after"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
    sns.heatmap(mat_before, annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=disp_labels,
                yticklabels=disp_labels[:6], cbar=False, ax=ax1, vmin=0, vmax=100)
    ax1.set_title("(A) Baseline Persian Vowel Confusion (% Row-Normalized)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Predicted Vowel / Deletion", fontsize=11)
    ax1.set_ylabel("Ground-Truth Reference Vowel", fontsize=11)

    sns.heatmap(mat_after, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=disp_labels,
                yticklabels=disp_labels[:6], cbar=False, ax=ax2, vmin=0, vmax=100)
    ax2.set_title("(B) Fine-Tuned Persian Vowel Confusion (% Row-Normalized)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Predicted Vowel / Deletion", fontsize=11)
    ax2.set_ylabel("Ground-Truth Reference Vowel", fontsize=11)
    plt.tight_layout()
    save_both(fig, "vowel_confusion_matrix.png")

    # 8. Figure 8: Word Edit Distance Reduction (20 Held-Out Challenge Words)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from heldout_challenge_set import HELDOUT_CHALLENGE_SET

    challenge_translit = [t for triples in HELDOUT_CHALLENGE_SET.values() for _, _, t in triples]
    top20_indices = [i for start in range(0, 60, 15) for i in range(start, start + 5)]
    labels_clean = [challenge_translit[i] for i in top20_indices]
    b_d20 = [m["challenge_records_before"][i]["edit_dist"] for i in top20_indices]
    f_d20 = [m["challenge_records_after"][i]["edit_dist"] for i in top20_indices]

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
    save_both(fig, "word_edit_distance_reduction.png")

    # 9. Figure 9: Architecture Diagram
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
    ax.axis('off')
    boxes = [
        ("Persian Input (Graphemes)\n'<fas>: Algoritm / Shabake'", 0.10, 0.5, '#E0F2FE', '#0284C7'),
        ("Raw UTF-8 Byte Stream\n[60, 102, 97, 115, ...]", 0.34, 0.5, '#FEF3C7', '#D97706'),
        ("Fine-Tuned ByT5 Seq2Seq\n(8,559 Persian Corpus Pairs)", 0.60, 0.5, '#EDE9FE', '#7C3AED'),
        ("Persian IPA Output\n/ʔælɡoɾiːtm/ | /ʃæbæke/", 0.86, 0.5, '#DCFCE7', '#16A34A')
    ]
    for text, x_c, y_c, bg, border in boxes:
        ax.text(x_c, y_c, text, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.7", fc=bg, ec=border, lw=2))
    for x1, x2 in [(0.20, 0.24), (0.44, 0.49), (0.71, 0.76)]:
        ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="->", color='#334155', lw=2.5))
    ax.set_title("Large-Scale Persian Grapheme-to-Phoneme (G2P) Architecture with ByT5", fontsize=13.5, fontweight='bold', pad=10)
    plt.tight_layout()
    save_both(fig, "g2p_byt5_architecture.png")

if __name__ == "__main__":
    main()
