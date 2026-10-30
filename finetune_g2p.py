#!/usr/bin/env python3
"""
Large-Scale Persian (Farsi) G2P Implementation, Benchmarking & Fine-Tuning Script
Supports both GPU (FP16) and CPU execution.
Trains on 8,559 Persian word-phoneme instances (Wiktionary + Curated Linguistic/Tech Lexicon)
and evaluates across 9 phonetic metrics on both the 500-word Wiktionary test split
and the 60-word Stratified Linguistic & Tech Challenge Suite.
"""

import argparse
import inspect
import json
import os
import re
import time
from collections import Counter
import numpy as np
import torch
from datasets import Dataset
from tabulate import tabulate
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

IPA_TOKEN_REGEX = re.compile(r"t͡ʃ|d͡ʒ|ɒː|iː|uː|ɔː|oʊ|eɪ|.")

def tokenize_ipa(ipa_str):
    cleaned = ipa_str.replace(" ", "").replace("ˈ", "").replace("ˌ", "").replace(".", "")
    return [t for t in IPA_TOKEN_REGEX.findall(cleaned) if t.strip()]

def levenshtein_alignment(pred_seq, ref_seq):
    m, n = len(pred_seq), len(ref_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_seq[i - 1] == ref_seq[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    i, j = m, n
    sub = dele = ins = 0
    aligned_pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred_seq[i - 1] == ref_seq[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
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
    total_dist = total_sub = total_del = total_ins = total_ref_len = 0
    word_errors = 0
    sim_scores = []
    chrf_scores = []
    per_word_records = []

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
    }

def main():
    parser = argparse.ArgumentParser(description="Large-Scale Persian G2P Fine-Tuning & Evaluation")
    parser.add_argument("--model_id", type=str, default="charsiu/g2p_multilingual_byT5_tiny_16_layers_100")
    parser.add_argument("--tokenizer_id", type=str, default="google/byt5-small")
    parser.add_argument("--train_file", type=str, default="data/persian_g2p_train_expanded.tsv")
    parser.add_argument("--dev_file", type=str, default="data/persian_g2p_dev.tsv")
    parser.add_argument("--test_wiki_file", type=str, default="data/persian_g2p_test_wiktionary500.tsv")
    parser.add_argument("--test_chal_file", type=str, default="data/persian_g2p_test_challenge60.tsv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--output_dir", type=str, default="./finetuned_byt5_g2p_model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)

    def batch_predict(words, bs=64):
        preds = []
        model.eval()
        for i in range(0, len(words), bs):
            bw = words[i:i + bs]
            pref = [f"<fas>: {w}" for w in bw]
            enc = tokenizer(pref, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=1, max_length=64)
            dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
            preds.extend([d.strip().replace(" ", "") for d in dec])
        return preds

    # Load datasets
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_pairs = [l.strip().split("\t")[:2] for l in f if "\t" in l]
    with open(args.dev_file, "r", encoding="utf-8") as f:
        val_pairs = [l.strip().split("\t")[:2] for l in f if "\t" in l]
    with open(args.test_wiki_file, "r", encoding="utf-8") as f:
        wiki_rows = [l.strip().split("\t") for l in f if "\t" in l]
    with open(args.test_chal_file, "r", encoding="utf-8") as f:
        chal_rows = [l.strip().split("\t") for l in f if "\t" in l]

    wiki_words = [r[0] for r in wiki_rows]
    wiki_refs = [r[1].split(",") for r in wiki_rows]
    chal_words = [r[0] for r in chal_rows]
    chal_refs = [[r[1]] for r in chal_rows]

    print(f"[*] Loaded {len(train_pairs):,} train | {len(val_pairs):,} dev | {len(wiki_words):,} Wiktionary test | {len(chal_words):,} Challenge test")

    # Baseline Evaluation
    base_wiki = evaluate_predictions_comprehensive(batch_predict(wiki_words), wiki_refs, wiki_words)
    base_chal = evaluate_predictions_comprehensive(batch_predict(chal_words), chal_refs, chal_words)
    print(f"[-] Baseline Wiktionary (N=500): PER={base_wiki['PER']:.2f}% | WER={base_wiki['WER']:.2f}% | chrF={base_wiki['chrF']:.2f}%")
    print(f"[-] Baseline Challenge  (N=60) : PER={base_chal['PER']:.2f}% | WER={base_chal['WER']:.2f}% | chrF={base_chal['chrF']:.2f}%")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(train_pairs))
    train_pairs = [train_pairs[i] for i in perm]

    def make_ds(pairs):
        ds = Dataset.from_dict({
            "input_text": [f"<fas>: {w}" for w, _ in pairs],
            "target_text": [p for _, p in pairs]
        })
        def _prep(ex):
            inp = tokenizer(ex["input_text"], max_length=64, truncation=True, padding=False)
            lbl = tokenizer(text_target=ex["target_text"], max_length=64, truncation=True, padding=False)
            inp["labels"] = lbl["input_ids"]
            return inp
        return ds.map(_prep, batched=True, remove_columns=["input_text", "target_text"])

    train_ds = make_ds(train_pairs)
    val_ds = make_ds(val_pairs)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    train_args = Seq2SeqTrainingArguments(
        output_dir="./checkpoints_persian_g2p",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        report_to="none",
        seed=42,
    )

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer.train()

    ft_wiki = evaluate_predictions_comprehensive(batch_predict(wiki_words), wiki_refs, wiki_words)
    ft_chal = evaluate_predictions_comprehensive(batch_predict(chal_words), chal_refs, chal_words)

    rows = [
        ["Wiktionary Test (N=500)", f"{base_wiki['PER']:.2f}% -> {ft_wiki['PER']:.2f}%",
         f"{base_wiki['WER']:.2f}% -> {ft_wiki['WER']:.2f}%", f"{base_wiki['chrF']:.2f}% -> {ft_wiki['chrF']:.2f}%"],
        ["Challenge Suite (N=60)", f"{base_chal['PER']:.2f}% -> {ft_chal['PER']:.2f}%",
         f"{base_chal['WER']:.2f}% -> {ft_chal['WER']:.2f}%", f"{base_chal['chrF']:.2f}% -> {ft_chal['chrF']:.2f}%"],
    ]
    print("\n" + tabulate(rows, headers=["Benchmark", "PER (Before -> After)", "WER (Before -> After)", "chrF (Before -> After)"], tablefmt="grid"))

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[+] Saved fine-tuned model to '{args.output_dir}'")

if __name__ == "__main__":
    main()
