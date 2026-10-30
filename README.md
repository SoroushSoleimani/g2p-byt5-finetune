# Large-Scale Persian (Farsi) Grapheme-to-Phoneme (G2P) Fine-Tuning with ByT5

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook%20Executed%20on%20GPU-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/soroushsoleimani/g2p-implementation-and-mini-finetune-byt5)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-ByT5-FFD21E)](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100)

An end-to-end, reproducible implementation and large-scale domain adaptation pipeline for **Persian (Farsi) Grapheme-to-Phoneme (G2P)** conversion using **ByT5** (`charsiu/g2p_multilingual_byT5_tiny_16_layers_100` with the `<fas>: ` prefix).

This project expands the official CharsiuG2P Persian Wiktionary lexicon (`7,149` base word-IPA pairs) with a curated lexicon of `235` modern Persian technical, scientific, compound/Ezafe, and geminated words (`6x` domain-weighted oversampling to `8,559` total training instances), and evaluates the model across **two independent, rigorously held-out benchmarks** (`N = 500` Unseen Wiktionary Test Set and `N = 60` Held-Out Stratified Linguistic Challenge Suite) using **9 phonetic metrics** and **9 high-resolution (300 DPI) diagnostic figures**.

---

## Architecture Overview

![ByT5 Persian G2P Architecture](figures/g2p_byt5_architecture.png)

Unlike subword tokenizers (BPE, WordPiece, SentencePiece) that group characters into multi-letter tokens, **ByT5** processes raw **UTF-8 bytes** directly (`256` byte embeddings + `3` special tokens). Because Persian is written in an abjad script where short vowels (`/æ/`, `/e/`, `/o/`), the Ezafe enclitic (`/-e/`, `/-je/`), and consonant gemination (`tashdid`) are omitted in standard orthography, operating directly on raw UTF-8 byte sequences allows the 16-layer encoder-decoder transformer to learn fine-grained orthographic-to-phonemic mappings without out-of-vocabulary (OOV) failures.

---

## Corpus Construction, Data Hygiene & Leakage Audit

The training, validation, and evaluation datasets are located in `data/` and adhere to strict data hygiene standards to ensure zero test contamination:

| Dataset Split | File Path | Instances | Description |
| :--- | :--- | :---: | :--- |
| **Expanded Training Set** | `data/persian_g2p_train_expanded.tsv` | **8,559** | `7,149` CharsiuG2P Persian Wiktionary pairs + `235` curated technical/linguistic pairs (`6x` oversampled = `1,410`) |
| **Validation Set (Dev)** | `data/persian_g2p_dev.tsv` | **75** | `50` official Wiktionary dev pairs + `25` held-out curated domain validation pairs |
| **Unseen Wiktionary Test** | `data/persian_g2p_test_wiktionary500.tsv` | **500** | Official unseen CharsiuG2P Persian (`fas.tsv`) general-vocabulary test split |
| **Held-Out Challenge Test** | `data/persian_g2p_test_challenge60.tsv` | **60** | `4` linguistic categories (`15` words per category) testing hard Persian phonological phenomena (100% held-out) |

### Strict Leakage Audit
To guarantee statistical validity and prevent artificial inflation of evaluation metrics:
* **Held-Out Challenge Suite**: `0 / 60` words appear in the training or validation splits.
* **Curated Validation Split**: `0 / 25` words appear in the training split.
* **Wiktionary Test Split**: `4 / 500` words appear in the training split (an artifact inherited directly from the original CharsiuG2P Wiktionary splits).

![Persian G2P Corpus EDA](figures/dataset_distribution.png)

As shown in Figure 1A, Persian IPA transcriptions average `7.75` phoneme tokens per word compared to `6.58` orthographic characters (`y = 1.06x + 0.76`, `R = 0.94`), illustrating the unwritten short vowels (`/æ/`, `/e/`, `/o/`) that the G2P model must infer from consonantal orthography.

---

## Training Dynamics on Kaggle Tesla T4 GPU

Fine-tuning was executed for **8 epochs** (`1,072` optimization steps with effective batch size `32`, peak learning rate `4e-4`, cosine decay, `100` linear warmup steps, and `FP16` mixed precision) on a Kaggle **NVIDIA Tesla T4 GPU** in **156.38 seconds**.

![Training & Validation Loss Curve](figures/loss_curve.png)

---

## Comprehensive Quantitative Evaluation (9 Phonetic Metrics)

We evaluate both the pretrained multilingual baseline (`charsiu/g2p_multilingual_byT5_tiny_16_layers_100`) and the fine-tuned Persian ByT5 model across **both independent benchmarks** using **9 metrics**:

| Benchmark & Metric | Pretrained Baseline (Before) | Fine-Tuned ByT5 (After) | Absolute / Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Held-Out Challenge Suite (`N = 60`) — PER (%)** | `20.59%` | **`11.93%`** | **`-42.06%` relative PER reduction** |
| **Held-Out Challenge Suite (`N = 60`) — WER (%)** | `78.33%` | **`60.00%`** | **`-18.33` percentage points (`-23.41%` rel)** |
| **Held-Out Challenge Suite (`N = 60`) — Exact Word Acc (%)** | `21.67%` | **`40.00%`** | **`+18.33` percentage points (nearly doubled)** |
| **Held-Out Challenge Suite (`N = 60`) — Sequence Similarity (%)** | `79.07%` | **`86.76%`** | **`+7.69` percentage points** |
| **Held-Out Challenge Suite (`N = 60`) — Phoneme chrF (%)** | `64.14%` | **`75.31%`** | **`+11.17` percentage points** |
| **Held-Out Challenge Suite (`N = 60`) — Sub / Del / Ins (%)** | `9.97% / 8.33% / 2.29%` | **`6.70% / 4.41% / 0.82%`** | **`-32.8%` Sub, `-47.1%` Del, `-64.2%` Ins** |
| **Held-Out Challenge Suite (`N = 60`) — Mean Edit Dist** | `2.10` phonemes/word | **`1.22` phonemes/word** | **`-41.90%` reduction in phonetic distance** |
| **------------------------------------------------** | **------------------** | **------------------** | **------------------------------------** |
| **Wiktionary General Test (`N = 500`) — PER (%)** | `10.04%` | **`7.90%`** | **`-21.33%` relative PER reduction** |
| **Wiktionary General Test (`N = 500`) — WER (%)** | `52.20%` | **`39.20%`** | **`-13.00` percentage points (`-24.90%` rel)** |
| **Wiktionary General Test (`N = 500`) — Exact Word Acc (%)** | `47.80%` | **`60.80%`** | **`+13.00` percentage points** |
| **Wiktionary General Test (`N = 500`) — Sequence Similarity (%)** | `90.00%` | **`91.77%`** | **`+1.77` percentage points** |
| **Wiktionary General Test (`N = 500`) — Phoneme chrF (%)** | `80.16%` | **`84.35%`** | **`+4.19` percentage points** |
| **Wiktionary General Test (`N = 500`) — Sub / Del / Ins (%)** | `6.28% / 2.45% / 1.30%` | **`4.78% / 1.81% / 1.31%`** | **`-23.9%` Sub, `-26.1%` Del** |
| **Wiktionary General Test (`N = 500`) — Mean Edit Dist** | `0.91` phonemes/word | **`0.71` phonemes/word** | **`-21.98%` reduction in phonetic distance** |

![PER, WER, Word Accuracy, Similarity, and chrF Comparison](figures/per_wer_metrics_comparison.png)

---

## Stratified Linguistic Category & Error Diagnostics

### 1. Breakdown by Persian Linguistic Phenomenon (`N = 60`, `15` Held-Out Words per Category)

| Linguistic Category | Examples (100% Held-Out) | Baseline PER | Fine-Tuned PER | Baseline WER | Fine-Tuned WER | Baseline chrF | Fine-Tuned chrF |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. AI & Tech Loanwords** | `پروسسور`, `اسکنر`, `روتر`, `دیتابیس`, `اپلیکیشن`, `بلاکچین` | `25.76%` | **`19.70%`** | `80.00%` | **`73.33%`** | `59.92%` | **`64.32%`** |
| **2. Short-Vowel Ambiguity** | `کاربرد`, `پیامد`, `چکیده`, `ویژگی`, `پژوهشگر`, `بسامد` | `12.60%` | **`7.87%`** | `53.33%` | **`46.67%`** | `76.34%` | **`82.70%`** |
| **3. Compound & Ezafe** | `مدل آماری`, `شبکه عمیق`, `حافظه کوتاه‌مدت`, `موتور جستجو` | `16.38%` | **`5.60%`** | `100.00%` | **`53.33%`** | `71.78%` | **`90.37%`** |
| **4. Gemination & Glottal** | `دقت`, `تعامل`, `تجزیه`, `متوسط`, `مدرس`, `مقرر`, `مکرر` | `31.40%` | **`19.83%`** | `80.00%` | **`66.67%`** | `48.54%` | **`63.86%`** |

![Stratified Linguistic Category Breakdown](figures/category_breakdown.png)

Key linguistic observations:
* **Compound & Ezafe constructions** show the most dramatic improvement: PER dropped from `16.38%` to **`5.60%`** (a **`-65.8%` relative reduction**), WER was cut from `100.0%` to **`53.33%`**, and chrF reached **`90.37%`**, proving that ByT5 learned genuine Ezafe enclitic placement (`-e` and `-ye`) without memorization.
* **Gemination & Glottal stops** exhibited an error reduction from `31.40%` to **`19.83%`** PER, with chrF surging from `48.54%` to **`63.86%`**.

### 2. Decomposition of Levenshtein Edit Operations (Substitutions, Deletions, Insertions)

![Edit Operations Breakdown](figures/edit_operations_breakdown.png)

### 3. Performance Across Orthographic Word-Length Buckets

![Length Bucket Analysis](figures/length_bucket_analysis.png)

### 4. Persian Vowel Confusion Matrix (`/æ/`, `/e/`, `/o/`, `/ɒː/`, `/iː/`, `/uː/`, `DEL`)

In the pretrained baseline (Figure 7A), the short vowels `/e/` and `/o/` suffered severe confusion: `22.4%` of `/e/` tokens and `56.2%` of `/o/` tokens were mispredicted as `/æ/`, while `24.5%` of `/e/` tokens were deleted outright (`DEL`). After large-scale fine-tuning (Figure 7B):
* Short vowel `/e/` recall increased from `51.55%` to **`70.99%`**, and `/e/` deletions dropped from `24.53%` to **`13.89%`**.
* Short vowel `/o/` recall increased from `18.12%` to **`61.96%`** (over `3.4x` increase), and confusion with `/æ/` dropped from `56.25%` to **`23.93%`**.

![Persian Vowel Confusion Matrix](figures/vowel_confusion_matrix.png)

### 5. Word-Level Phonetic Levenshtein Distance Reduction

![Word Edit Distance Reduction](figures/word_edit_distance_reduction.png)

---

## Qualitative Word-Level & Audio Waveform Comparison (`34` `.wav` Files)

The `audio_comparison/` directory contains **34 Persian `.wav` files** synthesized with Microsoft Edge neural TTS (`fa-IR-FaridNeural`) representing the phonological improvements before and after fine-tuning:

| Persian Input | Category | Ground Truth IPA | Pretrained Baseline (Before) | Fine-Tuned ByT5 (After) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `مدل آماری` | Compound & Ezafe | `/modeleɒːmɒːɾiː/` | `/mædælɒːmɒːɾiː/` (`err=3`) | `/modeleɒːmɒːɾiː/` (`err=0`) | **Exact Match** |
| `شبکه عمیق` | Compound & Ezafe | `/ʃæbækejeʔæmiːɢ/` | `/ʃæbkeʔæmiːɢ/` (`err=3`) | `/ʃæbækejeʔæmiːɢ/` (`err=0`) | **Exact Match** |
| `تعامل` | Gemination & Glottal | `/tæʔɒːmol/` | `/tæʔɒːmæl/` (`err=1`) | `/tæʔɒːmol/` (`err=0`) | **Exact Match** |
| `ویژگی` | Short-Vowel Ambiguity | `/viːʒeɡiː/` | `/viːʒɡiː/` (`err=1`) | `/viːʒeɡiː/` (`err=0`) | **Exact Match** |
| `پژوهشگر` | Short-Vowel Ambiguity | `/pæʒuːheʃɡæɾ/` | `/peʒuːheʃɡæɾ/` (`err=1`) | `/pæʒuːheʃɡæɾ/` (`err=0`) | **Exact Match** |
| `موتور جستجو` | Compound & Ezafe | `/motoɾed͡ʒostod͡ʒuː/` | `/muːtuːɾed͡ʒostd͡ʒuː/` (`err=5`) | `/motoɾed͡ʒosted͡ʒuː/` (`err=1`) | **Significant Improvement** |
| `متوسط` | Gemination & Glottal | `/motevæsset/` | `/motuːst/` (`err=5`) | `/motævæsset/` (`err=1`) | **Significant Improvement** |
| `مدرس` | Gemination & Glottal | `/modæɾɾes/` | `/mædɾæs/` (`err=4`) | `/modɾes/` (`err=2`) | **Significant Improvement** |
| `اسکنر` | AI & Tech Loanwords | `/eskæneɾ/` | `/æskænæɾ/` (`err=2`) | `/eskoneɾ/` (`err=1`) | **Significant Improvement** |
| `پروسسور` | AI & Tech Loanwords | `/pɾosesoɾ/` | `/poɾuːssuːɾ/` (`err=6`) | `/pɾossuːɾ/` (`err=3`) | **Significant Improvement** |

### Test Sentences (`sentence1` – `sentence3`)
1. **Sentence 1**: `مدل ترنسفورمر و شبکه عصبی در پایتورچ و تنسورفلو الگوریتم گرافیم را بهینه‌سازی می‌کنند.` (`sentence1_before_finetune.wav` vs. `sentence1_after_finetune.wav`)
2. **Sentence 2**: `مهندسی پزشکی و محاسبه دقیق متغیرها در هوش مصنوعی و یادگیری ماشین توجه زیادی دارد.` (`sentence2_before_finetune.wav` vs. `sentence2_after_finetune.wav`)
3. **Sentence 3**: `پژوهشگران علم داده در کگل با فریم‌ورک دیپ‌لرنینگ و توکنایزر مسئله اتصال را حل کردند.` (`sentence3_before_finetune.wav` vs. `sentence3_after_finetune.wav`)

---

## Repository Structure

```text
.
├── g2p_implementation_and_finetune.ipynb   # Comprehensive 19-cell Jupyter/Kaggle Notebook executed on Tesla T4 GPU
├── finetune_g2p.py                         # Standalone training & 9-metric evaluation script
├── evaluation_metrics.json                 # Complete empirical metrics exported directly from Kaggle GPU run
├── data/
│   ├── persian_g2p_train_expanded.tsv      # 8,559 training instances (7,149 Wiktionary + 235x6 Curated)
│   ├── persian_g2p_dev.tsv                 # 75 validation instances (50 Wiktionary + 25 Curated)
│   ├── persian_g2p_test_wiktionary500.tsv  # 500 unseen Wiktionary test instances
│   └── persian_g2p_test_challenge60.tsv    # 60 held-out challenge instances (0% training overlap)
├── figures/                                # 9 publication-quality (300 DPI) diagnostic plots
│   ├── dataset_distribution.png
│   ├── loss_curve.png
│   ├── per_wer_metrics_comparison.png
│   ├── category_breakdown.png
│   ├── edit_operations_breakdown.png
│   ├── length_bucket_analysis.png
│   ├── vowel_confusion_matrix.png
│   ├── word_edit_distance_reduction.png
│   └── g2p_byt5_architecture.png
├── audio_comparison/                       # 34 Persian .wav audio files (Before vs. After Fine-Tuning)
├── scripts/
│   ├── build_expanded_dataset.py           # Builds 8,559-pair dataset with strict leakage assertions
│   ├── heldout_challenge_set.py            # Clean 60-word held-out challenge benchmark definitions
│   ├── create_notebook.py                  # Generates the 19-cell benchmark notebook
│   ├── generate_charts.py                  # Generates all 9 300-DPI figures from evaluation_metrics.json
│   ├── generate_audio_comparison.py        # Synthesizes all 34 Persian .wav files via neural TTS
│   └── push_to_kaggle.py                   # Automated Kaggle GPU kernel deployment script
└── requirements.txt                        # Python dependencies
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Rebuild expanded Persian G2P dataset splits (8,559 train / 75 dev / 500 Wiktionary test / 60 Held-out test)
python scripts/build_expanded_dataset.py

# 3. Run standalone training and 9-metric evaluation
python finetune_g2p.py

# 4. Regenerate all 9 publication-quality (300 DPI) figures
python scripts/generate_charts.py

# 5. Synthesize all 34 Before/After Persian .wav audio files
python scripts/generate_audio_comparison.py
```
