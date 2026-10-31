#!/usr/bin/env python3
"""
Builds a focused, clean, interactive GitHub Pages site (index.html)
strictly dedicated to:
1. Playable Before vs. After .wav audio comparison for all 23 word and sentence pairs.
2. The 9 high-resolution diagnostic charts/figures.
Zero redundant text or repeated README content.
"""

import os
import json

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
OUT_HTML = os.path.join(BASE_DIR, "index.html")

SAMPLES = [
    # Held-out challenge words
    {
        "id": "modele_amari",
        "persian": "مدل آماری",
        "category": "Held-Out: Compound & Ezafe",
        "gt": "/modeleɒːmɒːɾiː/",
        "before_ipa": "/mædælɒːmɒːɾiː/",
        "after_ipa": "/modeleɒːmɒːɾiː/",
        "highlight": "Kasre Ezafe (-e-) restored"
    },
    {
        "id": "shabake_amigh",
        "persian": "شبکه عمیق",
        "category": "Held-Out: Compound & Ezafe",
        "gt": "/ʃæbækejeʔæmiːɢ/",
        "before_ipa": "/ʃæbkeʔæmiːɢ/",
        "after_ipa": "/ʃæbækejeʔæmiːɢ/",
        "highlight": "Ezafe glide (-ye-) & glottal /ʔ/ restored"
    },
    {
        "id": "taamol",
        "persian": "تعامل",
        "category": "Held-Out: Gemination & Glottal",
        "gt": "/tæʔɒːmol/",
        "before_ipa": "/tæʔɒːmæl/",
        "after_ipa": "/tæʔɒːmol/",
        "highlight": "Glottal stop /ʔ/ and Zamme /o/ corrected"
    },
    {
        "id": "vizhegi",
        "persian": "ویژگی",
        "category": "Held-Out: Short Vowels",
        "gt": "/viːʒeɡiː/",
        "before_ipa": "/viːʒɡiː/",
        "after_ipa": "/viːʒeɡiː/",
        "highlight": "Unwritten Kasre /e/ restored"
    },
    {
        "id": "pazhuheshgar",
        "persian": "پژوهشگر",
        "category": "Held-Out: Short Vowels",
        "gt": "/pæʒuːheʃɡæɾ/",
        "before_ipa": "/peʒuːheʃɡæɾ/",
        "after_ipa": "/pæʒuːheʃɡæɾ/",
        "highlight": "Initial Fathe /æ/ distinction corrected"
    },
    {
        "id": "processor",
        "persian": "پروسسور",
        "category": "Held-Out: Tech Loanword",
        "gt": "/pɾosesoɾ/",
        "before_ipa": "/poɾuːssuːɾ/",
        "after_ipa": "/pɾosesoɾ/",
        "highlight": "Epenthesis removed, accurate tech pronunciation"
    },
    # Core Tech Loanwords
    {
        "id": "pytorch",
        "persian": "پایتورچ",
        "category": "Tech Loanword",
        "gt": "/pɒːjtɔːɾt͡ʃ/",
        "before_ipa": "/pɒːjtuːɾt͡ʃ/",
        "after_ipa": "/pɒːjtɔːɾt͡ʃ/",
        "highlight": "Proper English loanword adaptation"
    },
    {
        "id": "tensorflow",
        "persian": "تنسورفلو",
        "category": "Tech Loanword",
        "gt": "/tensoɾfloʊ/",
        "before_ipa": "/tænsuːɾfluː/",
        "after_ipa": "/tensoɾfloʊ/",
        "highlight": "Vowel diphthong /oʊ/ & vowels corrected"
    },
    {
        "id": "transformer",
        "persian": "ترنسفورمر",
        "category": "Tech Loanword",
        "gt": "/tɾænsfoɾmeɾ/",
        "before_ipa": "/teɾænsfuːɾmeɾ/",
        "after_ipa": "/tɾænsfoɾmeɾ/",
        "highlight": "Natural consonant cluster without spurious vowel"
    },
    {
        "id": "algoritm",
        "persian": "الگوریتم",
        "category": "Tech Loanword",
        "gt": "/ʔælɡoɾiːtm/",
        "before_ipa": "/ælɡuːɾiːtm/",
        "after_ipa": "/ʔælɡoɾiːtm/",
        "highlight": "Accurate Zamme /o/ vocalization"
    },
    {
        "id": "kaggle",
        "persian": "کگل",
        "category": "Tech Loanword",
        "gt": "/kæɡel/",
        "before_ipa": "/kæɡl/",
        "after_ipa": "/kæɡel/",
        "highlight": "Vowel epenthesis /e/ placed correctly"
    },
    {
        "id": "keras",
        "persian": "کراس",
        "category": "Tech Loanword",
        "gt": "/keɾɒːs/",
        "before_ipa": "/kɾɒːs/",
        "after_ipa": "/keɾɒːs/",
        "highlight": "Initial Kasre /e/ vocalization"
    },
    {
        "id": "grapheme",
        "persian": "گرافیم",
        "category": "Linguistic Term",
        "gt": "/ɡeɾɒːfem/",
        "before_ipa": "/ɡɾɒːfiːm/",
        "after_ipa": "/ɡeɾɒːfem/",
        "highlight": "Short Kasre vowel accuracy"
    },
    # Phonological phenomena
    {
        "id": "shabake",
        "persian": "شبکه",
        "category": "Short Vowels",
        "gt": "/ʃæbæke/",
        "before_ipa": "/ʃæbke/",
        "after_ipa": "/ʃæbæke/",
        "highlight": "Interior Fathe /æ/ restored"
    },
    {
        "id": "model",
        "persian": "مدل",
        "category": "Short Vowels",
        "gt": "/model/",
        "before_ipa": "/mædæl/",
        "after_ipa": "/model/",
        "highlight": "Accurate /o/ and /e/ vowels"
    },
    {
        "id": "mohandesi",
        "persian": "مهندسی",
        "category": "Short Vowels",
        "gt": "/mohændesiː/",
        "before_ipa": "/mehændesiː/",
        "after_ipa": "/mohændesiː/",
        "highlight": "Initial Zamme /o/ distinction"
    },
    {
        "id": "mohasebe",
        "persian": "محاسبه",
        "category": "Short Vowels",
        "gt": "/mohɒːsebe/",
        "before_ipa": "/mæhɒːsebe/",
        "after_ipa": "/mohɒːsebe/",
        "highlight": "Initial Zamme /o/ vs Fathe /æ/"
    },
    {
        "id": "behine",
        "persian": "بهینه‌سازی",
        "category": "Compound & ZWNJ",
        "gt": "/behiːnesɒːziː/",
        "before_ipa": "/bæhiːnesɒːziː/",
        "after_ipa": "/behiːnesɒːziː/",
        "highlight": "Half-space (ZWNJ) boundary preserved"
    },
    {
        "id": "motaghayyer",
        "persian": "متغیر",
        "category": "Gemination (Tashdid)",
        "gt": "/motæɢæjjeɾ/",
        "before_ipa": "/motæɢɒːjeɾ/",
        "after_ipa": "/motæɢæjjeɾ/",
        "highlight": "Doubled /jj/ Tashdid gemination"
    },
    {
        "id": "tavajjoh",
        "persian": "توجه",
        "category": "Gemination (Tashdid)",
        "gt": "/tævæd͡ʒd͡ʒoh/",
        "before_ipa": "/tævæd͡ʒoh/",
        "after_ipa": "/tævæd͡ʒd͡ʒoh/",
        "highlight": "Geminated /d͡ʒd͡ʒ/ Tashdid restored"
    },
    # Sentences
    {
        "id": "sentence1",
        "persian": "مدل ترنسفورمر و شبکه عصبی در پایتورچ و تنسورفلو الگوریتم گرافیم را بهینه‌سازی می‌کنند.",
        "category": "Full Sentence 1 (AI Pipeline)",
        "gt": "Sentence 1 (Deep Learning)",
        "before_ipa": "Baseline: Omitted Ezafe & Distorted Vowels",
        "after_ipa": "Fine-Tuned: Fluent Persian Technical Speech",
        "highlight": "Continuous phonetic prosody and Ezafe"
    },
    {
        "id": "sentence2",
        "persian": "مهندسی پزشکی و محاسبه دقیق متغیرها در هوش مصنوعی و یادگیری ماشین توجه زیادی دارد.",
        "category": "Full Sentence 2 (Scientific Speech)",
        "gt": "Sentence 2 (Medical & AI)",
        "before_ipa": "Baseline: Missing Geminates & Glottal",
        "after_ipa": "Fine-Tuned: Exact Tashdid & Natural Flow",
        "highlight": "Tashdid on Daqiq, Motaghayyer & Tavajjoh"
    },
    {
        "id": "sentence3",
        "persian": "پژوهشگران علم داده در کگل با فریم‌ورک دیپ‌لرنینگ و توکنایزر مسئله اتصال را حل کردند.",
        "category": "Full Sentence 3 (Data Science)",
        "gt": "Sentence 3 (Data Science)",
        "before_ipa": "Baseline: Mispronounced Loanwords",
        "after_ipa": "Fine-Tuned: Precise Tech Loanword Cadence",
        "highlight": "Kaggle, Deep Learning, Tokenizer loanwords"
    }
]

FIGURES = [
    {
        "file": "figures/dataset_distribution.png",
        "title": "Figure 1: Corpus EDA & Phoneme Distribution",
        "caption": "Grapheme vs. phoneme length regression on 116K+ KaamelDict words, and top 20 Persian phonemes."
    },
    {
        "file": "figures/loss_curve.png",
        "title": "Figure 2: Loss Convergence & Cosine Learning Rate",
        "caption": "Cross-entropy loss trajectory on Tesla T4 GPU over 3 epochs (5,400 steps) with validation checkpoints."
    },
    {
        "file": "figures/per_wer_metrics_comparison.png",
        "title": "Figure 3: Multi-Metric Benchmark Comparison",
        "caption": "Dual-panel bar chart comparing PER, WER, Word Acc, Sim, and chrF before vs. after fine-tuning."
    },
    {
        "file": "figures/category_breakdown.png",
        "title": "Figure 4: Stratified Linguistic Category Breakdown",
        "caption": "Error reduction across AI Loanwords, Short Vowels, Ezafe, and Gemination/Glottal phenomena."
    },
    {
        "file": "figures/edit_operations_breakdown.png",
        "title": "Figure 5: Levenshtein Edit Operations Breakdown",
        "caption": "Substitutions (vowel errors), deletions (dropped short vowels/geminates), and insertions."
    },
    {
        "file": "figures/length_bucket_analysis.png",
        "title": "Figure 6: Word Length Bucket Error Analysis",
        "caption": "Error rate progression across short (1-4), medium (5-7), long (8-10), and compound (11+) words."
    },
    {
        "file": "figures/vowel_confusion_matrix.png",
        "title": "Figure 7: Persian Vowel Confusion Matrix",
        "caption": "Row-normalized confusion among Persian vowels /æ, e, o, ɒː, iː, uː/ before and after fine-tuning."
    },
    {
        "file": "figures/word_edit_distance_reduction.png",
        "title": "Figure 8: Word-Level Edit Distance Reduction",
        "caption": "Levenshtein distance error reduction across 20 held-out Persian challenge words."
    },
    {
        "file": "figures/g2p_byt5_architecture.png",
        "title": "Figure 9: ByT5 Persian G2P Architecture",
        "caption": "Raw byte sequence-to-sequence conversion pipeline from Persian orthography to IPA phonemes."
    }
]

html = f"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Persian G2P ByT5: Audio Comparison (.wav) & Diagnostic Charts</title>
  <meta name="description" content="Interactive Before vs. After audio comparison (.wav) and diagnostic visualizations for Persian G2P ByT5 fine-tuned on KaamelDict (116K+ words).">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@500;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', sans-serif; }}
    .font-fa {{ font-family: 'Vazirmatn', sans-serif; }}
    .font-mono {{ font-family: 'Fira Code', monospace; }}
    audio {{ filter: drop-shadow(0 1px 1px rgba(0,0,0,0.05)); }}
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Minimal Header -->
  <header class="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <span class="px-2.5 py-1 rounded bg-indigo-600 text-white font-bold text-xs tracking-wider">ByT5 G2P</span>
        <h1 class="text-base sm:text-lg font-bold text-white tracking-tight">Persian Audio Comparison & Diagnostic Charts</h1>
      </div>
      <div class="flex items-center space-x-3 text-sm">
        <a href="#audio-showcase" class="text-slate-300 hover:text-white transition">Audio (.wav)</a>
        <span class="text-slate-600">&bull;</span>
        <a href="#diagnostic-charts" class="text-slate-300 hover:text-white transition">9 Charts</a>
        <span class="text-slate-600">&bull;</span>
        <a href="https://github.com/SoroushSoleimani/g2p-byt5-finetune" target="_blank" rel="noopener"
           class="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition">
          GitHub Repo
        </a>
      </div>
    </div>
  </header>

  <!-- Hero Summary -->
  <section class="max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-6">
    <div class="flex flex-col md:flex-row md:items-end justify-between gap-4 pb-6 border-b border-slate-800">
      <div>
        <p class="text-xs uppercase font-semibold text-indigo-400 tracking-wider">Before vs. After Fine-Tuning on KaamelDict (116K+ Words)</p>
        <h2 class="text-2xl sm:text-3xl font-extrabold text-white mt-1">Interactive Audio & Visual Comparison</h2>
      </div>
      <div class="flex flex-wrap gap-2 text-xs">
        <span class="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 font-mono">23 Comparison Pairs (46 .wav files)</span>
        <span class="px-3 py-1.5 rounded-lg bg-emerald-950/80 border border-emerald-700 text-emerald-300 font-mono">-52.4% PER on Challenge Set</span>
        <span class="px-3 py-1.5 rounded-lg bg-indigo-950/80 border border-indigo-700 text-indigo-300 font-mono">Word Acc: 21.7% &rarr; 51.7%</span>
      </div>
    </div>
  </section>

  <!-- SECTION 1: AUDIO SAMPLES (.WAV) -->
  <section id="audio-showcase" class="max-w-7xl mx-auto px-4 sm:px-6 py-6">
    <div class="flex items-center justify-between mb-6">
      <h3 class="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
        <span>Playable Audio Comparison (.wav)</span>
        <span class="text-xs font-normal text-slate-400">({len(SAMPLES)} Examples)</span>
      </h3>
      <span class="text-xs text-slate-400">Synthesized with Neural TTS (fa-IR-FaridNeural)</span>
    </div>

    <!-- Cards Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
"""

for s in SAMPLES:
    b_wav = f"audio_comparison/{s['id']}_before_finetune.wav"
    a_wav = f"audio_comparison/{s['id']}_after_finetune.wav"

    html += f"""
      <div class="bg-slate-900 border border-slate-800 hover:border-slate-700 rounded-xl p-5 shadow-sm flex flex-col justify-between transition-colors">
        <div>
          <!-- Header of Card -->
          <div class="flex items-center justify-between mb-2">
            <span class="text-[11px] font-semibold px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-indigo-300">{s['category']}</span>
            <span class="text-[11px] text-emerald-400 font-medium">{s['highlight']}</span>
          </div>

          <!-- Persian text -->
          <div class="my-3 text-right">
            <div class="font-fa text-2xl font-bold text-white tracking-wide" dir="rtl">{s['persian']}</div>
          </div>

          <!-- IPA Comparison Box -->
          <div class="space-y-1.5 text-xs font-mono bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 my-3">
            <div class="flex justify-between items-baseline text-slate-400">
              <span class="text-[10px] uppercase font-sans text-slate-500">Target IPA:</span>
              <span class="text-slate-200">{s['gt']}</span>
            </div>
            <div class="flex justify-between items-baseline text-rose-400">
              <span class="text-[10px] uppercase font-sans text-rose-500/80">Before (Base):</span>
              <span class="line-through opacity-80">{s['before_ipa']}</span>
            </div>
            <div class="flex justify-between items-baseline text-emerald-400 font-semibold">
              <span class="text-[10px] uppercase font-sans text-emerald-500/80">After (FT):</span>
              <span>{s['after_ipa']}</span>
            </div>
          </div>
        </div>

        <!-- Audio Players -->
        <div class="mt-4 pt-4 border-t border-slate-800/80 space-y-3">
          <div>
            <div class="flex items-center justify-between text-[11px] font-medium text-rose-400 mb-1">
              <span>Before Fine-Tuning</span>
              <span class="text-[10px] font-mono text-slate-500">.wav</span>
            </div>
            <audio controls class="w-full h-8 rounded" preload="none">
              <source src="{b_wav}" type="audio/wav">
              Your browser does not support audio.
            </audio>
          </div>

          <div>
            <div class="flex items-center justify-between text-[11px] font-medium text-emerald-400 mb-1">
              <span>After Fine-Tuning</span>
              <span class="text-[10px] font-mono text-emerald-500/60">.wav</span>
            </div>
            <audio controls class="w-full h-8 rounded" preload="none">
              <source src="{a_wav}" type="audio/wav">
              Your browser does not support audio.
            </audio>
          </div>
        </div>
      </div>
"""

html += f"""
    </div>
  </section>

  <!-- SECTION 2: 9 DIAGNOSTIC CHARTS -->
  <section id="diagnostic-charts" class="max-w-7xl mx-auto px-4 sm:px-6 py-12 border-t border-slate-800">
    <div class="flex items-center justify-between mb-8">
      <div>
        <h3 class="text-xl font-bold text-white tracking-tight">9 Diagnostic Charts & Visualizations (300 DPI)</h3>
        <p class="text-xs text-slate-400 mt-1">Generated from empirical Kaggle evaluation metrics on KaamelDict (115K train, 500 test, 60 challenge words).</p>
      </div>
      <a href="g2p_implementation_and_finetune.ipynb" download
         class="text-xs font-semibold px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 transition">
        Download Executed Notebook (.ipynb)
      </a>
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
"""

for fig in FIGURES:
    html += f"""
      <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden hover:border-slate-700 transition flex flex-col justify-between">
        <a href="{fig['file']}" target="_blank" rel="noopener" class="block bg-slate-950 p-3 group">
          <img src="{fig['file']}" alt="{fig['title']}" class="w-full h-52 object-contain group-hover:scale-[1.02] transition duration-200" loading="lazy">
        </a>
        <div class="p-4 border-t border-slate-800">
          <h4 class="font-bold text-sm text-white">{fig['title']}</h4>
          <p class="text-xs text-slate-400 mt-1 leading-relaxed">{fig['caption']}</p>
          <a href="{fig['file']}" target="_blank" class="mt-3 inline-flex items-center text-xs font-medium text-indigo-400 hover:text-indigo-300 transition">
            View Full 300 DPI Image &rarr;
          </a>
        </div>
      </div>
"""

html += f"""
    </div>
  </section>

  <!-- Minimal Footer -->
  <footer class="border-t border-slate-800 py-8 bg-slate-950">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-4">
      <div>
        Persian G2P ByT5 Fine-Tuning &bull; KaamelDict Corpus (116K+ Words) &bull; 2026
      </div>
      <div class="flex items-center space-x-4">
        <a href="g2p_implementation_and_finetune.ipynb" download class="hover:text-slate-300 transition">Download Notebook</a>
        <a href="evaluation_metrics.json" download class="hover:text-slate-300 transition">Metrics JSON</a>
        <a href="https://github.com/SoroushSoleimani/g2p-byt5-finetune" target="_blank" class="hover:text-slate-300 transition">GitHub Repository</a>
      </div>
    </div>
  </footer>

</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[+] Successfully generated pure audio & charts index.html ({len(html):,} chars).")
