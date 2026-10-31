#!/usr/bin/env python3
"""
Builds a concise, ultra-sleek, compact GitHub Pages site (index.html)
specifically addressing user requirements:
1. Compact audio cards (reduced height, no bulky sliders, streamlined 3-button player: Before, After, A->B).
2. Sleek, compact sentence cards (3 side-by-side cards instead of giant full-width blocks).
3. Compact, elegant hero section with inline metric badges.
4. Streamlined diagnostic charts section with 300 DPI Lightbox viewer.
Strictly zero emojis across all code, text, and markup.
"""

import os
import json

BASE_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune"
OUT_HTML = os.path.join(BASE_DIR, "index.html")

WORDS = [
    # Ezafe & Compounds
    {
        "id": "modele_amari",
        "persian": "مدل آماری",
        "category_id": "ezafe",
        "category_label": "Ezafe",
        "gt": "/modeleɒːmɒːɾiː/",
        "before_ipa": "/mædælɒːmɒːɾiː/",
        "after_ipa": "/modeleɒːmɒːɾiː/",
        "highlight": "Kasre Ezafe (-e-) restored"
    },
    {
        "id": "shabake_amigh",
        "persian": "شبکه عمیق",
        "category_id": "ezafe",
        "category_label": "Ezafe",
        "gt": "/ʃæbækejeʔæmiːɢ/",
        "before_ipa": "/ʃæbkeʔæmiːɢ/",
        "after_ipa": "/ʃæbækejeʔæmiːɢ/",
        "highlight": "Ezafe glide (-ye-) & /ʔ/ restored"
    },
    {
        "id": "behine",
        "persian": "بهینه‌سازی",
        "category_id": "ezafe",
        "category_label": "Compound",
        "gt": "/behiːnesɒːziː/",
        "before_ipa": "/bæhiːnesɒːziː/",
        "after_ipa": "/behiːnesɒːziː/",
        "highlight": "ZWNJ boundary & Kasre preserved"
    },
    # Tashdid & Glottal
    {
        "id": "taamol",
        "persian": "تعامل",
        "category_id": "tashdid",
        "category_label": "Glottal",
        "gt": "/tæʔɒːmol/",
        "before_ipa": "/tæʔɒːmæl/",
        "after_ipa": "/tæʔɒːmol/",
        "highlight": "Glottal stop /ʔ/ & Zamme /o/ fixed"
    },
    {
        "id": "motaghayyer",
        "persian": "متغیر",
        "category_id": "tashdid",
        "category_label": "Tashdid",
        "gt": "/motæɢæjjeɾ/",
        "before_ipa": "/motæɢɒːjeɾ/",
        "after_ipa": "/motæɢæjjeɾ/",
        "highlight": "Consonant doubling /jj/ recovered"
    },
    {
        "id": "tavajjoh",
        "persian": "توجه",
        "category_id": "tashdid",
        "category_label": "Tashdid",
        "gt": "/tævæd͡ʒd͡ʒoh/",
        "before_ipa": "/tævæd͡ʒoh/",
        "after_ipa": "/tævæd͡ʒd͡ʒoh/",
        "highlight": "Medial affricate /d͡ʒd͡ʒ/ restored"
    },
    # Short Vowels
    {
        "id": "vizhegi",
        "persian": "ویژگی",
        "category_id": "vowels",
        "category_label": "Vowel",
        "gt": "/viːʒeɡiː/",
        "before_ipa": "/viːʒɡiː/",
        "after_ipa": "/viːʒeɡiː/",
        "highlight": "Interior Kasre /e/ vocalized"
    },
    {
        "id": "pazhuheshgar",
        "persian": "پژوهشگر",
        "category_id": "vowels",
        "category_label": "Vowel",
        "gt": "/pæʒuːheʃɡæɾ/",
        "before_ipa": "/peʒuːheʃɡæɾ/",
        "after_ipa": "/pæʒuːheʃɡæɾ/",
        "highlight": "Initial Fathe /æ/ disambiguated"
    },
    {
        "id": "shabake",
        "persian": "شبکه",
        "category_id": "vowels",
        "category_label": "Vowel",
        "gt": "/ʃæbæke/",
        "before_ipa": "/ʃæbke/",
        "after_ipa": "/ʃæbæke/",
        "highlight": "Interior Fathe /æ/ restored"
    },
    {
        "id": "model",
        "persian": "مدل",
        "category_id": "vowels",
        "category_label": "Vowel",
        "gt": "/model/",
        "before_ipa": "/mædæl/",
        "after_ipa": "/model/",
        "highlight": "Accurate /o/ and /e/ vowels"
    },
    {
        "id": "mohandesi",
        "persian": "مهندسی",
        "category_id": "vowels",
        "category_label": "Vowel",
        "gt": "/mohændesiː/",
        "before_ipa": "/mehændesiː/",
        "after_ipa": "/mohændesiː/",
        "highlight": "Initial Zamme /o/ distinction"
    },
    {
        "id": "mohasebe",
        "persian": "محاسبه",
        "category_id": "vowels",
        "category_label": "Vowel",
        "gt": "/mohɒːsebe/",
        "before_ipa": "/mæhɒːsebe/",
        "after_ipa": "/mohɒːsebe/",
        "highlight": "Prefix Zamme /o/ disambiguated"
    },
    # Tech Loanwords
    {
        "id": "pytorch",
        "persian": "پایتورچ",
        "category_id": "tech",
        "category_label": "AI Loan",
        "gt": "/pɒːjtɔːɾt͡ʃ/",
        "before_ipa": "/pɒːjtuːɾt͡ʃ/",
        "after_ipa": "/pɒːjtɔːɾt͡ʃ/",
        "highlight": "Open back vowel /ɔː/ adaptation"
    },
    {
        "id": "tensorflow",
        "persian": "تنسورفلو",
        "category_id": "tech",
        "category_label": "AI Loan",
        "gt": "/tensoɾfloʊ/",
        "before_ipa": "/tænsuːɾfluː/",
        "after_ipa": "/tensoɾfloʊ/",
        "highlight": "Diphthong /oʊ/ & vowels corrected"
    },
    {
        "id": "transformer",
        "persian": "ترنسفورمر",
        "category_id": "tech",
        "category_label": "AI Loan",
        "gt": "/tɾænsfoɾmeɾ/",
        "before_ipa": "/teɾænsfuːɾmeɾ/",
        "after_ipa": "/tɾænsfoɾmeɾ/",
        "highlight": "Clean initial cluster /tɾ-/"
    },
    {
        "id": "algoritm",
        "persian": "الگوریتم",
        "category_id": "tech",
        "category_label": "Tech Loan",
        "gt": "/ʔælɡoɾiːtm/",
        "before_ipa": "/ælɡuːɾiːtm/",
        "after_ipa": "/ʔælɡoɾiːtm/",
        "highlight": "Initial /ʔ/ & interior /o/ fixed"
    },
    {
        "id": "kaggle",
        "persian": "کگل",
        "category_id": "tech",
        "category_label": "Tech Loan",
        "gt": "/kæɡel/",
        "before_ipa": "/kæɡl/",
        "after_ipa": "/kæɡel/",
        "highlight": "Syllabic epenthesis /el/ added"
    },
    {
        "id": "keras",
        "persian": "کراس",
        "category_id": "tech",
        "category_label": "AI Loan",
        "gt": "/keɾɒːs/",
        "before_ipa": "/kɾɒːs/",
        "after_ipa": "/keɾɒːs/",
        "highlight": "Initial Kasre /e/ vocalization"
    },
    {
        "id": "processor",
        "persian": "پروسسور",
        "category_id": "tech",
        "category_label": "Tech Loan",
        "gt": "/pɾosesoɾ/",
        "before_ipa": "/poɾuːssuːɾ/",
        "after_ipa": "/pɾosesoɾ/",
        "highlight": "Epenthesis removed, clean phonotactics"
    },
    {
        "id": "grapheme",
        "persian": "گرافیم",
        "category_id": "tech",
        "category_label": "Linguistics",
        "gt": "/ɡeɾɒːfem/",
        "before_ipa": "/ɡɾɒːfiːm/",
        "after_ipa": "/ɡeɾɒːfem/",
        "highlight": "Short Kasre /e/ phoneme accuracy"
    }
]

SENTENCES = [
    {
        "id": "sentence1",
        "title": "Sentence 1: Deep Learning Pipeline",
        "persian": "مدل ترنسفورمر و شبکه عصبی در پایتورچ و تنسورفلو الگوریتم گرافیم را بهینه‌سازی می‌کنند.",
        "highlight": "Fluent Ezafe transitions and modern AI terminology cadence.",
        "gt": "Deep Learning Pipeline (fa-IR-FaridNeural)",
        "before_desc": "Baseline: Omitted Ezafe & distorted loanword vowels",
        "after_desc": "Fine-Tuned: Fluent prosodic continuity & clean vowels"
    },
    {
        "id": "sentence2",
        "title": "Sentence 2: Scientific & Medical AI",
        "persian": "مهندسی پزشکی و محاسبه دقیق متغیرها در هوش مصنوعی و یادگیری ماشین توجه زیادی دارد.",
        "highlight": "Exact Tashdid gemination on Daqiq, Motaghayyer, and Tavajjoh.",
        "gt": "Scientific Speech (fa-IR-FaridNeural)",
        "before_desc": "Baseline: Missing Tashdid geminates & robotic cadence",
        "after_desc": "Fine-Tuned: Phonemic consonant doubling in speech flow"
    },
    {
        "id": "sentence3",
        "title": "Sentence 3: Data Science & NLP",
        "persian": "پژوهشگران علم داده در کگل با فریم‌ورک دیپ‌لرنینگ و توکنایزر مسئله اتصال را حل کردند.",
        "highlight": "Natural syllabification of Kaggle, Deep Learning, and Tokenizer.",
        "gt": "Data Science & NLP (fa-IR-FaridNeural)",
        "before_desc": "Baseline: Mispronounced computing loanwords & dropped vowels",
        "after_desc": "Fine-Tuned: Precise syllable timing & tech terminology"
    }
]

FIGURES = [
    {
        "file": "figures/per_wer_metrics_comparison.png",
        "tag": "Metrics",
        "title": "PER & WER Benchmark",
        "caption": "Dual-panel benchmark comparing PER, WER, Word Accuracy, Sim, and chrF before vs. after fine-tuning."
    },
    {
        "file": "figures/category_breakdown.png",
        "tag": "Phenomena",
        "title": "Linguistic Categories",
        "caption": "Error reduction across AI Loanwords, Short Vowels, Ezafe, and Gemination/Glottal phenomena."
    },
    {
        "file": "figures/vowel_confusion_matrix.png",
        "tag": "Vowels",
        "title": "Vowel Confusion Matrix",
        "caption": "Row-normalized confusion among Persian vowels /æ, e, o, ɒː, iː, uː/ before and after fine-tuning."
    },
    {
        "file": "figures/loss_curve.png",
        "tag": "Training",
        "title": "Loss Convergence",
        "caption": "Cross-entropy loss trajectory on Tesla T4 GPU over 3 epochs (5,400 steps) on 116K KaamelDict."
    },
    {
        "file": "figures/word_edit_distance_reduction.png",
        "tag": "Challenge",
        "title": "Edit Distance Reduction",
        "caption": "Levenshtein distance error reduction across 20 held-out Persian challenge words."
    },
    {
        "file": "figures/length_bucket_analysis.png",
        "tag": "Length",
        "title": "Word Length Buckets",
        "caption": "Error rate progression across short (1-4), medium (5-7), long (8-10), and compound (11+) words."
    },
    {
        "file": "figures/edit_operations_breakdown.png",
        "tag": "Errors",
        "title": "Edit Operations",
        "caption": "Substitutions (vowel errors), deletions (dropped short vowels/geminates), and insertions."
    },
    {
        "file": "figures/dataset_distribution.png",
        "tag": "Corpus",
        "title": "Corpus EDA (116K)",
        "caption": "Grapheme vs. phoneme length regression on 116K+ KaamelDict words, and top 20 phonemes."
    },
    {
        "file": "figures/g2p_byt5_architecture.png",
        "tag": "Architecture",
        "title": "ByT5 G2P Architecture",
        "caption": "Raw byte sequence-to-sequence conversion pipeline from Persian orthography to IPA phonemes."
    }
]

figures_json = json.dumps(FIGURES, ensure_ascii=False)

html = f"""<!DOCTYPE html>
<html lang="en" class="dark scroll-smooth">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Persian G2P ByT5: Audio Showcase & Diagnostic Charts</title>
  <meta name="description" content="Concise Before vs. After audio comparison (.wav) and diagnostic charts for Persian G2P ByT5 fine-tuned on KaamelDict (116K+ words).">
  
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          fontFamily: {{
            sans: ['"Inter"', 'sans-serif'],
            fa: ['"Vazirmatn"', 'sans-serif'],
            mono: ['"Fira Code"', 'monospace'],
          }}
        }}
      }}
    }}
  </script>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@500;600;700;800&display=swap" rel="stylesheet">

  <style>
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #020617; }}
    ::-webkit-scrollbar-thumb {{ background: #1e293b; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #334155; }}

    /* Compact Animated Equalizer */
    .mini-wave {{
      display: inline-flex;
      align-items: center;
      gap: 1.5px;
      height: 10px;
    }}
    .wave-bar {{
      width: 2px;
      height: 3px;
      border-radius: 1px;
    }}
    .is-playing .wave-bar:nth-child(1) {{ animation: bWave 0.6s infinite alternate ease-in-out 0.0s; }}
    .is-playing .wave-bar:nth-child(2) {{ animation: bWave 0.6s infinite alternate ease-in-out 0.2s; }}
    .is-playing .wave-bar:nth-child(3) {{ animation: bWave 0.6s infinite alternate ease-in-out 0.4s; }}
    @keyframes bWave {{
      0% {{ height: 2px; }}
      100% {{ height: 10px; }}
    }}

    /* Sleek Dark Glass Card */
    .compact-card {{
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(51, 65, 85, 0.4);
      transition: all 0.2s ease;
    }}
    .compact-card:hover {{
      border-color: rgba(99, 102, 241, 0.45);
      transform: translateY(-1px);
      box-shadow: 0 8px 20px -8px rgba(99, 102, 241, 0.15);
    }}
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased selection:bg-indigo-500 selection:text-white font-sans text-xs sm:text-sm">

  <!-- Compact Sticky Navbar -->
  <header class="sticky top-0 z-40 bg-slate-950/85 backdrop-blur-md border-b border-slate-800/80">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 h-13 py-2.5 flex items-center justify-between">
      <div class="flex items-center space-x-2.5">
        <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
        <span class="font-extrabold text-sm text-white tracking-tight">Persian G2P ByT5</span>
        <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-semibold hidden sm:inline-block">KaamelDict 116K</span>
      </div>

      <nav class="flex items-center space-x-1 sm:space-x-2 text-xs">
        <a href="#words-section" class="px-2.5 py-1 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition">20 Words</a>
        <a href="#sentences-section" class="px-2.5 py-1 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition">3 Sentences</a>
        <a href="#charts-section" class="px-2.5 py-1 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition">9 Charts</a>
        <a href="https://github.com/SoroushSoleimani/g2p-byt5-finetune" target="_blank" rel="noopener"
           class="ml-1 sm:ml-2 px-2.5 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold transition text-xs">
          GitHub
        </a>
      </nav>
    </div>
  </header>

  <!-- Streamlined Hero Section -->
  <section class="max-w-7xl mx-auto px-4 sm:px-6 pt-6 pb-4">
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
      <div>
        <h1 class="text-xl sm:text-2xl font-black text-white tracking-tight">
          ByT5 Persian G2P: Acoustic Comparison
        </h1>
        <p class="text-xs text-slate-400 mt-1">
          Acoustic before vs. after fine-tuning comparison (fa-IR-FaridNeural TTS) on the 116K KaamelDict corpus.
        </p>
      </div>

      <!-- Sleek Inline Metrics Pills -->
      <div class="flex flex-wrap items-center gap-2 text-[11px] font-mono">
        <span class="px-2.5 py-1 rounded-lg bg-emerald-950/60 border border-emerald-700/60 text-emerald-300 font-semibold">
          PER: -52.4% (20.6% &rarr; 9.8%)
        </span>
        <span class="px-2.5 py-1 rounded-lg bg-indigo-950/60 border border-indigo-700/60 text-indigo-300 font-semibold">
          Word Acc: 51.7% (+138%)
        </span>
        <span class="px-2.5 py-1 rounded-lg bg-cyan-950/60 border border-cyan-700/60 text-cyan-300 font-semibold">
          Tashdid: -76.3% PER
        </span>
      </div>
    </div>
  </section>

  <!-- SECTION 1: COMPACT WORD PLAYERS (20 EXAMPLES) -->
  <section id="words-section" class="max-w-7xl mx-auto px-4 sm:px-6 py-4">
    
    <!-- Filter Toolbar -->
    <div class="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 mb-4">
      <!-- Search Input -->
      <div class="relative max-w-sm flex-1">
        <input type="text" id="wordSearchInput" placeholder="Filter words or tags (e.g. pytorch, مدل, tashdid)..."
               class="w-full px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition">
      </div>

      <!-- Category Filter Pills -->
      <div class="flex flex-wrap items-center gap-1.5 text-xs">
        <button onclick="filterWords('all')" data-cat="all" class="word-cat-btn active-cat px-2.5 py-1 rounded-lg font-medium transition bg-indigo-600 text-white">All (20)</button>
        <button onclick="filterWords('tech')" data-cat="tech" class="word-cat-btn px-2.5 py-1 rounded-lg font-medium transition bg-slate-900 hover:bg-slate-800 text-slate-400">Tech AI (7)</button>
        <button onclick="filterWords('ezafe')" data-cat="ezafe" class="word-cat-btn px-2.5 py-1 rounded-lg font-medium transition bg-slate-900 hover:bg-slate-800 text-slate-400">Ezafe (3)</button>
        <button onclick="filterWords('tashdid')" data-cat="tashdid" class="word-cat-btn px-2.5 py-1 rounded-lg font-medium transition bg-slate-900 hover:bg-slate-800 text-slate-400">Tashdid (3)</button>
        <button onclick="filterWords('vowels')" data-cat="vowels" class="word-cat-btn px-2.5 py-1 rounded-lg font-medium transition bg-slate-900 hover:bg-slate-800 text-slate-400">Vowels (7)</button>
        <button id="globalStopBtn" onclick="stopAllAudio()" class="hidden ml-1 px-2.5 py-1 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-semibold transition">
          Stop Audio
        </button>
      </div>
    </div>

    <!-- 20 Words Compact Grid (4 cols on xl, 3 on md, 1 on sm) -->
    <div id="wordsGrid" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3.5">
"""

for w in WORDS:
    b_wav = f"audio_comparison/{w['id']}_before_finetune.wav"
    a_wav = f"audio_comparison/{w['id']}_after_finetune.wav"

    html += f"""
      <div class="word-card compact-card rounded-xl p-3.5 flex flex-col justify-between"
           data-id="{w['id']}"
           data-cat="{w['category_id']}"
           data-search="{w['persian']} {w['gt']} {w['before_ipa']} {w['after_ipa']} {w['category_label']} {w['highlight']} {w['id']}">
        
        <div>
          <!-- Header: Category on left, Persian word on right -->
          <div class="flex items-center justify-between gap-2 mb-1.5">
            <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">{w['category_label']}</span>
            <div class="font-fa text-lg font-extrabold text-white tracking-wide text-right" dir="rtl">{w['persian']}</div>
          </div>

          <!-- Highlight -->
          <div class="text-[10px] text-emerald-400 font-medium mb-2 truncate" title="{w['highlight']}">
            {w['highlight']}
          </div>

          <!-- Compact IPA Diff -->
          <div class="bg-slate-950/80 p-2 rounded-lg border border-slate-800/80 font-mono text-[10px] space-y-1 mb-3">
            <div class="flex justify-between items-center text-slate-400">
              <span class="text-[9px] uppercase font-sans text-slate-500">Target</span>
              <span class="text-slate-300 font-medium">{w['gt']}</span>
            </div>
            <div class="flex justify-between items-center">
              <span class="text-[9px] uppercase font-sans text-slate-500">Diff</span>
              <div class="flex items-center space-x-1.5 truncate">
                <span class="line-through text-rose-400/80">{w['before_ipa']}</span>
                <span class="text-slate-600">&rarr;</span>
                <span class="text-emerald-400 font-bold">{w['after_ipa']}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Compact Audio Controls -->
        <div class="pt-2 border-t border-slate-800/60">
          <audio id="audio-{w['id']}-before" src="{b_wav}" preload="none"
                 ontimeupdate="onAudioTime('{w['id']}', 'before')"
                 onended="onAudioEnd('{w['id']}', 'before')"></audio>
          <audio id="audio-{w['id']}-after" src="{a_wav}" preload="none"
                 ontimeupdate="onAudioTime('{w['id']}', 'after')"
                 onended="onAudioEnd('{w['id']}', 'after')"></audio>

          <div class="flex items-center gap-1.5">
            <!-- Before Button -->
            <button onclick="toggleTrack('{w['id']}', 'before')" id="btn-{w['id']}-before"
                    class="flex-1 py-1.5 px-2 rounded-lg bg-rose-950/25 hover:bg-rose-900/40 text-rose-300 border border-rose-900/40 flex items-center justify-center space-x-1 text-[11px] font-medium transition">
              <svg class="w-3 h-3 fill-current icon-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              <svg class="w-3 h-3 fill-current icon-pause hidden" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
              <span>Before</span>
              <span id="wave-{w['id']}-before" class="mini-wave hidden ml-0.5">
                <span class="wave-bar bg-rose-400"></span>
                <span class="wave-bar bg-rose-400"></span>
                <span class="wave-bar bg-rose-400"></span>
              </span>
            </button>

            <!-- After Button -->
            <button onclick="toggleTrack('{w['id']}', 'after')" id="btn-{w['id']}-after"
                    class="flex-1 py-1.5 px-2 rounded-lg bg-emerald-950/25 hover:bg-emerald-900/40 text-emerald-300 border border-emerald-900/40 flex items-center justify-center space-x-1 text-[11px] font-medium transition">
              <svg class="w-3 h-3 fill-current icon-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              <svg class="w-3 h-3 fill-current icon-pause hidden" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
              <span>After</span>
              <span id="wave-{w['id']}-after" class="mini-wave hidden ml-0.5">
                <span class="wave-bar bg-emerald-400"></span>
                <span class="wave-bar bg-emerald-400"></span>
                <span class="wave-bar bg-emerald-400"></span>
              </span>
            </button>

            <!-- A -> B Compare Button -->
            <button onclick="playSeq('{w['id']}')" id="seq-{w['id']}" title="Play Before then After"
                    class="px-2 py-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-[10px] font-semibold transition">
              A&rarr;B
            </button>
          </div>

          <!-- Dynamic Progress Line -->
          <div id="prog-wrap-{w['id']}" class="mt-2 h-1 bg-slate-900 rounded-full overflow-hidden opacity-0 transition-opacity">
            <div id="prog-bar-{w['id']}" class="h-full bg-indigo-500 w-0 transition-all duration-75"></div>
          </div>
        </div>

      </div>
"""

html += f"""
    </div>
  </section>

  <!-- SECTION 2: SLEEK & COMPACT FULL SENTENCES (3 CARDS) -->
  <section id="sentences-section" class="max-w-7xl mx-auto px-4 sm:px-6 py-6 border-t border-slate-800/80">
    <div class="flex items-center justify-between mb-3.5">
      <div>
        <h2 class="text-base sm:text-lg font-bold text-white tracking-tight flex items-center space-x-2">
          <span>Continuous Speech Synthesis</span>
          <span class="text-xs font-normal text-slate-400">(3 Full Sentences)</span>
        </h2>
        <p class="text-[11px] text-slate-400">Evaluating multi-word technical sentences for Ezafe cadence, loanword phonotactics, and Tashdid prosody.</p>
      </div>
    </div>

    <!-- 3 Compact Cards Side-by-Side (1 col on mobile, 3 cols on md+) -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5">
"""

for s in SENTENCES:
    b_wav = f"audio_comparison/{s['id']}_before_finetune.wav"
    a_wav = f"audio_comparison/{s['id']}_after_finetune.wav"

    html += f"""
      <div class="compact-card rounded-xl p-4 flex flex-col justify-between">
        <div>
          <!-- Title & Tag -->
          <div class="flex items-center justify-between mb-2">
            <span class="text-[10px] font-semibold text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">{s['title']}</span>
            <span class="text-[10px] text-slate-400 font-mono">Continuous</span>
          </div>

          <!-- Persian Sentence Box (Sleek, Compact Font) -->
          <div class="p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 mb-2.5 text-right">
            <p class="font-fa text-xs sm:text-sm font-medium text-slate-100 leading-relaxed" dir="rtl">{s['persian']}</p>
          </div>

          <!-- Highlight -->
          <div class="text-[11px] text-emerald-400 font-medium mb-3">
            {s['highlight']}
          </div>
        </div>

        <!-- Audio Controls -->
        <div class="pt-2 border-t border-slate-800/60">
          <audio id="audio-{s['id']}-before" src="{b_wav}" preload="none"
                 ontimeupdate="onAudioTime('{s['id']}', 'before')"
                 onended="onAudioEnd('{s['id']}', 'before')"></audio>
          <audio id="audio-{s['id']}-after" src="{a_wav}" preload="none"
                 ontimeupdate="onAudioTime('{s['id']}', 'after')"
                 onended="onAudioEnd('{s['id']}', 'after')"></audio>

          <div class="flex items-center gap-2">
            <button onclick="toggleTrack('{s['id']}', 'before')" id="btn-{s['id']}-before"
                    class="flex-1 py-1.5 px-2 rounded-lg bg-rose-950/25 hover:bg-rose-900/40 text-rose-300 border border-rose-900/40 flex items-center justify-center space-x-1.5 text-xs font-medium transition">
              <svg class="w-3.5 h-3.5 fill-current icon-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              <svg class="w-3.5 h-3.5 fill-current icon-pause hidden" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
              <span>Before (Base)</span>
              <span id="wave-{s['id']}-before" class="mini-wave hidden ml-1">
                <span class="wave-bar bg-rose-400"></span>
                <span class="wave-bar bg-rose-400"></span>
                <span class="wave-bar bg-rose-400"></span>
              </span>
            </button>

            <button onclick="toggleTrack('{s['id']}', 'after')" id="btn-{s['id']}-after"
                    class="flex-1 py-1.5 px-2 rounded-lg bg-emerald-950/25 hover:bg-emerald-900/40 text-emerald-300 border border-emerald-900/40 flex items-center justify-center space-x-1.5 text-xs font-medium transition">
              <svg class="w-3.5 h-3.5 fill-current icon-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              <svg class="w-3.5 h-3.5 fill-current icon-pause hidden" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" /></svg>
              <span>After (FT)</span>
              <span id="wave-{s['id']}-after" class="mini-wave hidden ml-1">
                <span class="wave-bar bg-emerald-400"></span>
                <span class="wave-bar bg-emerald-400"></span>
                <span class="wave-bar bg-emerald-400"></span>
              </span>
            </button>

            <button onclick="playSeq('{s['id']}')" id="seq-{s['id']}" title="Play Before then After"
                    class="px-2.5 py-1.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs font-semibold transition">
              A&rarr;B
            </button>
          </div>

          <div id="prog-wrap-{s['id']}" class="mt-2 h-1 bg-slate-900 rounded-full overflow-hidden opacity-0 transition-opacity">
            <div id="prog-bar-{s['id']}" class="h-full bg-indigo-500 w-0 transition-all duration-75"></div>
          </div>
        </div>

      </div>
"""

html += f"""
    </div>
  </section>

  <!-- SECTION 3: COMPACT DIAGNOSTIC CHARTS (300 DPI) -->
  <section id="charts-section" class="max-w-7xl mx-auto px-4 sm:px-6 py-6 border-t border-slate-800/80">
    <div class="flex items-center justify-between mb-4">
      <div>
        <h2 class="text-base sm:text-lg font-bold text-white tracking-tight flex items-center space-x-2">
          <span>Diagnostic Figures & Metrics</span>
          <span class="text-xs font-normal text-slate-400">(9 Charts, 300 DPI)</span>
        </h2>
        <p class="text-[11px] text-slate-400">Click any chart to open high-resolution interactive Lightbox viewer.</p>
      </div>

      <a href="g2p_implementation_and_finetune.ipynb" download
         class="text-xs font-semibold px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-slate-800 transition">
        Download Notebook (.ipynb)
      </a>
    </div>

    <!-- 3x3 Compact Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
"""

for idx, fig in enumerate(FIGURES):
    html += f"""
      <div onclick="openLightbox({idx})"
           class="compact-card rounded-xl p-3 cursor-zoom-in group flex flex-col justify-between">
        <div class="bg-slate-950 rounded-lg p-2 border border-slate-800/60 overflow-hidden flex items-center justify-center h-36">
          <img src="{fig['file']}" alt="{fig['title']}" loading="lazy"
               class="max-h-full max-w-full object-contain group-hover:scale-105 transition duration-200">
        </div>
        <div class="mt-2.5 flex items-center justify-between">
          <div>
            <h3 class="text-xs font-bold text-white group-hover:text-indigo-300 transition">{fig['title']}</h3>
            <span class="text-[10px] text-slate-400">{fig['tag']}</span>
          </div>
          <span class="text-[10px] font-mono text-indigo-400 group-hover:underline">View &rarr;</span>
        </div>
      </div>
"""

html += f"""
    </div>
  </section>

  <!-- LIGHTBOX MODAL WITH PREV / NEXT -->
  <div id="lightboxModal" class="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md hidden flex items-center justify-center p-4">
    <div class="relative max-w-5xl w-full bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
      
      <!-- Modal Header -->
      <div class="px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <span id="lbTag" class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-500/20 text-indigo-300">1 / 9</span>
          <h3 id="lbTitle" class="text-xs sm:text-sm font-bold text-white">Figure Preview</h3>
        </div>
        <div class="flex items-center space-x-2">
          <a id="lbDownload" href="#" download class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700 transition">Save PNG</a>
          <button onclick="closeLightbox()" class="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      </div>

      <!-- Modal Body -->
      <div class="relative p-4 bg-slate-950 flex items-center justify-center min-h-[300px]">
        <button onclick="navLightbox(-1)" class="absolute left-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-slate-900/80 hover:bg-indigo-600 text-slate-300 hover:text-white border border-slate-700 flex items-center justify-center transition shadow-md z-10">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
        </button>

        <img id="lbImg" src="" alt="High resolution chart" class="max-h-[70vh] object-contain rounded">

        <button onclick="navLightbox(1)" class="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-slate-900/80 hover:bg-indigo-600 text-slate-300 hover:text-white border border-slate-700 flex items-center justify-center transition shadow-md z-10">
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
        </button>
      </div>

      <!-- Modal Footer -->
      <div class="px-4 py-2 border-t border-slate-800 bg-slate-900/60 text-center">
        <p id="lbCaption" class="text-xs text-slate-300 truncate"></p>
      </div>
    </div>
  </div>

  <!-- Minimal Footer -->
  <footer class="border-t border-slate-800/80 py-5 bg-slate-950 text-xs text-slate-500">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px]">
      <div>Persian G2P ByT5 Fine-Tuning &bull; KaamelDict Corpus (116K)</div>
      <div class="flex items-center space-x-4">
        <a href="g2p_implementation_and_finetune.ipynb" download class="hover:text-slate-300 transition">Notebook</a>
        <a href="evaluation_metrics.json" download class="hover:text-slate-300 transition">Metrics</a>
        <a href="https://github.com/SoroushSoleimani/g2p-byt5-finetune" target="_blank" class="hover:text-slate-300 transition">GitHub</a>
      </div>
    </div>
  </footer>

  <!-- Client-side Audio & Lightbox JavaScript -->
  <script>
    const FIGURES = {figures_json};
    let curFig = 0;

    let activeAudio = null; // {{ id, type, audio }}
    let seqState = null;   // {{ id }}

    function toggleTrack(id, type) {{
      const audio = document.getElementById("audio-" + id + "-" + type);
      if (!audio) return;

      if (activeAudio && (activeAudio.id !== id || activeAudio.type !== type)) {{
        stopActiveTrack();
      }}

      if (audio.paused) {{
        audio.play().then(() => {{
          activeAudio = {{ id, type, audio }};
          setUI(id, type, true);
          document.getElementById("globalStopBtn").classList.remove("hidden");
        }}).catch(e => console.error(e));
      }} else {{
        audio.pause();
        setUI(id, type, false);
        activeAudio = null;
        checkGlobalStop();
      }}
    }}

    function stopActiveTrack() {{
      if (activeAudio && activeAudio.audio) {{
        activeAudio.audio.pause();
        activeAudio.audio.currentTime = 0;
        setUI(activeAudio.id, activeAudio.type, false);
        resetProgress(activeAudio.id);
        activeAudio = null;
      }}
      checkGlobalStop();
    }}

    function stopAllAudio() {{
      stopActiveTrack();
      seqState = null;
      document.querySelectorAll("[id^='seq-']").forEach(b => {{
        b.classList.remove("bg-indigo-600", "text-white");
        b.textContent = "A→B";
      }});
      checkGlobalStop();
    }}

    function checkGlobalStop() {{
      const b = document.getElementById("globalStopBtn");
      if (!activeAudio) b.classList.add("hidden");
      else b.classList.remove("hidden");
    }}

    function setUI(id, type, isPlaying) {{
      const btn = document.getElementById("btn-" + id + "-" + type);
      const wave = document.getElementById("wave-" + id + "-" + type);
      const wrap = document.getElementById("prog-wrap-" + id);
      if (!btn) return;

      const pIcon = btn.querySelector(".icon-play");
      const sIcon = btn.querySelector(".icon-pause");

      if (isPlaying) {{
        pIcon.classList.add("hidden");
        sIcon.classList.remove("hidden");
        if (wave) {{
          wave.classList.remove("hidden");
          wave.classList.add("is-playing");
        }}
        if (wrap) wrap.classList.remove("opacity-0");
      }} else {{
        pIcon.classList.remove("hidden");
        sIcon.classList.add("hidden");
        if (wave) {{
          wave.classList.add("hidden");
          wave.classList.remove("is-playing");
        }}
        if (wrap && !seqState) wrap.classList.add("opacity-0");
      }}
    }}

    function resetProgress(id) {{
      const bar = document.getElementById("prog-bar-" + id);
      const wrap = document.getElementById("prog-wrap-" + id);
      if (bar) bar.style.width = "0%";
      if (wrap) wrap.classList.add("opacity-0");
    }}

    function onAudioTime(id, type) {{
      const audio = document.getElementById("audio-" + id + "-" + type);
      const bar = document.getElementById("prog-bar-" + id);
      if (!audio || isNaN(audio.duration) || audio.duration === 0) return;
      if (bar) {{
        const pct = (audio.currentTime / audio.duration) * 100;
        bar.style.width = pct + "%";
        if (type === "before") bar.className = "h-full bg-rose-500 w-0 transition-all duration-75";
        else bar.className = "h-full bg-emerald-500 w-0 transition-all duration-75";
      }}
    }}

    function onAudioEnd(id, type) {{
      setUI(id, type, false);
      if (activeAudio && activeAudio.id === id && activeAudio.type === type) {{
        activeAudio = null;
      }}
      checkGlobalStop();

      // Sequenced A -> B transition
      if (seqState && seqState.id === id && type === "before") {{
        const btn = document.getElementById("seq-" + id);
        if (btn) btn.textContent = "Playing B...";
        setTimeout(() => {{
          toggleTrack(id, "after");
        }}, 300);
      }} else if (seqState && seqState.id === id && type === "after") {{
        const btn = document.getElementById("seq-" + id);
        if (btn) {{
          btn.classList.remove("bg-indigo-600", "text-white");
          btn.textContent = "A→B";
        }}
        resetProgress(id);
        seqState = null;
      }} else {{
        resetProgress(id);
      }}
    }}

    function playSeq(id) {{
      stopAllAudio();
      const btn = document.getElementById("seq-" + id);
      if (btn) {{
        btn.classList.add("bg-indigo-600", "text-white");
        btn.textContent = "Playing A...";
      }}
      seqState = {{ id }};
      toggleTrack(id, "before");
    }}

    // Filter Words by Category
    function filterWords(cat) {{
      document.querySelectorAll(".word-cat-btn").forEach(b => {{
        if (b.getAttribute("data-cat") === cat) {{
          b.className = "word-cat-btn active-cat px-2.5 py-1 rounded-lg font-medium transition bg-indigo-600 text-white";
        }} else {{
          b.className = "word-cat-btn px-2.5 py-1 rounded-lg font-medium transition bg-slate-900 hover:bg-slate-800 text-slate-400";
        }}
      }});
      applyWordFilter();
    }}

    function applyWordFilter() {{
      const q = (document.getElementById("wordSearchInput").value || "").toLowerCase().trim();
      const activeBtn = document.querySelector(".word-cat-btn.active-cat");
      const activeCat = activeBtn ? activeBtn.getAttribute("data-cat") : "all";

      document.querySelectorAll(".word-card").forEach(card => {{
        const cardCat = card.getAttribute("data-cat");
        const blob = card.getAttribute("data-search").toLowerCase();

        const matchCat = (activeCat === "all") || (cardCat === activeCat);
        const matchQ = !q || blob.includes(q);

        if (matchCat && matchQ) card.classList.remove("hidden");
        else card.classList.add("hidden");
      }});
    }}

    document.getElementById("wordSearchInput").addEventListener("input", applyWordFilter);

    // Lightbox Modal
    function openLightbox(idx) {{
      curFig = (idx + FIGURES.length) % FIGURES.length;
      const f = FIGURES[curFig];
      document.getElementById("lbImg").src = f.file;
      document.getElementById("lbTitle").textContent = f.title;
      document.getElementById("lbCaption").textContent = f.caption;
      document.getElementById("lbDownload").href = f.file;
      document.getElementById("lbTag").textContent = (curFig + 1) + " / " + FIGURES.length;
      document.getElementById("lightboxModal").classList.remove("hidden");
      document.body.style.overflow = "hidden";
    }}

    function navLightbox(step) {{
      openLightbox(curFig + step);
    }}

    function closeLightbox() {{
      document.getElementById("lightboxModal").classList.add("hidden");
      document.body.style.overflow = "";
    }}

    document.getElementById("lightboxModal").addEventListener("click", function(e) {{
      if (e.target === this) closeLightbox();
    }});

    document.addEventListener("keydown", function(e) {{
      if (e.key === "Escape") closeLightbox();
      if (!document.getElementById("lightboxModal").classList.contains("hidden")) {{
        if (e.key === "ArrowLeft") navLightbox(-1);
        if (e.key === "ArrowRight") navLightbox(1);
      }}
    }});
  </script>

</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[+] Successfully built compact & sleek index.html ({len(html):,} bytes).")
