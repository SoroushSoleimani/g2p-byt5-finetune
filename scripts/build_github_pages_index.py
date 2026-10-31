#!/usr/bin/env python3
"""
Builds an ultra-modern, high-tech GitHub Pages site (index.html)
specifically showcasing:
1. Interactive custom-styled Before vs. After .wav audio players for all 23 word/sentence pairs.
   - Clean dark-mode controls with custom Play/Pause SVG buttons.
   - Animated soundwave equalizer bars (4 bouncing bars during playback).
   - Dynamic time progress indicator and scrubbable track.
   - Instant "Play A -> B Compare" sequenced playback button.
   - Instant search and category filter pills.
2. The 9 high-resolution diagnostic charts (300 DPI) with a built-in Modal Lightbox featuring Prev/Next navigation and keyboard arrow support.
Strictly zero emojis across all code and text.
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
        "category_id": "ezafe",
        "category_label": "Compound & Ezafe",
        "tag": "Ezafe Boundary",
        "gt": "/modeleɒːmɒːɾiː/",
        "before_ipa": "/mædælɒːmɒːɾiː/",
        "after_ipa": "/modeleɒːmɒːɾiː/",
        "highlight": "Kasre Ezafe (-e-) vowel accurately recovered",
        "is_sentence": False
    },
    {
        "id": "shabake_amigh",
        "persian": "شبکه عمیق",
        "category_id": "ezafe",
        "category_label": "Compound & Ezafe",
        "tag": "Ezafe & Glottal",
        "gt": "/ʃæbækejeʔæmiːɢ/",
        "before_ipa": "/ʃæbkeʔæmiːɢ/",
        "after_ipa": "/ʃæbækejeʔæmiːɢ/",
        "highlight": "Ezafe glide (-ye-) and glottal stop /ʔ/ restored",
        "is_sentence": False
    },
    {
        "id": "taamol",
        "persian": "تعامل",
        "category_id": "tashdid",
        "category_label": "Tashdid & Glottal",
        "tag": "Hamza & Vowel",
        "gt": "/tæʔɒːmol/",
        "before_ipa": "/tæʔɒːmæl/",
        "after_ipa": "/tæʔɒːmol/",
        "highlight": "Glottal stop /ʔ/ preserved and Zamme /o/ vocalized",
        "is_sentence": False
    },
    {
        "id": "vizhegi",
        "persian": "ویژگی",
        "category_id": "vowels",
        "category_label": "Short Vowels",
        "tag": "Short Vowel",
        "gt": "/viːʒeɡiː/",
        "before_ipa": "/viːʒɡiː/",
        "after_ipa": "/viːʒeɡiː/",
        "highlight": "Unwritten interior Kasre /e/ restored without vowel deletion",
        "is_sentence": False
    },
    {
        "id": "pazhuheshgar",
        "persian": "پژوهشگر",
        "category_id": "vowels",
        "category_label": "Short Vowels",
        "tag": "Short Vowel",
        "gt": "/pæʒuːheʃɡæɾ/",
        "before_ipa": "/peʒuːheʃɡæɾ/",
        "after_ipa": "/pæʒuːheʃɡæɾ/",
        "highlight": "Fathe /æ/ vs Kasre /e/ prefix vowel contrast resolved",
        "is_sentence": False
    },
    {
        "id": "processor",
        "persian": "پروسسور",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Loanword Epenthesis",
        "gt": "/pɾosesoɾ/",
        "before_ipa": "/poɾuːssuːɾ/",
        "after_ipa": "/pɾosesoɾ/",
        "highlight": "Spurious epenthesis removed, accurate tech pronunciation",
        "is_sentence": False
    },
    # Tech Loanwords
    {
        "id": "pytorch",
        "persian": "پایتورچ",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Modern AI Corpus",
        "gt": "/pɒːjtɔːɾt͡ʃ/",
        "before_ipa": "/pɒːjtuːɾt͡ʃ/",
        "after_ipa": "/pɒːjtɔːɾt͡ʃ/",
        "highlight": "Open back vowel /ɔː/ adaptation in computing loanword",
        "is_sentence": False
    },
    {
        "id": "tensorflow",
        "persian": "تنسورفلو",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Modern AI Corpus",
        "gt": "/tensoɾfloʊ/",
        "before_ipa": "/tænsuːɾfluː/",
        "after_ipa": "/tensoɾfloʊ/",
        "highlight": "Standard technical diphthong /oʊ/ and initial /e/ preserved",
        "is_sentence": False
    },
    {
        "id": "transformer",
        "persian": "ترنسفورمر",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Modern AI Corpus",
        "gt": "/tɾænsfoɾmeɾ/",
        "before_ipa": "/teɾænsfuːɾmeɾ/",
        "after_ipa": "/tɾænsfoɾmeɾ/",
        "highlight": "Initial consonant cluster /tɾ-/ formed without spurious vowel",
        "is_sentence": False
    },
    {
        "id": "algoritm",
        "persian": "الگوریتم",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Tech Vocabulary",
        "gt": "/ʔælɡoɾiːtm/",
        "before_ipa": "/ælɡuːɾiːtm/",
        "after_ipa": "/ʔælɡoɾiːtm/",
        "highlight": "Initial glottal /ʔ/ and interior /o/ vocalization fixed",
        "is_sentence": False
    },
    {
        "id": "kaggle",
        "persian": "کگل",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Tech Vocabulary",
        "gt": "/kæɡel/",
        "before_ipa": "/kæɡl/",
        "after_ipa": "/kæɡel/",
        "highlight": "Appropriate syllable-final /el/ epenthesis applied",
        "is_sentence": False
    },
    {
        "id": "keras",
        "persian": "کراس",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Modern AI Corpus",
        "gt": "/keɾɒːs/",
        "before_ipa": "/kɾɒːs/",
        "after_ipa": "/keɾɒːs/",
        "highlight": "Short Kasre vowel /e/ inserted for natural Persian phonotactics",
        "is_sentence": False
    },
    {
        "id": "grapheme",
        "persian": "گرافیم",
        "category_id": "tech",
        "category_label": "Tech Loanwords",
        "tag": "Linguistic Term",
        "gt": "/ɡeɾɒːfem/",
        "before_ipa": "/ɡɾɒːfiːm/",
        "after_ipa": "/ɡeɾɒːfem/",
        "highlight": "Short Kasre /e/ phoneme accuracy verified",
        "is_sentence": False
    },
    # Phonological phenomena
    {
        "id": "shabake",
        "persian": "شبکه",
        "category_id": "vowels",
        "category_label": "Short Vowels",
        "tag": "Short Vowel",
        "gt": "/ʃæbæke/",
        "before_ipa": "/ʃæbke/",
        "after_ipa": "/ʃæbæke/",
        "highlight": "Interior Fathe /æ/ vowel restored preventing cluster syncopation",
        "is_sentence": False
    },
    {
        "id": "model",
        "persian": "مدل",
        "category_id": "vowels",
        "category_label": "Short Vowels",
        "tag": "Short Vowel",
        "gt": "/model/",
        "before_ipa": "/mædæl/",
        "after_ipa": "/model/",
        "highlight": "Accurate Zamme /o/ and Kasre /e/ recovered over default Fathe",
        "is_sentence": False
    },
    {
        "id": "mohandesi",
        "persian": "مهندسی",
        "category_id": "vowels",
        "category_label": "Short Vowels",
        "tag": "Short Vowel",
        "gt": "/mohændesiː/",
        "before_ipa": "/mehændesiː/",
        "after_ipa": "/mohændesiː/",
        "highlight": "Initial Zamme /o/ vowel disambiguated correctly",
        "is_sentence": False
    },
    {
        "id": "mohasebe",
        "persian": "محاسبه",
        "category_id": "vowels",
        "category_label": "Short Vowels",
        "tag": "Short Vowel",
        "gt": "/mohɒːsebe/",
        "before_ipa": "/mæhɒːsebe/",
        "after_ipa": "/mohɒːsebe/",
        "highlight": "Correct Arabic loan prefix vocalization /mo-/ resolved",
        "is_sentence": False
    },
    {
        "id": "behine",
        "persian": "بهینه‌سازی",
        "category_id": "ezafe",
        "category_label": "Compound & Ezafe",
        "tag": "Compound & ZWNJ",
        "gt": "/behiːnesɒːziː/",
        "before_ipa": "/bæhiːnesɒːziː/",
        "after_ipa": "/behiːnesɒːziː/",
        "highlight": "Zero-width non-joiner boundary parsed cleanly with initial /e/",
        "is_sentence": False
    },
    {
        "id": "motaghayyer",
        "persian": "متغیر",
        "category_id": "tashdid",
        "category_label": "Tashdid & Glottal",
        "tag": "Gemination (Tashdid)",
        "gt": "/motæɢæjjeɾ/",
        "before_ipa": "/motæɢɒːjeɾ/",
        "after_ipa": "/motæɢæjjeɾ/",
        "highlight": "Phonemic consonant doubling /jj/ fully recovered",
        "is_sentence": False
    },
    {
        "id": "tavajjoh",
        "persian": "توجه",
        "category_id": "tashdid",
        "category_label": "Tashdid & Glottal",
        "tag": "Gemination (Tashdid)",
        "gt": "/tævæd͡ʒd͡ʒoh/",
        "before_ipa": "/tævæd͡ʒoh/",
        "after_ipa": "/tævæd͡ʒd͡ʒoh/",
        "highlight": "Restored medial affricate geminate /d͡ʒd͡ʒ/ in speech synthesis",
        "is_sentence": False
    },
    # Full Sentences
    {
        "id": "sentence1",
        "persian": "مدل ترنسفورمر و شبکه عصبی در پایتورچ و تنسورفلو الگوریتم گرافیم را بهینه‌سازی می‌کنند.",
        "category_id": "sentences",
        "category_label": "Full Sentences",
        "tag": "Deep Learning Pipeline",
        "gt": "Sentence 1 (Deep Learning Pipeline)",
        "before_ipa": "Baseline: Omitted Ezafe & Distorted Vowels across technical terms",
        "after_ipa": "Fine-Tuned: Fluent prosodic continuity, exact Ezafe and loanword vowels",
        "highlight": "Continuous sentence-level phonetic alignment and fluent Ezafe handling",
        "is_sentence": True
    },
    {
        "id": "sentence2",
        "persian": "مهندسی پزشکی و محاسبه دقیق متغیرها در هوش مصنوعی و یادگیری ماشین توجه زیادی دارد.",
        "category_id": "sentences",
        "category_label": "Full Sentences",
        "tag": "Medical & Scientific AI",
        "gt": "Sentence 2 (Scientific Speech)",
        "before_ipa": "Baseline: Missing Tashdid, unvocalized prefixes, robotic prosody",
        "after_ipa": "Fine-Tuned: Exact Tashdid on Daqiq, Motaghayyer & Tavajjoh with natural flow",
        "highlight": "Multiple Tashdid geminates correctly vocalized in continuous speech",
        "is_sentence": True
    },
    {
        "id": "sentence3",
        "persian": "پژوهشگران علم داده در کگل با فریم‌ورک دیپ‌لرنینگ و توکنایزر مسئله اتصال را حل کردند.",
        "category_id": "sentences",
        "category_label": "Full Sentences",
        "tag": "Data Science & NLP",
        "gt": "Sentence 3 (Data Science & NLP)",
        "before_ipa": "Baseline: Mispronounced AI loanwords (Kaggle, Deep Learning, Tokenizer)",
        "after_ipa": "Fine-Tuned: Accurate syllable cadence and modern Persian NLP terminology",
        "highlight": "Accurate syllable timing and natural pronunciation of modern AI loanwords",
        "is_sentence": True
    }
]

FIGURES = [
    {
        "file": "figures/dataset_distribution.png",
        "tag": "EDA & Distribution",
        "title": "Figure 1: Corpus EDA & Phoneme Distribution",
        "caption": "Grapheme vs. phoneme length regression on 116K+ KaamelDict words, and top 20 Persian phonemes."
    },
    {
        "file": "figures/loss_curve.png",
        "tag": "Training Dynamics",
        "title": "Figure 2: Loss Convergence & Cosine Learning Rate",
        "caption": "Cross-entropy loss trajectory on Tesla T4 GPU over 3 epochs (5,400 steps) with validation checkpoints."
    },
    {
        "file": "figures/per_wer_metrics_comparison.png",
        "tag": "Benchmark Metrics",
        "title": "Figure 3: Multi-Metric Benchmark Comparison",
        "caption": "Dual-panel bar chart comparing PER, WER, Word Acc, Sim, and chrF before vs. after fine-tuning."
    },
    {
        "file": "figures/category_breakdown.png",
        "tag": "Category Analysis",
        "title": "Figure 4: Stratified Linguistic Category Breakdown",
        "caption": "Error reduction across AI Loanwords, Short Vowels, Ezafe, and Gemination/Glottal phenomena."
    },
    {
        "file": "figures/edit_operations_breakdown.png",
        "tag": "Levenshtein Breakdown",
        "title": "Figure 5: Levenshtein Edit Operations Breakdown",
        "caption": "Substitutions (vowel errors), deletions (dropped short vowels/geminates), and insertions."
    },
    {
        "file": "figures/length_bucket_analysis.png",
        "tag": "Length Progression",
        "title": "Figure 6: Word Length Bucket Error Analysis",
        "caption": "Error rate progression across short (1-4), medium (5-7), long (8-10), and compound (11+) words."
    },
    {
        "file": "figures/vowel_confusion_matrix.png",
        "tag": "Confusion Matrix",
        "title": "Figure 7: Persian Vowel Confusion Matrix",
        "caption": "Row-normalized confusion among Persian vowels /æ, e, o, ɒː, iː, uː/ before and after fine-tuning."
    },
    {
        "file": "figures/word_edit_distance_reduction.png",
        "tag": "Challenge Subset",
        "title": "Figure 8: Word-Level Edit Distance Reduction",
        "caption": "Levenshtein distance error reduction across 20 held-out Persian challenge words."
    },
    {
        "file": "figures/g2p_byt5_architecture.png",
        "tag": "Model Pipeline",
        "title": "Figure 9: ByT5 Persian G2P Architecture",
        "caption": "Raw byte sequence-to-sequence conversion pipeline from Persian orthography to IPA phonemes."
    }
]

figures_json = json.dumps(FIGURES, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark scroll-smooth">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Persian G2P ByT5: Audio Comparison (.wav) & Diagnostic Charts</title>
  <meta name="description" content="Interactive Before vs. After audio comparison (.wav) and publication-grade diagnostic visualizations for Persian G2P ByT5 fine-tuned on KaamelDict (116K+ words).">
  
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            brand: {{
              500: '#6366f1',
              600: '#4f46e5',
            }}
          }},
          fontFamily: {{
            sans: ['"Inter"', 'sans-serif'],
            fa: ['"Vazirmatn"', 'sans-serif'],
            mono: ['"Fira Code"', 'monospace'],
          }}
        }}
      }}
    }}
  </script>

  <!-- Google Fonts: Inter, Vazirmatn, Fira Code -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@400;500;600;700;800;900&family=Vazirmatn:wght@500;700;800;900&display=swap" rel="stylesheet">

  <style>
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: #020617; }}
    ::-webkit-scrollbar-thumb {{ background: #1e293b; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #334155; }}

    /* Animated Soundwave Equalizer */
    .sound-wave {{
      display: inline-flex;
      align-items: center;
      gap: 2.5px;
      height: 14px;
    }}
    .wave-bar {{
      width: 2.5px;
      height: 3px;
      border-radius: 2px;
      transition: height 0.15s ease;
    }}
    .is-playing .wave-bar:nth-child(1) {{ animation: bounceWave 0.7s infinite alternate ease-in-out 0.0s; }}
    .is-playing .wave-bar:nth-child(2) {{ animation: bounceWave 0.7s infinite alternate ease-in-out 0.2s; }}
    .is-playing .wave-bar:nth-child(3) {{ animation: bounceWave 0.7s infinite alternate ease-in-out 0.4s; }}
    .is-playing .wave-bar:nth-child(4) {{ animation: bounceWave 0.7s infinite alternate ease-in-out 0.1s; }}
    
    @keyframes bounceWave {{
      0% {{ height: 3px; }}
      50% {{ height: 14px; }}
      100% {{ height: 6px; }}
    }}

    /* Card Glow & Glassmorphism */
    .glass-card {{
      background: rgba(15, 23, 42, 0.75);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(51, 65, 85, 0.45);
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .glass-card:hover {{
      border-color: rgba(99, 102, 241, 0.45);
      transform: translateY(-2px);
      box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.12);
    }}

    /* Range slider custom styling */
    input[type=range].audio-slider {{
      -webkit-appearance: none;
      background: transparent;
      cursor: pointer;
    }}
    input[type=range].audio-slider::-webkit-slider-runnable-track {{
      height: 4px;
      border-radius: 9999px;
      background: #1e293b;
    }}
    input[type=range].audio-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      height: 10px;
      width: 10px;
      border-radius: 9999px;
      background: #f8fafc;
      margin-top: -3px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.5);
      transition: transform 0.1s ease;
    }}
    input[type=range].audio-slider:hover::-webkit-slider-thumb {{
      transform: scale(1.3);
    }}
    input[type=range].audio-slider::-moz-range-track {{
      height: 4px;
      border-radius: 9999px;
      background: #1e293b;
    }}
    input[type=range].audio-slider::-moz-range-thumb {{
      height: 10px;
      width: 10px;
      border-radius: 9999px;
      background: #f8fafc;
      border: none;
    }}

    /* Background Mesh Ambient */
    .ambient-bg {{
      position: absolute;
      top: 0;
      left: 50%;
      transform: translateX(-50%);
      width: 100%;
      max-width: 1200px;
      height: 500px;
      background: radial-gradient(circle at 50% 10%, rgba(99, 102, 241, 0.12) 0%, rgba(14, 165, 233, 0.05) 35%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }}
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen antialiased selection:bg-indigo-500 selection:text-white relative font-sans">

  <!-- Ambient Light Effect -->
  <div class="ambient-bg"></div>

  <!-- Sleek Sticky Header -->
  <header class="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/80">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      
      <!-- Brand & Project -->
      <div class="flex items-center space-x-3">
        <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-[1px] flex items-center justify-center shadow-lg shadow-indigo-500/20">
          <div class="w-full h-full bg-slate-950 rounded-[11px] flex items-center justify-center">
            <svg class="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
            </svg>
          </div>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="font-black text-sm tracking-tight text-white">ByT5 Persian G2P</span>
            <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-semibold">KaamelDict 116K</span>
          </div>
          <p class="text-[11px] text-slate-400 hidden sm:block">Acoustic Audio Comparison & Diagnostic Metrics</p>
        </div>
      </div>

      <!-- Quick Nav & GitHub Link -->
      <div class="flex items-center space-x-3">
        <nav class="hidden md:flex items-center space-x-1 text-xs font-medium">
          <a href="#audio-section" class="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition">Audio Player (.wav)</a>
          <a href="#charts-section" class="px-3 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/60 transition">9 Figures (300 DPI)</a>
        </nav>
        
        <div class="h-4 w-px bg-slate-800 hidden md:block"></div>

        <a href="https://github.com/SoroushSoleimani/g2p-byt5-finetune" target="_blank" rel="noopener"
           class="inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700/80 text-white text-xs font-semibold shadow-sm transition">
          <svg class="w-4 h-4 fill-current text-slate-300" viewBox="0 0 24 24">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
          </svg>
          <span>Repository</span>
        </a>
      </div>
    </div>
  </header>

  <!-- Hero Section -->
  <section class="max-w-7xl mx-auto px-4 sm:px-6 pt-10 pb-6 relative z-10">
    <div class="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-8 border-b border-slate-800/80">
      <div class="max-w-3xl">
        <div class="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold mb-3">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Acoustic Before vs. After Fine-Tuning Evaluation</span>
        </div>
        <h1 class="text-3xl sm:text-4xl lg:text-5xl font-black text-white tracking-tight leading-tight">
          Persian G2P ByT5
          <span class="block text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400">
            Acoustic Verification & Charts
          </span>
        </h1>
        <p class="text-sm sm:text-base text-slate-400 mt-3 leading-relaxed max-w-2xl">
          Empirical comparison of token-free ByT5 fine-tuned on the 116,629-word KaamelDict lexicon.
          Auditory synthesis performed with Neural TTS (<span class="font-mono text-slate-300 text-xs">fa-IR-FaridNeural</span>) demonstrating error reductions across Tashdid, Ezafe, short vowels, and modern technical loanwords.
        </p>
      </div>

      <!-- Live Benchmarks Summary Cards -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div class="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800/80 flex flex-col justify-between">
          <span class="text-[11px] font-medium text-slate-400">PER Reduction</span>
          <div class="mt-2 flex items-baseline space-x-1">
            <span class="text-2xl font-black text-emerald-400">-52.4%</span>
          </div>
          <span class="text-[10px] text-slate-500 mt-1 font-mono">20.59% &rarr; 9.80%</span>
        </div>

        <div class="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800/80 flex flex-col justify-between">
          <span class="text-[11px] font-medium text-slate-400">Word Accuracy</span>
          <div class="mt-2 flex items-baseline space-x-1">
            <span class="text-2xl font-black text-indigo-400">51.7%</span>
          </div>
          <span class="text-[10px] text-slate-500 mt-1 font-mono">+138% (21.7% Base)</span>
        </div>

        <div class="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800/80 flex flex-col justify-between">
          <span class="text-[11px] font-medium text-slate-400">Tashdid & Glottal</span>
          <div class="mt-2 flex items-baseline space-x-1">
            <span class="text-2xl font-black text-cyan-400">-76.3%</span>
          </div>
          <span class="text-[10px] text-slate-500 mt-1 font-mono">31.4% &rarr; 7.4% PER</span>
        </div>

        <div class="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800/80 flex flex-col justify-between">
          <span class="text-[11px] font-medium text-slate-400">Training Corpus</span>
          <div class="mt-2 flex items-baseline space-x-1">
            <span class="text-2xl font-black text-white">116.6K</span>
          </div>
          <span class="text-[10px] text-slate-500 mt-1 font-mono">KaamelDict + AI words</span>
        </div>
      </div>
    </div>
  </section>

  <!-- SECTION 1: INTERACTIVE AUDIO SHOWCASE -->
  <section id="audio-section" class="max-w-7xl mx-auto px-4 sm:px-6 py-6 relative z-10">
    
    <!-- Controls Toolbar: Search, Filters, Stop All -->
    <div class="bg-slate-900/90 backdrop-blur-md border border-slate-800/90 rounded-2xl p-4 mb-8 shadow-xl">
      <div class="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
        
        <!-- Search Input -->
        <div class="relative flex-1 max-w-md">
          <div class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
            <svg class="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input type="text" id="sampleSearchInput" placeholder="Filter by Persian word, IPA, or tag (e.g. pytorch, تشدید, ezafe)..."
                 class="w-full pl-10 pr-4 py-2 bg-slate-950/80 border border-slate-800 rounded-xl text-xs sm:text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
        </div>

        <!-- Global Stop Audio & Quick Info -->
        <div class="flex items-center space-x-3 self-end md:self-auto">
          <button id="globalStopBtn" onclick="stopAllAudio()"
                  class="hidden inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-semibold transition">
            <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
              <rect x="5" y="5" width="14" height="14" rx="2" />
            </svg>
            <span>Stop Audio</span>
          </button>
          
          <div class="text-xs text-slate-400 font-mono">
            Showing <span id="visibleCount" class="text-white font-bold">{len(SAMPLES)}</span> of {len(SAMPLES)} pairs
          </div>
        </div>
      </div>

      <!-- Category Filter Tabs -->
      <div class="flex flex-wrap items-center gap-2 mt-4 pt-4 border-t border-slate-800/60">
        <button onclick="setCategoryFilter('all')" data-cat="all"
                class="cat-filter-btn active-filter px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-indigo-600 text-white shadow-sm">
          All Samples ({len(SAMPLES)})
        </button>
        <button onclick="setCategoryFilter('tech')" data-cat="tech"
                class="cat-filter-btn px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-slate-800/80 hover:bg-slate-700/80 text-slate-300">
          Tech & AI Loanwords (7)
        </button>
        <button onclick="setCategoryFilter('ezafe')" data-cat="ezafe"
                class="cat-filter-btn px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-slate-800/80 hover:bg-slate-700/80 text-slate-300">
          Compound & Ezafe (3)
        </button>
        <button onclick="setCategoryFilter('tashdid')" data-cat="tashdid"
                class="cat-filter-btn px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-slate-800/80 hover:bg-slate-700/80 text-slate-300">
          Tashdid & Glottal (3)
        </button>
        <button onclick="setCategoryFilter('vowels')" data-cat="vowels"
                class="cat-filter-btn px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-slate-800/80 hover:bg-slate-700/80 text-slate-300">
          Short Vowels (6)
        </button>
        <button onclick="setCategoryFilter('sentences')" data-cat="sentences"
                class="cat-filter-btn px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-slate-800/80 hover:bg-slate-700/80 text-slate-300">
          Full Sentences (3)
        </button>
      </div>
    </div>

    <!-- Cards Grid -->
    <div id="samplesGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
"""

for s in SAMPLES:
    b_wav = f"audio_comparison/{s['id']}_before_finetune.wav"
    a_wav = f"audio_comparison/{s['id']}_after_finetune.wav"
    span_class = "col-span-1 md:col-span-2 lg:col-span-3" if s["is_sentence"] else "col-span-1"
    
    html_content += f"""
      <div class="sample-card glass-card rounded-2xl p-5 flex flex-col justify-between {span_class}"
           data-id="{s['id']}"
           data-cat="{s['category_id']}"
           data-search="{s['persian']} {s['gt']} {s['category_label']} {s['tag']} {s['highlight']} {s['id']}">
        
        <!-- Card Top Section -->
        <div>
          <!-- Header Badges -->
          <div class="flex items-center justify-between gap-2 mb-3">
            <span class="inline-flex items-center px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-indigo-500/10 border border-indigo-500/20 text-indigo-300">
              {s['category_label']}
            </span>
            <span class="text-[11px] font-medium text-slate-400 bg-slate-800/60 px-2 py-0.5 rounded-md border border-slate-700/50">
              {s['tag']}
            </span>
          </div>

          <!-- Persian Text Hero Box -->
          <div class="my-4 p-4 rounded-xl bg-gradient-to-l from-slate-900/90 to-slate-950/90 border-r-4 border-indigo-500 border-t border-b border-l border-slate-800/60 text-right">
            <div class="font-fa {'text-lg sm:text-xl font-bold leading-relaxed' if s['is_sentence'] else 'text-3xl font-black'} text-white tracking-wide drop-shadow-sm" dir="rtl">
              {s['persian']}
            </div>
            <div class="text-[11px] text-emerald-400 font-medium mt-2 flex items-center justify-end space-x-1" dir="ltr">
              <svg class="w-3.5 h-3.5 text-emerald-400 mr-1 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
              </svg>
              <span>{s['highlight']}</span>
            </div>
          </div>

          <!-- IPA Phonetic Diff Comparison -->
          <div class="space-y-2 text-xs font-mono bg-slate-950/80 p-3.5 rounded-xl border border-slate-800/80 mb-4">
            <div class="flex justify-between items-baseline text-slate-400">
              <span class="text-[10px] uppercase font-sans tracking-wider text-slate-500 font-semibold">Target IPA</span>
              <span class="text-slate-200 font-semibold">{s['gt']}</span>
            </div>
            <div class="flex justify-between items-baseline text-rose-400">
              <span class="text-[10px] uppercase font-sans tracking-wider text-rose-500/80 font-semibold">Before (Base)</span>
              <span class="line-through decoration-rose-500/70 text-rose-300/80">{s['before_ipa']}</span>
            </div>
            <div class="flex justify-between items-baseline text-emerald-400">
              <span class="text-[10px] uppercase font-sans tracking-wider text-emerald-500/80 font-semibold">After (FT)</span>
              <span class="font-bold text-emerald-300">{s['after_ipa']}</span>
            </div>
          </div>
        </div>

        <!-- Audio Custom Control Center -->
        <div class="pt-4 border-t border-slate-800/80 space-y-3">
          
          <!-- Sequenced Compare Button (A & B) -->
          <div class="flex items-center justify-between">
            <span class="text-[11px] text-slate-400 font-medium">Auditory Verification</span>
            <button onclick="playSequenced('{s['id']}')" id="seq-btn-{s['id']}"
                    class="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-[11px] font-semibold transition">
              <svg class="w-3 h-3 fill-current text-indigo-400" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
              <span>Play A &rarr; B Compare</span>
            </button>
          </div>

          <!-- BEFORE AUDIO ROW -->
          <div id="player-row-{s['id']}-before" class="audio-control-row group/b rounded-xl p-2.5 bg-rose-950/20 border border-rose-900/30 hover:border-rose-700/50 transition">
            <audio id="audio-{s['id']}-before" src="{b_wav}" preload="metadata"
                   onloadedmetadata="onAudioLoaded('{s['id']}', 'before')"
                   ontimeupdate="onAudioTimeUpdate('{s['id']}', 'before')"
                   onended="onAudioEnded('{s['id']}', 'before')"></audio>
            
            <div class="flex items-center justify-between gap-3">
              <!-- Play Button & Label -->
              <div class="flex items-center space-x-2.5">
                <button onclick="toggleAudio('{s['id']}', 'before')" id="btn-{s['id']}-before"
                        class="w-8 h-8 rounded-full bg-rose-500/20 group-hover/b:bg-rose-500/30 text-rose-300 border border-rose-500/40 flex items-center justify-center transition shadow-sm">
                  <svg class="w-4 h-4 fill-current ml-0.5 icon-play" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  <svg class="w-4 h-4 fill-current icon-pause hidden" viewBox="0 0 24 24">
                    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                  </svg>
                </button>
                <div>
                  <div class="flex items-center space-x-1.5">
                    <span class="text-xs font-semibold text-rose-300">Before Fine-Tuning</span>
                    <span class="text-[9px] font-mono text-rose-400/60 uppercase">Base</span>
                  </div>
                  <!-- Soundwave Equalizer -->
                  <div id="wave-{s['id']}-before" class="sound-wave">
                    <span class="wave-bar bg-rose-400"></span>
                    <span class="wave-bar bg-rose-400"></span>
                    <span class="wave-bar bg-rose-400"></span>
                    <span class="wave-bar bg-rose-400"></span>
                  </div>
                </div>
              </div>

              <!-- Time display -->
              <span id="time-{s['id']}-before" class="text-[11px] font-mono text-rose-300/80">0:00</span>
            </div>

            <!-- Scrubber Track -->
            <div class="mt-2 flex items-center">
              <input type="range" id="seek-{s['id']}-before" value="0" min="0" max="100" step="0.1"
                     oninput="onAudioSeek('{s['id']}', 'before', this.value)"
                     class="audio-slider w-full accent-rose-500">
            </div>
          </div>

          <!-- AFTER AUDIO ROW -->
          <div id="player-row-{s['id']}-after" class="audio-control-row group/a rounded-xl p-2.5 bg-emerald-950/20 border border-emerald-900/30 hover:border-emerald-700/50 transition">
            <audio id="audio-{s['id']}-after" src="{a_wav}" preload="metadata"
                   onloadedmetadata="onAudioLoaded('{s['id']}', 'after')"
                   ontimeupdate="onAudioTimeUpdate('{s['id']}', 'after')"
                   onended="onAudioEnded('{s['id']}', 'after')"></audio>
            
            <div class="flex items-center justify-between gap-3">
              <!-- Play Button & Label -->
              <div class="flex items-center space-x-2.5">
                <button onclick="toggleAudio('{s['id']}', 'after')" id="btn-{s['id']}-after"
                        class="w-8 h-8 rounded-full bg-emerald-500/20 group-hover/a:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 flex items-center justify-center transition shadow-sm">
                  <svg class="w-4 h-4 fill-current ml-0.5 icon-play" viewBox="0 0 24 24">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  <svg class="w-4 h-4 fill-current icon-pause hidden" viewBox="0 0 24 24">
                    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                  </svg>
                </button>
                <div>
                  <div class="flex items-center space-x-1.5">
                    <span class="text-xs font-semibold text-emerald-300">After Fine-Tuning</span>
                    <span class="text-[9px] font-mono text-emerald-400/80 font-bold px-1 rounded bg-emerald-500/20">KaamelDict</span>
                  </div>
                  <!-- Soundwave Equalizer -->
                  <div id="wave-{s['id']}-after" class="sound-wave">
                    <span class="wave-bar bg-emerald-400"></span>
                    <span class="wave-bar bg-emerald-400"></span>
                    <span class="wave-bar bg-emerald-400"></span>
                    <span class="wave-bar bg-emerald-400"></span>
                  </div>
                </div>
              </div>

              <!-- Time display -->
              <span id="time-{s['id']}-after" class="text-[11px] font-mono text-emerald-300/80">0:00</span>
            </div>

            <!-- Scrubber Track -->
            <div class="mt-2 flex items-center">
              <input type="range" id="seek-{s['id']}-after" value="0" min="0" max="100" step="0.1"
                     oninput="onAudioSeek('{s['id']}', 'after', this.value)"
                     class="audio-slider w-full accent-emerald-500">
            </div>
          </div>

        </div>

      </div>
"""

html_content += f"""
    </div>
  </section>

  <!-- SECTION 2: 9 DIAGNOSTIC CHARTS (300 DPI) WITH LIGHTBOX -->
  <section id="charts-section" class="max-w-7xl mx-auto px-4 sm:px-6 py-12 border-t border-slate-800/80 relative z-10">
    <div class="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8">
      <div>
        <div class="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold mb-2">
          <span>Empirical Evaluation & Architecture</span>
        </div>
        <h2 class="text-2xl sm:text-3xl font-black text-white tracking-tight">9 Diagnostic Charts (300 DPI)</h2>
        <p class="text-xs sm:text-sm text-slate-400 mt-1 max-w-2xl">
          Visualizations generated during the Kaggle Tesla T4 GPU evaluation run on 116,629 KaamelDict word pairs, 500 Wiktionary test words, and 60 held-out phonological challenges.
        </p>
      </div>

      <div class="flex items-center space-x-3">
        <a href="g2p_implementation_and_finetune.ipynb" download
           class="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold shadow-sm transition">
          <svg class="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <span>Download Executed Notebook (.ipynb)</span>
        </a>
      </div>
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
"""

for idx, fig in enumerate(FIGURES):
    html_content += f"""
      <div class="glass-card rounded-2xl overflow-hidden flex flex-col justify-between group">
        
        <!-- Image Preview (Click to open Lightbox) -->
        <div onclick="openLightboxByIndex({idx})"
             class="cursor-zoom-in relative bg-slate-950 p-4 border-b border-slate-800/80 overflow-hidden">
          <img src="{fig['file']}" alt="{fig['title']}" loading="lazy"
               class="w-full h-52 object-contain group-hover:scale-[1.03] transition duration-300">
          <div class="absolute top-3 right-3 px-2 py-1 rounded-md bg-slate-900/90 border border-slate-700 text-[10px] font-mono text-slate-300 backdrop-blur-sm">
            {fig['tag']}
          </div>
          <div class="absolute inset-0 bg-indigo-600/0 group-hover:bg-indigo-600/5 transition flex items-center justify-center">
            <span class="opacity-0 group-hover:opacity-100 transition duration-200 px-3 py-1.5 rounded-xl bg-slate-900/90 text-white text-xs font-semibold shadow-lg border border-slate-700 flex items-center space-x-1.5">
              <svg class="w-3.5 h-3.5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
              <span>Click to Enlarge</span>
            </span>
          </div>
        </div>

        <!-- Info & Actions -->
        <div class="p-4 flex flex-col justify-between flex-1">
          <div>
            <h3 class="font-bold text-sm text-white">{fig['title']}</h3>
            <p class="text-xs text-slate-400 mt-1.5 leading-relaxed">{fig['caption']}</p>
          </div>
          
          <div class="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-xs">
            <button onclick="openLightboxByIndex({idx})"
                    class="text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center space-x-1 transition">
              <span>View Full Resolution</span>
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
            </button>
            <a href="{fig['file']}" download
               class="text-slate-400 hover:text-white transition flex items-center space-x-1">
              <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <span>Save PNG</span>
            </a>
          </div>
        </div>

      </div>
"""

html_content += f"""
    </div>
  </section>

  <!-- LIGHTBOX MODAL VIEWER WITH PREV / NEXT -->
  <div id="lightboxModal" class="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md hidden flex items-center justify-center p-4">
    <div class="relative max-w-5xl w-full bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
      
      <!-- Modal Header -->
      <div class="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <div>
          <div class="flex items-center space-x-2">
            <span id="lightboxIndexTag" class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-500/20 text-indigo-300">1 / 9</span>
            <h3 id="lightboxTitle" class="text-sm font-bold text-white">Chart Preview</h3>
          </div>
          <p id="lightboxCaption" class="text-xs text-slate-400 truncate max-w-xl mt-0.5"></p>
        </div>
        <div class="flex items-center space-x-3">
          <a id="lightboxDownload" href="#" download class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700 transition">Download 300 DPI</a>
          <button onclick="closeLightbox()" class="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      <!-- Modal Image View with Prev / Next Navigation -->
      <div class="relative p-4 bg-slate-950 overflow-auto flex items-center justify-center min-h-[350px]">
        
        <!-- Prev Button -->
        <button onclick="navigateLightbox(-1)"
                class="absolute left-3 top-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-slate-900/80 hover:bg-indigo-600 text-slate-300 hover:text-white border border-slate-700 flex items-center justify-center transition shadow-lg z-10">
          <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        <!-- Image -->
        <img id="lightboxImage" src="" alt="High resolution diagnostic chart" class="max-h-[70vh] object-contain rounded-lg">

        <!-- Next Button -->
        <button onclick="navigateLightbox(1)"
                class="absolute right-3 top-1/2 -translate-y-1/2 w-9 h-9 rounded-full bg-slate-900/80 hover:bg-indigo-600 text-slate-300 hover:text-white border border-slate-700 flex items-center justify-center transition shadow-lg z-10">
          <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  </div>

  <!-- Minimal Footer -->
  <footer class="border-t border-slate-800/80 py-8 bg-slate-950 relative z-10">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-400 gap-4">
      <div>
        Persian G2P ByT5 Fine-Tuning &bull; KaamelDict Corpus (116K+ Words) &bull; 2026
      </div>
      <div class="flex items-center space-x-5">
        <a href="g2p_implementation_and_finetune.ipynb" download class="hover:text-white transition">Notebook (.ipynb)</a>
        <a href="evaluation_metrics.json" download class="hover:text-white transition">Metrics JSON</a>
        <a href="https://github.com/SoroushSoleimani/g2p-byt5-finetune" target="_blank" class="hover:text-white transition">GitHub Repo</a>
      </div>
    </div>
  </footer>

  <!-- Interactive JavaScript Application Logic -->
  <script>
    const FIGURES_DATA = {figures_json};
    let currentFigureIndex = 0;

    let currentPlaying = null;
    let sequencedQueue = null;

    function formatTime(sec) {{
      if (isNaN(sec) || sec < 0) return "0:00";
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return m + ":" + (s < 10 ? "0" : "") + s;
    }}

    function onAudioLoaded(id, type) {{
      const audio = document.getElementById("audio-" + id + "-" + type);
      const timeSpan = document.getElementById("time-" + id + "-" + type);
      if (audio && !isNaN(audio.duration) && audio.duration > 0) {{
        timeSpan.textContent = "0:00 / " + formatTime(audio.duration);
      }}
    }}

    function toggleAudio(id, type) {{
      const audio = document.getElementById("audio-" + id + "-" + type);
      if (!audio) return;

      if (currentPlaying && (currentPlaying.id !== id || currentPlaying.type !== type)) {{
        pauseCurrentAudio();
      }}

      if (audio.paused) {{
        audio.play().then(() => {{
          currentPlaying = {{ id, type, audio }};
          setButtonState(id, type, true);
          document.getElementById("globalStopBtn").classList.remove("hidden");
        }}).catch(err => console.error("Playback error:", err));
      }} else {{
        audio.pause();
        setButtonState(id, type, false);
        currentPlaying = null;
        checkGlobalStopBtn();
      }}
    }}

    function pauseCurrentAudio() {{
      if (currentPlaying && currentPlaying.audio) {{
        currentPlaying.audio.pause();
        setButtonState(currentPlaying.id, currentPlaying.type, false);
        currentPlaying = null;
      }}
      checkGlobalStopBtn();
    }}

    function stopAllAudio() {{
      pauseCurrentAudio();
      sequencedQueue = null;
      document.querySelectorAll("[id^='seq-btn-']").forEach(b => {{
        b.classList.remove("bg-indigo-600", "text-white");
        b.querySelector("span").textContent = "Play A → B Compare";
      }});
      checkGlobalStopBtn();
    }}

    function checkGlobalStopBtn() {{
      const btn = document.getElementById("globalStopBtn");
      if (!currentPlaying) {{
        btn.classList.add("hidden");
      }} else {{
        btn.classList.remove("hidden");
      }}
    }}

    function setButtonState(id, type, isPlaying) {{
      const btn = document.getElementById("btn-" + id + "-" + type);
      const wave = document.getElementById("wave-" + id + "-" + type);
      if (!btn) return;

      const playIcon = btn.querySelector(".icon-play");
      const pauseIcon = btn.querySelector(".icon-pause");

      if (isPlaying) {{
        playIcon.classList.add("hidden");
        pauseIcon.classList.remove("hidden");
        if (wave) wave.classList.add("is-playing");
      }} else {{
        playIcon.classList.remove("hidden");
        pauseIcon.classList.add("hidden");
        if (wave) wave.classList.remove("is-playing");
      }}
    }}

    function onAudioTimeUpdate(id, type) {{
      const audio = document.getElementById("audio-" + id + "-" + type);
      const timeSpan = document.getElementById("time-" + id + "-" + type);
      const seek = document.getElementById("seek-" + id + "-" + type);

      if (!audio || isNaN(audio.duration)) return;

      const current = audio.currentTime;
      const duration = audio.duration;
      timeSpan.textContent = formatTime(current) + " / " + formatTime(duration);

      if (duration > 0) {{
        seek.value = (current / duration) * 100;
      }}
    }}

    function onAudioSeek(id, type, value) {{
      const audio = document.getElementById("audio-" + id + "-" + type);
      if (!audio || isNaN(audio.duration)) return;
      audio.currentTime = (value / 100) * audio.duration;
    }}

    function onAudioEnded(id, type) {{
      setButtonState(id, type, false);
      if (currentPlaying && currentPlaying.id === id && currentPlaying.type === type) {{
        currentPlaying = null;
      }}
      checkGlobalStopBtn();

      if (sequencedQueue && sequencedQueue.id === id && type === "before") {{
        const seqBtn = document.getElementById("seq-btn-" + id);
        if (seqBtn) {{
          seqBtn.querySelector("span").textContent = "Playing After (B)...";
        }}
        setTimeout(() => {{
          toggleAudio(id, "after");
        }}, 350);
      }} else if (sequencedQueue && sequencedQueue.id === id && type === "after") {{
        const seqBtn = document.getElementById("seq-btn-" + id);
        if (seqBtn) {{
          seqBtn.classList.remove("bg-indigo-600", "text-white");
          seqBtn.querySelector("span").textContent = "Play A → B Compare";
        }}
        sequencedQueue = null;
      }}
    }}

    function playSequenced(id) {{
      stopAllAudio();
      const seqBtn = document.getElementById("seq-btn-" + id);
      if (seqBtn) {{
        seqBtn.classList.add("bg-indigo-600", "text-white");
        seqBtn.querySelector("span").textContent = "Playing Before (A)...";
      }}
      sequencedQueue = {{ id }};
      toggleAudio(id, "before");
    }}

    function setCategoryFilter(category) {{
      document.querySelectorAll(".cat-filter-btn").forEach(btn => {{
        if (btn.getAttribute("data-cat") === category) {{
          btn.className = "cat-filter-btn active-filter px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-indigo-600 text-white shadow-sm";
        }} else {{
          btn.className = "cat-filter-btn px-3 py-1.5 rounded-xl text-xs font-semibold transition bg-slate-800/80 hover:bg-slate-700/80 text-slate-300";
        }}
      }});

      applyFilters();
    }}

    function applyFilters() {{
      const query = (document.getElementById("sampleSearchInput").value || "").toLowerCase().trim();
      const activeBtn = document.querySelector(".cat-filter-btn.active-filter");
      const activeCat = activeBtn ? activeBtn.getAttribute("data-cat") : "all";

      const cards = document.querySelectorAll(".sample-card");
      let visible = 0;

      cards.forEach(card => {{
        const cardCat = card.getAttribute("data-cat");
        const searchBlob = card.getAttribute("data-search").toLowerCase();

        const matchCat = (activeCat === "all") || (cardCat === activeCat);
        const matchQuery = !query || searchBlob.includes(query);

        if (matchCat && matchQuery) {{
          card.classList.remove("hidden");
          visible++;
        }} else {{
          card.classList.add("hidden");
        }}
      }});

      document.getElementById("visibleCount").textContent = visible;
    }}

    document.getElementById("sampleSearchInput").addEventListener("input", applyFilters);

    function openLightboxByIndex(index) {{
      currentFigureIndex = (index + FIGURES_DATA.length) % FIGURES_DATA.length;
      const fig = FIGURES_DATA[currentFigureIndex];
      document.getElementById("lightboxImage").src = fig.file;
      document.getElementById("lightboxTitle").textContent = fig.title;
      document.getElementById("lightboxCaption").textContent = fig.caption;
      document.getElementById("lightboxDownload").href = fig.file;
      document.getElementById("lightboxIndexTag").textContent = (currentFigureIndex + 1) + " / " + FIGURES_DATA.length;
      document.getElementById("lightboxModal").classList.remove("hidden");
      document.body.style.overflow = "hidden";
    }}

    function navigateLightbox(step) {{
      openLightboxByIndex(currentFigureIndex + step);
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
        if (e.key === "ArrowLeft") navigateLightbox(-1);
        if (e.key === "ArrowRight") navigateLightbox(1);
      }}
    }});
  </script>

</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[+] Successfully generated modern UI index.html ({len(html_content):,} bytes).")
