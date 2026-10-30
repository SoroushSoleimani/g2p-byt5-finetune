#!/usr/bin/env python3
import os
import base64

ARTIFACT_DIR = "/home/soroush/.gemini/antigravity/brain/cc374521-ca11-4d6e-9e2b-4a6513ad3b45"
AUDIO_DIR = "/home/soroush/.gemini/antigravity/scratch/g2p-finetune/audio_comparison"

samples = [
    {
        "id": "modele_amari",
        "title": "مدل آماری (Held-Out Compound & Ezafe)",
        "desc": "آزمون کلمه مرکب نادیده: تشخیص دقیق کسره اضافه (-e-) به جای سکون یا مصوت نادرست",
        "ground_truth": "/modeleɒːmɒːɾiː/",
        "before_pred": "/mædælɒːmɒːɾiː/ (۳ خطا)",
        "after_pred": "/modeleɒːmɒːɾiː/ (تطابق کامل)",
        "before_file": "modele_amari_before_finetune.wav",
        "after_file": "modele_amari_after_finetune.wav"
    },
    {
        "id": "shabake_amigh",
        "title": "شبکه عمیق (Held-Out Compound & Ezafe)",
        "desc": "آزمون ترکیب نادیده: پیش‌بینی دقیق همزه و یای اضافه (-ye-) بین مصوت e و همزه عمیق",
        "ground_truth": "/ʃæbækejeʔæmiːɢ/",
        "before_pred": "/ʃæbkeʔæmiːɢ/ (۳ خطا)",
        "after_pred": "/ʃæbækejeʔæmiːɢ/ (تطابق کامل)",
        "before_file": "shabake_amigh_before_finetune.wav",
        "after_file": "shabake_amigh_after_finetune.wav"
    },
    {
        "id": "taamol",
        "title": "تعامل (Held-Out Gemination & Glottal)",
        "desc": "آزمون کلمه نادیده با انسداد چاکنایی: پیش‌بینی ضمه /o/ روی میم به جای فتحه",
        "ground_truth": "/tæʔɒːmol/",
        "before_pred": "/tæʔɒːmæl/ (۱ خطا)",
        "after_pred": "/tæʔɒːmol/ (تطابق کامل)",
        "before_file": "taamol_before_finetune.wav",
        "after_file": "taamol_after_finetune.wav"
    },
    {
        "id": "vizhegi",
        "title": "ویژگی (Held-Out Short-Vowel Ambiguity)",
        "desc": "آزمون کلمه نادیده با مصوت کوتاه پنهان: تشخیص کسره /e/ بعد از ژ به جای سکون",
        "ground_truth": "/viːʒeɡiː/",
        "before_pred": "/viːʒɡiː/ (۱ خطا - حذف مصوت)",
        "after_pred": "/viːʒeɡiː/ (تطابق کامل)",
        "before_file": "vizhegi_before_finetune.wav",
        "after_file": "vizhegi_after_finetune.wav"
    },
    {
        "id": "pazhuheshgar",
        "title": "پژوهشگر (Held-Out Short-Vowel Ambiguity)",
        "desc": "آزمون کلمه نادیده: تفکیک دقیق فتحه ابتدایی /æ/ به جای کسره",
        "ground_truth": "/pæʒuːheʃɡæɾ/",
        "before_pred": "/peʒuːheʃɡæɾ/ (۱ خطا)",
        "after_pred": "/pæʒuːheʃɡæɾ/ (تطابق کامل)",
        "before_file": "pazhuheshgar_before_finetune.wav",
        "after_file": "pazhuheshgar_after_finetune.wav"
    },
    {
        "id": "processor",
        "title": "پروسسور (Held-Out AI & Tech Loanword)",
        "desc": "آزمون کلمه نادیده وام‌واژه: اصلاح خوشه صامت ابتدایی /pɾ/ و حذف مصوت‌های اضافه",
        "ground_truth": "/pɾosesoɾ/",
        "before_pred": "/poɾuːssuːɾ/ (۶ خطا)",
        "after_pred": "/pɾossuːɾ/ (۳ خطا - بهبود محسوس)",
        "before_file": "processor_before_finetune.wav",
        "after_file": "processor_after_finetune.wav"
    },
    {
        "id": "algoritm",
        "title": "الگوریتم (Domain Adaptation)",
        "desc": "اصلاح واکه /æ/ در هجای نخست و اصلاح توالی صامت‌های /tm/ به جای هجای اضافه",
        "ground_truth": "/ʔælɡoɾiːtm/",
        "before_pred": "/elɡuːɾiːtæm/ (۵ خطا)",
        "after_pred": "/ʔælɡoɾiːtm/ (تطابق کامل)",
        "before_file": "algoritm_before_finetune.wav",
        "after_file": "algoritm_after_finetune.wav"
    },
    {
        "id": "shabake",
        "title": "شبکه (Short Vowel Restoration)",
        "desc": "بازیابی واکه کوتاه /æ/ میانی در صامت‌های پشت‌سرهم",
        "ground_truth": "/ʃæbæke/",
        "before_pred": "/ʃæbke/ (۱ خطا - حذف واکه)",
        "after_pred": "/ʃæbæke/ (تطابق کامل)",
        "before_file": "shabake_before_finetune.wav",
        "after_file": "shabake_after_finetune.wav"
    },
    {
        "id": "model",
        "title": "مدل (Vowel Ambiguity)",
        "desc": "تمایز دقیق واکه /o/ و /e/ در برابر پیش‌فرض اشتباه /æ/ در مدل پایه",
        "ground_truth": "/model/",
        "before_pred": "/mædæl/ (۲ خطا)",
        "after_pred": "/model/ (تطابق کامل)",
        "before_file": "model_before_finetune.wav",
        "after_file": "model_after_finetune.wav"
    },
    {
        "id": "tavajjoh",
        "title": "توجه (Gemination / Tashdid)",
        "desc": "تشخیص دقیق تشدید صامت جیم /d͡ʒd͡ʒ/ و واکه کوتاه /æ/ به جای تلفظ کاملاً اشتباه",
        "ground_truth": "/tævæd͡ʒd͡ʒoh/",
        "before_pred": "/tuːd͡ʒe/ (۸ خطا)",
        "after_pred": "/tævæd͡ʒd͡ʒoh/ (تطابق کامل)",
        "before_file": "tavajjoh_before_finetune.wav",
        "after_file": "tavajjoh_after_finetune.wav"
    },
    {
        "id": "sentence1",
        "title": "جمله آزمایشی ۱ (Sentence Pipeline)",
        "desc": "مدل ترنسفورمر و شبکه عصبی در پایتورچ و تنسورفلو الگوریتم گرافیم را بهینه‌سازی می‌کنند.",
        "ground_truth": "جمله تخصصی هوش مصنوعی شامل ۷ واژه چالش‌برانگیز",
        "before_pred": "تلفظ با واکه‌های اشتباه (مَدَل، شَبکه، اِلگوریتَم)",
        "after_pred": "تلفظ اصلاح‌شده با زنجیره واجی طبیعی (مُدِل، شَبَکه، اَلگوریتم)",
        "before_file": "sentence1_before_finetune.wav",
        "after_file": "sentence1_after_finetune.wav"
    },
    {
        "id": "sentence2",
        "title": "جمله آزمایشی ۲ (Sentence Pipeline)",
        "desc": "مهندسی پزشکی و محاسبه دقیق متغیرها در هوش مصنوعی و یادگیری ماشین توجه زیادی دارد.",
        "ground_truth": "جمله حاوی ترکیبات اضافه و تشدید",
        "before_pred": "تلفظ با نادیده گرفتن تشدید و کسر اضافه (تُوجه، مَتغیرها)",
        "after_pred": "تلفظ دقیق با تشدید و هماهنگی واکه‌ای (تَوَجُّه، مُتَغَیِّرها)",
        "before_file": "sentence2_before_finetune.wav",
        "after_file": "sentence2_after_finetune.wav"
    },
    {
        "id": "sentence3",
        "title": "جمله آزمایشی ۳ (Sentence Pipeline)",
        "desc": "پژوهشگران علم داده در کگل با فریم‌ورک دیپ‌لرنینگ و توکنایزر مسئله اتصال را حل کردند.",
        "ground_truth": "جمله با وام‌واژه‌های یادگیری ماشین و همزه",
        "before_pred": "تلفظ بدون کسر اضافه با خطای فریم‌ورک و دیپ‌لرنینگ",
        "after_pred": "تلفظ روان با زنجیره واجی استاندارد فارسی",
        "before_file": "sentence3_before_finetune.wav",
        "after_file": "sentence3_after_finetune.wav"
    }
]

for s in samples:
    b_path = os.path.join(AUDIO_DIR, s["before_file"])
    a_path = os.path.join(AUDIO_DIR, s["after_file"])
    with open(b_path, "rb") as f:
        s["before_b64"] = "data:audio/wav;base64," + base64.b64encode(f.read()).decode("ascii")
    with open(a_path, "rb") as f:
        s["after_b64"] = "data:audio/wav;base64," + base64.b64encode(f.read()).decode("ascii")

cards_html = []
for i, s in enumerate(samples, 1):
    card = f"""
      <div class="bg-slate-800/90 border border-slate-700/80 rounded-xl p-5 shadow-sm hover:border-slate-500/50 transition">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 mb-3 border-b border-slate-700/50 pb-3">
          <div>
            <div class="flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-slate-700 text-xs flex items-center justify-center font-bold text-slate-300">{i}</span>
              <h2 class="font-bold text-base text-slate-100">{s['title']}</h2>
            </div>
            <p class="text-xs text-slate-400 mt-1">{s['desc']}</p>
          </div>
          <div class="text-left font-mono text-xs bg-slate-900/90 px-3 py-1.5 rounded-lg border border-slate-700/60 text-emerald-400" dir="ltr">
            Ref: {s['ground_truth']}
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
          <!-- Before Card -->
          <div class="bg-red-950/20 border border-red-900/40 rounded-lg p-3 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs font-bold text-red-400">مدل پایه چندزبانه (قبل)</span>
                <span class="text-xs text-red-300/80 font-mono" dir="ltr">{s['before_pred']}</span>
              </div>
            </div>
            <div class="mt-3 flex items-center gap-2">
              <button onclick="playAudio('{s['id']}_before')" id="btn_{s['id']}_before" 
                class="w-full py-2 px-3 bg-red-600/30 hover:bg-red-600/50 text-red-200 border border-red-500/40 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition">
                <span>پخش صدای قبل</span>
              </button>
              <audio id="audio_{s['id']}_before" src="{s['before_b64']}" preload="none" onended="onAudioEnded('{s['id']}_before')"></audio>
            </div>
          </div>

          <!-- After Card -->
          <div class="bg-emerald-950/20 border border-emerald-900/40 rounded-lg p-3 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs font-bold text-emerald-400">مدل فاین‌تیون‌شده فارسی (بعد)</span>
                <span class="text-xs text-emerald-300/80 font-mono" dir="ltr">{s['after_pred']}</span>
              </div>
            </div>
            <div class="mt-3 flex items-center gap-2">
              <button onclick="playAudio('{s['id']}_after')" id="btn_{s['id']}_after" 
                class="w-full py-2 px-3 bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-200 border border-emerald-500/40 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition">
                <span>پخش صدای بعد</span>
              </button>
              <audio id="audio_{s['id']}_after" src="{s['after_b64']}" preload="none" onended="onAudioEnded('{s['id']}_after')"></audio>
            </div>
          </div>
        </div>
      </div>
"""
    cards_html.append(card)

html_content = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; }}
    .playing {{ outline: 2px solid #3b82f6; background-color: rgba(59, 130, 246, 0.4) !important; }}
  </style>
</head>
<body class="bg-slate-900 text-slate-100 p-6 antialiased">
  <div class="max-w-4xl mx-auto space-y-6">
    <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-md">
      <div class="flex items-center justify-between border-b border-slate-700 pb-4 mb-4">
        <div>
          <h1 class="text-xl font-bold text-white">پلیر مقایسه صوتی خروجی‌های Before و After فاین‌تیون (Persian G2P ByT5)</h1>
          <p class="text-xs text-slate-400 mt-1">تولید شده با صدای عصبی fa-IR-FaridNeural بر مبنای توالی واج‌های استخراج‌شده</p>
        </div>
        <span class="px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full text-xs font-semibold">13 جفت مقایسه‌ای منتخب</span>
      </div>
      
      <p class="text-xs text-slate-300 leading-relaxed">
        روی دکمه‌های «پخش صدای قبل» و «پخش صدای بعد» کلیک کنید تا تفاوت در بازیابی مصوت‌های کوتاه پنهان (/e/، /o/)، پیاده‌سازی کسر اضافه و تلفظ صامت‌های مشدد را بشنوید.
      </p>
    </div>

    <div class="space-y-4">
      {"".join(cards_html)}
    </div>
  </div>

  <script>
    let currentAudio = null;
    let currentBtn = null;

    function playAudio(id) {{
      const audio = document.getElementById('audio_' + id);
      const btn = document.getElementById('btn_' + id);
      
      if (currentAudio && currentAudio !== audio) {{
        currentAudio.pause();
        currentAudio.currentTime = 0;
        if (currentBtn) currentBtn.classList.remove('playing');
      }}

      if (audio.paused) {{
        audio.play();
        btn.classList.add('playing');
        currentAudio = audio;
        currentBtn = btn;
      }} else {{
        audio.pause();
        audio.currentTime = 0;
        btn.classList.remove('playing');
        currentAudio = null;
        currentBtn = null;
      }}
    }}

    function onAudioEnded(id) {{
      const btn = document.getElementById('btn_' + id);
      if (btn) btn.classList.remove('playing');
      if (currentAudio === document.getElementById('audio_' + id)) {{
        currentAudio = null;
        currentBtn = null;
      }}
    }}
  </script>
</body>
</html>
"""

out_html_path = os.path.join(ARTIFACT_DIR, "persian_g2p_audio_player.html")
with open(out_html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[+] Saved interactive audio player to {out_html_path} ({round(len(html_content)/1024, 1)} KB)")
