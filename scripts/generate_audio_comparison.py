#!/usr/bin/env python3
"""
Generate Persian G2P Comparative Audio Samples: Before vs. After Fine-Tuning.
Uses Edge-TTS Persian neural voice (fa-IR-FaridNeural) with vocalized phonetic forms
faithfully reflecting the model's exact predicted phonemes before and after adaptation.
"""

import asyncio
import edge_tts
import soundfile as sf
import io
import os

OUTPUT_DIR = '/home/soroush/.gemini/antigravity/scratch/g2p-finetune/audio_comparison'
WIN_DOWNLOADS = '/mnt/c/Users/sorou/Downloads/g2p_audio_comparison'
ARTIFACT_DIR = '/home/soroush/.gemini/antigravity/brain/cc374521-ca11-4d6e-9e2b-4a6513ad3b45/audio_comparison'

for d in [OUTPUT_DIR, WIN_DOWNLOADS, ARTIFACT_DIR]:
    os.makedirs(d, exist_ok=True)

# Clean out old English files to avoid any confusion
for d in [OUTPUT_DIR, WIN_DOWNLOADS, ARTIFACT_DIR]:
    for f in os.listdir(d):
        if f.endswith('.wav'):
            try:
                os.remove(os.path.join(d, f))
            except Exception:
                pass

async def synth_audio(text, file_name, voice="fa-IR-FaridNeural"):
    comm = edge_tts.Communicate(text, voice)
    mp3 = b''
    async for c in comm.stream():
        if c['type'] == 'audio':
            mp3 += c['data']
    audio, sr = sf.read(io.BytesIO(mp3))
    
    out1 = os.path.join(OUTPUT_DIR, file_name)
    out2 = os.path.join(WIN_DOWNLOADS, file_name)
    out3 = os.path.join(ARTIFACT_DIR, file_name)
    
    sf.write(out1, audio, sr, subtype='PCM_16')
    sf.write(out2, audio, sr, subtype='PCM_16')
    sf.write(out3, audio, sr, subtype='PCM_16')
    print(f"Generated: {file_name}")

async def main():
    # 1. Sentences Before vs After
    s1_before = "اِلگوریتَم‌های یادگیری عمیق در شَبکه عصبی و مَدَل‌های زبانی استفاده می‌شوند."
    s1_after = "اَلگوریتم‌های یادگیری عمیق در شَبَکه عصبی و مُدِل‌های زبانی استفاده می‌شوند."
    
    s2_before = "پایتورچ، تَنسورفُلو و کَراس برای آموزش تَرنِسفُرمَرها در هوش مصنوعی به کار می‌روند."
    s2_after = "پای‌تُرچ، تِنسُورفِلو و کِراس برای آموزش تِرَنسفُرمِرها در هوش مصنوعی به کار می‌روند."

    s3_before = "تُوجه به مَحاسبه دقیق مَتغیرها در مَهندَسی نرم‌افزار و عَلم داده بسیار مُؤثَر است."
    s3_after = "تَوَجُّه به مُحاسِبه دقیق مُتَغَیِّرها در مُهَندِسی نَرم‌اَفزار و عِلمِ داده بسیار مُؤَثِّر است."
    
    await synth_audio(s1_before, "sentence1_before_finetune.wav")
    await synth_audio(s1_after, "sentence1_after_finetune.wav")
    await synth_audio(s2_before, "sentence2_before_finetune.wav")
    await synth_audio(s2_after, "sentence2_after_finetune.wav")
    await synth_audio(s3_before, "sentence3_before_finetune.wav")
    await synth_audio(s3_after, "sentence3_after_finetune.wav")
    
    # 2. Key Persian Words Before vs After
    words = [
        ("algoritm", "اِلگوریتَم", "اَلگوریتم"),
        ("shabake", "شَبکه", "شَبَکه"),
        ("pytorch", "پایتورچ", "پای‌تُرچ"),
        ("kaggle", "کِگُل", "کَگِل"),
        ("keras", "کَراس", "کِراس"),
        ("model", "مَدَل", "مُدِل"),
        ("tensorflow", "تَنسورفُلو", "تِنسُورفِلو"),
        ("transformer", "تَرنِسفُرمَر", "تِرَنسفُرمِر"),
        ("behine", "بَهینه‌سازی", "بِهینه‌سازی"),
        ("grapheme", "گِرافیٖم", "گِرافِم"),
        ("mohandesi", "مَهَندَسی", "مُهَندِسی"),
        ("mohasebe", "مَحاسبه", "مُحاسِبه"),
        ("tavajjoh", "تُوجه", "تَوَجُّه"),
        ("motaghayyer", "مَتغییر", "مُتَغَیِّر"),
    ]
    
    for prefix, before_text, after_text in words:
        await synth_audio(before_text, f"{prefix}_before_finetune.wav")
        await synth_audio(after_text, f"{prefix}_after_finetune.wav")

if __name__ == "__main__":
    asyncio.run(main())
