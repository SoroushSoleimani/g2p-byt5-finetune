"""
persiang2p.tts
Studio-Grade Neural Persian Text-to-Speech (TTS) Engine.
Directly couples persiang2p high-speed phonetic & Ezafe front-end
with Studio VITS ONNX neural acoustic models for natural, human-like Persian speech.
"""

import os
import shutil
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Union, List

# Supported voices
VOICE_CONFIGS = {
    "ganji": {
        "name": "vits-piper-fa-ganji-adabi",
        "description": "Studio Literary Audiobook Narrator (22.05 kHz, 8.5x Real-Time)",
        "sample_rate": 22050,
        "repo_subpath": "vits-piper-fa-ganji-adabi",
        "vowel_a": "keep",
    },
    "smartgiti": {
        "name": "male-SmartGitiCorp-coqui-vits",
        "description": "Studio Corporate Male (44.1 kHz High Fidelity)",
        "sample_rate": 44100,
        "repo_subpath": "male-SmartGitiCorp-coqui-vits",
        "vowel_a": "alpha",
    },
    "gptinformal": {
        "name": "female-GPTInformal-coqui-vits",
        "description": "Studio Conversational Female (24 kHz)",
        "sample_rate": 24000,
        "repo_subpath": "female-GPTInformal-coqui-vits",
        "vowel_a": "alpha",
    },
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class PersianTTS:
    """
    High-MOS Persian Neural Text-to-Speech Engine.
    
    Voices:
        - 'smartgiti': Studio Corporate Voice (44.1 kHz, High Fidelity)
        - 'ganji': Studio Literary Narrator (22.05 kHz, 8.5x Real-Time) [Recommended for narration]
        - 'gptinformal': Studio Conversational Female (24 kHz)
        - 'amir': Piper Baseline (Amateur Voice)
    """

    def __init__(
        self,
        voice: str = "ganji",
        model_path: Optional[str] = None,
        tokens_path: Optional[str] = None,
        num_threads: int = 4,
    ):
        self.voice_name = voice.lower()
        self.num_threads = num_threads

        # Import PersianG2P front-end
        from . import PersianG2P
        self.g2p = PersianG2P()

        # Handle 'amir' Piper baseline separately
        if self.voice_name == "amir":
            try:
                from piper.voice import PiperVoice
                onnx = model_path or os.path.join(MODELS_DIR, "piper", "fa_IR-amir-medium.onnx")
                cfg = tokens_path or os.path.join(MODELS_DIR, "piper", "fa_IR-amir-medium.onnx.json")
                self.backend = "piper"
                self.piper_voice = PiperVoice.load(onnx, config_path=cfg)
                self.sample_rate = self.piper_voice.config.sample_rate
                return
            except Exception as e:
                print(f"Failed to load Piper voice 'amir': {e}. Falling back to 'ganji'.")
                self.voice_name = "ganji"

        if self.voice_name not in VOICE_CONFIGS:
            valid_keys = list(VOICE_CONFIGS.keys()) + ["amir"]
            raise ValueError(f"Unknown voice '{voice}'. Choose from: {valid_keys}")

        vconfig = VOICE_CONFIGS[self.voice_name]
        self.vowel_mode = vconfig["vowel_a"]

        # Ensure model directory exists
        v_dir = os.path.join(MODELS_DIR, vconfig["name"])
        self.model_path = model_path or os.path.join(v_dir, "model.onnx")
        self.tokens_path = tokens_path or os.path.join(v_dir, "tokens.txt")

        self._ensure_model_files(vconfig["repo_subpath"], v_dir)

        # Initialize sherpa-onnx engine
        import sherpa_onnx
        self.sherpa_onnx = sherpa_onnx
        
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=self.model_path,
                    tokens=self.tokens_path,
                ),
                provider="cpu",
                num_threads=self.num_threads,
            )
        )
        if not cfg.validate():
            raise RuntimeError(f"Sherpa-ONNX configuration validation failed for {self.model_path}")

        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.sample_rate = self.tts.sample_rate
        self.backend = "sherpa-onnx"

    def _ensure_model_files(self, repo_subpath: str, target_dir: str):
        if not os.path.exists(self.model_path) or not os.path.exists(self.tokens_path):
            os.makedirs(target_dir, exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading studio voice model '{repo_subpath}' from Hugging Face...")
                for fname in ["model.onnx", "tokens.txt"]:
                    f_dl = hf_hub_download(repo_id="karim23657/persian-tts-vits", filename=f"{repo_subpath}/{fname}")
                    dest = os.path.join(target_dir, fname)
                    shutil.copy2(f_dl, dest)
                print("Model download completed successfully.")
            except Exception as e:
                raise RuntimeError(
                    f"Model files missing at {target_dir} and automatic download failed: {e}"
                )

    def _phonemes_to_tokens(self, ipa_string: str) -> List[str]:
        """Maps persiang2p IPA string to model vocabulary tokens."""
        # Clean affricate ties and punctuation
        cleaned = (
            ipa_string.replace("\u0361", "")
            .replace("،", ",")
            .replace("؛", ";")
            .replace("؟", "?")
        )
        if self.vowel_mode == "alpha":
            cleaned = cleaned.replace("ɒ", "ɑ")
        return list(cleaned)

    def synthesize(
        self,
        text: str,
        output_wav: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesizes Persian text into a high-fidelity studio waveform.

        Args:
            text: Input Persian text.
            output_wav: Optional path to save .wav file.
            speed: Speaking speed multiplier (1.0 = standard).

        Returns:
            (audio_waveform, sample_rate)
        """
        if self.backend == "piper":
            # Piper text path
            norm_text = self.g2p.normalizer.normalize(text, expand_numbers=True)
            chunks = list(self.piper_voice.synthesize(norm_text))
            if not chunks:
                return np.zeros(0, dtype=np.float32), self.sample_rate
            audio = np.concatenate([c.audio_float_array for c in chunks])
        else:
            # Sherpa-ONNX direct phonetic path
            ipa = self.g2p(text)
            tokens = self._phonemes_to_tokens(ipa)
            
            gen_cfg = self.sherpa_onnx.GenerationConfig()
            gen_cfg.tokens = [tokens]
            gen_cfg.speed = speed
            
            res = self.tts.generate("", config=gen_cfg)
            audio = np.array(res.samples, dtype=np.float32)

        if output_wav:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
            sf.write(output_wav, audio, self.sample_rate)

        return audio, self.sample_rate

    def __call__(self, text: str, output_wav: Optional[str] = None) -> np.ndarray:
        audio, _ = self.synthesize(text, output_wav=output_wav)
        return audio
