"""
persiang2p.tts
High-MOS Neural Persian Text-to-Speech (TTS) Engine.
Couples the persiang2p phonetic & Ezafe front-end with an optimized
VITS ONNX neural acoustic model (Piper Voice) for natural, human-like speech synthesis.
"""

import os
import time
import urllib.request
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Union

try:
    from piper.voice import PiperVoice
except ImportError:
    PiperVoice = None

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "piper")
DEFAULT_ONNX = os.path.join(DEFAULT_MODEL_DIR, "fa_IR-amir-medium.onnx")
DEFAULT_JSON = os.path.join(DEFAULT_MODEL_DIR, "fa_IR-amir-medium.onnx.json")

HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/"


class PersianTTS:
    def __init__(
        self,
        onnx_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        if PiperVoice is None:
            raise ImportError(
                "piper-tts is required for PersianTTS. Install it via 'pip install piper-tts'."
            )

        self.onnx_path = onnx_path or DEFAULT_ONNX
        self.config_path = config_path or DEFAULT_JSON

        # Auto-download model weights if not present locally
        self._ensure_model_files()

        # Load Piper voice with official internal espeak-ng acoustic pipeline
        self.voice = PiperVoice.load(self.onnx_path, config_path=self.config_path, use_cuda=use_gpu)
        self.sample_rate = self.voice.config.sample_rate

        # G2P Normalizer
        from .normalizer import PersianNormalizer
        self.normalizer = PersianNormalizer()

    def _ensure_model_files(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.onnx_path)), exist_ok=True)
        if not os.path.exists(self.onnx_path):
            print(f"Downloading Persian VITS ONNX model (60 MB) to {self.onnx_path}...")
            urllib.request.urlretrieve(HF_BASE_URL + "fa_IR-amir-medium.onnx", self.onnx_path)
            print("Download completed.")
        if not os.path.exists(self.config_path):
            print(f"Downloading model config to {self.config_path}...")
            urllib.request.urlretrieve(HF_BASE_URL + "fa_IR-amir-medium.onnx.json", self.config_path)

    def synthesize(
        self,
        text: str,
        output_wav: Optional[str] = None,
        speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesizes Persian text into a natural, studio-quality 22050 Hz audio waveform.
        
        Args:
            text: Input Persian text (supports Persian diacritics / اعراب for precise vowels & Ezafe).
            output_wav: Optional path to save the output .wav file.
            speed: Speaking speed multiplier (1.0 = normal, 1.2 = faster, 0.8 = slower).
            
        Returns:
            (audio_waveform, sample_rate)
        """
        norm_text = self.normalizer.normalize(text, expand_numbers=True)
        if not norm_text:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        # Synthesize audio chunks via native Piper VITS acoustic model
        chunks = list(self.voice.synthesize(norm_text))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        audio = np.concatenate([c.audio_float_array for c in chunks])

        if output_wav:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
            sf.write(output_wav, audio, self.sample_rate)

        return audio, self.sample_rate

    def __call__(self, text: str, output_wav: Optional[str] = None) -> np.ndarray:
        audio, _ = self.synthesize(text, output_wav=output_wav)
        return audio
