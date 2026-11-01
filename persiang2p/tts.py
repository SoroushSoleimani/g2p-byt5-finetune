"""
persiang2p.tts
High-MOS Neural Persian Text-to-Speech (TTS) Engine.
Couples the high-speed persiang2p phonetic front-end with an optimized
VITS ONNX neural acoustic model for fluent, natural, human-like speech synthesis.
"""

import os
import json
import time
import urllib.request
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Union

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
        if ort is None:
            raise ImportError(
                "onnxruntime is required for PersianTTS. Install it via 'pip install onnxruntime'."
            )

        self.onnx_path = onnx_path or DEFAULT_ONNX
        self.config_path = config_path or DEFAULT_JSON

        # Auto-download model weights if not present locally
        self._ensure_model_files()

        # Load Piper configuration and phoneme map
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 22050)
        self.pmap = self.config.get("phoneme_id_map", {})
        self.bos_id = self.pmap.get("^", [1])[0]
        self.eos_id = self.pmap.get("$", [2])[0]
        self.pad_id = self.pmap.get("_", [0])[0]

        # Initialize ONNX inference session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)

        # Lazy import of PersianG2P to avoid circular import issues
        from . import PersianG2P
        self.g2p = PersianG2P()

    def _ensure_model_files(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.onnx_path)), exist_ok=True)
        if not os.path.exists(self.onnx_path):
            print(f"Downloading Persian VITS ONNX model (60 MB) to {self.onnx_path}...")
            urllib.request.urlretrieve(HF_BASE_URL + "fa_IR-amir-medium.onnx", self.onnx_path)
            print("Download completed.")
        if not os.path.exists(self.config_path):
            print(f"Downloading model config to {self.config_path}...")
            urllib.request.urlretrieve(HF_BASE_URL + "fa_IR-amir-medium.onnx.json", self.config_path)

    def ipa_to_phoneme_ids(self, ipa_str: str) -> list:
        """
        Converts a Persian IPA string into interspersed VITS phoneme token IDs.
        """
        s = ipa_str.replace("t͡ʃ", "tʃ").replace("d͡ʒ", "dʒ")
        ids = [self.bos_id]
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if i + 1 < n and s[i:i+2] in self.pmap:
                token = s[i:i+2]
                ids.extend(self.pmap[token])
                ids.append(self.pad_id)
                i += 2
                continue
            
            if ch in self.pmap:
                ids.extend(self.pmap[ch])
                ids.append(self.pad_id)
            elif ch == ' ':
                ids.extend(self.pmap.get(' ', [3]))
                ids.append(self.pad_id)
            i += 1

        ids.append(self.eos_id)
        return ids

    def synthesize(
        self,
        text: str,
        is_ipa: bool = False,
        output_wav: Optional[str] = None,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesizes Persian text into a high-fidelity 22050 Hz audio waveform.
        
        Args:
            text: Input raw Persian text OR pre-transcribed IPA.
            is_ipa: If True, skips G2P transcription and uses text as raw IPA.
            output_wav: Optional path to save the output .wav file.
            speed: Speaking speed multiplier (1.0 = normal, 1.2 = faster, 0.8 = slower).
            noise_scale: Phoneme variability noise.
            noise_w: Phoneme duration variability.
            
        Returns:
            (audio_waveform, sample_rate)
        """
        if is_ipa:
            ipa = text
        else:
            ipa = self.g2p.transcribe(text)

        phoneme_ids = self.ipa_to_phoneme_ids(ipa)
        if len(phoneme_ids) <= 2:
            return np.zeros(0, dtype=np.float32), self.sample_rate

        inp = np.array([phoneme_ids], dtype=np.int64)
        inp_lens = np.array([len(phoneme_ids)], dtype=np.int64)
        length_scale = 1.0 / max(0.1, speed)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

        out = self.session.run(None, {
            "input": inp,
            "input_lengths": inp_lens,
            "scales": scales
        })

        audio = out[0].squeeze()

        if output_wav:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
            sf.write(output_wav, audio, self.sample_rate)

        return audio, self.sample_rate

    def __call__(self, text: str, output_wav: Optional[str] = None) -> np.ndarray:
        audio, _ = self.synthesize(text, output_wav=output_wav)
        return audio
