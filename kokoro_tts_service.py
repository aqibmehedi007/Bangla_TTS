"""
Kokoro TTS Service for QuteVoice
Based on the working implementation from RAG_Simplified project
"""

import io
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

VOICE_LANG_MAP = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}


class KokoroTTSService:
    """Text-to-Speech service using Kokoro model"""

    def __init__(self, model_path: str = "./models/Kokoro_espeak_Q4.gguf"):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.pipelines: Dict[str, Any] = {}
        self.model = None

        # Default voice fallback
        self.default_voice = "af_bella"
        self.sample_rate = 24000
        logger.info(f"Kokoro TTS service initialized on device: {self.device}")

    async def initialize(self) -> bool:
        """Flag service as ready; pipelines are created lazily per language."""
        if self.initialized:
            return True

        if not self.model_path.exists():
            logger.warning("Kokoro model file not found, will operate in demo/fallback mode")
            return False

        try:
            from kokoro import KModel  # noqa: F401
        except ImportError:
            logger.error("Kokoro library not installed. Please run: pip install kokoro")
            return False

        self.initialized = True
        return True

    async def _get_pipeline(self, lang_code: str):
        """Create or reuse a KPipeline for the requested language."""
        from kokoro import KPipeline

        lang_code = (lang_code or "a").lower()
        if lang_code not in VOICE_LANG_MAP:
            lang_code = "a"

        if lang_code in self.pipelines:
            return self.pipelines[lang_code]

        logger.info(f"Creating Kokoro pipeline for language '{lang_code}' on device {self.device}")
        pipeline = KPipeline(lang_code=lang_code, device=self.device)
        self.pipelines[lang_code] = pipeline
        return pipeline

    def _extract_lang_from_voice(self, voice: Optional[str], explicit_lang: Optional[str]) -> str:
        if explicit_lang and explicit_lang.lower() in VOICE_LANG_MAP:
            return explicit_lang.lower()
        if voice:
            prefix = voice.strip().split(",")[0]
            if prefix:
                lang_code = prefix[0].lower()
                if lang_code in VOICE_LANG_MAP:
                    return lang_code
        return "a"

    async def synthesize_speech(self, text: str, voice: Optional[str] = None, speed: float = 1.0,
                                language: Optional[str] = None) -> bytes:
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return b""

        if not self.initialized and not await self.initialize():
            raise RuntimeError("Kokoro service failed to initialize")

        voice_id = voice or self.default_voice
        lang_code = self._extract_lang_from_voice(voice_id, language)

        try:
            pipeline = await self._get_pipeline(lang_code)
        except Exception as exc:
            logger.error(f"Could not create Kokoro pipeline: {exc}")
            raise RuntimeError("Pipeline creation failed") from exc

        # Ensure requested voice can be loaded; fall back if necessary
        try:
            pipeline.load_voice(voice_id)
        except Exception as exc:
            logger.warning(
                "Voice '%s' unavailable (%s). Falling back to '%s'.",
                voice_id, exc, self.default_voice
            )
            voice_id = self.default_voice
            try:
                pipeline.load_voice(voice_id)
            except Exception as inner_exc:
                logger.error("Fallback voice '%s' also failed: %s", voice_id, inner_exc)
                raise RuntimeError(f"Unable to load voice '{voice}'") from inner_exc

        try:
            logger.info(
                "Synthesizing speech (%s chars) voice=%s lang=%s speed=%s",
                len(text), voice_id, lang_code, speed,
            )
            audio_segments = []
            for result in pipeline(text, voice=voice_id, speed=float(speed)):
                audio = getattr(result, "audio", None)
                if audio is not None and len(audio) > 0:
                    audio_segments.append(audio)

            if not audio_segments:
                logger.error("No audio produced by Kokoro for voice %s", voice_id)
                return b""

            full_audio = np.concatenate([segment.numpy() if hasattr(segment, "numpy") else segment
                                          for segment in audio_segments])
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, full_audio, self.sample_rate, format="WAV")
            return audio_bytes.getvalue()
        except Exception as exc:
            logger.exception("Kokoro synthesis failed: %s", exc)
            raise RuntimeError(str(exc))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "Kokoro TTS" if self.initialized else "Kokoro TTS (uninitialized)",
            "model_file": str(self.model_path),
            "device": self.device,
            "sample_rate": self.sample_rate,
            "initialized": self.initialized,
        }

    def is_available(self) -> bool:
        return self.initialized

    def unload_model(self) -> bool:
        try:
            self.pipelines.clear()
            self.initialized = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as exc:
            logger.error("Failed to unload Kokoro model: %s", exc)
            return False


def generate_demo_tone(text: str, sample_rate: int = 24000) -> bytes:
    import math

    duration = max(1.5, min(12.0, len(text) * 0.08))
    t = np.linspace(0, duration, int(sample_rate * duration))
    base_freq = 440 + (hash(text) % 160)
    waveform = 0.3 * np.sin(2 * math.pi * base_freq * t)
    envelope = np.exp(-t * 1.8) * (1 - np.exp(-t * 6))
    audio = waveform * envelope
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio, sample_rate, format="WAV")
    return audio_bytes.getvalue()


class DemoTTSService:
    """Demo TTS service for when Kokoro is not available"""

    def __init__(self):
        self.device = "cpu"
        self.initialized = True
        self.sample_rate = 24000
        self.available_voices = [
            "af_heart", "af_bella", "am_onyx", "bf_emma", "zf_xiaoxiao",
        ]
        self.default_voice = "af_heart"
        logger.info("🎭 Demo TTS service initialized")

    async def initialize(self):
        return True

    def is_available(self):
        return True

    async def synthesize_speech(self, text: str, voice: Optional[str] = None,
                                speed: float = 1.0, language: Optional[str] = None) -> bytes:
        return generate_demo_tone(text, self.sample_rate)

    def get_model_info(self):
        return {
            "model_name": "Demo TTS",
            "model_file": "demo_mode",
            "device": self.device,
            "sample_rate": self.sample_rate,
            "initialized": True,
            "available_voices": len(self.available_voices),
        }
