"""
Kokoro TTS Service for Aurora Voice Studio
Integrated with the official kokoro library (v0.9.2+)
"""

import io
import logging
import asyncio
from typing import Optional, Dict, Any, List
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
    "bn": "Bengali",
}

# Mapping from Bengali characters to IPA-like phonemes (simplified for Kokoro/IPA)
BN_IPA_MAP = {
    'অ': 'ɔ', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u', 'ঋ': 'ri',
    'এ': 'e', 'ঐ': 'oj', 'ও': 'o', 'ঔ': 'ow',
    'ক': 'k', 'খ': 'kʰ', 'গ': 'g', 'ঘ': 'gʰ', 'ঙ': 'ŋ',
    'চ': 'c', 'ছ': 'cʰ', 'জ': 'ɟ', 'ঝ': 'ɟʰ', 'ঞ': 'n',
    'ট': 'ʈ', 'ঠ': 'ʈʰ', 'ড': 'ɖ', 'ঢ': 'ɖʰ', 'ণ': 'n',
    'ত': 't', 'থ': 'tʰ', 'দ': 'd', 'ধ': 'dʰ', 'ন': 'n',
    'প': 'p', 'ফ': 'pʰ', 'ব': 'b', 'ভ': 'bʰ', 'ম': 'm',
    'য': 'ɟ', 'র': 'r', 'ল': 'l', 'শ': 'ʃ', 'ষ': 'ʃ', 'স': 's', 'হ': 'h',
    'ড়': 'ɽ', 'ঢ়': 'ɽʰ', 'য়': 'j', 'ৎ': 't',
    'ং': 'ŋ', 'ঃ': 'h', 'ঁ': '̃',
    'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u', 'ৃ': 'ri',
    'ে': 'e', 'ৈ': 'oj', 'ো': 'o', 'ৌ': 'ow', '্': '',
}

# Mapping for Bengali to Romanized English (manual quality)
BN_ROMAN_WORDS = {
    'নমস্কার': 'nomoshkar', 'আপনি': 'apni', 'কেমন': 'kemon', 'আছেন': 'achhen',
    'সুন্দর': 'shundor', 'আমার': 'amar', 'নাম': 'nam', 'আকাশ': 'akash',
    'আপনার': 'apnar', 'কি': 'ki', 'আজকের': 'ajker', 'আবহাওয়া': 'abohawa',
    'খুব': 'khub', 'আমি': 'ami', 'ভাত': 'bhat', 'খেতে': 'khete',
    'ভালোবাসি': 'bhalobashi', 'কালকে': 'kalke', 'বাজারে': 'bajare', 'যাবেন': 'jaben',
    'বই': 'boi', 'পড়া': 'pora', 'প্রিয়': 'priyo', 'শখ': 'shokh',
    'এই': 'ei', 'কাজটি': 'kajti', 'করা': 'kora', 'কঠিন': 'kothin',
    'সূর্য': 'shurjo', 'পূর্ব': 'purbo', 'দিকে': 'dike', 'ওঠে': 'othe',
    'এবং': 'ebong', 'পশ্চিম': 'poshchim', 'অস্ত': 'osto', 'যায়': 'jay',
    'ঢাকা': 'dhaka', 'বাংলাদেশ': 'bangladesh', 'রাজধানী': 'rajdhani',
}

def bengali_to_roman(text: str) -> str:
    """Manual/Heuristic conversion of Bengali text to English alphabet pronunciation."""
    import re
    # If text is already mostly English alphabet, return it (user might type Romanized)
    if re.search(r'[a-zA-Z]{3,}', text):
        return text

    words = text.split()
    roman_words = []
    for word in words:
        clean = re.sub(r'[,.?!]', '', word)
        punc = word[len(clean):]
        if clean in BN_ROMAN_WORDS:
            roman_words.append(BN_ROMAN_WORDS[clean] + punc)
        else:
            # Simple heuristic for unmapped words
            roman_part = ""
            for char in clean:
                if char in BN_IPA_MAP:
                    # Reuse the character map but map to closer English sounds
                    m = BN_IPA_MAP[char]
                    m = m.replace('ɔ', 'o').replace('ɟ', 'j').replace('c', 'ch').replace('ʃ', 'sh')
                    roman_part += m
                else:
                    roman_part += char
            roman_words.append(roman_part + punc)
    return " ".join(roman_words).strip()

class KokoroTTSService:
    """Text-to-Speech service using official Kokoro library"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Kokoro TTS service
        
        Args:
            model_path: Optional path to a specific model file (.pth or .onnx)
        """
        self.model_path = Path(model_path) if model_path else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.pipelines: Dict[str, Any] = {}
        self.available_voices: Dict[str, Any] = {}
        
        # Default configuration
        self.default_voice = "af_heart"
        self.default_bn_voice = "bnm_custom" # Default to Male for Bengali
        self.sample_rate = 24000
        logger.info(f"Kokoro TTS service initialized on device: {self.device}")

    async def initialize(self) -> bool:
        """Initialize the service and verify library availability"""
        if self.initialized:
            return True

        try:
            from kokoro import KPipeline
            logger.info("Successfully imported KPipeline from kokoro library")
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
        try:
            # Note: KPipeline handles model downloading/loading automatically
            pipeline = KPipeline(lang_code=lang_code, device=self.device)
            self.pipelines[lang_code] = pipeline
            return pipeline
        except Exception as e:
            if lang_code == "bn":
                logger.warning(f"KPipeline does not support 'bn', will use custom handler: {e}")
                return None
            logger.error(f"Failed to create pipeline for '{lang_code}': {e}")
            raise

    def _extract_lang_from_voice(self, voice: Optional[str], explicit_lang: Optional[str]) -> str:
        """Heuristic to determine language code from voice ID or explicit parameter"""
        if explicit_lang and explicit_lang.lower() in VOICE_LANG_MAP:
            return explicit_lang.lower()
        
        # Use parsed voices for more accurate language code
        if voice and voice in self.available_voices:
            return self.available_voices[voice]["language_code"]

        if voice:
            # Fallback to old heuristic if voice not found in parsed list
            # Voice IDs usually start with lang code (e.g., 'af_heart' -> 'a')
            prefix = voice.strip().split(",")[0]
            if prefix:
                lang_code = prefix[0].lower()
                if lang_code in VOICE_LANG_MAP:
                    return lang_code
            # Special case for Bengali
            if "bn" in prefix:
                return "bn"
        return "a"

    async def synthesize_speech(self, text: str, voice: Optional[str] = None, speed: float = 1.0,
                                language: Optional[str] = None) -> bytes:
        """
        Synthesize speech from text
        
        Args:
            text: Input text
            voice: Voice ID (e.g., 'af_heart')
            speed: Speaking speed multiplier
            language: Explicit language code (e.g., 'a', 'b', 'e'...)
            
        Returns:
            WAV audio data in bytes
        """
        if not text or not text.strip():
            return b""

        if not self.initialized and not await self.initialize():
            raise RuntimeError("Kokoro service failed to initialize")

        # Initial assignment
        voice_id = (voice or self.default_voice).replace("\\", "")
        lang_code = self._extract_lang_from_voice(voice_id, language)

        # Re-check defaults if lang is specifically bn
        if lang_code == 'bn' and not voice:
             voice_id = self.default_bn_voice
        
        # Normalization for custom Bengali voices to avoid 404s
        original_voice_id = voice_id
        if voice_id in ["bnm_custom", "bnf_custom"]:
            voice_id = "am_adam" if "bnm" in voice_id else "af_heart"
            logger.info(f"Using base style '{voice_id}' for Bengali voice '{original_voice_id}'")

        try:
            if lang_code == 'bn':
                logger.info(f"Custom Bengali synthesis: text='{text[:30]}...' | base_style='{voice_id}'")
                
                # Load custom model if not cached
                if 'bn_model' not in self.pipelines:
                    from kokoro.model import KModel
                    from huggingface_hub import hf_hub_download
                    base_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1_0.pth")
                    model = KModel(config=None, model=base_path).to(self.device)
                    
                    checkpoint_path = Path("training/checkpoints/kokoro_bn_epoch_50.pth")
                    if checkpoint_path.exists():
                        logger.info(f"Applying custom Bengali weights from {checkpoint_path}")
                        state_dict = torch.load(checkpoint_path, map_location=self.device)
                        model.load_state_dict(state_dict)
                    
                    model.eval()
                    self.pipelines['bn_model'] = model
                
                model = self.pipelines['bn_model']
                
                # Use a wrapper pipeline for handling phonemes and styles
                if 'bn_pipeline_wrapper' not in self.pipelines:
                    from kokoro import KPipeline
                    wrapper = KPipeline(lang_code='a', device=self.device)
                    wrapper.model = model
                    self.pipelines['bn_pipeline_wrapper'] = wrapper
                
                pipeline = self.pipelines['bn_pipeline_wrapper']
                
                # Convert to Romanized English alphabet
                roman_text = bengali_to_roman(text)
                logger.info(f"Bengali Romanized: {roman_text[:50]}")
                
                # Load the 512-dim style vector for the custom voice
                custom_style_path = Path("core/voices/bnm_custom.pt")
                if custom_style_path.exists():
                    logger.info(f"Loading custom optimized voice style from {custom_style_path}")
                    ref_s = torch.load(custom_style_path, map_location=self.device)
                else:
                    ref_s = pipeline.load_voice(voice_id)
                
                audio_segments = []
                # Use as plain text to leverage the model's native English G2P/training
                for _, _, audio in pipeline(roman_text, voice=ref_s, speed=float(speed)):
                    if audio is not None:
                        if hasattr(audio, "cpu"):
                            audio = audio.cpu().numpy()
                        audio_segments.append(audio)
                
                if not audio_segments:
                    raise ValueError("No audio segments produced")
                full_audio = np.concatenate(audio_segments)
                
            else:
                pipeline = await self._get_pipeline(lang_code)
                audio_segments = []
                for _, _, audio in pipeline(text, voice=voice_id, speed=float(speed)):
                    if audio is not None:
                        if hasattr(audio, "cpu"):
                            audio = audio.cpu().numpy()
                        audio_segments.append(audio)

                if not audio_segments:
                    return b""
                full_audio = np.concatenate(audio_segments)
            
            # Export to WAV
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, full_audio, self.sample_rate, format="WAV")
            return audio_bytes.getvalue()
            
        except Exception as exc:
            logger.exception(f"Synthesis failed: {exc}")
            raise RuntimeError(f"Synthesis error: {exc}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return status and info about the service"""
        return {
            "model_name": "Kokoro TTS (Official Library)",
            "device": self.device,
            "sample_rate": self.sample_rate,
            "initialized": self.initialized,
            "loaded_pipelines": list(self.pipelines.keys()),
            "available_voices_count": len(self.available_voices),
        }

    def is_available(self) -> bool:
        """Check if service is initialized and ready"""
        return self.initialized

    def unload_model(self) -> bool:
        """Free up resources"""
        try:
            self.pipelines.clear()
            self.initialized = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as exc:
            logger.error(f"Failed to unload models: {exc}")
            return False

class DemoTTSService:
    """Mock service for fallback/demo purposes"""

    def __init__(self):
        self.device = "cpu"
        self.initialized = True
        self.sample_rate = 24000
        logger.info("🎭 Demo TTS service initialized")

    async def initialize(self):
        return True

    def is_available(self):
        return True

    def get_model_info(self):
        return {
            "model_name": "Demo Mode (Library Unavailable)",
            "device": "cpu",
            "sample_rate": 24000,
            "initialized": True,
        }

    async def synthesize_speech(self, text: str, voice: Optional[str] = None,
                                speed: float = 1.0, language: Optional[str] = None) -> bytes:
        """Generate a simple sine tone for demo"""
        duration = max(1.0, len(text) * 0.05 / float(speed))
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        freq = 440 + (hash(text) % 200)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, self.sample_rate, format="WAV")
        return audio_bytes.getvalue()
