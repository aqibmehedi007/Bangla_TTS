"""
Generate custom Bengali audio using XTTS v2 Zero-Shot Voice Cloning.
"""
import os
import torch
import functools
from pathlib import Path
from TTS.api import TTS

# Fix for PyTorch 2.4+ security restriction
torch.load = functools.partial(torch.load, weights_only=False)

# Paths
PROJECT_ROOT = Path("d:/GitHub/Kokoro_tts_module")
REFERENCE_VOICE = PROJECT_ROOT / "data" / "bengali" / "xtts" / "reference_voice.wav"
OUTPUT_PATH = PROJECT_ROOT / "training" / "xtts_checkpoints" / "custom_bangla_test.wav"

def main():
    text = "Amar shonar bangla, ami tomay bhalobashi."
    print(f"🎙️ Synthesizing: '{text}'")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    
    if not REFERENCE_VOICE.exists():
        print(f"❌ Error: Reference voice not found at {REFERENCE_VOICE}")
        return

    # Generate audio
    tts.tts_to_file(
        text=text,
        file_path=str(OUTPUT_PATH),
        speaker_wav=str(REFERENCE_VOICE),
        language="en" # Using English phonetics for Romanized Bengali
    )
    
    print(f"✅ Audio generated successfully at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
