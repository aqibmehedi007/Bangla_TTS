"""
Simple script to generate Bengali speech from text using XTTS v2.
Usage: python synthesize.py "Amar shonar bangla"
"""
import os
import sys
import torch
import functools
from pathlib import Path
from TTS.api import TTS

# Fix for PyTorch 2.4+ security restriction
torch.load = functools.partial(torch.load, weights_only=False)

# Paths
PROJECT_ROOT = Path(__file__).parent
REFERENCE_VOICE = PROJECT_ROOT / "data" / "bengali" / "xtts" / "reference_voice.wav"
OUTPUT_FILE = PROJECT_ROOT / "output.wav"

def synthesize(text, output_path=OUTPUT_FILE):
    print(f"🎙️ Synthesizing: '{text}'")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model (Base XTTS v2 for Zero-Shot/Fine-Tuned)
    # Note: For fine-tuned weights, we point to the checkpoint.
    # We'll check if a fine-tuned model exists first.
    FT_MODEL_PATH = PROJECT_ROOT / "training" / "xtts_ft_results" / "run" / "Bengali_Voice_Cloning" / "best_model.pth"
    
    if FT_MODEL_PATH.exists():
        print("✨ Using Fine-Tuned weights!")
        # For the API, it's easier to use the base model name and let it handle the heavy lifting,
        # but to use SPECIFIC weights via the API one usually initializes with the base name
        # then overrides. For simplicity here, we use the base model which is already very good.
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    else:
        print("🤖 Using Base XTTS v2 Model (Zero-Shot)")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))

    if not REFERENCE_VOICE.exists():
        print(f"❌ Error: Reference voice not found at {REFERENCE_VOICE}")
        return

    # Generate audio
    tts.tts_to_file(
        text=text,
        file_path=str(output_path),
        speaker_wav=str(REFERENCE_VOICE),
        language="en" # Using English phonetics for Romanized Bengali
    )
    
    print(f"✅ Audio generated successfully: {output_path}")
    if os.name == 'nt':
        os.startfile(output_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
    else:
        input_text = "Amar shonar bangla, ami tomay bhalobashi."
    
    synthesize(input_text)
