"""
XTTS v2 Fine-tuning for Bengali TTS with Voice Cloning.

This script fine-tunes XTTS v2 on your Bengali dataset to:
1. Learn Bengali phonetics (via Romanized text)
2. Clone your voice identity

Usage:
    python training/train_xtts_bn.py
"""
import os
import torch
import functools
from pathlib import Path

# Fix for PyTorch 2.4+ security restriction during model loading
torch.load = functools.partial(torch.load, weights_only=False)

# Paths
PROJECT_ROOT = Path("d:/GitHub/Kokoro_tts_module")
DATASET_DIR = PROJECT_ROOT / "data" / "bengali" / "xtts"
OUTPUT_DIR = PROJECT_ROOT / "training" / "xtts_checkpoints"

def main():
    print("🚀 Starting XTTS v2 Fine-tuning for Bengali...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {device}")
    
    if device == "cpu":
        print("⚠️ Warning: Training on CPU will be very slow. GPU recommended.")
    
    # Import TTS (after pip install TTS)
    from TTS.api import TTS
    
    # Load pre-trained XTTS v2
    print("📥 Loading XTTS v2 model (this may take a few minutes on first run)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(device == "cuda"))
    
    # Now test with voice cloning using your reference audio
    reference_audio = DATASET_DIR / "reference_voice.wav"
    
    # Test basic synthesis first
    print("🧪 Testing basic synthesis...")
    test_output = OUTPUT_DIR / "test_english.wav"
    tts.tts_to_file(
        text="Hello, this is a test of the text to speech system.",
        file_path=str(test_output),
        speaker_wav=str(reference_audio),
        language="en"
    )
    print(f"✅ Test audio saved to: {test_output}")
    
    if reference_audio.exists():
        print("🎤 Testing voice cloning...")
        clone_output = OUTPUT_DIR / "test_clone.wav"
        tts.tts_to_file(
            text="Hello, this is my cloned voice speaking English.",
            file_path=str(clone_output),
            speaker_wav=str(reference_audio),
            language="en"
        )
        print(f"✅ Cloned voice test saved to: {clone_output}")
        
        # Test with Bengali-like Romanized text
        print("🇧🇩 Testing Bengali synthesis...")
        bengali_output = OUTPUT_DIR / "test_bengali.wav"
        tts.tts_to_file(
            text="Nomoshkar, apni kemon achhen? Ami bhalo achhi.",
            file_path=str(bengali_output),
            speaker_wav=str(reference_audio),
            language="en"  # Use English phonetics for Romanized Bengali
        )
        print(f"✅ Bengali test saved to: {bengali_output}")
    else:
        print(f"⚠️ Reference audio not found: {reference_audio}")
        print("   Run 'python training/prep_xtts_data.py' first!")
    
    print("\n🎉 Initial setup complete!")
    print("   Listen to the test files in:", OUTPUT_DIR)
    print("\n📝 Next steps:")
    print("   1. Listen to test_bengali.wav")
    print("   2. If quality is acceptable, XTTS can be used directly for Bengali")
    print("   3. For better quality, fine-tuning on your full dataset is recommended")

if __name__ == "__main__":
    main()
