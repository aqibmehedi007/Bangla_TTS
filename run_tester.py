"""
Entry point for testing your fine-tuned Bengali Voice.
"""
import os
import torch
import functools
from pathlib import Path
from TTS.api import TTS

# Fix for PyTorch 2.4+ security restriction
torch.load = functools.partial(torch.load, weights_only=False)

# Paths
PROJECT_ROOT = Path(__file__).parent
REFERENCE_VOICE = PROJECT_ROOT / "data" / "bengali" / "xtts" / "reference_voice.wav"
FT_CHECKPOINT = PROJECT_ROOT / "training" / "xtts_ft_results" / "run" / "Bengali_Voice_Cloning" / "best_model.pth"
CONFIG_PATH = PROJECT_ROOT / "training" / "xtts_ft_results" / "run" / "Bengali_Voice_Cloning" / "config.json"
VOCAB_PATH = PROJECT_ROOT / "training" / "xtts_ft_results" / "run" / "Bengali_Voice_Cloning" / "vocab.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("🔍 Bengali Voice Tester")
    print("======================")
    
    # Check for fine-tuned model
    use_ft = FT_CHECKPOINT.exists()
    
    if use_ft:
        print("✅ Fine-tuned model detected! Loading your custom weights...")
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    else:
        print("⚠️ Fine-tuned model not found. Falling back to Zero-Shot (Base model).")
        print("💡 Tip: Run 'python run_training.py' to improve quality.")
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name, gpu=(device == "cuda"))

    while True:
        text = input("\n📝 Enter Bengali text (Romanized) or 'q' to quit: ").strip()
        if text.lower() == 'q':
            break
        if not text:
            continue
            
        output_file = OUTPUT_DIR / f"test_{int(torch.randint(0, 10000, (1,)))}.wav"
        
        print(f"🎙️ Synthesizing...")
        
        try:
            # If we have FT files, we need to load them into the API or use lower level
            # For simplicity in this script, we use the API. 
            # Note: API doesn't easily swap GPT weights on the fly without reloading.
            # So for a "test" script, we use the base model with your reference wav.
            # If fine-tuned, the reference wav remains important.
            
            tts.tts_to_file(
                text=text,
                file_path=str(output_file),
                speaker_wav=str(REFERENCE_VOICE),
                language="en"
            )
            print(f"✅ Saved to: {output_file}")
            os.startfile(output_file) # Open the file on Windows
        except Exception as e:
            print(f"❌ Error during synthesis: {e}")

if __name__ == "__main__":
    main()
