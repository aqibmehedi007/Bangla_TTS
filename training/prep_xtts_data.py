"""
Prepare Bengali dataset for XTTS v2 fine-tuning.
Converts master_transcripts.json to the format expected by Coqui TTS.
"""
import os
import json
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path("d:/GitHub/Kokoro_tts_module")
DATASET_DIR = PROJECT_ROOT / "data" / "bengali" / "dataset"
TRANSCRIPTS_FILE = PROJECT_ROOT / "data" / "bengali" / "master_transcripts.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "bengali" / "xtts"
WAVS_DIR = OUTPUT_DIR / "wavs"

def main():
    print("📦 Preparing Bengali dataset for XTTS v2...")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load transcripts
    with open(TRANSCRIPTS_FILE, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)
    
    metadata_lines = []
    valid_count = 0
    
    print("🎵 Converting audio to standard WAV (22050Hz mono)...")
    for item in tqdm(transcripts):
        folder_id = item.get("folder_id", item.get("id", ""))
        text = item.get("romanized", item.get("text", ""))
        
        if not folder_id or not text:
            continue
            
        audio_src = DATASET_DIR / folder_id / "audio.wav"
        audio_dst = WAVS_DIR / f"{folder_id}.wav"
        
        if audio_src.exists():
            try:
                # Load with librosa (handles WebM/Opus headers)
                # Resample to 22050Hz as per XTTS standard
                y, sr = librosa.load(str(audio_src), sr=22050, mono=True)
                
                # Save as standard WAV
                sf.write(str(audio_dst), y, 22050)
                
                # Metadata expects: audio_file|text|speaker_name
                metadata_lines.append(f"{audio_dst.absolute()}|{text}|user")
                valid_count += 1
            except Exception as e:
                print(f"❌ Failed to convert {audio_src}: {e}")
        else:
            print(f"⚠️ Missing audio: {audio_src}")
    
    # Shuffle and Split (190 train, 10 val)
    import random
    random.seed(42)
    random.shuffle(metadata_lines)
    
    train_lines = metadata_lines[:190]
    val_lines = metadata_lines[190:]
    
    # Write metadata files
    train_path = OUTPUT_DIR / "metadata_train.csv"
    val_path = OUTPUT_DIR / "metadata_val.csv"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    print(f"\n✅ Created {train_path} ({len(train_lines)} samples)")
    print(f"✅ Created {val_path} ({len(val_lines)} samples)")
    
    # Create reference_voice.wav from the first valid sample
    if metadata_lines:
        import shutil
        first_audio_abs = metadata_lines[0].split('|')[0]
        ref_audio_path = OUTPUT_DIR / "reference_voice.wav"
        shutil.copy(first_audio_abs, ref_audio_path)
        print(f"🎤 Reference voice saved to: {ref_audio_path}")
    
    print("\n🎉 Dataset preparation and conversion complete!")

if __name__ == "__main__":
    main()
