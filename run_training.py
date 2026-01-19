"""
Entry point for XTTS v2 Fine-tuning on your Bengali Voice.
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("🔥 Starting XTTS v2 Fine-tuning...")
    print("📊 This will use your 200 Bengali recordings to train the model.")
    
    # 1. Ensure metadata is fresh
    prep_path = Path("training/prep_xtts_data.py")
    train_path = Path("training/train_xtts_bn_finetune.py")
    
    if not prep_path.exists() or not train_path.exists():
        print("❌ Error: Training scripts not found in training/ directory.")
        return

    print("📦 Step 1: Preparing data split...")
    subprocess.run([sys.executable, str(prep_path)], check=True)
    
    print("🚀 Step 2: Starting training loop (Expected time: 1-2 hours)...")
    try:
        subprocess.run([sys.executable, str(train_path)], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")

if __name__ == "__main__":
    main()
