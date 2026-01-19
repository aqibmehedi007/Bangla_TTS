# Bangla TTS - Voice Cloning (XTTS v2) 🇧🇩

This project enables high-quality Bengali text-to-speech with **Authentic Voice Cloning** using Coqui XTTS v2. It allows you to clone your voice with as little as 6 seconds of audio and fine-tune it for a natural Bengali accent using a dataset of 200 recordings.

## 💻 Current System Configuration
The project is optimized and tested on this configuration:
- **CPU**: AMD Ryzen 5 7500F 6-Core Processor
- **RAM**: 16 GB
- **GPU**: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **OS**: Windows (PowerShell/CMD)

---

## 🚀 Setup & Installation

### 1. Repository Setup
```bash
# Clone the repository
git clone git@github.com:aqibmehedi007/Bangla_TTS.git
cd Bangla_TTS

# Install dependencies
pip install -r requirements.txt

# Install XTTS v2 specialized version
pip install TTS==0.22.0
pip install transformers==4.43.0
```

### 2. Environment Dependencies
- Python 3.9 - 3.11
- CUDA 11.8 or 12.1 (for GPU acceleration)
- FFmpeg (required for audio processing)

---

## 🛠️ Workflow Guides

The system is designed with a 3-step modular workflow:

### 🎙️ Step 1: Voice Recording
Collect or refine your Bengali audio recordings.
```bash
python run_recorder.py
```
- Open [http://localhost:5000/recorder](http://localhost:5000/recorder) 
- Review existing samples or record new ones.

### 🔥 Step 2: Model Training
Fine-tune the XTTS v2 engine on your specific voice identity.
```bash
python run_training.py
```
- This script first runs `prep_xtts_data.py` to organize your WAV files and transcripts.
- Then it kicks off a 10-20 epoch fine-tuning session.
- **Estimated time**: 1-2 hours on RTX 3060.

### ⚡ Step 3: Synthesis & Testing
Generate high-quality Bengali speech from text.
```bash
# Interactive Mode
python run_tester.py

# Direct CLI Mode
python synthesize.py "Ami bangla bolte pari"
```

---

## 📂 Project Structure
- `run_recorder.py`: Launcher for the Recording Web Interface.
- `run_training.py`: One-click training pipeline.
- `run_tester.py`: Interactive testing menu.
- `synthesize.py`: Single-shot CLI synthesis tool.
- `data/bengali/`: Contains `wavs/` and Romanized transcripts.
- `training/`: Contains fine-tuning logic and local checkpoints.

## 📝 Usage Tips
- **Romanized Text**: The model performs best with Romanized Bengali (e.g., `Nomoshkar`) as it maps more cleanly to the English backbone of XTTS v2.
- **Voice Clarity**: Ensure your recordings in `data/bengali/` are clear for higher quality cloning.
- **Audio Output**: Generated files are saved as `output.wav` or in the `output/` folder.
