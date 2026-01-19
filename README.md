# Kokoro TTS Module - Bengali Voice Cloning 🇧🇩

This project enables high-quality Bengali text-to-speech with **Authentic Voice Cloning** using Coqui XTTS v2.

## 🚀 Quick Start

### 1. Installation
Clone this repository and install the dependencies:
```bash
# Pull the latest code
git pull origin main

# Install requirements
pip install -r requirements.txt

# Install XTTS v2 specialized version
pip install TTS==0.22.0
pip install transformers==4.43.0
```

### 2. Hardware Requirements
- **OS**: Windows (tested on Windows 10/11)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060) is highly recommended for training.
- **Python**: 3.9 - 3.11

---

## 🛠️ Usage Workflow

The project is organized into three simple entry points:

### 🎙️ Step 1: Record Your Voice
Launch the web-based recording interface to manage your 200 Bengali recordings.
```bash
python run_recorder.py
```
- Open [http://localhost:5000/recorder](http://localhost:5000/recorder) in your browser.
- Record or upload audio for each Bengali sentence.

### 🔥 Step 2: Start Training (Fine-tuning)
Once you have recorded your voice, start the fine-tuning process to adapt the model to your unique accent.
```bash
python run_training.py
```
- **Time**: ~1-2 hours on an RTX 3060.
- **Output**: Checkpoints will be saved in `training/xtts_ft_results/`.

### 🔍 Step 3: Test & Synthesize
Test the quality of your fine-tuned voice with any Bengali text.
```bash
python run_tester.py
```
- Enter Romanized Bengali text (e.g., `Amar nam Akash`).
- The system will generate a `.wav` file in the `output/` directory.

---

## 📂 Project Structure
- `run_recorder.py`: Web UI for audio collection.
- `run_training.py`: Model fine-tuning pipeline.
- `run_tester.py`: Interactive speech synthesis test.
- `data/bengali/`: Your recordings and transcripts.
- `training/`: Training scripts and model checkpoints.

## 📝 Notes
- **Romanization**: For best results, use the English alphabet to write Bengali phonetically (e.g., use `nomoshkar` instead of `নমস্কার`).
- **Data Quality**: Ensure your recordings are clear and free of background noise for the best cloning result.
