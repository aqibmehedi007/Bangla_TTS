"""
QuteVoice - Advanced Kokoro TTS Web Interface
Automated setup with dependency installation and model downloading
"""

import os
import sys
import tempfile
import asyncio
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import time
import json
import re
from functools import lru_cache

VOICE_LANGUAGE_LABELS = {
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

GENDER_MAP = {
    "f": "female",
    "m": "male",
}

VOICE_FILE_CANDIDATES = [
    Path("kokoro_voices.md"),
    Path("docs/kokoro_voices.md"),
]


def _clean_cell(value: str) -> str:
    value = value.replace("**", "").strip()
    value = re.sub(r"[`]", "", value)
    return value


def _parse_voice_markdown(content: str):
    voices = []
    current_language = None
    columns = []

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line:
            continue
        if line.startswith("### "):
            current_language = line[4:].strip()
            columns = []
            continue
        if not line.startswith("|"):
            continue

        cells = [c.strip() for c in line.strip(" |\n").split("|")]

        # Header row
        if not columns:
            columns = cells
            continue

        # Separator row (----)
        if all(set(cell) <= {"-", " "} for cell in cells):
            continue

        record = dict(zip(columns, cells))
        name = _clean_cell(record.get("Name", ""))
        if not name:
            continue

        lang_code = name[0].lower() if name else "a"
        gender_code = name[1].lower() if len(name) > 1 else ""

        voice_entry = {
            "id": name,
            "language_code": lang_code,
            "language_label": VOICE_LANGUAGE_LABELS.get(lang_code, current_language or "Unknown"),
            "gender": GENDER_MAP.get(gender_code, "unknown"),
            "category": current_language or VOICE_LANGUAGE_LABELS.get(lang_code, "Unknown"),
            "traits": _clean_cell(record.get("Traits", "")),
            "target_quality": _clean_cell(record.get("Target Quality", "")),
            "training_duration": _clean_cell(record.get("Training Duration", "")),
            "overall_grade": _clean_cell(record.get("Overall Grade", "")),
            "sha": _clean_cell(record.get("SHA256", "")),
        }
        voices.append(voice_entry)

    return voices


def _fallback_voice_list():
    fallback = []
    voice_list_path = Path("voices_list.txt")
    if voice_list_path.exists():
        for line in voice_list_path.read_text(encoding="utf-8").splitlines():
            voice = line.split("/")[-1].replace(".pt", "").strip()
            if not voice:
                continue
            lang_code = voice[0].lower()
            gender_code = voice[1].lower() if len(voice) > 1 else ""
            fallback.append({
                "id": voice,
                "language_code": lang_code,
                "language_label": VOICE_LANGUAGE_LABELS.get(lang_code, "Unknown"),
                "gender": GENDER_MAP.get(gender_code, "unknown"),
                "category": VOICE_LANGUAGE_LABELS.get(lang_code, "Unknown"),
                "traits": "",
                "target_quality": "",
                "training_duration": "",
                "overall_grade": "",
                "sha": "",
            })
    return fallback


@lru_cache(maxsize=1)
def get_voice_catalog():
    for candidate in VOICE_FILE_CANDIDATES:
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8")
                voices = _parse_voice_markdown(content)
                if voices:
                    break
            except Exception as exc:
                print(f"⚠️  Failed to parse {candidate}: {exc}")
                voices = []
                continue
    else:
        voices = _fallback_voice_list()

    if not voices:
        voices = [{
            "id": "af_heart",
            "language_code": "a",
            "language_label": VOICE_LANGUAGE_LABELS.get("a"),
            "gender": "female",
            "category": "American English",
            "traits": "",
            "target_quality": "",
            "training_duration": "",
            "overall_grade": "A",
            "sha": "",
        }]

    language_map = {}
    for voice in voices:
        code = voice.get("language_code", "a")
        label = voice.get("language_label") or VOICE_LANGUAGE_LABELS.get(code)
        if code not in language_map:
            language_map[code] = label

    language_options = [
        {"code": code, "label": label}
        for code, label in sorted(language_map.items(), key=lambda item: item[1] or "")
    ]

    return {
        "voices": voices,
        "languages": language_options,
    }


def run_async(coroutine):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# Check and install requirements automatically
def check_and_install_requirements():
    """Check if requirements are installed and install them if needed"""
    print("🔍 Checking Python dependencies...")
    
    try:
        # Try importing key packages
        import torch
        import flask
        import soundfile
        import numpy
        print("✅ All required packages are already installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Installing requirements automatically...")
        
        try:
            # Install requirements
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            print("💡 Please run: pip install -r requirements.txt")
            return False

def check_gpu_availability():
    """Check if GPU/CUDA is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("🚀 GPU detected - CUDA acceleration will be enabled")
            return True
        else:
            print("💻 No GPU detected - will run on CPU (slower)")
            return False
    except:
        print("💻 Could not detect GPU - will run on CPU (slower)")
        return False

def check_and_install_llama_cpp():
    """Check if llama-cpp-python is installed with CUDA support, install if missing."""
    try:
        import llama_cpp
        print("✅ llama-cpp-python found")
        
        # Test if CUDA support is available
        print("🔧 Testing CUDA support...")
        try:
            # Set CUDA environment variables
            os.environ['GGML_CUDA_FORCE_CUBLAS'] = '1'
            
            # Try to create a simple model to test CUDA
            test_model = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=1,  # Test with 1 layer
                n_ctx=512,       # Small context
                verbose=False    # Disable verbose output
            )
            
            # Test a simple generation to verify GPU is working
            test_output = test_model("Test", max_tokens=5, stop=["\n"])
            print("✅ CUDA support verified - GPU offload working!")
            return True
        except Exception as e:
            print(f"⚠️ CUDA test failed: {e}")
            print("🔧 Installing CUDA-enabled version...")
            return install_cuda_llama_cpp()
            
    except ImportError:
        print("🔧 llama-cpp-python not found. Installing with CUDA support...")
        return install_cuda_llama_cpp()

def install_cuda_llama_cpp():
    """Install the CUDA-enabled version of llama-cpp-python using JamePeng wheel."""
    print("🔧 Installing llama-cpp-python with CUDA support using JamePeng wheel...")
    
    # Create pre-build-wheel directory
    wheel_dir = Path("pre-build-wheel")
    wheel_dir.mkdir(exist_ok=True)
    
    # JamePeng wheel URL for Python 3.11 with CUDA 12.8 support
    wheel_url = "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.16-cu128-AVX2-win-20250831/llama_cpp_python-0.3.16-cp311-cp311-win_amd64.whl"
    wheel_file = wheel_dir / "llama_cpp_python-0.3.16-cp311-cp311-win_amd64.whl"
    
    try:
        # First uninstall any existing version
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "llama-cpp-python", "-y"], 
                      capture_output=True, text=True)
        
        # Download wheel if not exists
        if not wheel_file.exists():
            print("📥 Downloading JamePeng CUDA wheel...")
            print(f"📡 Downloading from: {wheel_url}")
            print(f"💾 Saving to: {wheel_file}")
            
            download_file(wheel_url, wheel_file)
            print("✅ Wheel downloaded successfully!")
        else:
            print(f"✅ Wheel already exists: {wheel_file}")
        
        # Install the wheel
        print("🔧 Installing JamePeng CUDA wheel...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            str(wheel_file)
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ JamePeng CUDA llama-cpp-python installed successfully!")
            print("🚀 This wheel provides full CUDA 12.8 support for optimal GPU performance")
            return True
        else:
            print(f"❌ Failed to install JamePeng wheel: {result.stderr}")
            
            # Fallback to standard CUDA installation
            print("🔄 Falling back to standard CUDA installation...")
            result2 = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python==0.3.16", 
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu128"
            ], capture_output=True, text=True, timeout=300)
            
            if result2.returncode == 0:
                print("✅ Standard CUDA llama-cpp-python installed")
                return True
            else:
                print(f"❌ All installation methods failed: {result2.stderr}")
                return False
                
    except subprocess.TimeoutExpired:
        print("❌ Installation timed out. Please install manually:")
        print(f"💡 pip install {wheel_file}")
        return False
    except Exception as e:
        print(f"❌ Failed to install llama-cpp-python: {e}")
        return False

def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    import requests
    from tqdm import tqdm
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filepath.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def download_kokoro_model():
    """Download the Kokoro model file if it doesn't exist"""
    model_path = "./models/Kokoro_espeak_Q4.gguf"
    models_dir = Path("./models")
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"✅ Model already exists: {model_path}")
        return True
    
    print("📥 Model not found. Downloading Kokoro TTS model...")
    print("⚠️  This may take several minutes depending on your internet connection...")
    
    # Model download URL from the correct Hugging Face repository (Q4 model)
    model_urls = [
        "https://huggingface.co/mmwillet2/Kokoro_GGUF/resolve/main/Kokoro_espeak_Q4.gguf"
    ]
    
    for i, url in enumerate(model_urls, 1):
        try:
            print(f"🔄 Attempting download from source {i}/{len(model_urls)}...")
            print(f"📡 URL: {url}")
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded * 100) / total_size)
                    print(f"\r📥 Downloading: {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
            
            urllib.request.urlretrieve(url, model_path, show_progress)
            print(f"\n✅ Model downloaded successfully to: {model_path}")
            
            # Verify file size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size > 100:  # Model should be at least 100MB
                print(f"📁 Model size: {file_size:.1f} MB")
                return True
            else:
                print(f"⚠️  Downloaded file seems too small ({file_size:.1f} MB), trying next source...")
                os.remove(model_path)
                
        except Exception as e:
            print(f"\n❌ Download failed from source {i}: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            continue
    
    print("❌ Failed to download model from all sources.")
    print("💡 Please manually download the Kokoro Q4 model and place it in the models/ folder")
    print("🔗 You can find the model at: https://huggingface.co/mmwillet2/Kokoro_GGUF/tree/main")
    print("📋 Required model: Kokoro_espeak_Q4.gguf (178 MB)")
    return False

def setup_environment():
    """Complete environment setup"""
    print("🚀 Setting up Kokoro TTS Module Environment...")
    print("=" * 60)
    
    # Step 1: Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Step 2: Check and install requirements
    if not check_and_install_requirements():
        return False
    
    # Step 3: Check and install llama-cpp-python for GPU support
    if gpu_available:
        if not check_and_install_llama_cpp():
            print("⚠️  GPU support installation failed - continuing with CPU mode")
    
    # Step 4: Download model if needed
    if not download_kokoro_model():
        print("⚠️  Continuing without model - some features may not work")
    
    print("=" * 60)
    print("✅ Environment setup complete!")
    if gpu_available:
        print("🚀 GPU acceleration enabled - optimal performance expected")
    else:
        print("💻 CPU mode - functional but slower performance")
    return True

# Import after setup
def import_required_modules():
    """Import required modules after setup"""
    try:
        import torch
        from flask import Flask, render_template, request, jsonify, send_file
        from kokoro_tts_service import KokoroTTSService
        return True
    except ImportError as e:
        print(f"❌ Critical imports failed: {e}")
        print("💡 Please check your Python environment.")
        return False

# Global variables
app = None
tts_service = None
model_path = "./models/Kokoro_espeak_Q4.gguf"

def load_model():
    """Load the Kokoro TTS model"""
    global tts_service
    try:
        print(f"Loading TTS model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"⚠️  Model file not found: {model_path}")
            print("💡 Creating demo TTS service...")
            tts_service = DemoTTSService()
            return True
        
        # Try to load the actual Kokoro service
        try:
            from kokoro_tts_service import KokoroTTSService
            print("✅ KokoroTTSService imported successfully")
            tts_service = KokoroTTSService(model_path)
            
            # Test if we can initialize
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tts_service.initialize())
            loop.close()
            
            if result:
                print("✅ TTS model loaded successfully!")
                return True
            else:
                print("❌ Failed to load TTS model")
                print("💡 Falling back to demo mode...")
                tts_service = DemoTTSService()
                return True
                
        except ImportError as e:
            print(f"⚠️  Kokoro library not available: {e}")
            print("💡 Using demo TTS service instead...")
            tts_service = DemoTTSService()
            return True
        except NameError as e:
            print(f"⚠️  NameError: {e}")
            print("💡 Using demo TTS service instead...")
            tts_service = DemoTTSService()
            return True
            
    except Exception as e:
        print(f"❌ Error loading TTS model: {e}")
        print("💡 Using demo TTS service instead...")
        tts_service = DemoTTSService()
        return True

class DemoTTSService:
    """Demo TTS service for when Kokoro is not available"""
    
    def __init__(self):
        self.device = "cpu"
        self.initialized = True
        self.available_voices = ["demo_voice"]
        self.default_voice = "demo_voice"
        self.sample_rate = 24000
        print("🎭 Demo TTS service initialized")
    
    async def initialize(self):
        return True
    
    def is_available(self):
        return True
    
    async def synthesize_speech(self, text, voice=None, speed=1.0, language=None):
        """Generate demo audio"""
        import numpy as np
        import soundfile as sf
        import io
        speed = max(0.5, min(2.0, float(speed or 1.0)))
        duration = max(1.0, len(text) * 0.08 / speed)
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        frequency = 440 + (hash(f"{voice}:{text}") % 200)
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        envelope = np.exp(-t * 2 * speed) * (1 - np.exp(-t * 10))
        audio = audio * envelope
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, self.sample_rate, format='WAV')
        return audio_bytes.getvalue()
    
    def get_model_info(self):
        return {
            "model_name": "Demo TTS",
            "model_file": "demo_mode",
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "initialized": self.initialized,
            "available_voices": len(self.available_voices)
        }

def create_flask_app():
    """Create Flask application with routes"""
    from flask import Flask, render_template, request, jsonify, send_file
    import torch
    
    global app
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Main page"""
        catalog = get_voice_catalog()
        model_info = {}
        if tts_service:
            try:
                model_info = tts_service.get_model_info()
            except Exception:
                model_info = {}
        return render_template(
            'app.html',
            voices=catalog['voices'],
            languages=catalog['languages'],
            model_info=model_info,
        )

    @app.route('/api/voices')
    def api_voices():
        catalog = get_voice_catalog()
        return jsonify(catalog)

    @app.route('/generate_speech', methods=['POST'])
    def generate_speech():
        """Generate speech from text"""
        try:
            data = request.get_json() or {}
            text = data.get('text', '')
            voice = data.get('voice') or None
            language = data.get('language') or None
            speed = data.get('speed', 1.0)

            if not text or not text.strip():
                return jsonify({'error': 'No text provided'}), 400

            try:
                speed = max(0.5, min(2.0, float(speed)))
            except (TypeError, ValueError):
                speed = 1.0

            if tts_service is None or not tts_service.is_available():
                return jsonify({'error': 'Model not loaded'}), 500

            try:
                audio_data = run_async(tts_service.synthesize_speech(
                    text.strip(), voice=voice, speed=speed, language=language
                ))
            except Exception as exc:
                return jsonify({'error': str(exc)}), 500

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(audio_data)
            temp_file.close()

            return jsonify({
                'success': True,
                'generated_text': f"Audio generated for: {text[:50]}{'...' if len(text) > 50 else ''}",
                'message': f'Speech generation completed! Audio size: {len(audio_data)} bytes',
                'audio_ready': True,
                'text_length': len(text),
                'audio_path': os.path.basename(temp_file.name),
                'audio_size': len(audio_data),
                'voice': voice,
                'language': language,
                'speed': speed,
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/batch_generate', methods=['POST'])
    def batch_generate():
        data = request.get_json() or {}
        items = data.get('items', [])
        if not isinstance(items, list) or not items:
            return jsonify({'error': 'No batch items provided'}), 400

        results = []
        for idx, item in enumerate(items):
            text = (item.get('text') or '').strip()
            voice = item.get('voice') or None
            language = item.get('language') or None
            speed = item.get('speed', 1.0)

            try:
                speed = float(speed)
            except (TypeError, ValueError):
                speed = 1.0

            if not text:
                results.append({'index': idx, 'success': False, 'error': 'Text is empty'})
                continue

            try:
                audio_data = run_async(tts_service.synthesize_speech(
                    text, voice=voice, speed=speed, language=language
                ))
            except Exception as exc:
                results.append({'index': idx, 'success': False, 'error': str(exc)})
                continue

            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.write(audio_data)
            temp_file.close()

            results.append({
                'index': idx,
                'success': True,
                'audio_path': os.path.basename(temp_file.name),
                'audio_size': len(audio_data),
                'voice': voice,
                'language': language,
                'speed': speed,
            })

        return jsonify({'success': True, 'results': results})

    @app.route('/model_status')
    def model_status():
        """Check if model is loaded"""
        try:
            loaded = tts_service is not None and tts_service.is_available()
            device_info = 'unknown'
            model_name = 'Unknown'
            available_voices = len(get_voice_catalog().get('voices', []))

            if tts_service:
                try:
                    device_info = getattr(tts_service, 'device', 'cpu')
                    model_info = tts_service.get_model_info()
                    model_name = model_info.get('model_name', 'Unknown')
                except:
                    device_info = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model_name = 'Demo TTS' if isinstance(tts_service, DemoTTSService) else 'Kokoro TTS'

            return jsonify({
                'loaded': loaded,
                'model_path': model_path,
                'model_exists': os.path.exists(model_path),
                'device': device_info,
                'model_name': model_name,
                'sample_rate': 24000,
                'available_voices': available_voices,
                'is_demo_mode': isinstance(tts_service, DemoTTSService) if tts_service else False
            })
        except Exception as e:
            return jsonify({
                'loaded': False,
                'error': str(e),
                'model_path': model_path,
                'model_exists': os.path.exists(model_path),
                'is_demo_mode': False
            })

    @app.route('/download_audio/<filename>')
    def download_audio(filename):
        """Download generated audio file"""
        try:
            # Security check - only allow files from temp directory
            if not filename.endswith('.wav'):
                return jsonify({'error': 'Invalid file type'}), 400
                
            audio_path = os.path.join(tempfile.gettempdir(), filename)
            if os.path.exists(audio_path):
                return send_file(audio_path, as_attachment=True)
            else:
                return jsonify({'error': 'Audio file not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎤 Kokoro TTS Module Test - Advanced Application")
    print("🚀 Automated Setup & Launch")
    print("=" * 60)
    
    # Run complete environment setup
    if not setup_environment():
        print("❌ Environment setup failed!")
        print("💡 Please check your Python installation and internet connection")
        sys.exit(1)
    
    # Import required modules after setup
    if not import_required_modules():
        print("❌ Failed to import required modules!")
        sys.exit(1)
    
    print("\n🎯 Starting TTS Application...")
    print("=" * 60)
    
    # Check if model exists after setup
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
        print(f"📁 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("💡 The application will run in demo mode")
    
    print("\n🔄 Loading TTS model...")
    
    # Load the model on startup
    if load_model():
        print("\n🌐 Creating web application...")
        create_flask_app()
        
        print("🌐 Starting web server...")
        print("📍 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        try:
            app.run(debug=True, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n👋 Server stopped. Goodbye!")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
            print("💡 Please check the logs above for more details")
    else:
        print("\n⚠️  Model failed to load - running in limited mode")
        print("🌐 Creating web application...")
        create_flask_app()
        
        print("🌐 Starting web server anyway...")
        print("📍 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        try:
            app.run(debug=True, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n👋 Server stopped. Goodbye!")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
            print("💡 Please check the logs above for more details")
