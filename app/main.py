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
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
    "bn": "Bengali",
}

GENDER_MAP = {
    "f": "female",
    "m": "male",
}

VOICE_FILE_CANDIDATES = [
    Path("core/kokoro_voices.md"),
    Path("kokoro_voices.md"),
    Path("docs/kokoro_voices.md"),
]


def _clean_cell(value: str) -> str:
    value = value.replace("**", "").replace("\\", "").strip()
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
            if "lang_code" in line and "=" in line:
                match = re.search(r"lang_code=['\"]([^'\"]+)['\"]", line)
                if match:
                    # Logic to potentially override lang_code if needed
                    pass
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

        if name.startswith("bn"):
            lang_code = "bn"
            gender_code = name[2].lower() if len(name) > 2 else ""
        else:
            lang_code = name[0].lower() if name else "a"
            gender_code = name[1].lower() if len(name) > 1 else ""

        voice_entry = {
            "id": name,
            "language_code": lang_code,
            "language_label": VOICE_LANGUAGE_LABELS.get(lang_code, current_language or "Unknown"),
            "gender": GENDER_MAP.get(gender_code, "unknown"),
            "display_name": name.split('_', 1)[1].replace('_', ' ').title() if '_' in name else name.title(),
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
    voice_list_path = Path("core/voices_list.txt")
    if voice_list_path.exists():
        for line in voice_list_path.read_text(encoding="utf-8").splitlines():
            voice = line.split("/")[-1].replace(".pt", "").strip()
            if not voice:
                continue
            if voice.startswith("bn"):
                lang_code = "bn"
                gender_code = voice[2].lower() if len(voice) > 2 else ""
            else:
                lang_code = voice[0].lower()
                gender_code = voice[1].lower() if len(voice) > 1 else ""
            fallback.append({
                "id": voice,
                "language_code": lang_code,
                "language_label": VOICE_LANGUAGE_LABELS.get(lang_code, "Unknown"),
                "gender": GENDER_MAP.get(gender_code, "unknown"),
                "display_name": voice.split('_', 1)[1].replace('_', ' ').title() if '_' in voice else voice.title(),
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
                    print(f"✅ Loaded {len(voices)} voices from {candidate}")
                    bn_voices = [v for v in voices if v.get("language_code") == "bn"]
                    if bn_voices:
                        print(f" debug: Found {len(bn_voices)} Bengali voices: {[v['id'] for v in bn_voices]}")
                    else:
                        print(" debug: No Bengali voices found in this file.")
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
            "display_name": "Heart",
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
        import kokoro
        import soundfile
        import numpy
        print("✅ Core packages (including kokoro) are installed!")
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
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("💻 No CUDA GPU detected - running on CPU")
            return False
    except:
        print("💻 Could not detect GPU status - defaulting to CPU")
        return False

def setup_environment():
    """Complete environment setup for official Kokoro library"""
    print("🚀 Setting up Kokoro TTS Module Environment...")
    print("=" * 60)
    
    # Step 1: Check and install requirements
    if not check_and_install_requirements():
        return False
    
    # Step 2: Check GPU availability (just for logging/info)
    gpu_available = check_gpu_availability()
    
    print("=" * 60)
    print("✅ Environment setup complete!")
    if gpu_available:
        print("🚀 GPU acceleration will be used by Kokoro library")
    else:
        print("💻 CPU mode - functional but slower performance")
    return True

# Import after setup
def import_required_modules():
    """Import required modules after setup"""
    try:
        import torch
        from flask import Flask, render_template, request, jsonify, send_file, redirect
        from core.service import KokoroTTSService
        return True
    except ImportError as e:
        print(f"❌ Critical imports failed: {e}")
        print("💡 Please check your Python environment.")
        return False

# Global variables
app = None
tts_service = None

def load_model():
    """Load the Kokoro TTS model using the service"""
    global tts_service
    try:
        print("🔄 Loading Kokoro TTS service...")
        
        # Try to load the actual Kokoro service
        try:
            from core.service import KokoroTTSService
            tts_service = KokoroTTSService()
            
            # Test if we can initialize
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tts_service.initialize())
            loop.close()
            
            if result:
                print("✅ TTS service loaded successfully!")
                return True
            else:
                print("❌ Failed to initialize TTS service")
                print("💡 Falling back to demo mode...")
                from core.service import DemoTTSService
                tts_service = DemoTTSService()
                return True
                
        except (ImportError, NameError) as e:
            print(f"⚠️  Kokoro library not fully available: {e}")
            print("💡 Using demo TTS service instead...")
            from core.service import DemoTTSService
            tts_service = DemoTTSService()
            return True
            
    except Exception as e:
        print(f"❌ Error loading TTS model: {e}")
        from core.service import DemoTTSService
        tts_service = DemoTTSService()
        return True

# DemoTTSService is now imported from kokoro_tts_service

def create_flask_app():
    """Create Flask application with routes"""
    from flask import Flask, render_template, request, jsonify, send_file, redirect
    import torch
    
    global app
    app = Flask(__name__, template_folder='templates', static_folder='static')
    
    @app.route('/')
    def index():
        """Redirect to recorder for the current training phase"""
        return redirect('/recorder')

    @app.route('/studio')
    def studio():
        """Main TTS Studio page (Aurora Voice Studio)"""
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
            from core.service import DemoTTSService
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

    @app.route('/recorder')
    def recorder():
        """Recorder page"""
        return render_template('recorder.html')

    @app.route('/api/recorder/status')
    def recorder_status():
        """Get status of all folders in the dataset"""
        dataset_path = Path("data/bengali/dataset")
        if not dataset_path.exists():
            return jsonify({'error': 'Dataset directory not found', 'folders': []})
            
        folders = []
        # Sort folders numerically
        folder_paths = sorted([d for d in dataset_path.iterdir() if d.is_dir()], key=lambda x: x.name)
        
        for folder in folder_paths:
            has_audio = (folder / "audio.wav").exists()
            folders.append({
                'id': folder.name,
                'recorded': has_audio
            })
            
        return jsonify({'folders': folders})

    @app.route('/api/recorder/transcript/<folder_id>')
    def recorder_transcript(folder_id):
        """Get transcript for a specific folder"""
        transcript_path = Path(f"data/bengali/dataset/{folder_id}/transcript.txt")
        if transcript_path.exists():
            return jsonify({
                'id': folder_id,
                'transcript': transcript_path.read_text(encoding="utf-8").strip()
            })
        return jsonify({'error': 'Transcript not found'}), 404

    @app.route('/api/recorder/upload/<folder_id>', methods=['POST'])
    def recorder_upload(folder_id):
        """Upload recorded audio for a folder"""
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        folder_path = Path(f"data/bengali/dataset/{folder_id}")
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            
        save_path = folder_path / "audio.wav"
        audio_file.save(str(save_path))
        
        return jsonify({'success': True, 'message': f'Audio saved for {folder_id}'})

    @app.route('/api/recorder/audio/<folder_id>')
    def recorder_audio(folder_id):
        """Serve the recorded audio for a folder"""
        audio_path = Path(f"data/bengali/dataset/{folder_id}/audio.wav")
        if audio_path.exists():
            return send_file(str(audio_path))
        return jsonify({'error': 'Audio not found'}), 404

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
    
    print("\n🔄 Loading TTS model...")
    
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
