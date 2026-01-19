"""
Entry point for the Bengali Voice Recording Interface.
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Launching Recording Interface...")
    print("📍 URL: http://localhost:5000/recorder")
    
    app_path = Path("app/main.py")
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found.")
        return

    try:
        subprocess.run([sys.executable, str(app_path)], check=True)
    except KeyboardInterrupt:
        print("\n👋 Recorder stopped.")

if __name__ == "__main__":
    main()
