import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from core.service import KokoroTTSService, bengali_to_roman

async def test_bengali_synthesis():
    print("🧪 Starting Terminal-based Bengali Synthesis Test")
    print("=" * 50)
    
    # Initialize service
    service = KokoroTTSService()
    success = await service.initialize()
    if not success:
        print("❌ Failed to initialize service")
        return

    # Test sentence
    test_text = "tumi kemon acho?"
    
    # Voice selection
    voice = "bnm_custom"
    
    print(f"🚀 Generating speech for: '{test_text}' using voice '{voice}'")
    
    try:
        audio_data = await service.synthesize_speech(test_text, voice=voice)
        
        if audio_data and len(audio_data) > 100:
            output_path = "test_bn_output.wav"
            with open(output_path, "wb") as f:
                f.write(audio_data)
            print(f"✅ Success! Audio saved to: {os.path.abspath(output_path)}")
            print(f"📏 File size: {len(audio_data)} bytes")
        else:
            print("❌ Failed to generate meaningful audio data")
            
    except Exception as e:
        print(f"❌ Error during synthesis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bengali_synthesis())
