import json
from pathlib import Path
import re

# Comprehensive Romanization Mapping (Bengali to English Pronunciation)
ROMAN_MAP = {
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'n',
    'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
    'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh', 'স': 's', 'হ': 'h',
    'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y', 'ৎ': 't',
    'ং': 'ng', 'ঃ': 'h', 'ঁ': 'n',
    'অ': 'o', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u', 'ঋ': 'ri',
    'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
    'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u', 'ৃ': 'ri',
    'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou', '্': '',
}

def romanize_bengali(text):
    # This is a heuristic-based romanization
    # We'll use some common patterns for better pronunciation
    result = ""
    for char in text:
        if char in ROMAN_MAP:
            result += ROMAN_MAP[char]
        elif char in " \n\t":
            result += " "
        elif char in ",.?!-":
            result += char
        else:
            # Keep other characters (punctuation/numbers)
            if re.match(r'[A-Za-z0-9]', char):
                result += char
    
    # Post-processing for common patterns
    result = result.replace("aa", "a").replace("oo", "o").replace("ii", "i")
    
    # Specific common word refinements
    refinements = {
        "nomoskar": "nomoshkar",
        "apni": "apni",
        "kæmon": "kemon",
        "achen": "achhen",
        "sundor": "shundor",
        "akash": "akash",
        "bangladesh": "bangladesh",
        "rajdhani": "rajdhani",
    }
    
    words = result.split()
    refined_words = []
    for word in words:
        clean = re.sub(r'[,.?!]', '', word).lower()
        punc = word[len(clean):]
        if clean in refinements:
            refined_words.append(refinements[clean] + punc)
        else:
            refined_words.append(word)
            
    return " ".join(refined_words).strip()

def prepare_romanized_json():
    master_json = Path("d:/GitHub/Kokoro_tts_module/data/bengali/master_transcripts.json")
    output_json = Path("d:/GitHub/Kokoro_tts_module/data/bengali/master_transcripts_romanized.json")
    
    if not master_json.exists():
        print(f"❌ Master JSON not found at {master_json}")
        return

    with open(master_json, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    romanized_data = []
    for entry in transcripts:
        roman_text = romanize_bengali(entry["text"])
        romanized_data.append({
            "id": entry["id"],
            "original": entry["text"],
            "romanized": roman_text
        })
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(romanized_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created Romanized master JSON at {output_json}")

if __name__ == "__main__":
    prepare_romanized_json()
