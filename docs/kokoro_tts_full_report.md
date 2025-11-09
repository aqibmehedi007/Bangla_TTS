# Kokoro TTS Research Dossier

## 1. Executive Summary
Kokoro TTS is an Apache-licensed, 82M-parameter text-to-speech model derived from StyleTTS 2 and ISTFTNet components. The system delivers near state-of-the-art quality while remaining lightweight enough for commodity hardware, supports eight language families via language-aware pipelines, and ships with a catalogue of 54 style embeddings ("voices") covering male and female speakers across American/British English, Japanese, Mandarin, Spanish, French, Hindi, Italian, and Brazilian Portuguese.[^hf-readme][^voices]

## 2. Model Overview
- **Architecture:** Hybrid StyleTTS 2 encoder/decoder with ISTFTNet vocoder; decoder-only release.[^hf-readme]
- **Model size:** 82 million parameters; default weights `kokoro-v1_0.pth` (`sha256 496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`).[^hf-readme]
- **Sampling rate:** 24 kHz mono output as used in reference pipeline and CLI writer.[^kokoro-main]
- **License:** Apache 2.0 (per model card).[^hf-readme]
- **Training data:** "Few hundred" hours of permissive or synthetic audio with IPA labels; total training cost ≈1,000 A100 GPU hours (~$1,000).[^hf-readme]
- **Deployment note:** Average inference cost when served via commercial APIs is <$1 per million characters (April 2025 market survey).[^hf-readme]

## 3. Language Support & G2P Stack
Kokoro uses language-aware grapheme-to-phoneme (G2P) processing via the `KPipeline` helper. Each language is keyed by a single-letter code, with optional aliases for locale-style identifiers.[^deepwiki]

| Code | Alias Examples | Language | G2P Provider | Extra Dependencies |
|------|----------------|----------|--------------|---------------------|
| `a` | `en-us` | American English | `misaki.en.G2P` + espeak fallback | `pip install misaki[en]`; optional `espeak-ng` | 
| `b` | `en-gb` | British English | `misaki.en.G2P` + British espeak fallback | `pip install misaki[en]`; `espeak-ng` |
| `e` | `es` | Spanish | `espeak.EspeakG2P` | System `espeak-ng` |
| `f` | `fr-fr` | French | `espeak.EspeakG2P` | System `espeak-ng` |
| `h` | `hi` | Hindi | `espeak.EspeakG2P` | System `espeak-ng` |
| `i` | `it` | Italian | `espeak.EspeakG2P` | System `espeak-ng` |
| `p` | `pt-br` | Brazilian Portuguese | `espeak.EspeakG2P` | System `espeak-ng` |
| `j` | `ja` | Japanese | `misaki.ja.JAG2P` | `pip install misaki[ja]` |
| `z` | `zh` | Mandarin Chinese | `misaki.zh.ZHG2P` (with optional English callable) | `pip install misaki[zh]` |

- Pipelines warn when a voice-language mismatch occurs but allow cross-language synthesis (quality may degrade).[^deepwiki][^pipeline]
- English tokenization includes chunking heuristics to keep phoneme sequences within the 510-token model context; other languages implement sentence-aware chunking to ~400 characters per segment.[^pipeline]

## 4. Voice Catalogue
The official `VOICES.md` catalogue enumerates 54 embeddings (style vectors) organized by language + speaker gender (female prefixes `f`, male `m`).[^voices]

| Language | Female Voices | Male Voices | Notes |
|----------|---------------|-------------|-------|
| American English (`af`, `am`) | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky` (11) | `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` (9) | Quality grades range A–F; `af_heart` and `af_bella` earn top marks. |
| British English (`bf`, `bm`) | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily` (4) | `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis` (4) | Mix of MM-minute to HH-hour training durations. |
| Japanese (`jf`, `jm`) | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro` (4) | `jm_kumo` (1) | Includes CC BY audiobook sources (Koniwa texts). |
| Mandarin Chinese (`zf`, `zm`) | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi` (4) | `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang` (4) | All graded C/D; limited-duration dataset. |
| Spanish (`ef`, `em`) | `ef_dora` (1) | `em_alex`, `em_santa` (2) | Espeak G2P pathway. |
| French (`ff`) | `ff_siwis` (1) | — | Only female voice; derived from SIWIS corpus. |
| Hindi (`hf`, `hm`) | `hf_alpha`, `hf_beta` (2) | `hm_omega`, `hm_psi` (2) | All with MM-minute grade systems. |
| Italian (`if`, `im`) | `if_sara` (1) | `im_nicola` (1) | |
| Brazilian Portuguese (`pf`, `pm`) | `pf_dora` (1) | `pm_alex`, `pm_santa` (2) | |

**Voice metadata:** Each entry lists subjective quality tiers (`A`–`F`), training duration buckets (minutes vs hours), and SHA256 prefixes for integrity verification. Voices are downloaded lazily from the Hugging Face repo on first use and cached locally by `KPipeline`.[^voices][^pipeline]

**Voice blending:** Comma-separated identifiers (e.g. `af_bella,af_heart`) are averaged into a composite embedding; results are cached for reuse.[^pipeline]

## 5. Inference Parameters & Controls
- **Voice (`voice`):** Required when synthesizing with a loaded model. Accepts single ID, comma-separated blend, or a custom `.pt` tensor path.[^pipeline]
- **Language (`lang_code`):** Determined by pipeline instance or inferred from voice prefix; CLI allows explicit `-l/--language` selection.[^pipeline][^kokoro-main]
- **Speed (`speed`):** Float multiplier applied to predicted durations; recommended operational range ≈0.7–1.3 to avoid artifacts, though larger deviations are technically accepted. Speed can also be a callable for dynamic scaling.[^pipeline]
- **Split pattern:** Regex controlling text segmentation (`\n+` default) to manage chunk sizes.[^pipeline]
- **Phoneme overrides:** Advanced users can feed phoneme strings directly or override word pronunciations with `[word](phonemes)` syntax.[^pipeline]
- **Temperature/Randomness:** No sampling-temperature parameter is exposed; Kokoro follows a deterministic duration-prediction + vocoder path. Any "temperature" UI control would need to adjust speed or future custom decoding heuristics—not currently part of upstream API.
- **Outputs:** CLI writes little-endian 16-bit PCM WAV at 24,000 Hz.[^kokoro-main]

## 6. Dependencies & Runtime Environment
- **Python package:** `kokoro>=0.9.4` installs the pipeline, model loader, and CLI.[^repo-readme]
- **Core packages:** `torch`, `soundfile`, `numpy`, `loguru`, `huggingface_hub`. Model uses Hugging Face for weight downloads.[^kokoro_model]
- **Language extras:** `misaki[en]`, `misaki[ja]`, `misaki[zh]` as needed; `espeak-ng` binary for espeak-backed languages.[^repo-readme][^deepwiki]
- **Hardware:** Auto-selects CUDA GPU if available, MPS on Apple Silicon when `PYTORCH_ENABLE_MPS_FALLBACK=1`, else CPU. Voice inference runs comfortably on mid-range GPUs; CPU mode is slower but supported.[^pipeline]

## 7. Evaluation & Benchmarks
- Featured in Hugging Face TTS Arena and external leaderboards with competitive rankings (screenshots as of Feb 2025).[^eval]
- Voice grades in `VOICES.md` reflect dataset richness and perceived fidelity—useful for default voice selection.

## 8. UI/UX Planning for Kokoro Module
Goal: deliver a modern web UI that empowers both casual users and power users to synthesize single clips or batch renders while exposing core Kokoro controls.

### 8.1 User Roles & Flows
- **Quick Preview:** Paste text, pick language/voice, tweak speed, preview audio inline, download WAV.
- **Batch Producer:** Upload multiple text items (files or table), apply shared or per-item settings, queue batch render with progress tracking.
- **Voice Explorer:** Filter by language, gender, quality grade; preview voice snippets.

### 8.2 Information Architecture
1. **Navigation Sidebar / Tabs**
   - `Create Single`
   - `Batch Jobs`
   - `Voice Library`
   - `Settings & Resources` (model status, GPU info, docs)
2. **Global Header**
   - Model availability badge (e.g., `Kokoro v1.0 — Loaded on CPU/GPU`)
   - Quick links to documentation and troubleshooting.

### 8.3 Key UI Components
- **Language Selector:** Dropdown grouped by region; displays language code + friendly name + dependency warnings when language is missing required modules.
- **Voice Selector:** Dual-pane control:
  - Left pane lists voices filtered by selected language and gender toggle (Female/Male/All).
  - Right pane shows metadata (quality grade, duration bucket, SHA prefix) pulled from `VOICES.md` cache and offers a "Preview" button (plays stored demo sample).
- **Parameter Controls:**
  - `Speed` slider (0.5–2.0, default 1.0) with tooltip explaining trade-offs.
  - `Temperature` placeholder switch: disabled by default with tooltip explaining the feature is not supported yet; optionally provide link to documentation for advanced experimentation (e.g., adjust speed or post-processing instead).
  - Advanced accordion for `Split Pattern`, custom phoneme input, and voice blending field.
- **Text Workspace:**
  - Markdown-capable editor for single mode with character count, phoneme preview toggle.
  - Batch mode table with columns: `Text`, `Language`, `Voice`, `Speed`, `Status`, `Output`. Supports CSV upload/export.
- **Generation Queue:** Real-time status cards per job (Pending → Processing → Complete / Error). Each card surfaces audio playback, download buttons, and logs.
- **Notifications:** Toasts for missing dependencies, voice-language mismatches (reflecting pipeline warnings).

### 8.4 Accessibility & Responsiveness
- WCAG-compliant color palette, keyboard navigation, aria labels for sliders/buttons.
- Responsive layout: two-column on desktop, collapsible drawers on mobile.
- Provide waveform visualization only when GPU resources suffice (configurable).

### 8.5 Tech Considerations
- Cache voice metadata (names, gender, grades) locally on first load.
- Pre-download sample previews asynchronously to prevent blocking UI.
- Batch processing handled via background queue (Celery/RQ) with API endpoints mirroring existing Flask routes.
- Provide REST/WS events for progress to keep front-end responsive.

## 9. Recommendations & Next Steps
1. **Data Integration:** Ingest `VOICES.md` at runtime to populate voice filters and metadata; mirror in database for offline use.
2. **User Experience:** Prioritize `af_heart` as default (top-grade American English voice) but expose all 54 voices with clear quality indicators.
3. **Error Handling:** Surface pipeline warnings (e.g., espeak missing, voice-language mismatch) directly to the UI.
4. **Extensibility:** Design `Temperature` control as future stub; when upstream introduces stochastic sampling, map to actual parameters.
5. **Documentation:** Embed key usage tips (speed guidance, dependency installs, licensing) inside the app and README using references from this dossier.

---
[^hf-readme]: Hugging Face model card — `hexgrad/Kokoro-82M/README.md` (retrieved 2025-11-09).
[^voices]: Hugging Face voice catalogue — `hexgrad/Kokoro-82M/VOICES.md` (retrieved 2025-11-09).
[^deepwiki]: DeepWiki article “hexgrad/kokoro — Languages and Voices” (indexed 2025-06-08).
[^pipeline]: `kokoro/pipeline.py` source (retrieved 2025-11-09).
[^kokoro-main]: `kokoro/__main__.py` CLI writer (retrieved 2025-11-09).
[^repo-readme]: GitHub repository README `hexgrad/kokoro` (retrieved 2025-11-09).
[^kokoro_model]: `kokoro/model.py` architecture definition (retrieved 2025-11-09).
[^eval]: Hugging Face evaluation summary `hexgrad/Kokoro-82M/EVAL.md` (retrieved 2025-11-09).
