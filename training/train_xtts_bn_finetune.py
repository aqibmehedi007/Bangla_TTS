"""
Standardized XTTS v2 Fine-tuning Script for Bengali TTS.
Optimized for 200 samples on RTX 3060.
"""
import os
import torch
import functools
import gc
from pathlib import Path
from trainer import Trainer, TrainerArgs

# Fix for PyTorch 2.4+ security restriction
torch.load = functools.partial(torch.load, weights_only=False)

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# Paths
PROJECT_ROOT = Path("d:/GitHub/Kokoro_tts_module")
DATASET_DIR = PROJECT_ROOT / "data" / "bengali" / "xtts"
TRAIN_CSV = DATASET_DIR / "metadata_train.csv"
EVAL_CSV = DATASET_DIR / "metadata_val.csv"
OUTPUT_DIR = PROJECT_ROOT / "training" / "xtts_ft_results"

def main():
    print("🚀 Starting XTTS v2 Fine-tuning Workflow...")
    
    # 1. Environment Setup
    RUN_NAME = "Bengali_Voice_Cloning"
    PROJECT_NAME = "Kokoro_XTTS_Project"
    OUT_PATH = str(OUTPUT_DIR / "run")
    os.makedirs(OUT_PATH, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Training on: {device}")
    
    # 2. Base Model Files Setup (Automated)
    CHECKPOINTS_OUT_PATH = str(OUTPUT_DIR / "base_model_files")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
    
    # URLs for base XTTS v2 components
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "config.json")

    # Download if missing
    if not os.path.isfile(XTTS_CHECKPOINT):
        print("📥 Downloading base XTTS v2 files (once only)...")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK, TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK],
            CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # 3. Dataset Configuration
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="bengali_cloning_dataset",
        path=str(DATASET_DIR),
        meta_file_train=str(TRAIN_CSV),
        meta_file_val=str(EVAL_CSV),
        language="en", # We use 'en' because our transcripts are Romanized
    )

    # 4. Model Training Arguments
    model_args = GPTArgs(
        max_conditioning_length=132300, # 6 seconds
        min_conditioning_length=66150,  # 3 seconds
        max_wav_length=250000,          # ~11 seconds
        max_text_length=400,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    # 5. Training Config
    config = GPTTrainerConfig(
        epochs=10, 
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="Bengali Voice Cloning Fine-tuning",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=2, # Optimized for 12GB VRAM
        batch_group_size=48,
        eval_batch_size=1,
        num_loader_workers=0, # Windows fix
        eval_split_max_size=256,
        print_step=25,
        plot_step=50,
        log_model_step=100,
        save_step=250,
        save_n_checkpoints=2,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=False,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06, 
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [1000, 2000, 4000], "gamma": 0.5},
        test_sentences=[
            "Amar shonar bangla, ami tomay bhalobashi.",
            "Apni kemon achhen? Ami bhalo achhi."
        ],
    )

    # 6. Initialize Model & Data
    print("🏗️ Initializing Model and Loading Samples...")
    model = GPTTrainer.init_from_config(config)
    
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
    )

    # 7. Start Training
    print("🔥 Starting Trainer. Training loss will be printed every 25 steps.")
    trainer = Trainer(
        TrainerArgs(
            restore_path=None, 
            skip_train_epoch=False,
            # Windows stability params
            grad_accum_steps=1,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    trainer.fit()
    
    print("\n✅ Training Complete!")
    print(f"📂 Checkpoints saved in: {OUT_PATH}/{RUN_NAME}")
    print("\n📝 Next Step: Run inference with 'best_model.pth' and your reference voice.")

if __name__ == "__main__":
    main()
