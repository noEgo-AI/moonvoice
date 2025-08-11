import argparse
import glob
import json
import os
from pathlib import Path
import random
from typing import List

from loguru import logger
from tqdm import tqdm

import torch

# OuteTTS imports
from outetts.models.info import InterfaceVersion, Backend
from outetts.models.config import ModelConfig
from outetts.version.v3.prompt_processor import PromptProcessor as V3PromptProcessor


def find_audio_files(data_dir: str) -> List[str]:
    """Find candidate WAV files under lsy/ (prefers lsy/mfa_input/*.wav)."""
    mfa_wavs = sorted(glob.glob(os.path.join(data_dir, "mfa_input", "*.wav")))
    if mfa_wavs:
        return mfa_wavs
    return sorted(glob.glob(os.path.join(data_dir, "*.wav")))


def main():
    parser = argparse.ArgumentParser(description="Prepare v3 training lsy from lsy/ using Whisper + DAC.")
    parser.add_argument(
        "--data_dir", default="lsy", help="Root lsy directory containing WAVs (default: lsy)"
    )
    parser.add_argument(
        "--tokenizer_path",
        default="OuteTTS-1.0-0.6B",
        help="Path/name to v3 tokenizer (directory with tokenizer.json).",
    )
    parser.add_argument(
        "--whisper_model", default="turbo", help="Whisper model name (e.g., turbo, base, small, medium, large)."
    )
    parser.add_argument(
        "--whisper_device", default=None, help="Device for Whisper inference (e.g., cuda, cpu)."
    )
    # Output controls
    parser.add_argument(
        "--out_train_jsonl",
        default="lsy/training_data_v3.jsonl",
        help="Output JSONL for training prompts.",
    )
    parser.add_argument(
        "--out_valid_jsonl",
        default="lsy/valid_v3.jsonl",
        help="Output JSONL for validation prompts.",
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.1,
        help="Portion of samples to route to validation set (0.0-1.0). Set to 0 to disable split.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/valid split"
    )
    parser.add_argument(
        "--speakers_dir",
        default="lsy/v3_speakers",
        help="Directory to save extracted speaker JSONs.",
    )
    parser.add_argument(
        "--max_files", type=int, default=None, help="Process at most N files (for quick runs)."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    speakers_dir = Path(args.speakers_dir)
    speakers_dir.mkdir(parents=True, exist_ok=True)

    # Configure v3 environment
    # Note: We configure only what v3 needs: tokenizer and audio processor (DAC)
    config = ModelConfig(
        model_path="OuteAI/Llama-OuteTTS-1.0-1B",
        tokenizer_path=args.tokenizer_path,
        interface_version=InterfaceVersion.V3,
        backend=Backend.HF,  # backend isn't used here
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=None,
    )

    # Instantiate v3 helpers
    audio_processor = config.audio_processor(config)
    prompt_processor = V3PromptProcessor(args.tokenizer_path)

    wavs = find_audio_files(data_dir)
    if args.max_files is not None:
        wavs = wavs[: args.max_files]

    if not wavs:
        logger.error(f"No WAV files found under {data_dir} or {data_dir}/mfa_input.")
        return

    logger.info(f"Found {len(wavs)} audio files to process (v3 pipeline).")

    train_path = Path(args.out_train_jsonl)
    valid_path = Path(args.out_valid_jsonl)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    train_count = 0
    valid_count = 0

    # Init RNG for deterministic split
    random.seed(args.seed)

    # Open both outputs; if valid_ratio==0.0 we will only write to train
    with train_path.open("w", encoding="utf-8") as ftrain, valid_path.open("w", encoding="utf-8") as fvalid:
        for wav in tqdm(wavs, desc="Preparing v3 training lsy"):
            try:
                base = Path(wav).stem
                speaker_json_path = speakers_dir / f"{base}.json"

                # Extract speaker representation via Whisper word-level timestamps + DAC encoding
                speaker = audio_processor.create_speaker_from_whisper(
                    audio=wav, whisper_model=args.whisper_model, device=args.whisper_device
                )

                # Save the raw speaker JSON for reuse/debugging
                with speaker_json_path.open("w", encoding="utf-8") as sf:
                    json.dump(speaker, sf, ensure_ascii=False, indent=2)

                # Build v3 training prompt
                prompt = prompt_processor.get_training_prompt(speaker)

                # Write JSONL line expected by training pipelines
                record = json.dumps({"id": base, "audio_path": wav, "text": prompt}, ensure_ascii=False)
                if args.valid_ratio and args.valid_ratio > 0 and random.random() < args.valid_ratio:
                    fvalid.write(record + "\n")
                    valid_count += 1
                else:
                    ftrain.write(record + "\n")
                    train_count += 1

                processed += 1
            except Exception as e:
                logger.warning(f"Skip {wav}: {e}")
                skipped += 1

    msg = (
        f"Done. Processed={processed}, Skipped={skipped}, "
        f"Train JSONL={train_path} ({train_count}), Valid JSONL={valid_path} ({valid_count}), "
        f"Speakers dir={speakers_dir}"
    )
    logger.success(msg)


if __name__ == "__main__":
    main()
