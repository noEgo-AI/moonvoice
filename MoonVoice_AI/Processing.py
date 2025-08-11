import argparse
import glob
import json
import os
from pathlib import Path
import random
from typing import List

from loguru import logger
from tqdm import tqdm

import torch, outetts

# OuteTTS imports
# from outetts.models.info import InterfaceVersion, Backend
# from outetts.models.config import ModelConfig
# from outetts.version.v3.prompt_processor import PromptProcessor as V3PromptProcessor


class Processing:
    def __init(self):
        self.model = "OuteTTS-1.0-0.6B"
        self.interface_version=outetts.InterfaceVersion.V3
        self.backend=outetts.Backend.HF
        self.device = "cpu"
        self.dtype = torch.bfloat16
        self.seed = 42

    def build_interface(self):
        cfg = outetts.ModelConfig(
            model_path=self.model,
            tokenizer_path=self.model,
            interface_version=self.interface_version,  # ← 필수
            backend=self.backend,
            device = self.device,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=self.dtype,
        )
        return outetts.Interface(cfg)

    def run(self, path):
        train = Path(f"{path}/training_data_v3.jsonl")
        valid = Path(f"{path}/valid_v3.jsonl")
        avg = Path(f"{path}/avg_speaker.jsonl")
        seed = random.seed(self.seed)
        interface = self.build_interface()
        paths = sorted(Path().glob(f"{path}/*.wav"))
        print(paths)

        # with train.open("w", encoding="utf-8") as ftrain, valid.open("w", encoding="utf-8") as fvalid:
        #     for wav in tqdm(wavs, desc="Preparing v3 training lsy"):
        #         try:
        #             base = Path(wav).stem
        #             speaker_json_path = speakers_dir / f"{base}.json"
        #
        #             # Extract speaker representation via Whisper word-level timestamps + DAC encoding
        #             speaker = audio_processor.create_speaker_from_whisper(
        #                 audio=wav, whisper_model=args.whisper_model, device=args.whisper_device
        #             )
        #
        #             # Build v3 training prompt
        #             prompt = prompt_processor.get_training_prompt(speaker)
        #
        #             # Write JSONL line expected by training pipelines
        #             record = json.dumps({"id": base, "audio_path": wav, "text": prompt}, ensure_ascii=False)
        #             if args.valid_ratio and args.valid_ratio > 0 and random.random() < args.valid_ratio:
        #                 fvalid.write(record + "\n")
        #                 valid_count += 1
        #             else:
        #                 ftrain.write(record + "\n")
        #                 train_count += 1
        #
        #             processed += 1
        #         except Exception as e:
        #             logger.warning(f"Skip {wav}: {e}")
        #             skipped += 1
        #
        #
        #     if not paths:
        #         raise SystemExit(f"No files matched --speaker_wavs_glob={path}")
        #     speakers = [interface.create_speaker(str(p)) for p in paths]
        #     base = speakers[0]
        #     for s in speakers[1:]:
        #         base = base + s
        #     speaker = base / len(speakers)

pro = Processing().run("/home/server1/AI2/OuteTTS/datas/wavs/lsy")

