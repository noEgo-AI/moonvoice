import glob
import json
from pathlib import Path
import random
from loguru import logger
from tqdm import tqdm
import os
import hashlib
import multiprocessing as mp
from typing import Optional

import torch, outetts

# Global per-process interface for multiprocessing workers
_g_interface = None

def _mp_init_worker(model_path, interface_version, backend, device, dtype_name):
    """Initializer for each worker process: build and cache outetts.Interface."""
    global _g_interface
    import torch as _t, outetts as _o
    try:
        _t.set_num_threads(1)
    except Exception:
        pass
    _dtype = getattr(_t, dtype_name, _t.bfloat16)
    cfg = _o.ModelConfig(
        model_path=model_path,
        tokenizer_path=model_path,
        interface_version=interface_version,
        backend=backend,
        device=device,
        dtype=_dtype,
    )
    _g_interface = _o.Interface(cfg)

def _mp_process_wav(wav_path: str):
    """Worker function to create speaker and training prompt for a single wav."""
    try:
        global _g_interface
        base = Path(wav_path).stem
        speaker = _g_interface.create_speaker(wav_path)
        prompt = _g_interface.prompt_processor.get_training_prompt(speaker)
        record = {"id": base, "audio_path": wav_path, "text": prompt}
        return {"ok": True, "wav": wav_path, "speaker": speaker, "record": record}
    except Exception as e:
        return {"ok": False, "wav": wav_path, "error": str(e)}


class Processing:
    def __init__(self):
        self.model = "OuteTTS-1.0-0.6B"
        self.interface_version=outetts.InterfaceVersion.V3
        self.backend=outetts.Backend.HF
        # Default to GPU if available for faster preprocessing
        self.device="cuda"
        self.dtype = torch.bfloat16
        self.seed = 42
        self.sep_ratio = 0.2
        # Optional override for where to read the LoRA hparams template from
        # If set, expected to point to a file like "<base_model_path>/lora_hparams.json"
        self.hparams_template_path: Optional[str] = None

    def _avg_speakers_dict(self, dict_list):
        """Average numeric contents of speaker dicts; keep metadata as first value."""
        if not dict_list:
            return {}

        special_keep = {"version", "interface_version", "format", "backend", "type"}

        def merge(values):
            if not values:
                return None
            # Dict → recurse per key
            if all(isinstance(v, dict) for v in values):
                keys = set().union(*(v.keys() for v in values))
                out = {}
                for k in keys:
                    sub = [v[k] for v in values if k in v]
                    if k in special_keep and sub:
                        out[k] = sub[0]
                    else:
                        out[k] = merge(sub)
                return out
            # Scalars
            if all(isinstance(v, (int, float)) for v in values):
                if all(isinstance(v, int) for v in values) and len(set(values)) == 1:
                    return values[0]
                return float(sum(values) / len(values))
            # Flat numeric lists
            if all(isinstance(v, list) and all(isinstance(x, (int, float)) for x in v) for v in values):
                L = min(len(v) for v in values)
                if L == 0:
                    return []
                return [float(sum(v[i] for v in values) / len(values)) for i in range(L)]
            # Fallback: take first
            return values[0]

        return merge(dict_list)


    # outetts model interface를 만드는 부분 -> 이 부분도 최적화 시켜야 함.
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

    def run(self, path, run_lora: bool = False, *, output_path: Optional[str] = None):
        out_dir = Path(output_path) if output_path else Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        train = out_dir / "training_data_v3.jsonl"
        valid = out_dir / "valid_v3.jsonl"
        avg = out_dir / "avg_speaker.jsonl"
        random.seed(self.seed)
        interface = self.build_interface()
        paths = sorted(glob.glob(f"{path}/*.wav"))
        print(paths)
        with train.open("w", encoding="utf-8") as ftrain, valid.open("w", encoding="utf-8") as fvalid:
            valid_count = 0
            train_count = 0
            processed = 0
            skipped = 0
            speakers = []
            for wav in tqdm(paths, desc="Preparing v3 training lsy"):
                try:
                    base = Path(wav).stem
                    speaker = interface.create_speaker(wav)
                    speakers.append(speaker)
                    prompt = interface.prompt_processor.get_training_prompt(speaker)

                    # Write JSONL line expected by training pipelines
                    record = json.dumps({"id": base, "audio_path": wav, "text": prompt}, ensure_ascii=False)
                    if self.sep_ratio and self.sep_ratio > 0 and random.random() < self.sep_ratio:
                        fvalid.write(record + "\n")
                        valid_count += 1
                    else:
                        ftrain.write(record + "\n")
                        train_count += 1

                    processed += 1
                except Exception as e:
                    logger.warning(f"Skip {wav}: {e}")
                    skipped += 1

            # 평균 스피커(dict 기반 평균) 저장
            try:
                if speakers:
                    avg_speaker = self._avg_speakers_dict(speakers)
                    with avg.open("w", encoding="utf-8") as sf:
                        json.dump(avg_speaker, sf, ensure_ascii=False, indent=2)
                else:
                    logger.warning("No speakers collected; skip avg speaker file.")
            except Exception as e:
                logger.warning(f"Failed to create average speaker: {e}")

            # Write LoRA hparams into the audio path based on the template
            try:
                template_path = Path(self.hparams_template_path) if self.hparams_template_path else (Path(__file__).parent / "lora_hparams.json")
                if template_path.exists():
                    with template_path.open("r", encoding="utf-8") as tf:
                        hparams = json.load(tf)

                    # Update model and dataset paths
                    hparams["model_path"] = str(self.model)
                    hparams["tokenizer_path"] = str(self.model)
                    hparams["train_jsonl"] = str(train)
                    hparams["eval_jsonl"] = str(valid)
                    # Ensure training artifacts are written to the chosen output directory
                    hparams["output_dir"] = str(out_dir)

                    out_hparams_path = out_dir / "lora_hparams.json"
                    with out_hparams_path.open("w", encoding="utf-8") as of:
                        json.dump(hparams, of, ensure_ascii=False, indent=2)
                    logger.success(f"Wrote LoRA hparams → {out_hparams_path}")

                    # Optionally kick off LoRA finetuning using the new class-based API
                    if run_lora:
                        try:
                            from lora import LoraFinetuner
                            logger.info("Starting LoRA finetuning from generated hparams…")
                            LoraFinetuner.from_hparams(str(out_hparams_path)).run()
                        except Exception as e:
                            logger.warning(f"LoRA finetuning failed to start: {e}")
                else:
                    logger.warning(f"LoRA hparams template not found at {template_path}")
            except Exception as e:
                logger.warning(f"Failed to write lora_hparams.json: {e}")

    # ------------------------
    # Multiprocessing pipeline
    # ------------------------
    def run_mp(
        self,
        path: str,
        *,
        processes: Optional[int] = None,
        chunksize: int = 1,
        flush_every: int = 0,
        run_lora: bool = False,
        output_path: Optional[str] = None,
    ) -> None:
        """Prepare JSONL/avg speaker using multiprocessing workers.

        - Creates `training_data_v3.jsonl`, `valid_v3.jsonl`, `avg_speaker.jsonl`.
        - Uses a stable hash-based split for validation controlled by `self.sep_ratio`.
        - Each worker builds its own lightweight `outetts.Interface` once.
        """
        path = str(path)
        out_dir = Path(output_path) if output_path else Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        train = out_dir / "training_data_v3.jsonl"
        valid = out_dir / "valid_v3.jsonl"
        avg = out_dir / "avg_speaker.jsonl"

        wavs = sorted(glob.glob(os.path.join(path, "*.wav")))
        if not wavs:
            logger.warning(f"No WAV files found under {path}")
            return

        # Stable hash-based split function (independent of RNG order)
        def is_valid_split(wav_path: str) -> bool:
            if not self.sep_ratio or self.sep_ratio <= 0:
                return False
            base = Path(wav_path).stem
            h = hashlib.sha1(base.encode("utf-8")).hexdigest()
            v = int(h[:12], 16) / float(16 ** 12)  # map to [0,1)
            return v < float(self.sep_ratio)

        logger.info(
            f"MP preparing {len(wavs)} wavs (processes={processes or os.cpu_count()}, chunksize={chunksize}, sep_ratio={self.sep_ratio})"
        )

        # Serialize dtype for child processes
        try:
            dtype_name = str(self.dtype).split(".")[-1]  # 'torch.bfloat16' -> 'bfloat16'
        except Exception:
            dtype_name = "bfloat16"

        processed = skipped = train_count = valid_count = 0
        speakers = []
        with train.open("w", encoding="utf-8") as ftrain, valid.open("w", encoding="utf-8") as fvalid:
            # Use 'spawn' context to safely use CUDA in subprocesses
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=processes or os.cpu_count(),
                initializer=_mp_init_worker,
                initargs=(self.model, self.interface_version, self.backend, self.device, dtype_name),
            ) as pool:
                for res in tqdm(pool.imap_unordered(_mp_process_wav, wavs, chunksize=chunksize), total=len(wavs), desc="Preparing v3 (MP)"):
                    if res.get("ok"):
                        record = res["record"]
                        if is_valid_split(res["wav"]):
                            fvalid.write(json.dumps(record, ensure_ascii=False) + "\n")
                            valid_count += 1
                        else:
                            ftrain.write(json.dumps(record, ensure_ascii=False) + "\n")
                            train_count += 1
                        spk = res.get("speaker")
                        if isinstance(spk, dict):
                            speakers.append(spk)
                        processed += 1
                        if flush_every and (processed % flush_every == 0):
                            ftrain.flush(); fvalid.flush()
                    else:
                        skipped += 1
                        logger.warning(f"Skip {res.get('wav')}: {res.get('error')}")

        # 평균 스피커(dict 기반 평균) 저장
        try:
            if speakers:
                avg_speaker = self._avg_speakers_dict(speakers)
                with avg.open("w", encoding="utf-8") as sf:
                    json.dump(avg_speaker, sf, ensure_ascii=False, indent=2)
            else:
                logger.warning("No speakers collected; skip avg speaker file.")
        except Exception as e:
            logger.warning(f"Failed to create average speaker: {e}")

        # Write LoRA hparams from template
        try:
            template_path = Path(self.hparams_template_path) if self.hparams_template_path else (Path(__file__).parent / "lora_hparams.json")
            if template_path.exists():
                with template_path.open("r", encoding="utf-8") as tf:
                    hparams = json.load(tf)
                hparams["model_path"] = str(self.model)
                hparams["tokenizer_path"] = str(self.model)
                hparams["train_jsonl"] = str(train)
                hparams["eval_jsonl"] = str(valid)
                # Ensure training artifacts are written to the chosen output directory
                hparams["output_dir"] = str(out_dir)
                out_hparams_path = out_dir / "lora_hparams.json"
                with out_hparams_path.open("w", encoding="utf-8") as of:
                    json.dump(hparams, of, ensure_ascii=False, indent=2)
                logger.success(f"Wrote LoRA hparams → {out_hparams_path}")
                if run_lora:
                    try:
                        from lora import LoraFinetuner
                        logger.info("Starting LoRA finetuning from generated hparams…")
                        LoraFinetuner.from_hparams(str(out_hparams_path)).run()
                    except Exception as e:
                        logger.warning(f"LoRA finetuning failed to start: {e}")
            else:
                logger.warning(f"LoRA hparams template not found at {template_path}")
        except Exception as e:
            logger.warning(f"Failed to write lora_hparams.json: {e}")

        logger.success(
            f"Prepared: processed={processed}, skipped={skipped}, train={train_count}, valid={valid_count} (MP)"
        )


# pro = Processing()
# pro.run("/home/server1/AI2/OuteTTS/datas/wavs/lsy")
