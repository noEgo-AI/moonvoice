"""
voice_interface.py

High-level handler to run the repository pipeline:
  1) Prepare data (JSONL, averaged/reference speakers, hparams)
  2) Train LoRA from generated hparams
  3) Single-sentence inference with candidate selection/caching
  4) Chapter-level inference for a list/file of sentences

Example:
  from voice_interface import VoiceTTSHandler
  v = VoiceTTSHandler(device='cuda', dtype='bf16')
  v.prepare_data('datas/wavs/lsy')
  v.train_lora('datas/wavs/lsy')
  v.infer_candidates('datas/wavs/lsy', text='안녕하세요.', n_candidates=5,
                     prosody_override={'energy':60,'pitch':45},
                     postprocess={'normalize_rms_db':-20.0,'trim_db':-45.0,'fade_ms':8})
  v.synthesize_chapter_from_file('datas/wavs/lsy', 'chapter1.txt', evaluate_once=True)
"""

from __future__ import annotations

import os, glob
from pydub import AudioSegment
from pathlib import Path
from typing import Iterable, Optional
from loguru import logger

from utils.Processing import Processing
# from processing2 import Processing
from utils.lora import LoraFinetuner
from utils.Inference import LoraInference
from utils.book_tts import synthesize_chapter, synthesize_chapter_from_file


class VoiceTTSHandler:
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[str] = 'bf16',  # 'bf16' | 'fp16' | 'fp32'
        seeds_pool_size: int = 12,
        val_sample_size: int = 3,
        default_temperature: float = 0.4,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.seeds_pool_size = seeds_pool_size
        self.val_sample_size = val_sample_size
        self.default_temperature = default_temperature

    # ---------- Stage 1: Prepare ----------
    def prepare_data(
        self,
        wav_dir: str,
        *,
        output_path: str | None = None,
        usercode: str | None = None,
        voicecode: str | None = None,
        processes: int | None = None,
        chunksize: int = 1,
        flush_every: int = 0,
        sep_ratio: float | None = None,
        use_multiprocessing: bool = True,
        # Allow explicit base model override (absolute path recommended)
        base_model_path: str | None = None,
        # Force device for preprocessing (e.g., 'cpu' to avoid GPU)
        device_override: str | None = None,
    ) -> None:
        """Prepare JSONL + averaged/reference speakers (+ hparams) for training.

        - By default uses multiprocessing via `Processing.run_mp`.
        - `sep_ratio` > 0 splits a portion into `valid_v3.jsonl` using a hash-based stable split.
        - If `use_multiprocessing=False`, falls back to single-process if available; otherwise uses MP.
        - If `output_path` is provided, JSONL and hparams are written there while WAVs are read from `audio_dir`.
        - `wav_dir` may be a directory or a `.zip` file. If a ZIP is provided,
          it is extracted alongside the ZIP (to `<zip_stem>/`) and the first
          directory containing WAV files is used.
        - If both `output_path` and `usercode` are provided, outputs are written under `output_path/usercode[/voicecode]`.
        """
        wav_dir = str(wav_dir)

        def _looks_like_zip(p: str) -> bool:
            try:
                return os.path.isfile(p) and p.lower().endswith('.zip')
            except Exception:
                return False

        def _extract_zip(zip_path: str) -> str:
            import zipfile
            zpath = Path(zip_path)
            target_dir = zpath.with_suffix("")  # ./data/choi.zip -> ./data/choi
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to ensure extract dir {target_dir}: {e}")
            # Heuristic: if target has wavs already, skip re-extract
            existing_wavs = list(target_dir.rglob('*.wav'))
            if existing_wavs:
                logger.info(f"Using existing extracted directory: {target_dir}")
                return str(target_dir)
            logger.info(f"Extracting ZIP {zpath} → {target_dir}")
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(target_dir)
            return str(target_dir)

        def _find_wav_root(root: str) -> str:
            # If WAVs exist directly under root, use it; otherwise pick parent of first found WAV
            direct = list(Path(root).glob('*.wav'))
            if direct:
                return root
            found = list(Path(root).rglob('*.wav'))
            if found:
                return str(found[0].parent)
            return root

        # Resolve ZIP → directory containing WAVs
        if _looks_like_zip(wav_dir):
            try:
                extracted = _extract_zip(wav_dir)
                wav_dir = _find_wav_root(extracted)
            except Exception as e:
                logger.error(f"Failed to extract/process ZIP {wav_dir}: {e}")
                raise

        logger.info(f"Preparing data under: {wav_dir}")
        # Compute effective output directory (optionally nested by usercode)
        eff_output_path = None
        if output_path:
            if usercode and voicecode:
                eff_output_path = str(Path(output_path) / usercode / voicecode)
            elif usercode:
                eff_output_path = str(Path(output_path) / usercode)
            else:
                eff_output_path = str(output_path)
        pro = Processing()
        # Optional override of base model path (used for speaker extraction and hparams)
        if base_model_path:
            try:
                pro.model = str(base_model_path)
                # If a lora_hparams.json exists under base_model_path, use it as template
                from pathlib import Path as _P
                cand = _P(str(base_model_path)) / "lora_hparams.json"
                if cand.exists():
                    pro.hparams_template_path = str(cand)
                    logger.info(f"Using LoRA hparams template from base_model_path: {cand}")
                logger.info(f"Using base model for preprocessing: {pro.model}")
            except Exception as e:
                logger.warning(f"Failed to set base model path on Processing: {e}")
        # Optional device override (e.g., 'cpu')
        if device_override:
            try:
                pro.device = str(device_override)
                logger.info(f"Using device for preprocessing: {pro.device}")
            except Exception as e:
                logger.warning(f"Failed to set device on Processing: {e}")
        if sep_ratio is not None:
            try:
                pro.sep_ratio = float(sep_ratio)
            except Exception:
                pass
        # Prefer MP path. Fallback to SP run() if explicitly disabled and available.
        if use_multiprocessing:
            pro.run_mp(wav_dir, processes=processes, chunksize=chunksize, flush_every=flush_every, output_path=eff_output_path)
        else:
            if hasattr(pro, "run") and callable(getattr(pro, "run")):
                pro.run(wav_dir, output_path=eff_output_path)
            else:
                logger.warning("Single-process run() not available; using multiprocessing instead.")
                pro.run_mp(wav_dir, processes=processes, chunksize=chunksize, flush_every=flush_every, output_path=eff_output_path)
        logger.success("Data preparation complete.")

    # ---------- Stage 2: Train ----------
    def train_lora(
        self,
        model_dir: str,
        # hpramas_path: str | None = None,
        output_path: str | None = None,
        *,
        usercode: str | None = None,
        voicecode: str | None = None,
        # Allow explicit base model override (absolute path recommended)
        base_model_path: str | None = None,
        hparams_path: str | None = None,
    ) -> str:
        model_dir = str(model_dir)
        import json as _json
        # Resolve hparams path preference: explicit > output_path/usercode[/voicecode]
        if hparams_path:
            hp = str(hparams_path)
        else:
            hp_dir = Path(output_path) / (usercode or "")
            if voicecode:
                hp_dir = hp_dir / voicecode
            hp = str(hp_dir / 'lora_hparams.json')

        # if hparams_path:
        #     hp = str(hparams_path)
        # elif output_path:
        #     hp_dir = Path(output_path) / (usercode or "")
        #     hp = str(hp_dir / 'lora_hparams.json')
        # else:
        #     hp = str(Path(model_dir) / 'lora_hparams.json')
        if not Path(hp).exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {hp}")
        # Ensure output_dir in hparams points to the directory containing the hparams file
        try:
            with open(hp, 'r', encoding='utf-8') as f:
                _cfg = _json.load(f)
            desired_out = str(Path(hp).parent)
            _cfg['output_dir'] = desired_out
            # Optional: override model/tokenizer paths for training
            if base_model_path:
                _cfg['model_path'] = str(base_model_path)
                _cfg['tokenizer_path'] = str(base_model_path)
            with open(hp, 'w', encoding='utf-8') as f:
                _json.dump(_cfg, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated output_dir in hparams → {desired_out}")
            if base_model_path:
                logger.info(f"Overriding base model/tokenizer in hparams → {base_model_path}")
        except Exception as e:
            logger.warning(f"Could not adjust output_dir in hparams: {e}")
        logger.info(f"Training LoRA using hparams: {hp}")
        LoraFinetuner.from_hparams(hp).run()
        # Return adapter directory from hparams
        with open(hp, 'r', encoding='utf-8') as f:
            cfg = _json.load(f)
        outdir = cfg.get('output_dir') or str(Path(hp).parent)
        path = Path(outdir)
        logger.success(f"Training finished. Adapter at: {path}")
        return str(path)

    # ---------- Stage 3: Single-sentence inference ----------
    def infer_candidates(
        self,
        audio_dir: str,
        text: str,
        n_candidates: int = 5,
        *,
        evaluate: bool = True,
        lora_dir: str | None = None,
        # Optional explicit base model path; when provided we use path-based inference
        base_model_path: str | None = None,
        tokenizer_path: str | None = None,
        prosody_override: dict | None = None,
        postprocess: dict | None = None,
        seeds_override: list[int] | None = None,
        out: str | None = None,
        temperature: float | None = None,
    ) -> list[str]:
        audio_dir = str(audio_dir)
        temperature = float(temperature if temperature is not None else self.default_temperature)
        logger.info(f"Synthesizing {n_candidates} candidates for text (len={len(text)}).")
        # If base_model_path explicitly provided, switch to path-based inference
        if base_model_path:
            # Resolve lora_dir if not provided: try reading hparams under audio_dir
            lora_dir_resolved = lora_dir
            if lora_dir_resolved is None:
                try:
                    import json as _json
                    hp_path = Path(audio_dir) / 'lora_hparams.json'
                    if hp_path.exists():
                        with open(hp_path, 'r', encoding='utf-8') as f:
                            hp = _json.load(f)
                        outdir = hp.get('output_dir')
                        if outdir:
                            # If relative, resolve relative to audio_dir
                            p = Path(outdir)
                            lora_dir_resolved = str(p if p.is_absolute() else (Path(audio_dir) / p))
                except Exception as e:
                    logger.warning(f"Could not resolve lora_dir from hparams: {e}")
            if lora_dir_resolved is None:
                raise FileNotFoundError("lora_dir not provided and could not be inferred from hparams.")
            return LoraInference.from_paths(
                base_model_path=str(base_model_path),
                lora_dir=str(lora_dir_resolved),
                tokenizer_path=(str(tokenizer_path) if tokenizer_path else None),
                text=text,
                n_candidates=int(n_candidates),
                out=out,
                temperature=temperature,
                device=self.device,
                dtype=self.dtype,
                evaluate=bool(evaluate),
                seeds_pool_size=self.seeds_pool_size,
                val_sample_size=self.val_sample_size,
                prosody_override=prosody_override,
                postprocess=postprocess,
                seeds_override=seeds_override,
                context_dir=str(audio_dir),
            ).synthesize()
        # Default: use audio_dir-based flow (reads model/tokenizer from hparams)
        return LoraInference.from_audio_dir(
            audio_dir,
            text=text,
            n_candidates=int(n_candidates),
            out=out,
            lora_dir=lora_dir,
            temperature=temperature,
            device=self.device,
            dtype=self.dtype,
            evaluate=bool(evaluate),
            seeds_pool_size=self.seeds_pool_size,
            val_sample_size=self.val_sample_size,
            prosody_override=prosody_override,
            postprocess=postprocess,
            seeds_override=seeds_override,
        ).synthesize()

    # ---------- Stage 4: Chapter inference ----------
    def synthesize_chapter(
        self,
        audio_dir: str,
        text_lines: Iterable[str],
        *,
        n_candidates_per_sentence: int = 1,
        evaluate_once: bool = True,
        skip_eval_if_cached: bool = True,
        save_best_speaker: bool = False,
        best_speaker_filename: str | None = None,
        # Explicit path-based mode: provide base and adapter paths
        base_model_path: str | None = None,
        model_path: str | None = None,
        output_path: str | None = None,
        usercode: str | None = None,
        prosody_override: dict | None = None,
        postprocess: dict | None = None,
        synth_workers: int | None = None,
        eval_workers: int | None = None,
    ) -> list[str]:
        # Path-based mode if both provided
        if base_model_path and model_path:
            from utils.book_tts import synthesize_chapter_paths as _synth_paths
            return _synth_paths(
                base_model_path=str(base_model_path),
                model_path=str(model_path),
                text_lines=list(text_lines),
                output_path=(str(output_path) if output_path else None),
                usercode=(str(usercode) if usercode else None),
                n_candidates_per_sentence=n_candidates_per_sentence,
                evaluate_once=evaluate_once,
                skip_eval_if_cached=skip_eval_if_cached,
                save_best_speaker=save_best_speaker,
                best_speaker_filename=best_speaker_filename,
                seeds_pool_size=self.seeds_pool_size,
                val_sample_size=self.val_sample_size,
                prosody_override=prosody_override,
                postprocess=postprocess,
                synth_workers=synth_workers,
                eval_workers=eval_workers,
            )
        # Backward-compatible audio_dir flow
        audio_dir = str(audio_dir)
        return synthesize_chapter(
            audio_dir,
            list(text_lines),
            n_candidates_per_sentence=n_candidates_per_sentence,
            evaluate_once=evaluate_once,
            skip_eval_if_cached=skip_eval_if_cached,
            save_best_speaker=save_best_speaker,
            best_speaker_filename=best_speaker_filename,
            seeds_pool_size=self.seeds_pool_size,
            val_sample_size=self.val_sample_size,
            prosody_override=prosody_override,
            postprocess=postprocess,
            synth_workers=synth_workers,
            eval_workers=eval_workers,
        )

    def synthesize_chapter_from_file(
        self,
        audio_dir: str,
        text_file: str,
        *,
        n_candidates_per_sentence: int = 1,
        evaluate_once: bool = True,
        skip_eval_if_cached: bool = True,
        save_best_speaker: bool = False,
        best_speaker_filename: str | None = None,
        prosody_override: dict | None = None,
        postprocess: dict | None = None,
        synth_workers: int | None = None,
        # New explicit path-based mode
        base_model_path: str | None = None,
        model_path: str | None = None,
        output_path: str | None = None,
        usercode: str | None = None,
        eval_workers: int | None = None,
    ) -> list[str]:
        # If base_model_path/model_path provided, use explicit paths and direct outputs to output_path/usercode
        if base_model_path and model_path:
            return synthesize_chapter_from_file(
                "_unused_",
                str(text_file),
                n_candidates_per_sentence=n_candidates_per_sentence,
                evaluate_once=evaluate_once,
                skip_eval_if_cached=skip_eval_if_cached,
                save_best_speaker=save_best_speaker,
                best_speaker_filename=best_speaker_filename,
                seeds_pool_size=self.seeds_pool_size,
                val_sample_size=self.val_sample_size,
                prosody_override=prosody_override,
                postprocess=postprocess,
                synth_workers=synth_workers,
                base_model_path=str(base_model_path),
                model_path=str(model_path),
                output_path=(str(output_path) if output_path else None),
                usercode=(str(usercode) if usercode else None),
                eval_workers=eval_workers,
            )
        # Backward compatible: use audio_dir-based flow
        audio_dir = str(audio_dir)
        return synthesize_chapter_from_file(
            audio_dir,
            str(text_file),
            n_candidates_per_sentence=n_candidates_per_sentence,
            evaluate_once=evaluate_once,
            skip_eval_if_cached=skip_eval_if_cached,
            save_best_speaker=save_best_speaker,
            best_speaker_filename=best_speaker_filename,
            seeds_pool_size=self.seeds_pool_size,
            val_sample_size=self.val_sample_size,
            prosody_override=prosody_override,
            postprocess=postprocess,
            synth_workers=synth_workers,
            eval_workers=eval_workers,
        )

    @staticmethod
    def audio_numkey(path):
        import re
        m = re.search(r'(\d+)\.wav$', os.path.basename(path), re.I)
        return int(m.group(1)) if m else float("inf")

    def audio_merge(self,d, out):
        gap_ms = 300
        # 합성 간격 소리
        files = sorted(glob.glob(os.path.join(d, "*.wav")), key=self.audio_numkey)
        if not files: raise SystemExit("wav ??")
        merged = AudioSegment.silent(0)
        for f in files:
            seg = AudioSegment.from_wav(f)
            seg = seg.set_frame_rate(24000)
            seg = seg.set_channels(1)
            seg = seg.set_sample_width(2)  # 16-bit PCM
            merged += seg
            if gap_ms > 0:
                merged += AudioSegment.silent(gap_ms)
        merged.export(out, format="wav")
        print(f"{len(files)}? ?? -> {out}")


__all__ = ["VoiceTTSHandler"]

# if __name__ == "__main__":
#
#     voice_path = "datas/wavs/lsy1"
#     out_path = "output/lsy"
#     wav_path = "output/teddy"

    # import os
    # os.environ["EVAL_WORKERS"] = "3"

    # v = VoiceTTSHandler(device='cuda', dtype='bf16', seeds_pool_size=3, val_sample_size=3, default_temperature=0.40)
    # Multiprocessing data prep example
    # v.prepare_data(voice_path, output_path=out_path,usercode="1", processes=2)
    # v.prepare_data(voice_path, use_multiprocessing=False)
    # v.train_lora(out_path, output_path=out_path, usercode="1")
    # v.infer_candidates(
    #     voice_path,
    #     text='옛날 옛적, 깊은 숲 속 작은 오두막에 현명한 할머니가 살았답니다. 할머니는 매일 아침 햇살을 받으며 갓 구운 빵을 굽고, 숲속 동물들에게 따뜻한 미소를 나누었어요.',
    #     n_candidates=100,
    #     prosody_override={'energy':60,'pitch':45},
    #     postprocess={'normalize_rms_db':-20.0,'trim_db':-45.0,'fade_ms':8},
    #     evaluate=False,
    #   )
    # v.synthesize_chapter_from_file("_", '반달가슴곰_달곰이.txt',base_model_path="./OuteTTS-1.0-0.6B",
    #                                model_path=out_path, usercode="1" , output_path=wav_path,
    #                                evaluate_once=True, n_candidates_per_sentence=1, eval_workers=3,
    #                                synth_workers=3, skip_eval_if_cached=False)
    # v.synthesize_chapter_from_file(voice_path, '반달가슴곰_달곰이.txt', evaluate_once=True, n_candidates_per_sentence=1, synth_workers=3, skip_eval_if_cached=False, save_best_speaker=True )
    # v.audio_merge(out_path, "반달가슴곰_달곰이_최훈원.wav")



# How to use
# - Save a best-seed speaker during chapter synthesis:
#   - Direct:
#     - `synthesize_chapter_from_file('datas/wavs/lsy', 'chapter1.txt', evaluate_o
# nce=True, skip_eval_if_cached=True, save_best_speaker=True)`
#     - Optional: `best_speaker_filename='speaker_reference.json'` to overwrite yo
# ur reference (use cautiously).
#   - Via handler:
#     - `v.synthesize_chapter_from_file('datas/wavs/lsy', 'chapter1.txt', evaluate
# _once=True, skip_eval_if_cached=True, save_best_speaker=True)`



# - Rank seeds with a single reference audio:
#   - `from voice_interface import VoiceTTSHandler`
#   - `v = VoiceTTSHandler(device='cuda', dtype='bf16')`
#   - `scored = v.evaluate_seeds_with_single_audio('datas/wavs/lsy', 'datas/wavs/l
# sy/sample.wav', text_for_eval='안녕하세요.')`
# - Reuse for chapter TTS (skips re-eval if cached):
#   - `v.synthesize_chapter_from_file('datas/wavs/lsy', 'chapter1.txt', evaluate_o
# nce=True, skip_eval_if_cached=True)`


#
# Usage example
# - from voice_interface import VoiceTTSHandler
# - v = VoiceTTSHandler(device='cuda', dtype='bf16')
# - v.prepare_data('datas/wavs/lsy')
# - v.train_lora('datas/wavs/lsy')
# - v.infer_candidates(
#     'datas/wavs/lsy',
#     text='안녕하세요.',
#     n_candidates=5,
#     prosody_override={'energy':60,'pitch':45},
#     postprocess={'normalize_rms_db':-20.0,'trim_db':-45.0,'fade_ms':8}
#   )
# - v.synthesize_chapter_from_file('datas/wavs/lsy', 'chapter1.txt', evaluate_once
# =True)
#
# If you want, I can add a minimal CLI wrapper around VoiceTTSHandler for terminal
#  usage (e.g., voice_interface_cli.py).
#
