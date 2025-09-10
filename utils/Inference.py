import os
import random
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM
from peft import PeftModel

import outetts


def _pick_dtype(name: str | None):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class LoraInference:
    """Class-based inference using a trained LoRA adapter.

    Typical use with Processing/lora outputs:
      LoraInference.from_audio_dir('/path/to/wavs', text='...', n_candidates=5).synthesize()
    """

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 lora_dir: str,
                 device: str | None = None,
                 dtype: str | None = None,
                 force_local: bool = True,
                 max_length: int = 8192,
                 temperature: float = 0.4,
                 speaker: dict | None = None,
                 default_speaker: str = "en-female-1-neutral",
                 merge_lora: bool = False,
                 ): 
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.lora_dir = lora_dir
        self.device = device
        self.dtype_name = dtype
        self.dtype = _pick_dtype(dtype)
        self.force_local = force_local
        self.max_length = max_length
        self.temperature = temperature
        self.speaker = speaker
        self.default_speaker = default_speaker
        self.merge_lora = merge_lora

        # Build interface
        if self.force_local or Path(self.model_path).is_dir() or Path(self.tokenizer_path).is_dir():
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        cfg = outetts.ModelConfig(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            interface_version=outetts.InterfaceVersion.V3,
            backend=outetts.Backend.HF,
            device=self.device,
            dtype=self.dtype,
        )
        self.interface = outetts.Interface(cfg)

        # Attach LoRA
        if not Path(self.lora_dir).is_dir():
            raise FileNotFoundError(f"LoRA adapter dir not found: {self.lora_dir}")
        logger.info(f"Loading base model from: {self.model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            local_files_only=bool(self.force_local),
        )
        logger.info(f"Loading LoRA adapter from: {self.lora_dir}")
        model = PeftModel.from_pretrained(base_model, self.lora_dir)
        if self.merge_lora:
            logger.info("Merging LoRA weights into base model for inference")
            model = model.merge_and_unload()
        self.interface.model.model = model.to(self.interface.model.device)

    @classmethod
    def from_audio_dir(cls,
                       audio_dir: str,
                       text: str,
                       n_candidates: int = 5,
                       out: str | None = None,
                       lora_dir: str | None = None,
                       temperature: float = 0.4,
                       device: str | None = None,
                       dtype: str | None = None,
                       merge_lora: bool = False,
                       force_local: bool = True,
                       evaluate: bool = True,
                       seeds_pool_size: int = 12,
                       val_sample_size: int = 5,
                       prosody_override: dict | None = None,
                       postprocess: dict | None = None,
                       seeds_override: list[int] | None = None):
        audio_dir = Path(audio_dir)
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio dir not found: {audio_dir}")

        # Load hparams written by Processing.py to get model/tokenizer and possibly output_dir
        hp_path = audio_dir / "lora_hparams.json"
        if not hp_path.exists():
            raise FileNotFoundError(f"Missing hparams: {hp_path}")
        import json as _json
        with open(hp_path, "r", encoding="utf-8") as f:
            hp = _json.load(f)
        model_path = hp.get("model_path")
        tokenizer_path = hp.get("tokenizer_path", model_path)
        outdir_candidate = hp.get("output_dir", "outetts_finetuned_v3")

        # Resolve lora_dir: explicit > absolute > audio_dir-relative > cwd-relative
        lora_dir_resolved = lora_dir
        if lora_dir_resolved is None:
            for cand in [outdir_candidate,
                         str(audio_dir / outdir_candidate),
                         str(Path.cwd() / outdir_candidate)]:
                if cand and Path(cand).is_dir():
                    lora_dir_resolved = cand
                    break
        if lora_dir_resolved is None:
            raise FileNotFoundError("Cannot resolve lora_dir; provide it explicitly or ensure output_dir exists.")

        # Load reference speaker first, then averaged
        speaker = None
        ref_path = audio_dir / "speaker_reference.json"
        if ref_path.exists():
            try:
                import json as _json
                with open(ref_path, "r", encoding="utf-8") as f:
                    speaker = _json.load(f)
                logger.info(f"Loaded reference speaker: {ref_path}")
            except Exception as e:
                logger.warning(f"Failed to load reference speaker: {e}")
        if speaker is None:
            avg_candidates = [
                audio_dir / "Speaker_avg.json",
                audio_dir / "speaker_avg.json",
                audio_dir / "avg_speaker.json",
                audio_dir / "avg_speaker.jsonl",
            ]
            avg_path = next((p for p in avg_candidates if p.exists()), None)
            if avg_path is not None and avg_path.exists():
                try:
                    import json as _json
                    with open(avg_path, "r", encoding="utf-8") as f:
                        speaker = _json.load(f)
                    logger.info(f"Loaded averaged speaker: {avg_path}")
                except Exception as e:
                    logger.warning(f"Failed to load averaged speaker: {e}")

        inst = cls(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            lora_dir=lora_dir_resolved,
            device=device,
            dtype=dtype,
            force_local=force_local,
            temperature=temperature,
            speaker=speaker,
            merge_lora=merge_lora,
        )
        inst._text = text
        inst._n = max(1, int(n_candidates))
        inst._audio_dir = str(audio_dir)
        inst._out = out
        inst._evaluate = bool(evaluate)
        inst._seeds_pool_size = max(n_candidates, int(seeds_pool_size))
        inst._val_sample_size = max(1, int(val_sample_size))
        inst._prosody_override = prosody_override if isinstance(prosody_override, dict) else None
        inst._postproc = postprocess if isinstance(postprocess, dict) else None
        inst._seeds_override = list(seeds_override) if isinstance(seeds_override, list) else None
        return inst

    @classmethod
    def from_paths(cls,
                   base_model_path: str,
                   lora_dir: str,
                   *,
                   text: str,
                   n_candidates: int = 5,
                   out: str | None = None,
                   tokenizer_path: str | None = None,
                   context_dir: str | None = None,
                   temperature: float = 0.4,
                   device: str | None = None,
                   dtype: str | None = None,
                   merge_lora: bool = False,
                   force_local: bool = True,
                   evaluate: bool = True,
                   seeds_pool_size: int = 12,
                   val_sample_size: int = 5,
                   prosody_override: dict | None = None,
                   postprocess: dict | None = None,
                   seeds_override: list[int] | None = None,
                   out_dir: str | None = None):
        ctx_dir = Path(context_dir) if context_dir else Path(lora_dir)
        # Load reference/avg speaker from context directory
        speaker = None
        ref_path = ctx_dir / "speaker_reference.json"
        if ref_path.exists():
            try:
                import json as _json
                with open(ref_path, "r", encoding="utf-8") as f:
                    speaker = _json.load(f)
                logger.info(f"Loaded reference speaker: {ref_path}")
            except Exception as e:
                logger.warning(f"Failed to load reference speaker: {e}")
        if speaker is None:
            avg_candidates = [
                ctx_dir / "Speaker_avg.json",
                ctx_dir / "speaker_avg.json",
                ctx_dir / "avg_speaker.json",
                ctx_dir / "avg_speaker.jsonl",
            ]
            avg_path = next((p for p in avg_candidates if p.exists()), None)
            if avg_path is not None and avg_path.exists():
                try:
                    import json as _json
                    with open(avg_path, "r", encoding="utf-8") as f:
                        speaker = _json.load(f)
                    logger.info(f"Loaded averaged speaker: {avg_path}")
                except Exception as e:
                    logger.warning(f"Failed to load averaged speaker: {e}")

        inst = cls(
            model_path=base_model_path,
            tokenizer_path=(tokenizer_path or base_model_path),
            lora_dir=str(lora_dir),
            device=device,
            dtype=dtype,
            force_local=force_local,
            temperature=temperature,
            speaker=speaker,
            merge_lora=merge_lora,
        )
        inst._text = text
        inst._n = max(1, int(n_candidates))
        inst._context_dir = str(ctx_dir)
        inst._out = out
        inst._evaluate = bool(evaluate)
        inst._seeds_pool_size = max(n_candidates, int(seeds_pool_size))
        inst._val_sample_size = max(1, int(val_sample_size))
        inst._prosody_override = prosody_override if isinstance(prosody_override, dict) else None
        inst._postproc = postprocess if isinstance(postprocess, dict) else None
        inst._seeds_override = list(seeds_override) if isinstance(seeds_override, list) else None
        inst._dest_dir = str(out_dir) if out_dir else None
        return inst

    def synthesize(self):
        text = getattr(self, "_text", None)
        if not text:
            raise ValueError("No text provided for synthesis")
        n_candidates = getattr(self, "_n", 1)
        out = getattr(self, "_out", None)

        # Prepare output directory
        if getattr(self, "_dest_dir", None):
            dest_dir = Path(self._dest_dir)
        else:
            repo_root = Path(__file__).resolve().parent
            name = Path(getattr(self, "_audio_dir", "default")).name
            dest_dir = repo_root / "output" / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Build output filenames inside dest_dir
        outs = []
        base_stem = Path(out).stem if out else "candidate"
        suffix = (Path(out).suffix if out else ".wav") or ".wav"
        if n_candidates == 1:
            outs = [dest_dir / f"{base_stem}{suffix}"]
        else:
            outs = [dest_dir / f"{base_stem}_{i+1}{suffix}" for i in range(n_candidates)]

        # Build base GenerationConfig components
        speaker_eff = self.speaker if self.speaker is not None else self.interface.load_default_speaker(self.default_speaker)
        # Apply prosody override (energy/spectral_centroid/pitch: 0-100) to speaker dict copy
        if getattr(self, "_prosody_override", None):
            import copy
            ov = {k: int(v) for k, v in self._prosody_override.items() if k in ("energy", "spectral_centroid", "pitch")}
            sp = copy.deepcopy(speaker_eff)
            try:
                if isinstance(sp.get("global_features"), dict):
                    sp["global_features"].update({k: int(v) for k, v in ov.items()})
            except Exception:
                pass
            try:
                if isinstance(sp.get("words"), list):
                    for w in sp["words"]:
                        if isinstance(w, dict):
                            feats = w.get("features")
                            if isinstance(feats, dict):
                                feats.update(ov)
                            else:
                                w["features"] = dict(ov)
            except Exception:
                pass
            speaker_eff = sp
        gen_base = dict(
            text=text,
            speaker=speaker_eff,
            generation_type=outetts.GenerationType.CHUNKED,
            sampler_config=outetts.SamplerConfig(temperature=self.temperature),
            max_length=self.max_length,
        )

        # Optionally evaluate seeds using valid_v3.jsonl and pick best n seeds
        # Respect explicit seed override if provided
        if getattr(self, "_seeds_override", None):
            seeds = list(self._seeds_override)[:n_candidates]
        else:
            seeds = [1000 + i for i in range(self._seeds_pool_size)]
        if getattr(self, "_evaluate", False) and getattr(self, "_seeds_override", None) is None:
            # Try to reuse cached seed scores for this text
            reused = False
            try:
                import json as _json
                scores_path = dest_dir / "seed_scores.json"
                if scores_path.exists():
                    with open(scores_path, "r", encoding="utf-8") as f:
                        cached = _json.load(f)
                    # Support dict format {text, scores} or list of such dicts
                    def _find_entry(obj):
                        if isinstance(obj, dict) and obj.get("text") == text:
                            return obj
                        if isinstance(obj, list):
                            for it in obj:
                                if isinstance(it, dict) and it.get("text") == text:
                                    return it
                        return None
                    entry = _find_entry(cached)
                    if entry and isinstance(entry.get("scores"), list):
                        cached_seeds = [int(i.get("seed")) for i in entry["scores"] if "seed" in i]
                        if cached_seeds:
                            seeds = cached_seeds[:n_candidates]
                            reused = True
                            logger.info(f"Reusing cached seeds for text → {seeds}")
            except Exception as e:
                logger.warning(f"Failed to read cached seed scores: {e}")

            try:
                if not reused:
                    # Prefer context dir for validation data if available
                    valid_base = Path(getattr(self, "_context_dir", getattr(self, "_audio_dir", ".")))
                    best = self._select_best_seeds(valid_base / "valid_v3.jsonl", seeds, self._val_sample_size)
                    if best:
                        # Save seed scores (single-entry format; simple and effective)
                        try:
                            import json as _json
                            scores_path = dest_dir / "seed_scores.json"
                            payload = {
                                "text": text,
                                "scores": [{"seed": s, "score": float(sc)} for s, sc in best]
                            }
                            with open(scores_path, "w", encoding="utf-8") as f:
                                _json.dump(payload, f, indent=2, ensure_ascii=False)
                            logger.info(f"Saved seed scores → {scores_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save seed scores: {e}")
                        seeds = [s for s, _ in best[:n_candidates]]
                        logger.info(f"Selected seeds by validation score: {seeds}")
                    else:
                        seeds = seeds[:n_candidates]
            except Exception as e:
                logger.warning(f"Seed evaluation failed: {e}")
                seeds = seeds[:n_candidates]
        else:
            seeds = seeds[:n_candidates]

        # Generate candidates with chosen seeds
        results = []
        for i, (seed, out_path) in enumerate(zip(seeds, outs)):
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            gen = outetts.GenerationConfig(**gen_base)
            logger.info(f"Generating candidate {i+1}/{len(seeds)} (seed={seed})…")
            output = self.interface.generate(gen)
            # Optional post-processing (normalize/trim/fade)
            try:
                if getattr(self, "_postproc", None):
                    output.audio = self._apply_postprocess(output.audio, 44100)
            except Exception as e:
                logger.warning(f"Post-processing failed: {e}")
            output.save(str(out_path))
            results.append(str(out_path))
        logger.success(f"Generated {len(results)} candidates.")
        return results

    # --- Validation-based seed selection ---
    @staticmethod
    def _extract_text_from_training_prompt(prompt: str) -> str:
        from outetts.version.v3.tokens import SpecialTokens
        st = SpecialTokens()
        start = prompt.find(st.text_start)
        end = prompt.find(st.text_end)
        if start == -1 or end == -1 or end <= start:
            return ""
        return prompt[start + len(st.text_start): end].strip()

    @staticmethod
    def _parse_global_features_from_prompt(prompt: str) -> dict | None:
        # Extract integers from tokens like <|energy_42|><|spectral_centroid_17|><|pitch_63|>
        import re
        feats = {}
        try:
            m = re.search(r"<\|energy_(\d+)\|>", prompt)
            if m:
                feats["energy"] = int(m.group(1))
            m = re.search(r"<\|spectral_centroid_(\d+)\|>", prompt)
            if m:
                feats["spectral_centroid"] = int(m.group(1))
            m = re.search(r"<\|pitch_(\d+)\|>", prompt)
            if m:
                feats["pitch"] = int(m.group(1))
            return feats if feats else None
        except Exception:
            return None

    @staticmethod
    def _l1_distance_feats(a: dict, b: dict) -> float:
        keys = ["energy", "spectral_centroid", "pitch"]
        total = 0.0
        count = 0
        for k in keys:
            if k in a and k in b:
                total += abs(float(a[k]) - float(b[k]))
                count += 1
        return total / max(1, count)

    def _select_best_seeds(self, valid_path: Path, seeds: list[int], val_sample_size: int):
        if not valid_path.exists():
            logger.warning(f"Validation file not found: {valid_path}")
            return []
        # Load a few validation prompts
        import json as _json
        lines = []
        with open(valid_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    lines.append(_json.loads(ln))
                except Exception:
                    pass
        if not lines:
            logger.warning("Validation file is empty or unreadable.")
            return []
        random.shuffle(lines)
        samples = lines[:val_sample_size]

        # Prepare refs by extracting text from training prompts
        refs = [self._extract_text_from_training_prompt(i.get("text", "")) for i in samples]
        refs = [t for t in refs if t]
        if not refs:
            logger.warning("Could not extract any reference text from validation JSONL.")
            return []

        # Prepare reference targets: prefer parsing global features from prompt; fallback to audio features
        targets = []
        for rec in samples:
            prompt = rec.get("text", "")
            feats = self._parse_global_features_from_prompt(prompt)
            if feats:
                targets.append(("feats", feats, None))
                continue
            # Fallback: use reference audio features
            ap = rec.get("audio_path")
            if ap and Path(ap).exists():
                try:
                    # Load audio via interface's codec to ensure consistent preprocessing
                    codec = self.interface.audio_processor.audio_codec
                    audio = codec.load_audio(ap).to(self.interface.audio_processor.device)
                    feats_ref = self.interface.audio_processor.features.extract_audio_features(
                        audio.squeeze(0), codec.sr
                    )
                    targets.append(("feats", feats_ref, None))
                    continue
                except Exception as e:
                    logger.warning(f"Failed loading ref audio features {ap}: {e}\n")
            # If both fail, skip this sample
        if not targets:
            logger.warning("No usable validation targets; skipping evaluation.")
            return []

        # Evaluate each seed by average feature distance over targets
        scored = []
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                total = 0.0
                for kind, target_feats, _ in targets:
                    # Use one of the validation texts for generation
                    ref_text = self._extract_text_from_training_prompt(samples[0].get("text", "")) or ""
                    # Apply the same prosody override during evaluation
                    import copy
                    speaker_eval = self.speaker if self.speaker is not None else self.interface.load_default_speaker(self.default_speaker)
                    if getattr(self, "_prosody_override", None):
                        ov = {k: int(v) for k, v in self._prosody_override.items() if k in ("energy", "spectral_centroid", "pitch")}
                        sp = copy.deepcopy(speaker_eval)
                        try:
                            if isinstance(sp.get("global_features"), dict):
                                sp["global_features"].update({k: int(v) for k, v in ov.items()})
                        except Exception:
                            pass
                        try:
                            if isinstance(sp.get("words"), list):
                                for w in sp["words"]:
                                    if isinstance(w, dict):
                                        feats = w.get("features")
                                        if isinstance(feats, dict):
                                            feats.update(ov)
                                        else:
                                            w["features"] = dict(ov)
                        except Exception:
                            pass
                        speaker_eval = sp
                    gen = outetts.GenerationConfig(
                        text=ref_text,
                        speaker=speaker_eval,
                        generation_type=outetts.GenerationType.CHUNKED,
                        sampler_config=outetts.SamplerConfig(temperature=self.temperature),
                        max_length=self.max_length,
                    )
                    out = self.interface.generate(gen)
                    # Extract features from generated audio tensor at 44.1k
                    gen_audio = out.audio
                    feats_gen = self.interface.audio_processor.features.extract_audio_features(
                        gen_audio.squeeze(0), 44100
                    )
                    total += self._l1_distance_feats(target_feats, feats_gen)
                avg = total / len(targets)
                scored.append((seed, avg))
            except Exception as e:
                logger.warning(f"Seed {seed} evaluation failed: {e}")
        # Sort by score ascending (lower is better)
        scored.sort(key=lambda x: x[1])
        return scored

    # --- Single-audio based seed selection ---
    def _build_speaker_from_audio(self, audio_path: str) -> dict:
        """Create a speaker dict by extracting global features from a single audio.

        This takes the default speaker as a base and updates its global_features
        with the features extracted from the provided audio sample.
        """
        base = self.speaker if self.speaker is not None else self.interface.load_default_speaker(self.default_speaker)
        import copy
        sp = copy.deepcopy(base)
        codec = self.interface.audio_processor.audio_codec
        audio = codec.load_audio(str(audio_path)).to(self.interface.audio_processor.device)
        feats_ref = self.interface.audio_processor.features.extract_audio_features(audio.squeeze(0), codec.sr)
        try:
            if isinstance(sp.get("global_features"), dict):
                sp["global_features"].update(feats_ref)
            else:
                sp["global_features"] = dict(feats_ref)
        except Exception:
            sp["global_features"] = dict(feats_ref)
        return sp

    def evaluate_seeds_with_single_audio(self,
                                         audio_path: str,
                                         seeds: list[int] | None = None,
                                         ref_text: str | None = None) -> list[tuple[int, float]]:
        """Rank seeds by distance to features extracted from a single reference audio.

        - Builds a temporary speaker from `audio_path` (global_features set from audio).
        - Generates audio with each seed using `ref_text` (or self._text) and computes
          L1 distance to the reference audio features.
        - Returns list of (seed, score) sorted ascending (lower is better).
        """
        # Prepare seeds
        if seeds is None:
            seeds = [1000 + i for i in range(getattr(self, "_seeds_pool_size", 12))]
        # Build reference features from audio
        codec = self.interface.audio_processor.audio_codec
        audio = codec.load_audio(str(audio_path)).to(self.interface.audio_processor.device)
        feats_ref = self.interface.audio_processor.features.extract_audio_features(audio.squeeze(0), codec.sr)
        # Build generation speaker from the same audio
        speaker_eval = self._build_speaker_from_audio(audio_path)
        # Choose a text for evaluation
        text_eval = ref_text or getattr(self, "_text", None) or "안녕하세요."

        scored: list[tuple[int, float]] = []
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                gen = outetts.GenerationConfig(
                    text=text_eval,
                    speaker=speaker_eval,
                    generation_type=outetts.GenerationType.CHUNKED,
                    sampler_config=outetts.SamplerConfig(temperature=self.temperature),
                    max_length=self.max_length,
                )
                out = self.interface.generate(gen)
                gen_audio = out.audio
                feats_gen = self.interface.audio_processor.features.extract_audio_features(
                    gen_audio.squeeze(0), 44100
                )
                dist = self._l1_distance_feats(feats_ref, feats_gen)
                scored.append((seed, dist))
            except Exception as e:
                logger.warning(f"Seed {seed} evaluation (single audio) failed: {e}")
        scored.sort(key=lambda x: x[1])
        return scored

    # --- Build/save speaker from a generated sample (seed-locked) ---
    def build_speaker_from_seed(self, text: str, seed: int) -> dict:
        """Generate once with a specific seed and build a speaker profile from it.

        - Uses the current default speaker as a base.
        - Extracts global features from the generated audio and sets them in the speaker dict.
        """
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        base = self.speaker if self.speaker is not None else self.interface.load_default_speaker(self.default_speaker)
        import copy
        sp = copy.deepcopy(base)
        gen = outetts.GenerationConfig(
            text=text,
            speaker=base,
            generation_type=outetts.GenerationType.CHUNKED,
            sampler_config=outetts.SamplerConfig(temperature=self.temperature),
            max_length=self.max_length,
        )
        out = self.interface.generate(gen)
        gen_audio = out.audio
        feats = self.interface.audio_processor.features.extract_audio_features(gen_audio.squeeze(0), 44100)
        try:
            if isinstance(sp.get("global_features"), dict):
                sp["global_features"].update(feats)
            else:
                sp["global_features"] = dict(feats)
        except Exception:
            sp["global_features"] = dict(feats)
        return sp

    def save_speaker_from_seed(self, text: str, seed: int, dest_path: str) -> str:
        """Build a speaker from a seed-generated sample and save to JSON."""
        sp = self.build_speaker_from_seed(text, seed)
        import json as _json
        p = Path(dest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            _json.dump(sp, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved speaker from best seed={seed} → {p}")
        return str(p)

    # --- Audio post-processing ---
    def _apply_postprocess(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        cfg = self._postproc or {}
        x = audio.detach().clone()
        # Ensure 1D waveform for processing (mono)
        if x.dim() == 2:
            # [C, S] -> take first channel
            x = x[0]
        elif x.dim() > 2:
            x = x.flatten()

        # Trim silence based on threshold in dBFS (e.g., -45.0)
        trim_db = cfg.get("trim_db")
        if isinstance(trim_db, (int, float)):
            thr = 10.0 ** (float(trim_db) / 20.0)
            thr = abs(thr)
            a = x.abs()
            idx = (a > thr).nonzero(as_tuple=False).squeeze()
            if idx.numel() > 0:
                start = int(idx[0].item())
                end = int(idx[-1].item()) + 1
                x = x[start:end]

        # RMS normalization to target dBFS (e.g., -20.0)
        norm_db = cfg.get("normalize_rms_db")
        if isinstance(norm_db, (int, float)):
            rms = float((x.pow(2).mean().sqrt()).item()) + 1e-12
            target = 10.0 ** (float(norm_db) / 20.0)
            gain = target / rms
            # Avoid excessive gain
            gain = max(min(gain, 50.0), 0.02)
            x = x * gain

        # Simple soft clip to prevent overflow
        x = torch.clamp(x, -0.999, 0.999)

        # Apply short fade in/out (ms)
        fade_ms = cfg.get("fade_ms")
        if isinstance(fade_ms, (int, float)) and fade_ms > 0:
            n = int(sr * (float(fade_ms) / 1000.0))
            n = max(1, min(n, x.numel() // 2))
            if n > 0:
                ramp_in = torch.linspace(0.0, 1.0, steps=n, device=x.device)
                ramp_out = torch.linspace(1.0, 0.0, steps=n, device=x.device)
                x[:n] = x[:n] * ramp_in
                x[-n:] = x[-n:] * ramp_out

        # Return to 2D [1, S]
        return x.unsqueeze(0)


__all__ = ["LoraInference"]
