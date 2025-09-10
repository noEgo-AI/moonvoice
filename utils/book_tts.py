"""
Chapter-level driver to synthesize a list of sentences with a fixed voice identity.

Typical use:
  from book_tts import synthesize_chapter
  synthesize_chapter(
      audio_dir='datas/wavs/lsy',
      text_lines=['첫 문장', '둘째 문장'],
      evaluate_once=True,
      n_candidates_per_sentence=1,
      prosody_override={'energy': 55, 'pitch': 45},
      postprocess={'normalize_rms_db': -20.0, 'trim_db': -45.0, 'fade_ms': 8},
  )

Or load from a file (one sentence per line):
  synthesize_chapter_from_file('datas/wavs/lsy', 'chapter1.txt')
"""

from pathlib import Path
from typing import Iterable, List, Optional
from loguru import logger

from utils.Inference import LoraInference

import os
import multiprocessing as _mp
from pathlib import Path as _Path

def _eval_worker(args):
    """Evaluate a chunk of seeds and return (seed, score) pairs."""
    (audio_dir, probe, seeds_chunk, val_sample_size, prosody_override) = args
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from utils.Inference import LoraInference as _LI
    inst = _LI.from_audio_dir(
        audio_dir,
        text=probe,
        n_candidates=1,
        evaluate=False,              # only load model
        seeds_pool_size=len(seeds_chunk),
        val_sample_size=val_sample_size,
        prosody_override=prosody_override,
        postprocess=None,
    )
    valid_path = _Path(audio_dir) / "valid_v3.jsonl"
    best = inst._select_best_seeds(valid_path, list(seeds_chunk), val_sample_size)  # list[(seed, score)]
    return [(int(s), float(sc)) for s, sc in (best or [])]


def _synth_worker(args):
    """Synthesize a chunk of (idx, text, out_name) with a single loaded model per worker.

    args:
      - audio_dir: base audio dir containing lora_hparams.json
      - items: list of tuples (idx, text, out_name)
      - n_candidates: candidates per sentence
      - seeds: list[int] to use (top-N already selected)
      - seeds_pool_size / val_sample_size: kept for parity (not used when evaluate=False)
      - prosody_override / postprocess: same as top-level
    """
    (
        audio_dir,
        items,
        n_candidates,
        seeds,
        seeds_pool_size,
        val_sample_size,
        prosody_override,
        postprocess,
    ) = args
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from utils.Inference import LoraInference as _LI
    results: list[str] = []
    if not items:
        return results
    # Load once per worker with the first sentence; we will override text/out per item
    first_text = items[0][1]
    inst = _LI.from_audio_dir(
        audio_dir,
        text=first_text,
        n_candidates=n_candidates,
        out=None,
        evaluate=False,
        seeds_pool_size=seeds_pool_size,
        val_sample_size=val_sample_size,
        prosody_override=prosody_override,
        postprocess=postprocess,
        seeds_override=(seeds[:n_candidates] if seeds else None),
    )
    for _, text, out_name in items:
        # Update per-utterance fields then synthesize
        inst._text = text
        inst._n = int(n_candidates)
        inst._out = out_name
        inst._evaluate = False
        if seeds:
            inst._seeds_override = list(seeds[:n_candidates])
        try:
            out_paths = inst.synthesize()
            results.extend(out_paths)
        except Exception as e:
            logger.warning(f"Worker synthesis failed for '{text[:24]}…': {e}")
    return results


def _eval_worker_paths(args):
    """Evaluate seeds for path-based mode using base_model_path + model_path.

    args:
      - base_model_path: path to base LM
      - model_path: path to LoRA adapter directory (also holds JSON files)
      - probe: text
      - seeds_chunk: list[int]
      - val_sample_size: int
      - prosody_override: dict | None
      - out_dir: str | None to store seed_scores (unused here but kept for parity)
    """
    (base_model_path, model_path, probe, seeds_chunk, val_sample_size, prosody_override, out_dir) = args
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from utils.Inference import LoraInference as _LI
    inst = _LI.from_paths(
        base_model_path=base_model_path,
        lora_dir=model_path,
        text=probe,
        n_candidates=1,
        evaluate=False,              # only load model
        seeds_pool_size=len(seeds_chunk),
        val_sample_size=val_sample_size,
        prosody_override=prosody_override,
        postprocess=None,
        context_dir=model_path,
        out_dir=out_dir,
    )
    valid_path = _Path(model_path) / "valid_v3.jsonl"
    best = inst._select_best_seeds(valid_path, list(seeds_chunk), val_sample_size)  # list[(seed, score)]
    return [(int(s), float(sc)) for s, sc in (best or [])]


def _load_sentences(text_file: Path) -> List[str]:
    lines = []
    with text_file.open('r', encoding='utf-8') as f:
        f.readline()
        for ln in f:
            s = ln.strip()
            if s:
                lines.append(s)
    return lines


def synthesize_chapter(
    audio_dir: str,
    text_lines: Iterable[str],
    *,
    n_candidates_per_sentence: int = 1,
    evaluate_once: bool = True,
    # When True and a cached seed_scores.json exists, reuse it and skip evaluation
    skip_eval_if_cached: bool = True,
    # When True, after evaluation determines the best seed, save a speaker built from it
    save_best_speaker: bool = False,
    # Optional custom filename for saved speaker (under audio_dir)
    best_speaker_filename: Optional[str] = None,
    evaluate_seed_text: Optional[str] = None,
    seeds_pool_size: int = 5,
    val_sample_size: int = 3,
    prosody_override: Optional[dict] = None,
    postprocess: Optional[dict] = None,
    synth_workers: Optional[int] = None,
    eval_workers: Optional[int] = None,
) -> list[str]:
    audio_dir = str(audio_dir)
    sentences = [s for s in text_lines if isinstance(s, str) and s.strip()]
    if not sentences:
        logger.warning('No sentences provided. Nothing to do.')
        return []

    # Evaluate once on a representative text to pick best seeds, then reuse
    seeds = None
    if evaluate_once:
        probe = evaluate_seed_text or sentences[0]
        repo_root = Path(__file__).resolve().parent
        dest_dir = repo_root / 'output' / Path(audio_dir).name
        scores_path = dest_dir / 'seed_scores.json'
        # If requested and cache exists, reuse and skip evaluation
        if skip_eval_if_cached and scores_path.exists():
            import json as _json
            try:
                with open(scores_path, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                scores = data.get('scores', []) if isinstance(data, dict) else []
                seeds = [int(i.get('seed')) for i in scores if 'seed' in i]
                if seeds:
                    logger.info(f"Found cached seed scores, skipping evaluation → {scores_path}")
                else:
                    logger.warning('Cached seed_scores.json has no seeds; will evaluate.')
            except Exception as e:
                logger.warning(f'Failed to read cached seed scores: {e}; will evaluate.')

        if not seeds:
            logger.info('Evaluating seeds once using a probe sentence...')
            # Prepare seed list identical to LoraInference default
            all_seeds = [1000 + i for i in range(seeds_pool_size)]
            if eval_workers is None:
                workers = max(1, int(os.getenv("EVAL_WORKERS", "2")))
            else:
                workers = max(1, int(eval_workers))
            if workers == 1 or len(all_seeds) <= 1:
                # Single-process evaluation without synthesis; compute and save seed scores
                best_list = _eval_worker((audio_dir, probe, all_seeds, val_sample_size, prosody_override))
                merged = list(best_list or [])
                merged.sort(key=lambda x: x[1])  # lower distance is better
                dest_dir.mkdir(parents=True, exist_ok=True)
                import json as _json
                payload = {"text": probe, "scores": [{"seed": s, "score": float(sc)} for s, sc in merged]}
                with open(scores_path, 'w', encoding='utf-8') as f:
                    _json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved seed scores (single-process) → {scores_path}")
            else:
                # Round-robin split for better load balance
                chunks = [all_seeds[i::workers] for i in range(workers) if all_seeds[i::workers]]
                ctx = _mp.get_context("spawn")
                with ctx.Pool(processes=min(workers, len(chunks))) as pool:
                    parts = pool.map(_eval_worker, [(audio_dir, probe, ch, val_sample_size, prosody_override) for ch in chunks])
                # Merge and save
                merged = []
                for lst in parts:
                    if lst:
                        merged.extend(lst)
                merged.sort(key=lambda x: x[1])  # lower distance is better
                dest_dir.mkdir(parents=True, exist_ok=True)
                import json as _json
                payload = {"text": probe, "scores": [{"seed": s, "score": float(sc)} for s, sc in merged]}
                with open(scores_path, 'w', encoding='utf-8') as f:
                    _json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info(f"Merged seed scores → {scores_path}")
            # Read saved scores and take top-N seeds
            if scores_path.exists():
                import json as _json
                try:
                    with open(scores_path, 'r', encoding='utf-8') as f:
                        data = _json.load(f)
                    # Expect {text, scores: [{seed, score}]}
                    scores = data.get('scores', []) if isinstance(data, dict) else []
                    seeds = [int(i.get('seed')) for i in scores if 'seed' in i]
                    logger.info(f'Reusing top seeds for chapter: {seeds}')
                except Exception as e:
                    logger.warning(f'Failed to read seed scores: {e}')

        # Optionally, build and save a speaker profile from the best seed
        if save_best_speaker and seeds:
            try:
                best_seed = int(seeds[0])
                probe_text = evaluate_seed_text or sentences[0]
                # Resolve destination path (under audio_dir)
                audio_dir_path = Path(audio_dir)
                default_name = best_speaker_filename or f"speaker_from_best_seed_{best_seed}.json"
                dest_path = audio_dir_path / default_name
                # Build an inference instance to generate once and extract features
                inst_for_spk = LoraInference.from_audio_dir(
                    audio_dir,
                    text=probe_text,
                    n_candidates=1,
                    evaluate=False,
                    seeds_pool_size=seeds_pool_size,
                    val_sample_size=val_sample_size,
                    prosody_override=prosody_override,
                    postprocess=None,
                )
                inst_for_spk.save_speaker_from_seed(probe_text, best_seed, str(dest_path))
            except Exception as e:
                logger.warning(f"Failed to save speaker from best seed: {e}")
    # Determine synthesis workers (default from env or 1)
    if synth_workers is None:
        synth_workers = int(os.getenv("SYNTH_WORKERS", os.getenv("GEN_WORKERS", "1")))

    outputs: list[str] = []
    if synth_workers <= 1 or len(sentences) == 1:
        # Single-process fallback
        for idx, s in enumerate(sentences, start=1):
            base_name = f"{idx:04d}.wav"
            res = LoraInference.from_audio_dir(
                audio_dir,
                text=s,
                n_candidates=n_candidates_per_sentence,
                out=base_name,
                evaluate=(not evaluate_once),
                seeds_pool_size=seeds_pool_size,
                val_sample_size=val_sample_size,
                prosody_override=prosody_override,
                postprocess=postprocess,
                seeds_override=(seeds[:n_candidates_per_sentence] if seeds else None),
            ).synthesize()
            outputs.extend(res)
        return outputs

    # Multi-process: split sentences into chunks and synthesize in parallel
    items = [(i, s, f"{i:04d}.wav") for i, s in enumerate(sentences, start=1)]
    # Round-robin split for load balance
    chunks = [items[i::synth_workers] for i in range(synth_workers) if items[i::synth_workers]]
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=min(synth_workers, len(chunks))) as pool:
        parts = pool.map(
            _synth_worker,
            [
                (
                    audio_dir,
                    ch,
                    n_candidates_per_sentence,
                    (seeds[:n_candidates_per_sentence] if seeds else None),
                    seeds_pool_size,
                    val_sample_size,
                    prosody_override,
                    postprocess,
                )
                for ch in chunks
            ],
        )
    for lst in parts:
        if lst:
            outputs.extend(lst)
    return outputs


def synthesize_chapter_from_file(
    audio_dir: str,
    text_file: str,
    *,
    skip_eval_if_cached: bool = True,
    save_best_speaker: bool = False,
    best_speaker_filename: Optional[str] = None,
    eval_workers: Optional[int] = None,
    # New params for explicit path-based mode
    base_model_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    usercode: Optional[str] = None,
    **kwargs,
) -> list[str]:
    lines = _load_sentences(Path(text_file))
    # If base_model_path and model_path provided, use explicit path-based synthesis
    if base_model_path and model_path:
        return synthesize_chapter_paths(
            base_model_path=base_model_path,
            model_path=model_path,
            text_lines=lines,
            output_path=output_path,
            usercode=usercode,
            skip_eval_if_cached=skip_eval_if_cached,
            save_best_speaker=save_best_speaker,
            best_speaker_filename=best_speaker_filename,
            eval_workers=eval_workers,
            **kwargs,
        )
    # Fallback to audio_dir-based flow (backward compatible)
    return synthesize_chapter(
        audio_dir,
        lines,
        skip_eval_if_cached=skip_eval_if_cached,
        save_best_speaker=save_best_speaker,
        best_speaker_filename=best_speaker_filename,
        eval_workers=eval_workers,
        **kwargs,
    )


def synthesize_chapter_paths(
    *,
    base_model_path: str,
    model_path: str,
    text_lines: Iterable[str],
    output_path: Optional[str] = None,
    usercode: Optional[str] = None,
    n_candidates_per_sentence: int = 1,
    evaluate_once: bool = True,
    skip_eval_if_cached: bool = True,
    save_best_speaker: bool = False,
    best_speaker_filename: Optional[str] = None,
    evaluate_seed_text: Optional[str] = None,
    seeds_pool_size: int = 5,
    val_sample_size: int = 3,
    prosody_override: Optional[dict] = None,
    postprocess: Optional[dict] = None,
    synth_workers: Optional[int] = None,
    eval_workers: Optional[int] = None,
) -> list[str]:
    sentences = [s for s in text_lines if isinstance(s, str) and s.strip()]
    if not sentences:
        logger.warning('No sentences provided. Nothing to do.')
        return []

    # Resolve output directory
    out_dir = None
    if output_path:
        out_dir = Path(output_path) / (usercode or "")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model directory (adapter + json live under model_path/<usercode>)
    model_dir = Path(model_path) / (usercode or "") if usercode else Path(model_path)

    # Evaluate once to pick best seeds, then reuse
    seeds = None
    if evaluate_once:
        probe = evaluate_seed_text or sentences[0]
        # Seed score cache under the actual output directory if available
        if out_dir:
            scores_path = out_dir / 'seed_scores.json'
        else:
            # fallback to repo-root output/model_path.name
            repo_root = Path(__file__).resolve().parent
            scores_path = (repo_root / 'output' / model_dir.name / 'seed_scores.json')

        if skip_eval_if_cached and scores_path.exists():
            import json as _json
            try:
                with open(scores_path, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                scores = data.get('scores', []) if isinstance(data, dict) else []
                seeds = [int(i.get('seed')) for i in scores if 'seed' in i]
                if seeds:
                    logger.info(f"Found cached seed scores, skipping evaluation → {scores_path}")
                else:
                    logger.warning('Cached seed_scores.json has no seeds; will evaluate.')
            except Exception as e:
                logger.warning(f'Failed to read cached seed scores: {e}; will evaluate.')

        if not seeds:
            logger.info('Evaluating seeds once using a probe sentence...')
            all_seeds = [1000 + i for i in range(seeds_pool_size)]
            if eval_workers is None:
                workers = max(1, int(os.getenv("EVAL_WORKERS", "2")))
            else:
                workers = max(1, int(eval_workers))
            if workers == 1 or len(all_seeds) <= 1:
                # Single-process evaluation without synthesis; compute and save seed scores
                best_list = _eval_worker_paths((
                    base_model_path,
                    str(model_dir),
                    probe,
                    all_seeds,
                    val_sample_size,
                    prosody_override,
                    (str(out_dir) if out_dir else None),
                ))
                merged = list(best_list or [])
                merged.sort(key=lambda x: x[1])
                (out_dir or (Path(__file__).resolve().parent / 'output' / Path(model_path).name)).mkdir(parents=True, exist_ok=True)
                import json as _json
                payload = {"text": probe, "scores": [{"seed": s, "score": float(sc)} for s, sc in merged]}
                with open(scores_path, 'w', encoding='utf-8') as f:
                    _json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved seed scores (single-process) → {scores_path}")
            else:
                chunks = [all_seeds[i::workers] for i in range(workers) if all_seeds[i::workers]]
                ctx = _mp.get_context("spawn")
                with ctx.Pool(processes=min(workers, len(chunks))) as pool:
                    parts = pool.map(
                        _eval_worker_paths,
                        [
                            (
                                base_model_path,
                                str(model_dir),
                                probe,
                                ch,
                                val_sample_size,
                                prosody_override,
                                (str(out_dir) if out_dir else None),
                            )
                            for ch in chunks
                        ],
                    )
                merged = []
                for lst in parts:
                    if lst:
                        merged.extend(lst)
                merged.sort(key=lambda x: x[1])
                (out_dir or (Path(__file__).resolve().parent / 'output' / Path(model_path).name)).mkdir(parents=True, exist_ok=True)
                import json as _json
                payload = {"text": probe, "scores": [{"seed": s, "score": float(sc)} for s, sc in merged]}
                with open(scores_path, 'w', encoding='utf-8') as f:
                    _json.dump(payload, f, indent=2, ensure_ascii=False)
                logger.info(f"Merged seed scores → {scores_path}")
            if scores_path.exists():
                import json as _json
                try:
                    with open(scores_path, 'r', encoding='utf-8') as f:
                        data = _json.load(f)
                    scores = data.get('scores', []) if isinstance(data, dict) else []
                    seeds = [int(i.get('seed')) for i in scores if 'seed' in i]
                    logger.info(f'Reusing top seeds for chapter: {seeds}')
                except Exception as e:
                    logger.warning(f'Failed to read seed scores: {e}')

        # Saving best speaker in paths-mode: save under context_dir (model_dir)
        if save_best_speaker and seeds:
            try:
                best_seed = int(seeds[0])
                probe_text = evaluate_seed_text or sentences[0]
                default_name = best_speaker_filename or f"speaker_from_best_seed_{best_seed}.json"
                dest_path = model_dir / default_name
                inst_for_spk = LoraInference.from_paths(
                    base_model_path=base_model_path,
                    lora_dir=str(model_dir),
                    text=probe_text,
                    n_candidates=1,
                    evaluate=False,
                    seeds_pool_size=seeds_pool_size,
                    val_sample_size=val_sample_size,
                    prosody_override=prosody_override,
                    postprocess=None,
                    context_dir=str(model_dir),
                    out_dir=str(out_dir) if out_dir else None,
                )
                inst_for_spk.save_speaker_from_seed(probe_text, best_seed, str(dest_path))
            except Exception as e:
                logger.warning(f"Failed to save speaker from best seed: {e}")

    if synth_workers is None:
        synth_workers = int(os.getenv("SYNTH_WORKERS", os.getenv("GEN_WORKERS", "1")))

    outputs: list[str] = []
    if synth_workers <= 1 or len(sentences) == 1:
        for idx, s in enumerate(sentences, start=1):
            base_name = f"{idx:04d}.wav"
            res = LoraInference.from_paths(
                base_model_path=base_model_path,
                lora_dir=str(model_dir),
                text=s,
                n_candidates=n_candidates_per_sentence,
                out=base_name,
                evaluate=False,
                seeds_pool_size=seeds_pool_size,
                val_sample_size=val_sample_size,
                prosody_override=prosody_override,
                postprocess=postprocess,
                seeds_override=(seeds[:n_candidates_per_sentence] if seeds else None),
                context_dir=str(model_dir),
                out_dir=str(out_dir) if out_dir else None,
            ).synthesize()
            outputs.extend(res)
        return outputs

    items = [(i, s, f"{i:04d}.wav") for i, s in enumerate(sentences, start=1)]
    chunks = [items[i::synth_workers] for i in range(synth_workers) if items[i::synth_workers]]
    ctx = _mp.get_context("spawn")
    args_list = [
        (
            base_model_path,
            str(model_dir),
            ch,
            n_candidates_per_sentence,
            (seeds[:n_candidates_per_sentence] if seeds else None),
            seeds_pool_size,
            val_sample_size,
            prosody_override,
            postprocess,
            str(out_dir) if out_dir else None,
        )
        for ch in chunks
    ]
    with ctx.Pool(processes=min(synth_workers, len(chunks))) as pool:
        # Use starmap since _synth_worker_paths expects multiple positional args
        parts = pool.starmap(_synth_worker_paths, args_list)
    for lst in parts:
        if lst:
            outputs.extend(lst)
    return outputs


def _synth_worker_paths(base_model_path, model_path, items, n_candidates, seeds, seeds_pool_size, val_sample_size, prosody_override, postprocess, out_dir):
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    results: list[str] = []
    if not items:
        return results
    first_text = items[0][1]
    inst = LoraInference.from_paths(
        base_model_path=base_model_path,
        lora_dir=model_path,
        text=first_text,
        n_candidates=n_candidates,
        out=None,
        evaluate=False,
        seeds_pool_size=seeds_pool_size,
        val_sample_size=val_sample_size,
        prosody_override=prosody_override,
        postprocess=postprocess,
        seeds_override=(seeds[:n_candidates] if seeds else None),
        context_dir=model_path,
        out_dir=out_dir,
    )
    for _, text, out_name in items:
        inst._text = text
        inst._n = int(n_candidates)
        inst._out = out_name
        inst._evaluate = False
        if seeds:
            inst._seeds_override = list(seeds[:n_candidates])
        try:
            out_paths = inst.synthesize()
            results.extend(out_paths)
        except Exception as e:
            logger.warning(f"Worker synthesis failed for '{text[:24]}…': {e}")
    return results


__all__ = ['synthesize_chapter', 'synthesize_chapter_from_file']
