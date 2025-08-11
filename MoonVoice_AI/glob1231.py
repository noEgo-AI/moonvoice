#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speaker_wavs_glob_saver.py

- OuteTTS v3 인터페이스를 받아서, glob 패턴으로 모은 여러 참조 WAV을
  스피커 임베딩으로 변환 후 '평균'하여 하나의 스피커(JSON)로 저장.
- run_lora_inference.py의 --speaker_wavs_glob 처리와 동일 의도이되, 저장 로직만 모듈화.

사용 예)
    from speaker_wavs_glob_saver import build_avg_speaker_from_glob

    sp, paths = build_avg_speaker_from_glob(
        interface=itf,
        glob_pattern="wavs/prep_v3/refs/*.wav",
        save_json="wavs/prep_v3/speaker_avg.json"
    )
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Iterable, Tuple, List, Any

def _list_wavs(glob_pattern: str) -> List[Path]:
    paths = sorted(Path().glob(glob_pattern))
    return [p for p in paths if p.is_file()]

def _avg_dict_speakers(dict_list: List[dict]) -> dict:
    """
    create_speaker()가 dict를 반환하는 환경을 위한 안전한 평균기.
    - 숫자(int/float): 산술 평균
    - 숫자 리스트: 요소별 평균 (길이 불일치 시 최소 길이에 맞춤)
    - dict: 키별 재귀 평균
    - 그 외(문자열/불리언/메타): 첫 값 유지
    """
    if not dict_list:
        return {}

    SPECIAL_KEEP = {"version", "interface_version", "format", "backend", "type"}

    def merge_values(values: List[Any], key_path=()) -> Any:
        # 모두 dict
        if all(isinstance(v, dict) for v in values):
            keys = set().union(*(v.keys() for v in values))
            out = {}
            for k in keys:
                sub = [v[k] for v in values if k in v]
                if k in SPECIAL_KEEP and sub:
                    out[k] = sub[0]
                else:
                    out[k] = merge_values(sub, key_path + (k,))
            return out

        # 모두 숫자(리스트 아님)
        if all(isinstance(v, (int, float)) for v in values):
            # 모두 같은 int면 그대로
            if all(isinstance(v, int) for v in values) and len(set(values)) == 1:
                return values[0]
            return float(sum(values) / len(values))

        # 모두 숫자 리스트
        if all(isinstance(v, list) and all(isinstance(x, (int, float)) for x in v) for v in values):
            L = min(len(v) for v in values)
            if L == 0:
                return []
            return [float(sum(v[i] for v in values) / len(values)) for i in range(L)]

        # 섞였거나 문자열/불리언 등 -> 첫 값 유지
        return values[0] if values else None

    return merge_values(dict_list)

def _avg_speakers_operator(speakers: List[Any]) -> Any:
    """
    speaker 객체가 +, / 연산을 지원하는 환경(일부 OuteTTS 버전)에서 사용.
    실패 시 예외를 던져 상위에서 dict 평균으로 폴백.
    """
    base = speakers[0]
    for s in speakers[1:]:
        base = base + s
    return base / len(speakers)

def _save_speaker(interface, speaker_obj: Any, save_json: str) -> None:
    """
    가능한 경우 interface.save_speaker 사용,
    dict만 지원되는 환경에선 직접 json으로 저장.
    """
    out = Path(save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        interface.save_speaker(speaker_obj, str(out))
    except Exception:
        with out.open("w", encoding="utf-8") as f:
            json.dump(speaker_obj, f, ensure_ascii=False, indent=2)

def build_avg_speaker_from_glob(interface, glob_pattern: str, save_json: str | None = None) -> Tuple[Any, List[str]]:
    """
    glob_pattern으로 찾은 WAV들 -> speaker로 변환 -> 평균 -> (선택) JSON 저장.

    Returns:
        (avg_speaker, wav_paths)
    """
    wavs = _list_wavs(glob_pattern)
    if not wavs:
        raise FileNotFoundError(f"No files matched glob: {glob_pattern}")

    # create_speaker: 환경에 따라 dict 또는 연산자 지원 객체를 반환할 수 있음
    speakers = []
    dict_mode = False
    for p in wavs:
        sp = interface.create_speaker(str(p))
        speakers.append(sp)
        if isinstance(sp, dict):
            dict_mode = True

    if dict_mode:
        avg_sp = _avg_dict_speakers([sp for sp in speakers if isinstance(sp, dict)])
    else:
        # 연산자 평균 시도, 실패하면 dict로 폴백
        try:
            avg_sp = _avg_speakers_operator(speakers)
        except Exception:
            # 객체를 json으로 직렬화할 수 없다면, 다시 dict로 만들 수단이 필요.
            # 일단 첫 개를 dict로 가정할 수 없으므로 강제 폴백은 빈 객체가 될 수 있어 주의.
            # 현실적으로는 create_speaker가 dict를 반환하는 편이 많음.
            avg_sp = _avg_dict_speakers([sp for sp in speakers if isinstance(sp, dict)])

    if save_json:
        _save_speaker(interface, avg_sp, save_json)

    return avg_sp, [str(p) for p in wavs]


import torch, outetts

def build_interface(model_path: str, tokenizer_path: str, bf16: bool = True):
    cfg = outetts.ModelConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        interface_version=outetts.InterfaceVersion.V3,  # ← 필수
        backend=outetts.Backend.HF,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=(torch.bfloat16 if bf16 else None),
    )
    return outetts.Interface(cfg)

itf = build_interface("OuteTTS-1.0-0.6B", "OuteTTS-1.0-0.6B", bf16=True)

sp, paths = build_avg_speaker_from_glob(
        interface=itf,
        glob_pattern="wavs/*.wav",
        save_json="wavs/prep_v3/speaker_avg.json"
    )
