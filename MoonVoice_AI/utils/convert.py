#!/usr/bin/env python3
"""
Simple MP4 -> WAV converter using ffmpeg.

Single file usage:
  python convert.py input.mp4 [-o output.wav] [--sr 16000] [--ch 1] [-y]

Directory (batch) usage:
  python convert.py /path/to/dir [--outdir ./wavs] [--recursive] [--pattern "*.mp4"] [--sr 16000] [--ch 1] [-y]

Requires ffmpeg installed and available on PATH.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import Any

class Converter:
    def __init__(self):
        # 경로들을 나중에 디렉토리 위치를 정하게 되면 절대경로로 바꿀 것
        self.check_ffmpeg()

        # self.main()
        # p.mkdir(parents=True, exist_ok=True)


    def check_ffmpeg(self) -> None:
        if which("ffmpeg") is None:
            raise RuntimeError(f"Error: ffmpeg is not installed or not on PATH.\n"
                               f"Install ffmpeg and try again. Example: https://ffmpeg.org/download.html"
                               f"{sys.stderr}")

    def convert_to_wav(self, input_path: Path, output_path: Path, overwrite: bool = False) -> Path:
        """
        MP4 -> WAV (PCM 16-bit, 24000 Hz, mono)
        ?? ??: self.dir_wavs/<basename>.wav
        """

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        # ?? ?? ??: self.dir_wavs/<basename>.wav
        out_path = output_path / f"{input_path.stem}.wav"
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            ("-y" if overwrite else "-n"),
            "-i", str(input_path),
            "-vn",  # ??? ??
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "24000",  # 24 kHz
            "-ac", "1",  # mono
            str(out_path),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg failed (exit {e.returncode}). Command: {shlex.join(cmd)}"
            ) from e

        return out_path

    def check_wav(self, arrival) -> list[Path] | None:
        dir_arrival : Path = Path(arrival)
        dir_wavs = Path(f"{dir_arrival.parent.parent}/wavs/{dir_arrival.name}")
        dir_etc = Path(f"{dir_arrival.parent.parent}/etc/{dir_arrival.name}")

        arrival_wav = []
        arrival_others = []

        for f in dir_arrival.iterdir():  # glob("*")?? iterdir()? ?? ??
            if not f.is_file():
                continue
            (arrival_wav if f.suffix.lower() == ".wav" else arrival_others).append(f)

        if len(arrival_wav) != 30 and len(arrival_wav) > 0:
            raise RuntimeError(f"Error: File {len(arrival_wav)} does not contain a .wav file.\n")
        elif len(arrival_wav) == 0:
            dir_etc.mkdir(parents=True, exist_ok=True)
            for f in arrival_others:
                self.convert_to_wav(f,dir_wavs)
                shutil.move(str(f), dir_etc)
            dir_arrival.rmdir()
        else:
            dir_wavs.mkdir(parents=True, exist_ok=True)
            for f in arrival_wav:
                shutil.move(str(f), dir_wavs)
            dir_arrival.rmdir()

Converter = Converter()
Converter.check_wav("/home/server1/AI2/OuteTTS/datas/arrival_point/choi")
