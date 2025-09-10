import asyncio
import json
import os
import sys
from typing import Any, Dict

from aiohttp import web

# Ensure repo root on sys.path to import voice_interface
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from voice_interface import VoiceTTSHandler  # noqa: E402


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


def handler_from_env() -> VoiceTTSHandler:
    device = os.environ.get("AI_DEVICE")
    dtype = os.environ.get("AI_DTYPE", "bf16")
    seeds_pool_size = int(os.environ.get("AI_SEEDS_POOL", "12") or 12)
    val_sample_size = int(os.environ.get("AI_VAL_SAMPLE", "3") or 3)
    default_temperature = float(os.environ.get("AI_TEMP", "0.4") or 0.4)
    return VoiceTTSHandler(
        device=device,
        dtype=dtype,
        seeds_pool_size=seeds_pool_size,
        val_sample_size=val_sample_size,
        default_temperature=default_temperature,
    )


async def handle_run(request: web.Request) -> web.Response:
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        return web.json_response({"status": "error", "stdout": "", "stderr": "invalid json"}, status=400)

    action = payload.get("action")
    v = handler_from_env()
    stdout_chunks = []
    try:
        if action == "prepare":
            v.prepare_data(payload["audio_dir"])  # blocking, but OK for now
        elif action == "train":
            v.train_lora(payload["audio_dir"], payload.get("hparams_path"))
        elif action == "infer":
            v.infer_candidates(
                payload["audio_dir"],
                payload.get("text", ""),
                int(payload.get("n_candidates", 1) or 1),
                evaluate=bool(payload.get("evaluate", True)),
            )
        elif action == "chapter":
            v.synthesize_chapter(payload["audio_dir"], payload.get("text_lines") or [])
        elif action == "chapter_file":
            v.synthesize_chapter_from_file(payload["audio_dir"], payload.get("text_file", ""))
        elif action == "merge":
            v.audio_merge(payload["dir"], payload["out"])  # out path
        else:
            return web.json_response({"status": "error", "stdout": "", "stderr": f"unknown action: {action}"}, status=400)
        return web.json_response({"status": "ok", "stdout": "\n".join(stdout_chunks), "stderr": ""})
    except Exception as e:
        return web.json_response({"status": "error", "stdout": "", "stderr": str(e)}, status=500)


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_post("/run", handle_run)
    return app


def main() -> None:
    host = os.environ.get("AI_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("AI_SERVICE_PORT", "8777") or 8777)
    web.run_app(build_app(), host=host, port=port)


if __name__ == "__main__":
    main()

