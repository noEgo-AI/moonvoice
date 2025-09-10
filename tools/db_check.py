"""
Asynchronous DB-driven worker that dispatches CPU vs GPU tasks concurrently.

- You decide which job goes to CPU or GPU by editing `route_job(row)` below.
- CPU and GPU workers run concurrently using asyncio Queues and concurrency limits.
- All status transitions are clearly marked; adjust as your schema dictates.

Run examples:
  CPU_WORKERS=2 GPU_WORKERS=1 POLL_INTERVAL=1.5 python tools/db_check.py
"""
from __future__ import annotations

import asyncio
import sys
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Tuple, Set
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import functools

import voice_interface as vt
from tools.db_util import create_open_pool, get_abs_env_file
import datetime


# -------------------------------
# Global cancellation/kill registry (best-effort)
# -------------------------------

# Keyed by (usercode, voicecode)
ACTIVE_CPU_FUTURES: Dict[Tuple[str, Optional[str]], asyncio.Future] = {}
ACTIVE_GPU_FUTURES: Dict[Tuple[str, Optional[str]], asyncio.Future] = {}
CANCEL_PAIRS: Set[Tuple[str, Optional[str]]] = set()


# -------------------------------
# Job model and routing hook
# -------------------------------

JobKind = Literal["cpu", "gpu"]
Action = Literal[
    "prepare", "cleanup", "train", "make_sample",
    "book_generate", "book_merge",
]


@dataclass
class Job:
    no: int
    usercode: str
    filename: str | None
    status: int
    kind: JobKind
    action: Action
    bookcode: int | None = None
    bookfile: str | None = None
    voicecode: str | None = None


def route_job(row: tuple) -> tuple[JobKind, Action] | None:
    """Decide CPU vs GPU and action per DB row.

    Edit this logic to your needs.
    Row format: (no, usercode, filename, status)

    Default mapping (VoiceTTSHandler 사용 단계는 GPU 전용):
      - status=4  → GPU 'prepare' (전처리: 모델 사용 → GPU)
      - status=7  → CPU 'cleanup' (폴더 정리: 파일 작업 → CPU)
      - status=9  → GPU 'train'   (LoRA 학습: 모델 사용 → GPU)
      - status=11 → CPU 'make_sample' (샘플 예약/생성 등 DB 작업)
    실패 상태(예: 8, 12, 14, 15)는 폴링 대상에서 제외하여 자동 재시도하지 않습니다.
    """
    no, usercode, filename, status = row[0], str(row[1]), row[2], int(row[3])
    if status == 4:
        return "gpu", "prepare"
    if status == 7:
        return "cpu", "cleanup"
    if status == 9:
        return "gpu", "train"
    if status == 11:
        return "cpu", "make_sample"
    return None


# -------------------------------
# DB helpers
# -------------------------------

async def _claim(conn, job: Job) -> bool:
    """Transition status to in-progress atomically and claim the job.

    - prepare: 4 → 6 (진입)
    - cleanup: 7 → 9 (완료로 바로 마킹: 실제 삭제는 워커에서 시도)
    - train:   9 → 10 (진입)
    - make_sample: 11 → 13 (완료로 바로 마킹: 실제 입력은 워커에서 시도)
    Returns True on success (rowcount=1), False if someone else claimed it.
    """
    if job.action == "prepare":
        q = 'UPDATE "AudioStatus" SET status = 6 WHERE "no" = %s AND status = 4'
    elif job.action == "cleanup":
        q = 'UPDATE "AudioStatus" SET status = 9 WHERE "no" = %s AND status = 7'
    elif job.action == "train":
        q = 'UPDATE "AudioStatus" SET status = 10 WHERE "no" = %s AND status = 9'
    elif job.action == "make_sample":
        q = 'UPDATE "AudioStatus" SET status = 13 WHERE "no" = %s AND status = 11'
    else:
        return False
    async with conn.cursor() as cur:
        await cur.execute(q, (job.no,))
        await conn.commit()
        return cur.rowcount == 1


async def _finalize(pool, job: Job) -> None:
    """Mark job as finished for its action.

    - prepare: 6 → 7
    - cleanup: 9 (already set) → no-op
    - train:   10 → 11
    """
    if job.action == "prepare":
        q = 'UPDATE "AudioStatus" SET status = 7 WHERE "no" = %s AND status = 6'
    elif job.action == "train":
        q = 'UPDATE "AudioStatus" SET status = 11 WHERE "no" = %s AND status = 10'
    else:
        return
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(q, (job.no,))
            await conn.commit()

# 실패 상태 매핑: 모든 처리 단계에 대응
FAIL_STATUS = {
    "prepare": 8,   # 6 중/후 실패
    "train": 12,    # 10 중/후 실패
    "cleanup": 14,  # 파일 정리 실패
    "make_sample": 15,  # 샘플 예약/생성 실패
}

async def _mark_failed(pool, job: Job, reason: Exception | str) -> None:
    code = FAIL_STATUS.get(job.action)
    if code is None:
        # No failure status defined; just log or extend if your schema allows.
        return
    msg = str(reason)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.execute(
                    'UPDATE "AudioStatus" SET status = %s, error_message = %s WHERE "no" = %s',
                    (code, msg[:800], job.no),
                )
            except Exception:
                await cur.execute('UPDATE "AudioStatus" SET status = %s WHERE "no" = %s', (code, job.no))
            await conn.commit()


async def _poll_and_dispatch(pool, cpu_q: asyncio.Queue[Job], gpu_q: asyncio.Queue[Job], *, interval: float) -> None:
    """Poll DB, decide routing, claim rows, then enqueue jobs."""
    while True:
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Fetch candidates; keep it small to avoid starving workers
                    await cur.execute('SELECT "no","usercode","filename","status","voicecode" FROM "AudioStatus" WHERE status IN (4,7,9,11) ORDER BY "no" ASC LIMIT 32')
                    rows = await cur.fetchall()
            claimed_any = False
            for r in rows:
                route = route_job(r)
                if not route:
                    continue
                kind, action = route
                vc = None
                try:
                    vc = (str(r[4]) if r[4] is not None else None)
                except Exception:
                    vc = None
                job = Job(no=int(r[0]), usercode=str(r[1]), filename=str(r[2]), status=int(r[3]), kind=kind, action=action, voicecode=vc)
                async with pool.connection() as conn:
                    if await _claim(conn, job):
                        if kind == "cpu":
                            await cpu_q.put(job)
                        else:
                            await gpu_q.put(job)
                        claimed_any = True
            if not claimed_any:
                await asyncio.sleep(interval)
        except Exception as e:
            print(f"Dispatcher error: {e}")
            await asyncio.sleep(interval)


# -------------------------------
# BookStatus helpers/dispatcher
# -------------------------------

async def _update_book_status(pool, no: int, *, expect: int | None = None, to: int, set_generate_time: bool = False) -> None:
    sql_time = '"generatetime" = (NOW() AT TIME ZONE \'UTC\')' if set_generate_time else None
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            if expect is None:
                if sql_time:
                    await cur.execute(f'UPDATE "BookStatus" SET "status"=%s, {sql_time} WHERE "no"=%s', (to, no))
                else:
                    await cur.execute('UPDATE "BookStatus" SET "status"=%s WHERE "no"=%s', (to, no))
            else:
                if sql_time:
                    await cur.execute(f'UPDATE "BookStatus" SET "status"=%s, {sql_time} WHERE "no"=%s AND "status"=%s', (to, no, expect))
                else:
                    await cur.execute('UPDATE "BookStatus" SET "status"=%s WHERE "no"=%s AND "status"=%s', (to, no, expect))
            await conn.commit()


async def _claim_book(conn, no: int, expect: int, to: int) -> bool:
    async with conn.cursor() as cur:
        await cur.execute('UPDATE "BookStatus" SET "status"=%s WHERE "no"=%s AND "status"=%s', (to, no, expect))
        await conn.commit()
        return cur.rowcount == 1


async def _poll_and_dispatch_books(pool, cpu_q: asyncio.Queue[Job], gpu_q: asyncio.Queue[Job], *, interval: float) -> None:
    """Poll BookStatus and enqueue work per provided status mapping.

    Mapping (0..6):
      - 0 → 1: GPU book_generate
      - 2 → 4: CPU book_merge
      - 1/4 in-progress, 3/6 failed, 5 done (skipped)
    """
    while True:
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute('SELECT "no","usercode","bookcode","bookfile","status","voicecode" FROM "BookStatus" WHERE "status" IN (0,2) ORDER BY "no" ASC LIMIT 32')
                    rows = await cur.fetchall()
            claimed = False
            for no, usercode, bookcode, bookfile, status, voicecode in rows:
                no_i = int(no)
                user_s = str(usercode)
                bc = int(bookcode) if bookcode is not None else None
                bf = str(bookfile) if bookfile is not None else None
                vc = str(voicecode) if voicecode is not None else None
                if int(status) == 0:
                    async with pool.connection() as conn:
                        if await _claim_book(conn, no_i, 0, 1):
                            job = Job(no=no_i, usercode=user_s, filename=None, status=1, kind='gpu', action='book_generate', bookcode=bc, bookfile=bf, voicecode=vc)
                            await gpu_q.put(job)
                            claimed = True
                elif int(status) == 2:
                    async with pool.connection() as conn:
                        if await _claim_book(conn, no_i, 2, 4):
                            job = Job(no=no_i, usercode=user_s, filename=None, status=4, kind='cpu', action='book_merge', bookcode=bc, bookfile=bf, voicecode=vc)
                            await cpu_q.put(job)
                            claimed = True
            if not claimed:
                await asyncio.sleep(interval)
        except Exception as e:
            print(f"Book dispatcher error: {e}")
            await asyncio.sleep(interval)


# -------------------------------
# Reset handler (status=999)
# -------------------------------

async def _reset_handler(pool, *, ai_root: str, book_root: str | None, interval: float) -> None:
    """Handle global resets signaled by AudioStatus.status=999.

    For each row with status=999, remove all files under AI_ROOT/<user>/<voicecode>
    and BOOK_ROOT/<user>/<voicecode>, and mark related BookStatus rows to 999.
    """
    import shutil
    while True:
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Fetch distinct (usercode, voicecode) pairs requesting reset
                    await cur.execute('SELECT DISTINCT "usercode","voicecode" FROM "AudioStatus" WHERE "status" = 999')
                    rows = await cur.fetchall()
            for usercode, voicecode in rows:
                uc = str(usercode)
                vc = (str(voicecode) if voicecode is not None else None)
                key = (uc, vc)
                # Record cancellation intent
                CANCEL_PAIRS.add(key)
                # Best-effort: cancel active CPU/GPU futures for this pair
                fut = ACTIVE_GPU_FUTURES.pop(key, None)
                if fut is not None:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                fut = ACTIVE_CPU_FUTURES.pop(key, None)
                if fut is not None:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                # Delete AI_ROOT/<user>/<voicecode>
                if vc:
                    ai_dir = os.path.join(ai_root, uc, vc)
                    try:
                        await asyncio.to_thread(shutil.rmtree, ai_dir)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
                # Delete BOOK_ROOT/<user>/<voicecode>
                if book_root and vc:
                    bdir = os.path.join(book_root, uc, vc)
                    try:
                        await asyncio.to_thread(shutil.rmtree, bdir)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
                # Mark related BookStatus rows to 999 (generating or completed)
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            'UPDATE "BookStatus" SET "status"=999 WHERE "usercode"=%s AND "voicecode"=%s AND "status" IN (1,2,4,5)',
                            (uc, vc),
                        )
                        await conn.commit()
                # After processing, flip AudioStatus 999->1000 and BookStatus 999->1000 for the pair
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            'UPDATE "AudioStatus" SET "status"=1000 WHERE "usercode"=%s AND "voicecode"=%s AND "status"=999',
                            (uc, vc),
                        )
                        await cur.execute(
                            'UPDATE "BookStatus" SET "status"=1000 WHERE "usercode"=%s AND "voicecode"=%s AND "status"=999',
                            (uc, vc),
                        )
                        await conn.commit()
                # Finally, purge rows that have status=1000 for this (usercode, voicecode)
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            'DELETE FROM "AudioStatus" WHERE "usercode"=%s AND "voicecode"=%s AND "status"=1000',
                            (uc, vc),
                        )
                        await cur.execute(
                            'DELETE FROM "BookStatus" WHERE "usercode"=%s AND "voicecode"=%s AND "status"=1000',
                            (uc, vc),
                        )
                        await conn.commit()
                # Remove cancel intent after cleanup
                try:
                    CANCEL_PAIRS.discard(key)
                except Exception:
                    pass
        except Exception as e:
            print(f"Reset handler error: {e}")
        finally:
            await asyncio.sleep(interval)


async def _get_booklist_name(pool, code: int) -> str | None:
    """Fetch BookList.name by code. Returns None if not found."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute('SELECT "name" FROM "BookList" WHERE "code"=%s', (int(code),))
            row = await cur.fetchone()
            return (row[0] if row else None)


# -------------------------------
# Worker implementations
# -------------------------------

def _merge_audio_job(out_dir: str, merged: str) -> None:
    # Separate process-safe function to merge wav files
    from voice_interface import VoiceTTSHandler as _H
    _H().audio_merge(out_dir, merged)


def _rmtree_job(path: str) -> None:
    import shutil as _sh
    _sh.rmtree(path)


async def cpu_worker(name: str, q: asyncio.Queue[Job], pool,
                     voice_root: str, ai_root: str, base_model_path: Optional[str], *,
                     book_root: Optional[str] = None,
                     cpu_exec: ProcessPoolExecutor | None = None) -> None:
    """CPU 전용 처리 워커.
    - cleanup: 로컬 폴더 삭제 (best-effort)
    - make_sample : DB에 Query 문 날리기
    """
    while True:
        job = await q.get()
        try:
            if job.action == "cleanup":
                # 폴더 정리: 실패 시 예외를 올려 실패 상태로 전환
                root_dir = f"{voice_root}/{job.usercode}/"
                loop = asyncio.get_running_loop()
                key = (job.usercode, job.voicecode)
                try:
                    if cpu_exec is None:
                        fut = asyncio.to_thread(shutil.rmtree, root_dir)
                    else:
                        fut = loop.run_in_executor(cpu_exec, _rmtree_job, root_dir)
                    ACTIVE_CPU_FUTURES[key] = asyncio.ensure_future(fut)  # type: ignore[arg-type]
                    await ACTIVE_CPU_FUTURES[key]
                finally:
                    ACTIVE_CPU_FUTURES.pop(key, None)
                # 성공 시 추가 상태 변경 없음 (이미 9)
            elif job.action == "book_merge":
                if not book_root:
                    raise RuntimeError("BOOK_ROOT not configured")
                out_dir = os.path.join(book_root, str(job.usercode), str(job.voicecode or ''), str(job.bookcode))
                os.makedirs(out_dir, exist_ok=True)
                # 최종 파일명: bookcode_usercode_generatetime.wav (UTC, yyyymmddHHMMSS)
                ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')
                code_str = str(job.bookcode or job.no)
                merged_filename = f"{code_str}_{job.usercode}_{ts}.wav"
                merged = os.path.join(out_dir, merged_filename)
                loop = asyncio.get_running_loop()
                key = (job.usercode, job.voicecode)
                try:
                    if cpu_exec is None:
                        fut = asyncio.to_thread(vt.VoiceTTSHandler().audio_merge, out_dir, merged)
                    else:
                        fut = loop.run_in_executor(cpu_exec, _merge_audio_job, out_dir, merged)
                    ACTIVE_CPU_FUTURES[key] = asyncio.ensure_future(fut)  # type: ignore[arg-type]
                    await ACTIVE_CPU_FUTURES[key]
                finally:
                    ACTIVE_CPU_FUTURES.pop(key, None)
                # 병합 후 개별 생성 wav 파일 삭제 (merged.wav 제외)
                try:
                    for p in Path(out_dir).glob("*.wav"):
                        try:
                            if p.name != merged_filename:
                                p.unlink(missing_ok=True)
                        except Exception:
                            pass
                except Exception:
                    pass
                # 병합된 파일명을 DB에 기록
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute('UPDATE "BookStatus" SET "bookfile"=%s, "generatetime"=(NOW() AT TIME ZONE \'UTC\') WHERE "no"=%s', (merged_filename, job.no))
                        await conn.commit()
                # 성공: 4 -> 5
                await _update_book_status(pool, job.no, expect=4, to=5)
            elif job.action == "make_sample":
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # IDENTITY/DEFAULT 기반: "no"는 DB가 생성 → RETURNING로 수령
                        sql = '''
                        INSERT INTO "BookStatus" ("usercode","bookcode","bookfile","status","reservetime","generatetime","voicecode")
                        VALUES (%s, %s, %s, %s, (NOW() AT TIME ZONE 'UTC'), NULL, %s)
                        RETURNING "no";
                        '''
                        await cur.execute(sql, (job.usercode, 0, None, 0, job.voicecode))
                        # await cur.execute(sql, (job.usercode, 2, None, 0, job.voicecode))
                        _row = await cur.fetchone()
                        await conn.commit()
        except Exception as e:
            # AudioStatus 실패 처리
            if job.action in ("prepare", "train", "cleanup", "make_sample"):
                await _mark_failed(pool, job, e)
            # BookStatus 실패 처리
            if job.action == "book_merge":
                await _update_book_status(pool, job.no, expect=4, to=6)
            print(f"[CPU {name}] Job {job.no} failed: {e}")
        finally:
            q.task_done()


async def gpu_worker(name: str, q: asyncio.Queue[Job], pool, handler: vt.VoiceTTSHandler,
                     voice_root: str, ai_root: str, base_model_path: Optional[str], *,
                     book_root: Optional[str] = None, booklist_root: Optional[str] = None,
                     gpu_sema: asyncio.Semaphore | None = None) -> None:
    """GPU 전용 처리 워커.
    - prepare: 전처리(VoiceTTSHandler 사용 → GPU)
    - train: LoRA 학습
    """
    while True:
        job = await q.get()
        try:
            # Serialize all GPU-bound work so only one runs at a time
            if gpu_sema is None:
                gpu_sema = asyncio.Semaphore(1)
            async with gpu_sema:
                if job.action == "prepare":
                    # GPU prepare (모델 사용 단계는 GPU 워커에서 처리)
                    root = f"{voice_root}/{job.usercode}/{job.filename}"
                    key = (job.usercode, job.voicecode)
                    try:
                        fut = asyncio.to_thread(
                            handler.prepare_data,
                            root,
                            output_path=ai_root,
                            usercode=job.usercode,
                            voicecode=job.voicecode,
                            processes=2,
                            base_model_path=base_model_path,
                        )
                        ACTIVE_GPU_FUTURES[key] = asyncio.ensure_future(fut)
                        await ACTIVE_GPU_FUTURES[key]
                    finally:
                        ACTIVE_GPU_FUTURES.pop(key, None)
                    # 후검증: 필수 산출물 존재 여부 확인 (없으면 실패로 간주해 raise)
                    user_out = Path(ai_root) / str(job.usercode) / str(job.voicecode or '')
                    must_exist = [
                        user_out / "training_data_v3.jsonl",
                        user_out / "lora_hparams.json",
                    ]
                    missing = [str(p) for p in must_exist if not p.exists()]
                    if missing:
                        raise RuntimeError(f"prepare outputs missing: {missing}")
                    await _finalize(pool, job)
                elif job.action == "train":
                    key = (job.usercode, job.voicecode)
                    try:
                        fut = asyncio.to_thread(
                            handler.train_lora,
                            ai_root,
                            output_path=ai_root,
                            usercode=job.usercode,
                            voicecode=job.voicecode,
                            base_model_path=base_model_path,
                        )
                        ACTIVE_GPU_FUTURES[key] = asyncio.ensure_future(fut)
                        adapter_dir = await ACTIVE_GPU_FUTURES[key]
                    finally:
                        ACTIVE_GPU_FUTURES.pop(key, None)
                    # 후검증: 어댑터 산출물 확인 (safetensors/bin 중 하나)
                    adir = Path(adapter_dir)
                    ok = (adir.exists() and any((adir / n).exists() for n in ("adapter_model.safetensors", "pytorch_model.bin")))
                    if not ok:
                        raise RuntimeError(f"train outputs missing under: {adir}")
                    await _finalize(pool, job)
                elif job.action == "book_generate":
                    if not (book_root and booklist_root and base_model_path):
                        raise RuntimeError("BOOK_ROOT/BOOKLIST_ROOT/OUTETTS_MODEL_ROOT not configured")
                    # 텍스트 파일 경로: BookList 테이블에서 bookcode로 name 조회
                    if job.bookcode is None:
                        raise RuntimeError("bookcode is required for book_generate")
                    book_name = await _get_booklist_name(pool, int(job.bookcode))
                    if not book_name:
                        raise RuntimeError(f"BookList not found for code={job.bookcode}")
                    # booklist_root/name 조합 (name이 절대경로라면 그대로 사용)
                    text_path = book_name if os.path.isabs(book_name) else os.path.join(booklist_root, book_name)
                    out_dir = os.path.join(book_root, str(job.usercode), str(job.voicecode or ''), str(job.bookcode))
                    os.makedirs(out_dir, exist_ok=True)
                    lora_dir = os.path.join(ai_root, str(job.usercode), str(job.voicecode or ''))
                    key = (job.usercode, job.voicecode)
                    try:
                        fut = asyncio.to_thread(
                            handler.synthesize_chapter_from_file,
                            "_",
                            text_path,
                            base_model_path=base_model_path,
                            model_path=lora_dir,
                            output_path=out_dir,
                            usercode=None,
                            evaluate_once=True,
                            n_candidates_per_sentence=1,
                            eval_workers=3,
                            synth_workers=3,
                            skip_eval_if_cached=True,
                        )
                        ACTIVE_GPU_FUTURES[key] = asyncio.ensure_future(fut)
                        await ACTIVE_GPU_FUTURES[key]
                    finally:
                        ACTIVE_GPU_FUTURES.pop(key, None)
                    if not list(Path(out_dir).glob("*.wav")):
                        raise RuntimeError(f"book generate produced no wavs in: {out_dir}")
                    # 성공: 1 -> 2 + generatetime
                    await _update_book_status(pool, job.no, expect=1, to=2, set_generate_time=True)
        except Exception as e:
            # AudioStatus 실패 처리
            if job.action in ("prepare", "train", "cleanup", "make_sample"):
                await _mark_failed(pool, job, e)
            # BookStatus 실패 처리
            if job.action == "book_generate":
                await _update_book_status(pool, job.no, expect=1, to=3)
            elif job.action == "book_merge":
                await _update_book_status(pool, job.no, expect=4, to=6)
            print(f"[GPU {name}] Job {job.no} failed: {e}")
        finally:
            # GPU 메모리 캐시 정리로 OOM 완화 시도
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            q.task_done()


# -------------------------------
# Bootstrap and run
# -------------------------------

async def main() -> int:
    # Load .env by absolute path (edit if needed)
    env_path = "/home/server1/AI2/OuteTTS/.env"
    env = get_abs_env_file(env_path)
    # Reduce CUDA fragmentation by default
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    db_url = env.get("DATABASE_URL")
    voice_root = env.get("VOICE_ROOT")
    ai_root = env.get("AI_ROOT")
    base_model_path = env.get("OUTETTS_MODEL_ROOT")
    book_root = env.get("BOOK_ROOT")
    booklist_root = env.get("BOOKLIST_ROOT")

    if not db_url or not voice_root or not ai_root:
        raise SystemExit("Missing required env: DATABASE_URL, VOICE_ROOT, AI_ROOT")

    poll_interval = float(os.getenv("POLL_INTERVAL", "1.5"))
    cpu_workers = max(1, int(os.getenv("CPU_WORKERS", "2")))
    gpu_workers = max(1, int(os.getenv("GPU_WORKERS", "1")))

    # GPU handler only (VoiceTTSHandler 사용 단계는 GPU 전용)
    gpu_handler = vt.VoiceTTSHandler(device='cuda', dtype='bf16', seeds_pool_size=3, val_sample_size=3, default_temperature=0.40)

    pool = await create_open_pool(db_url)
    cpu_q: asyncio.Queue[Job] = asyncio.Queue(maxsize=cpu_workers * 2)
    gpu_q: asyncio.Queue[Job] = asyncio.Queue(maxsize=gpu_workers * 2)

    # Launch dispatcher and workers
    tasks = []
    tasks.append(asyncio.create_task(_poll_and_dispatch(pool, cpu_q, gpu_q, interval=poll_interval)))
    tasks.append(asyncio.create_task(_poll_and_dispatch_books(pool, cpu_q, gpu_q, interval=poll_interval)))
    tasks.append(asyncio.create_task(_reset_handler(pool, ai_root=ai_root, book_root=book_root, interval=poll_interval)))
    # Dedicated process pool for CPU-bound heavy tasks (merge/cleanup)
    cpu_exec_workers = max(1, int(os.getenv("CPU_EXEC_WORKERS", str(cpu_workers))))
    cpu_exec = ProcessPoolExecutor(max_workers=cpu_exec_workers)
    for i in range(cpu_workers):
        tasks.append(asyncio.create_task(cpu_worker(
            f"cpu-{i+1}", cpu_q, pool, voice_root, ai_root, base_model_path,
            book_root=book_root, cpu_exec=cpu_exec
        )))
    # Single shared semaphore to serialize GPU-bound work
    gpu_sema = asyncio.Semaphore(1)
    for i in range(gpu_workers):
        tasks.append(asyncio.create_task(gpu_worker(
            f"gpu-{i+1}", gpu_q, pool, gpu_handler,
            voice_root, ai_root, base_model_path,
            book_root=book_root, booklist_root=booklist_root,
            gpu_sema=gpu_sema,
        )))

    # Run forever; cancel tasks on Ctrl-C
    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, KeyboardInterrupt):
        for t in tasks:
            t.cancel()
    finally:
        try:
            cpu_exec.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
