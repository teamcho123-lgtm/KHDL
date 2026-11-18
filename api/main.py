"""FastAPI service for emotion and calibrated vocal arousal/stress proxy analysis."""

import asyncio
import csv
import hmac
import json
import logging
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import soundfile as sf
import torch
import torchaudio
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.acoustic_extractor import AcousticExtractor
from src.pipeline import StressAnalysisPipeline

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 256 * 1024
MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
UPLOAD_CHUNK_BYTES = 1024 * 1024

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hparams_inference.yaml"
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "collected_benchmark"
BENCHMARK_WAVS_DIR = BENCHMARK_DIR / "wavs"
BENCHMARK_MANIFEST_PATH = BENCHMARK_DIR / "benchmark_manifest.json"
BENCHMARK_METADATA_PATH = BENCHMARK_DIR / "benchmark_metadata.csv"
benchmark_lock = asyncio.Lock()

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

pipeline: Optional[StressAnalysisPipeline] = None
inference_semaphore: Optional[asyncio.Semaphore] = None
inference_timeout_seconds: float = 120.0


def _load_runtime_limits():
    if CONFIG_PATH.is_file():
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}
        max_conc = int(cfg.get("max_concurrent_inferences", 2))
        timeout_s = float(cfg.get("inference_timeout_seconds", 120.0))
        return max(1, max_conc), max(5.0, timeout_s)
    return 2, 120.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, inference_semaphore, inference_timeout_seconds
    logger.info("Initializing stress analysis pipeline")
    max_conc, timeout_s = _load_runtime_limits()
    inference_semaphore = asyncio.Semaphore(max_conc)
    inference_timeout_seconds = timeout_s
    pipeline = StressAnalysisPipeline.from_config(str(CONFIG_PATH))
    yield
    pipeline = None
    inference_semaphore = None


app = FastAPI(
    title="Speech Emotion & Vocal Arousal/Stress Proxy API",
    description="Speech emotion recognition and personal-baseline vocal arousal/stress proxy analysis",
    version="1.2.0",
    lifespan=lifespan,
)


class BodySizeTooLargeError(Exception):
    pass


@app.middleware("http")
async def enforce_early_body_size_limit(request: Request, call_next):
    """Reject oversized requests before Starlette's multipart parser buffers them."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds maximum upload limit (20 MiB)"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )

    received_bytes = 0
    original_receive = request._receive

    async def _bounded_receive():
        nonlocal received_bytes
        message = await original_receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"")
            received_bytes += len(body)
            if received_bytes > MAX_REQUEST_BODY_BYTES:
                raise BodySizeTooLargeError()
        return message

    request._receive = _bounded_receive
    try:
        return await call_next(request)
    except BodySizeTooLargeError:
        return JSONResponse(
            status_code=413,
            content={"detail": "Audio upload is too large (exceeds 20 MiB limit)"},
        )


def _get_allowed_origins() -> list[str]:
    env_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return DEFAULT_ALLOWED_ORIGINS


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)


def verify_api_access(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """Verify API key when configured, or enforce loopback-only access when unconfigured."""
    configured_key = os.environ.get("API_KEY", "").strip()
    provided_key = x_api_key
    if not provided_key and authorization and authorization.lower().startswith("bearer "):
        provided_key = authorization[7:].strip()

    if configured_key:
        if not provided_key or not hmac.compare_digest(provided_key, configured_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return

    client_host = request.client.host if request.client else None
    if client_host not in LOOPBACK_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="Remote access is disabled unless API_KEY is configured on the server",
        )


async def _run_bounded_inference(func, *args, **kwargs):
    """Run CPU/GPU inference under concurrency and timeout guards."""
    sem = inference_semaphore or asyncio.Semaphore(2)
    try:
        await asyncio.wait_for(sem.acquire(), timeout=5.0)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="Server is busy processing other audio requests; please retry shortly",
        ) from exc

    try:
        return await asyncio.wait_for(
            run_in_threadpool(func, *args, **kwargs),
            timeout=inference_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Audio analysis timed out",
        ) from exc
    finally:
        sem.release()


async def _save_upload(file: UploadFile) -> str:
    """Write a bounded upload to a temporary audio path preserving suffix and clean up on error."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    if not suffix or len(suffix) > 8 or not suffix.startswith("."):
        suffix = ".wav"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = handle.name
    total = 0
    try:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Audio upload is too large")
            handle.write(chunk)
        handle.close()
        if total == 0:
            raise HTTPException(status_code=400, detail="Audio upload is empty")
        return path
    except Exception:
        handle.close()
        if os.path.exists(path):
            os.remove(path)
        raise
    finally:
        await file.close()


def _remove_temp(path: Optional[str]):
    if path and os.path.exists(path):
        os.remove(path)


@app.get("/health/live")
def liveness_check():
    """Lightweight liveness probe confirming the ASGI process is responsive."""
    return {"status": "alive"}


@app.get("/health")
@app.get("/health/ready")
def health_check():
    """Readiness probe reporting pipeline and model availability accurately."""
    pipeline_ready = pipeline is not None
    emotion_ready = bool(pipeline and pipeline.emotion_model)
    stress_ready = bool(pipeline and pipeline.stress_model)
    fully_ready = bool(pipeline and pipeline.ready)

    if not pipeline_ready:
        status = "starting"
    elif fully_ready:
        status = "healthy"
    else:
        status = "degraded"

    return {
        "status": status,
        "ready": fully_ready,
        "pipeline_ready": pipeline_ready,
        "emotion_model_ready": emotion_ready,
        "stress_model_ready": stress_ready,
        "enhancer_enabled": bool(pipeline and pipeline.enhancer is not None),
        "auth_required": bool(os.environ.get("API_KEY", "").strip()),
    }


@app.post("/api/v1/register-baseline", dependencies=[Depends(verify_api_access)])
async def register_baseline(
    user_id: str = Form(...),
    overwrite: bool = Form(False),
    file: UploadFile = File(...),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    tmp_path = await _save_upload(file)
    try:
        result = await _run_bounded_inference(
            pipeline.register_baseline,
            user_id=user_id,
            calm_audio_path=tmp_path,
            overwrite=overwrite,
        )
        return {"status": "success", "result": result}
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Baseline registration failed")
        raise HTTPException(status_code=500, detail="Baseline registration failed") from exc
    finally:
        _remove_temp(tmp_path)


@app.get("/api/v1/baseline/{user_id}", dependencies=[Depends(verify_api_access)])
def check_baseline_status(user_id: str):
    """Check whether a specific user_id has a registered baseline without enumerating all users."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        exists = pipeline.has_baseline(user_id)
        return {"user_id": user_id, "registered": exists}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/v1/baseline/{user_id}", dependencies=[Depends(verify_api_access)])
def delete_baseline_profile(user_id: str):
    """Delete a specific user's baseline profile."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        deleted = pipeline.delete_baseline(user_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"No baseline profile found for '{user_id}'"
            )
        return {"status": "deleted", "user_id": user_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/analyze-stress", dependencies=[Depends(verify_api_access)])
async def analyze_stress(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    identify_speaker: bool = Form(False),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    tmp_path = await _save_upload(file)
    try:
        return await _run_bounded_inference(
            pipeline.analyze,
            audio_path=tmp_path,
            user_id=user_id,
            verify_speaker=bool(user_id),
            identify_speaker=bool(identify_speaker and not user_id),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Audio analysis failed")
        raise HTTPException(status_code=500, detail="Audio analysis failed") from exc
    finally:
        _remove_temp(tmp_path)


@app.get("/api/v1/samples", dependencies=[Depends(verify_api_access)])
def list_samples():
    """List available demo audio samples."""
    sample_candidates = [
        {
            "id": "calm_male",
            "name": "Nam - Bình tĩnh (Actor 01 Calm)",
            "path": "data/raw/RAVDESS/Actor_01/03-01-02-01-01-01-01.wav",
            "expected": "CALM",
        },
        {
            "id": "angry_male",
            "name": "Nam - Giận dữ (Actor 01 Angry)",
            "path": "data/raw/RAVDESS/Actor_01/03-01-05-01-02-01-01.wav",
            "expected": "HIGH_STRESS",
        },
        {
            "id": "fearful_female",
            "name": "Nữ - Sợ hãi (Actor 02 Fearful)",
            "path": "data/raw/RAVDESS/Actor_02/03-01-06-01-01-01-02.wav",
            "expected": "HIGH_STRESS",
        },
    ]
    root = Path(__file__).resolve().parents[1]
    available = []
    for s in sample_candidates:
        full_path = root / s["path"]
        if full_path.is_file():
            available.append({
                "id": s["id"],
                "name": s["name"],
                "expected": s["expected"],
                "url": f"/api/v1/sample-audio/{s['id']}",
            })
    return {"samples": available}


@app.get("/api/v1/sample-audio/{sample_id}", dependencies=[Depends(verify_api_access)])
def get_sample_audio(sample_id: str):
    """Retrieve demo audio sample by ID."""
    root = Path(__file__).resolve().parents[1]
    mapping = {
        "calm_male": "data/raw/RAVDESS/Actor_01/03-01-02-01-01-01-01.wav",
        "angry_male": "data/raw/RAVDESS/Actor_01/03-01-05-01-02-01-01.wav",
        "fearful_female": "data/raw/RAVDESS/Actor_02/03-01-06-01-01-01-02.wav",
    }
    rel_path = mapping.get(sample_id)
    if not rel_path:
        raise HTTPException(status_code=404, detail="Sample not found")
    full_path = root / rel_path
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found on disk")
    return FileResponse(str(full_path), media_type="audio/wav")


ALLOWED_GROUND_TRUTH = {"normal", "stress", "calm", "angry", "fearful", "happy", "sad"}


@app.post("/api/v1/benchmarks/submit", dependencies=[Depends(verify_api_access)])
async def submit_benchmark_sample(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    ground_truth: str = Form(...),
    context: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    consent: bool = Form(True),
):
    """Store an in-the-wild audio recording for speaker-independent benchmark evaluation."""
    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Người dùng cần đồng ý điều khoản lưu trữ mẫu kiểm chứng ẩn danh.",
        )

    # 1. Path traversal guard & user_id validation
    clean_user = user_id.strip()
    if not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", clean_user):
        raise HTTPException(
            status_code=400,
            detail="user_id phải từ 1-64 ký tự chữ, số, dấu gạch dưới hoặc gạch ngang",
        )

    # 2. Ground truth label validation
    clean_label = ground_truth.strip().lower()
    if clean_label not in ALLOWED_GROUND_TRUTH:
        allowed_str = ", ".join(sorted(ALLOWED_GROUND_TRUTH))
        raise HTTPException(
            status_code=400,
            detail=f"Nhãn ground_truth '{ground_truth}' không hợp lệ. Phải thuộc: {allowed_str}",
        )

    clean_device = (device.strip()[:60] if device else "web_microphone").replace("\n", " ").replace(",", ";")
    clean_context = (context.strip()[:120] if context else "unspecified").replace("\n", " ").replace(",", ";")

    # 3. Save upload to temporary file
    tmp_path = await _save_upload(file)
    try:
        # 4. Check duration & basic audio validity
        try:
            info = sf.info(tmp_path)
            duration_s = float(info.duration)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Không đọc được tệp âm thanh: {exc}") from exc

        if duration_s < 1.5:
            raise HTTPException(
                status_code=400,
                detail=f"Thời lượng âm thanh ({duration_s:.1f}s) quá ngắn. Tối thiểu cần 1.5 giây.",
            )
        if duration_s > 60.0:
            raise HTTPException(
                status_code=400,
                detail=f"Thời lượng âm thanh ({duration_s:.1f}s) quá dài cho một mẫu benchmark. Tối đa 60.0 giây.",
            )

        # 5. Acoustic validation (Acoustic Quality Safeguard)
        extractor = pipeline.acoustic_extractor if pipeline else AcousticExtractor()
        extracted = extractor.extract_features_with_quality(tmp_path)
        raw_metrics = extracted.get("metrics", {})

        if raw_metrics.get("f0_mean") is None:
            raise HTTPException(
                status_code=422,
                detail="Chất lượng âm thanh không đạt chuẩn Benchmark (không phát hiện được tần số F0 tuần hoàn hoặc bị câm).",
            )

        # 6. Standardize audio: load with soundfile, mixdown to mono, resample to 16kHz
        try:
            sig, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
            mono_sig = sig.mean(axis=1)
            if sr != 16000:
                tensor_sig = torch.from_numpy(mono_sig).float()
                resampled_tensor = torchaudio.functional.resample(tensor_sig, orig_freq=sr, new_freq=16000)
                mono_sig = resampled_tensor.numpy()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Lỗi chuẩn hóa âm thanh về 16kHz PCM: {exc}",
            ) from exc

        # 7. Write normalized WAV atomically
        BENCHMARK_WAVS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:8]
        sample_id = f"bm_{clean_user}_{clean_label}_{timestamp_str}_{unique_suffix}"
        wav_filename = f"{sample_id}.wav"
        dest_wav_path = BENCHMARK_WAVS_DIR / wav_filename

        temp_dest = dest_wav_path.with_suffix(".tmp.wav")
        sf.write(str(temp_dest), mono_sig, 16000, format="WAV", subtype="PCM_16")
        os.replace(str(temp_dest), str(dest_wav_path))

        # 8. Thread-safe atomic manifest and metadata CSV update
        root_dir = Path(__file__).resolve().parents[1]
        rel_wav = str(dest_wav_path.relative_to(root_dir)).replace("\\", "/")

        async with benchmark_lock:
            manifest_data = {}
            if BENCHMARK_MANIFEST_PATH.is_file():
                try:
                    with open(BENCHMARK_MANIFEST_PATH, "r", encoding="utf-8") as mf:
                        manifest_data = json.load(mf)
                except Exception:
                    manifest_data = {}

            manifest_data[sample_id] = {
                "wav": rel_wav,
                "duration": round(duration_s, 2),
                "speaker_id": clean_user,
                "ground_truth": clean_label,
                "device": clean_device,
                "context": clean_context,
                "acoustic_physics": {
                    "f0_mean": round(raw_metrics["f0_mean"], 1) if raw_metrics.get("f0_mean") is not None else None,
                    "f0_std": round(raw_metrics["f0_std"], 1) if raw_metrics.get("f0_std") is not None else None,
                    "hnr": round(raw_metrics["hnr_db"], 1) if raw_metrics.get("hnr_db") is not None else None,
                    "jitter": round(raw_metrics["jitter_percent"], 2) if raw_metrics.get("jitter_percent") is not None else None,
                    "shimmer": round(raw_metrics["shimmer_percent"], 2) if raw_metrics.get("shimmer_percent") is not None else None,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            tmp_manifest = BENCHMARK_MANIFEST_PATH.with_suffix(".tmp")
            with open(tmp_manifest, "w", encoding="utf-8") as mf:
                json.dump(manifest_data, mf, indent=2, ensure_ascii=False)
            os.replace(str(tmp_manifest), str(BENCHMARK_MANIFEST_PATH))

            csv_exists = BENCHMARK_METADATA_PATH.is_file()
            with open(BENCHMARK_METADATA_PATH, "a", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                if not csv_exists:
                    writer.writerow([
                        "sample_id", "timestamp", "user_id", "ground_truth",
                        "duration_s", "f0_mean", "hnr", "device", "context", "filename"
                    ])
                writer.writerow([
                    sample_id,
                    datetime.now(timezone.utc).isoformat(),
                    clean_user,
                    clean_label,
                    round(duration_s, 2),
                    round(raw_metrics.get("f0_mean") or 0.0, 1),
                    round(raw_metrics.get("hnr_db") or 0.0, 1),
                    clean_device,
                    clean_context,
                    wav_filename,
                ])

        return {
            "status": "success",
            "sample_id": sample_id,
            "filename": wav_filename,
            "duration_seconds": round(duration_s, 2),
            "ground_truth": clean_label,
            "user_id": clean_user,
            "message": "Đã lưu mẫu kiểm chứng thành công vào tập dữ liệu benchmark.",
        }
    finally:
        _remove_temp(tmp_path)


@app.get("/api/v1/benchmarks/stats", dependencies=[Depends(verify_api_access)])
async def get_benchmark_stats():
    """Retrieve summary statistics of collected benchmark samples."""
    async with benchmark_lock:
        if not BENCHMARK_MANIFEST_PATH.is_file():
            return {
                "total_samples": 0,
                "speakers_count": 0,
                "speakers": [],
                "ground_truth_counts": {},
            }
        try:
            with open(BENCHMARK_MANIFEST_PATH, "r", encoding="utf-8") as mf:
                data = json.load(mf)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Không đọc được benchmark manifest") from exc

    speakers = set()
    label_counts: Dict[str, int] = {}
    for item in data.values():
        spk = item.get("speaker_id")
        if spk:
            speakers.add(spk)
        lbl = item.get("ground_truth", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    return {
        "total_samples": len(data),
        "speakers_count": len(speakers),
        "speakers": sorted(list(speakers)),
        "ground_truth_counts": label_counts,
    }


# Mount static frontend files at root (served after API endpoints)
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
