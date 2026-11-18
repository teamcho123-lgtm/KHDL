"""FastAPI service for emotion and calibrated vocal arousal/stress proxy analysis."""

import asyncio
import hmac
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.pipeline import StressAnalysisPipeline

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 256 * 1024
MAX_REQUEST_BODY_BYTES = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
UPLOAD_CHUNK_BYTES = 1024 * 1024

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hparams_inference.yaml"
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

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


# Mount static frontend files at root (served after API endpoints)
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
