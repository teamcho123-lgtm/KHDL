"""FastAPI service for emotion and calibrated vocal stress analysis."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.pipeline import StressAnalysisPipeline

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hparams_inference.yaml"
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
pipeline: Optional[StressAnalysisPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initializing stress analysis pipeline")
    pipeline = StressAnalysisPipeline.from_config(str(CONFIG_PATH))
    yield
    pipeline = None


app = FastAPI(
    title="Speech Emotion & Stress Analyzer API",
    description="Speech emotion and personal-baseline stress analysis",
    version="1.1.0",
    lifespan=lifespan,
)

# The API does not use cookie authentication; wildcard origins do not need
# credentialed cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def _save_upload(file: UploadFile) -> str:
    """Write a bounded upload to a temporary audio path preserving suffix and clean up on error."""
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    if not suffix or len(suffix) > 8:
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


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if pipeline is not None else "starting",
        "pipeline_ready": pipeline is not None,
        "emotion_model_ready": bool(pipeline and pipeline.emotion_model),
        "stress_model_ready": bool(pipeline and pipeline.stress_model),
    }


@app.post("/api/v1/register-baseline")
async def register_baseline(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    tmp_path = await _save_upload(file)
    try:
        result = await run_in_threadpool(
            pipeline.register_baseline, user_id=user_id, calm_audio_path=tmp_path
        )
        return {"status": "success", "result": result}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Baseline registration failed")
        raise HTTPException(status_code=500, detail="Baseline registration failed") from exc
    finally:
        _remove_temp(tmp_path)


@app.post("/api/v1/analyze-stress")
async def analyze_stress(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    tmp_path = await _save_upload(file)
    try:
        return await run_in_threadpool(
            pipeline.analyze,
            audio_path=tmp_path,
            user_id=user_id,
            verify_speaker=bool(user_id),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Audio analysis failed")
        raise HTTPException(status_code=500, detail="Audio analysis failed") from exc
    finally:
        _remove_temp(tmp_path)


@app.get("/api/v1/baselines")
def list_baselines():
    """List all registered user baseline profiles."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    profiles = []
    base_dir = str(pipeline.baseline_dir)
    if os.path.isdir(base_dir):
        for f in os.listdir(base_dir):
            if f.endswith("_profile.npz"):
                profiles.append(f[:-12])
    return {"registered_users": sorted(profiles)}


@app.get("/api/v1/samples")
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


@app.get("/api/v1/sample-audio/{sample_id}")
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

