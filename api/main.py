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

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "hparams_inference.yaml"
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
    """Write a bounded upload to a temporary WAV path and clean up on error."""
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
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
