from __future__ import annotations

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from config import configure_logging, ensure_directories
from job_manager import delete_job, download_path, list_jobs, read_status, require_job_dir, save_uploads, write_points, write_status
from models import GenerateResponse, JobsResponse, PointPrompts, UploadResponse
from pipeline import run_preview
from worker import worker_pool

configure_logging()
ensure_directories()

logger = logging.getLogger(__name__)

app = FastAPI(title="SAM2 Hunyuan3D Generation Server", version="1.0.0")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    image: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...),
) -> UploadResponse:
    prompts = _parse_prompts(points, labels)
    job_id = await save_uploads(image, prompts)
    logger.info("Created job %s", job_id)
    return UploadResponse(job_id=job_id)


@app.post("/points/{job_id}")
def update_points(job_id: str, prompts: PointPrompts) -> PointPrompts:
    status = read_status(job_id)
    if status.status == "running":
        raise HTTPException(status_code=409, detail="Cannot update points while the job is running")
    updated = write_points(job_id, prompts)
    write_status(job_id, "queued", 0, "Points updated")
    return updated


@app.post("/preview_mask/{job_id}")
def preview_mask(job_id: str) -> FileResponse:
    job_path = require_job_dir(job_id)
    path = run_preview(job_id, job_path)
    return FileResponse(path=path, filename="refined_mask.png", media_type="image/png")


@app.post("/generate/{job_id}", response_model=GenerateResponse)
def generate(job_id: str) -> GenerateResponse:
    status = read_status(job_id)
    if status.status == "completed":
        return GenerateResponse(job_id=job_id, status="completed", message="Job already completed")
    worker_pool.start(job_id)
    return GenerateResponse(job_id=job_id, status="queued", message="Generation started")


@app.get("/status/{job_id}")
def status(job_id: str):
    return read_status(job_id)


@app.get("/jobs", response_model=JobsResponse)
def jobs() -> JobsResponse:
    return JobsResponse(jobs=list_jobs())


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str) -> FileResponse:
    path = download_path(job_id, filename)
    return FileResponse(path=path, filename=filename)


@app.delete("/job/{job_id}")
def remove_job(job_id: str) -> dict[str, str]:
    try:
        delete_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    logger.info("Deleted job %s", job_id)
    return {"job_id": job_id, "status": "deleted"}


def _parse_prompts(points: str, labels: str) -> PointPrompts:
    try:
        return PointPrompts(points=json.loads(points), labels=json.loads(labels))
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid point prompts: {exc}") from exc
