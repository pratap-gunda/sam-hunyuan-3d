from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from config import settings
from models import JobStatus, PointPrompts


ALLOWED_DOWNLOADS = {
    "refined_mask.png",
    "object_rgba.png",
    "object_crop.png",
    "result.glb",
}

_status_lock = threading.RLock()


def new_job_id() -> str:
    return uuid.uuid4().hex[:8]


def validate_job_id(job_id: str) -> str:
    try:
        uuid.UUID(job_id if len(job_id) == 32 else job_id.ljust(32, "0"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job id") from exc
    if not job_id.isalnum() or len(job_id) not in {8, 32}:
        raise HTTPException(status_code=400, detail="Invalid job id")
    return job_id


def job_dir(job_id: str) -> Path:
    validate_job_id(job_id)
    path = (settings.jobs_dir / job_id).resolve()
    if settings.jobs_dir.resolve() not in path.parents and path != settings.jobs_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid job path")
    return path


def require_job_dir(job_id: str) -> Path:
    path = job_dir(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return path


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def write_status(job_id: str, status: str, progress: int, message: str) -> JobStatus:
    payload = JobStatus(job_id=job_id, status=status, progress=progress, message=message)
    with _status_lock:
        path = status_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)
    return payload


def read_status(job_id: str) -> JobStatus:
    require_job_dir(job_id)
    path = status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job status not found")
    with _status_lock:
        return JobStatus.model_validate_json(path.read_text(encoding="utf-8"))


def list_jobs() -> list[JobStatus]:
    jobs: list[JobStatus] = []
    for path in sorted(settings.jobs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.is_dir() and (path / "status.json").exists():
            try:
                jobs.append(read_status(path.name))
            except (json.JSONDecodeError, ValueError, HTTPException):
                continue
    return jobs


async def save_uploads(image: UploadFile, prompts: PointPrompts) -> str:
    job_id = new_job_id()
    path = job_dir(job_id)
    path.mkdir(parents=True, exist_ok=False)

    try:
        await _save_valid_image(image, path / "input.png", "RGB")
        write_points(job_id, prompts)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)
        raise

    write_status(job_id, "queued", 0, "Queued")
    return job_id


def points_path(job_id: str) -> Path:
    return job_dir(job_id) / "points.json"


def write_points(job_id: str, prompts: PointPrompts) -> PointPrompts:
    require_job_dir(job_id)
    path = points_path(job_id)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(prompts.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return prompts


def read_points(job_id: str) -> PointPrompts:
    require_job_dir(job_id)
    path = points_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Point prompts not found")
    return PointPrompts.model_validate_json(path.read_text(encoding="utf-8"))


async def _save_valid_image(upload: UploadFile, destination: Path, mode: str) -> None:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded file")
    try:
        with Image.open(upload.file) as img:
            img.verify()
        upload.file.seek(0)
        with Image.open(upload.file) as img:
            converted = img.convert(mode)
            converted.save(destination, format="PNG")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {upload.filename}") from exc
    finally:
        await upload.close()


def delete_job(job_id: str) -> None:
    path = require_job_dir(job_id)
    shutil.rmtree(path)


def download_path(job_id: str, filename: str) -> Path:
    if filename not in ALLOWED_DOWNLOADS:
        raise HTTPException(status_code=400, detail="File is not downloadable")
    path = require_job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return path
