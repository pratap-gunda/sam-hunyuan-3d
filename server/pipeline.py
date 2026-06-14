from __future__ import annotations

import importlib
import logging
from pathlib import Path

from job_manager import read_points, write_status

logger = logging.getLogger(__name__)


def run_pipeline(job_id: str, job_path: Path) -> None:
    input_path = job_path / "input.png"
    refined_mask_path = job_path / "refined_mask.png"
    object_rgba_path = job_path / "object_rgba.png"
    object_crop_path = job_path / "object_crop.png"
    result_path = job_path / "result.glb"

    _require_file(input_path, "input image")
    _require_file(job_path / "points.json", "point prompts")

    try:
        write_status(job_id, "running", 30, "Running SAM2")
        run_sam_stage(job_id, job_path)
        _require_file(refined_mask_path, "refined mask")
        _require_file(object_rgba_path, "RGBA cutout")
        _require_file(object_crop_path, "object crop")
        write_status(job_id, "running", 60, "SAM2 complete")

        write_status(job_id, "running", 80, "Running Hunyuan3D")
        hunyuan_template = importlib.import_module("templates.hunyuan_template")
        hunyuan_template.run_hunyuan3d(
            image_path=object_crop_path,
            output_path=result_path,
        )
        _require_file(result_path, "Hunyuan3D result")
        write_status(job_id, "completed", 100, "Complete")
        logger.info("Job %s completed", job_id)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        write_status(job_id, "failed", 100, str(exc))
        raise


def run_sam_stage(job_id: str, job_path: Path) -> None:
    prompts = read_points(job_id)
    sam_template = importlib.import_module("templates.sam_template")
    sam_template.run_sam2(
        input_image_path=job_path / "input.png",
        points=prompts.points,
        labels=prompts.labels,
        refined_mask_path=job_path / "refined_mask.png",
        object_rgba_path=job_path / "object_rgba.png",
        object_crop_path=job_path / "object_crop.png",
    )


def run_preview(job_id: str, job_path: Path) -> Path:
    _require_file(job_path / "input.png", "input image")
    _require_file(job_path / "points.json", "point prompts")
    write_status(job_id, "running", 20, "Running SAM2 preview")
    try:
        run_sam_stage(job_id, job_path)
        refined_mask_path = job_path / "refined_mask.png"
        _require_file(refined_mask_path, "refined mask")
        write_status(job_id, "queued", 40, "Mask preview ready")
        return refined_mask_path
    except Exception as exc:
        logger.exception("Preview failed for job %s", job_id)
        write_status(job_id, "failed", 100, str(exc))
        raise


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
