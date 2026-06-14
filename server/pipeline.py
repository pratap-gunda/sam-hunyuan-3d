from __future__ import annotations

import importlib
import logging
import shutil
import gc
from pathlib import Path

import torch

from config import settings
from job_manager import read_points, write_status

logger = logging.getLogger(__name__)


def run_pipeline(job_id: str, job_path: Path) -> None:
    input_path = job_path / "input.png"
    refined_mask_path = job_path / "refined_mask.png"
    object_rgba_path = job_path / "object_rgba.png"
    object_crop_path = job_path / "object_crop.png"
    front_path = job_path / "front.png"
    side_path = job_path / "side.png"
    back_path = job_path / "back.png"
    result_path = job_path / "result.glb"

    _require_file(input_path, "input image")
    _require_file(job_path / "points.json", "point prompts")

    try:
        write_status(job_id, "running", 30, "Running SAM2")
        run_sam_stage(job_id, job_path)
        _require_file(refined_mask_path, "refined mask")
        _require_file(object_rgba_path, "RGBA cutout")
        _require_file(object_crop_path, "object crop")
        release_cuda_memory()
        run_flux_stage(job_id, job_path)
        release_cuda_memory()
        _require_file(front_path, "front view")
        _require_file(side_path, "side view")
        _require_file(back_path, "back view")

        write_status(job_id, "running", 90, "Running Hunyuan3D")
        hunyuan_template = importlib.import_module("templates.hunyuan_template")
        hunyuan_template.run_hunyuan3d(
            front_path=front_path,
            side_path=side_path,
            back_path=back_path,
            output_path=result_path,
        )
        if settings.low_vram and hasattr(hunyuan_template, "unload_pipeline"):
            hunyuan_template.unload_pipeline()
        _require_file(result_path, "Hunyuan3D result")
        write_status(job_id, "completed", 100, "Completed")
        logger.info("Job %s completed", job_id)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        write_status(job_id, "failed", 100, str(exc))
        raise


def run_view_pipeline(job_id: str, job_path: Path) -> None:
    _require_file(job_path / "input.png", "input image")
    _require_file(job_path / "points.json", "point prompts")
    try:
        write_status(job_id, "running", 30, "Running SAM2")
        run_sam_stage(job_id, job_path)
        _require_file(job_path / "object_crop.png", "object crop")
        release_cuda_memory()
        run_flux_stage(job_id, job_path)
        release_cuda_memory()
        write_status(job_id, "queued", 80, "Multi-view package ready")
        logger.info("Views generated for job %s", job_id)
    except Exception as exc:
        logger.exception("View generation failed for job %s", job_id)
        write_status(job_id, "failed", 100, str(exc))
        raise


def run_sam_stage(job_id: str, job_path: Path) -> None:
    prompts = read_points(job_id)
    sam_template = importlib.import_module("templates.sam_template")
    try:
        sam_template.run_sam2(
            input_image_path=job_path / "input.png",
            points=prompts.points,
            labels=prompts.labels,
            refined_mask_path=job_path / "refined_mask.png",
            object_rgba_path=job_path / "object_rgba.png",
            object_crop_path=job_path / "object_crop.png",
        )
    finally:
        if settings.low_vram and hasattr(sam_template, "unload_predictor"):
            sam_template.unload_predictor()


def run_flux_stage(job_id: str, job_path: Path) -> None:
    object_crop_path = job_path / "object_crop.png"
    front_path = job_path / "front.png"
    side_path = job_path / "side.png"
    back_path = job_path / "back.png"

    _require_file(object_crop_path, "object crop")
    shutil.copyfile(object_crop_path, front_path)

    flux_template = importlib.import_module("templates.flux_template")
    write_status(job_id, "running", 50, "Generating Side View")
    flux_template.generate_side_view(object_crop_path, side_path)
    _require_file(side_path, "side view")

    write_status(job_id, "running", 70, "Generating Back View")
    flux_template.generate_back_view(object_crop_path, back_path)
    _require_file(back_path, "back view")

    write_status(job_id, "running", 80, "Generating Multi-View Package")
    _require_file(front_path, "front view")
    if settings.low_vram and hasattr(flux_template, "unload_pipe"):
        flux_template.unload_pipe()


def run_preview(job_id: str, job_path: Path) -> Path:
    _require_file(job_path / "input.png", "input image")
    _require_file(job_path / "points.json", "point prompts")
    write_status(job_id, "running", 20, "Running SAM2 preview")
    try:
        run_sam_stage(job_id, job_path)
        release_cuda_memory()
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


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
