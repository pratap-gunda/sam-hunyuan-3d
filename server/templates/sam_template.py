from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import settings


CHECKPOINT = settings.cache_dir / "checkpoints" / "sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG, str(CHECKPOINT), device=device))


def run_sam2(
    input_image_path: str | Path,
    points: list[list[int]],
    labels: list[int],
    refined_mask_path: str | Path,
    object_rgba_path: str | Path,
    object_crop_path: str | Path,
) -> None:
    input_image_path = Path(input_image_path)
    refined_mask_path = Path(refined_mask_path)
    object_rgba_path = Path(object_rgba_path)
    object_crop_path = Path(object_crop_path)

    if not points or len(points) != len(labels):
        raise ValueError("SAM2 requires matching non-empty points and labels")

    image = np.array(Image.open(input_image_path).convert("RGB"))
    predictor.set_image(image)

    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    with torch.inference_mode():
        masks, _scores, _logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    final_mask = masks[0].astype(np.uint8) * 255
    refined_mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_mask).save(refined_mask_path)

    original_rgba = np.array(Image.open(input_image_path).convert("RGBA"))
    mask = np.array(Image.open(refined_mask_path).convert("L"))

    mask_factor = mask / 255.0
    original_rgba[:, :, 0] = (original_rgba[:, :, 0] * mask_factor).astype(np.uint8)
    original_rgba[:, :, 1] = (original_rgba[:, :, 1] * mask_factor).astype(np.uint8)
    original_rgba[:, :, 2] = (original_rgba[:, :, 2] * mask_factor).astype(np.uint8)
    original_rgba[:, :, 3] = mask

    object_rgba = Image.fromarray(original_rgba, mode="RGBA")
    object_rgba.save(object_rgba_path)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("SAM2 produced an empty refined mask")

    crop_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    object_rgba.crop(crop_box).save(object_crop_path)


if __name__ == "__main__":
    base_dir = Path.home() / "hunyuan3d"
    run_sam2(
        input_image_path=base_dir / "assets" / "input.png",
        points=[[100, 100]],
        labels=[1],
        refined_mask_path=base_dir / "outputs" / "refined_mask.png",
        object_rgba_path=base_dir / "outputs" / "object_rgba.png",
        object_crop_path=base_dir / "outputs" / "object_crop.png",
    )
