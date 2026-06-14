from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

from config import settings


os.environ["HF_HOME"] = str(settings.cache_dir)
os.environ["TORCH_HOME"] = str(settings.cache_dir)

rembg = BackgroundRemover()
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv-turbo",
    variant="fp16",
)
pipeline.enable_flashvdm()


def run_hunyuan3d(image_path: str | Path, output_path: str | Path) -> None:
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = Image.open(image_path).convert("RGBA")
    if image.mode == "RGB":
        image = rembg(image)

    images = {
        "front": image,
        "left": image,
        "back": image,
    }

    start_time = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=5,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type="trimesh",
    )[0]

    print(f"--- {time.time() - start_time:.2f} seconds ---")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    base_dir = Path.home() / "hunyuan3d"
    run_hunyuan3d(
        image_path=base_dir / "outputs" / "object_crop.png",
        output_path=base_dir / "outputs" / "result.glb",
    )
