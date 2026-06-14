from __future__ import annotations

import os
import time
import gc
from pathlib import Path

import torch
from PIL import Image
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

from config import settings


os.environ["HF_HOME"] = str(settings.cache_dir)
os.environ["TORCH_HOME"] = str(settings.cache_dir)

rembg: BackgroundRemover | None = None
pipeline: Hunyuan3DDiTFlowMatchingPipeline | None = None


def load_pipeline() -> Hunyuan3DDiTFlowMatchingPipeline:
    global pipeline
    if pipeline is None:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv",
            subfolder="hunyuan3d-dit-v2-mv-turbo",
            variant="fp16",
        )
        pipeline.enable_flashvdm()
    return pipeline


def unload_pipeline() -> None:
    global pipeline
    if pipeline is not None:
        del pipeline
        pipeline = None
    gc.collect()
    torch.cuda.empty_cache()


def background_remover() -> BackgroundRemover:
    global rembg
    if rembg is None:
        rembg = BackgroundRemover()
    return rembg


def run_hunyuan3d(
    front_path: str | Path,
    side_path: str | Path,
    back_path: str | Path,
    output_path: str | Path,
) -> None:
    front_path = Path(front_path)
    side_path = Path(side_path)
    back_path = Path(back_path)
    output_path = Path(output_path)

    images = {
        "front": _load_rgba(front_path),
        "left": _load_rgba(side_path),
        "back": _load_rgba(back_path),
    }

    start_time = time.time()
    pipeline_instance = load_pipeline()
    mesh = pipeline_instance(
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


def _load_rgba(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.mode == "RGB":
        return background_remover()(image)
    return image


if __name__ == "__main__":
    base_dir = Path.home() / "hunyuan3d"
    run_hunyuan3d(
        front_path=base_dir / "outputs" / "front.png",
        side_path=base_dir / "outputs" / "side.png",
        back_path=base_dir / "outputs" / "back.png",
        output_path=base_dir / "outputs" / "result.glb",
    )
