from __future__ import annotations

import os
import shutil
import gc
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline
from PIL import Image

from config import settings


os.environ["HF_HOME"] = str(settings.cache_dir)
os.environ["TORCH_HOME"] = str(settings.cache_dir)

if not torch.cuda.is_available():
    raise RuntimeError("FLUX Kontext requires CUDA GPU execution")

pipe: FluxKontextPipeline | None = None


def load_pipe() -> FluxKontextPipeline:
    global pipe
    if pipe is None:
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
    return pipe


def unload_pipe() -> None:
    global pipe
    if pipe is not None:
        pipe.to("cpu")
        del pipe
        pipe = None
    gc.collect()
    torch.cuda.empty_cache()


SIDE_PROMPT = """This is a studio photograph of a single object on a transparent background.

Generate the exact same object viewed from the LEFT SIDE.

Requirements:

* preserve object identity
* preserve proportions
* preserve geometry
* preserve materials
* preserve texture
* preserve color
* preserve scale
* preserve lighting consistency
* no new parts
* no missing parts
* no deformation
* transparent background
* centered object
* realistic side view"""


BACK_PROMPT = """This is a studio photograph of a single object on a transparent background.

Generate the exact same object viewed from the BACK.

Requirements:

* preserve object identity
* preserve proportions
* preserve geometry
* preserve materials
* preserve texture
* preserve color
* preserve scale
* preserve lighting consistency
* no new parts
* no missing parts
* no deformation
* transparent background
* centered object
* realistic back view"""


def generate_multiview(
    object_crop_path: str | Path,
    front_path: str | Path,
    side_path: str | Path,
    back_path: str | Path,
) -> None:
    object_crop_path = Path(object_crop_path)
    front_path = Path(front_path)
    side_path = Path(side_path)
    back_path = Path(back_path)

    if not object_crop_path.exists():
        raise FileNotFoundError(f"Missing object crop: {object_crop_path}")

    front_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(object_crop_path, front_path)

    source = Image.open(object_crop_path).convert("RGBA")
    _generate_view(source, SIDE_PROMPT, side_path)
    _generate_view(source, BACK_PROMPT, back_path)


def generate_side_view(object_crop_path: str | Path, side_path: str | Path) -> None:
    source = Image.open(object_crop_path).convert("RGBA")
    _generate_view(source, SIDE_PROMPT, Path(side_path))


def generate_back_view(object_crop_path: str | Path, back_path: str | Path) -> None:
    source = Image.open(object_crop_path).convert("RGBA")
    _generate_view(source, BACK_PROMPT, Path(back_path))


def _generate_view(source: Image.Image, prompt: str, output_path: Path) -> None:
    pipeline = load_pipe()
    result = pipeline(
        image=source,
        prompt=prompt,
        guidance_scale=2.5,
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
