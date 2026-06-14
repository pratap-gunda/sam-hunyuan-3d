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
        if settings.flux_sequential_offload:
            pipe.enable_sequential_cpu_offload()
        elif settings.flux_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _enable_memory_helpers(pipe)
    return pipe


def unload_pipe() -> None:
    global pipe
    if pipe is not None:
        free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(free_hooks):
            free_hooks()
        elif not settings.flux_cpu_offload and not settings.flux_sequential_offload:
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

    source = _load_flux_source(object_crop_path)
    _generate_view(source, SIDE_PROMPT, side_path)
    _clear_cuda_cache()
    _generate_view(source, BACK_PROMPT, back_path)
    _clear_cuda_cache()


def generate_side_view(object_crop_path: str | Path, side_path: str | Path) -> None:
    source = _load_flux_source(object_crop_path)
    _generate_view(source, SIDE_PROMPT, Path(side_path))
    _clear_cuda_cache()


def generate_back_view(object_crop_path: str | Path, back_path: str | Path) -> None:
    source = _load_flux_source(object_crop_path)
    _generate_view(source, BACK_PROMPT, Path(back_path))
    _clear_cuda_cache()


def _generate_view(source: Image.Image, prompt: str, output_path: Path) -> None:
    pipeline = load_pipe()
    with torch.inference_mode():
        result = pipeline(
            image=source,
            prompt=prompt,
            guidance_scale=settings.flux_guidance_scale,
            num_inference_steps=settings.flux_steps,
            width=source.width,
            height=source.height,
        ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    del result


def _load_flux_source(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    return _resize_for_flux(image)


def _resize_for_flux(image: Image.Image) -> Image.Image:
    max_size = max(256, settings.flux_max_size)
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / longest
    new_width = max(64, int(width * scale) // 16 * 16)
    new_height = max(64, int(height * scale) // 16 * 16)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _enable_memory_helpers(pipeline: FluxKontextPipeline) -> None:
    for method_name in (
        "enable_vae_slicing",
        "enable_vae_tiling",
        "enable_attention_slicing",
    ):
        method = getattr(pipeline, method_name, None)
        if callable(method):
            method()


def _clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
