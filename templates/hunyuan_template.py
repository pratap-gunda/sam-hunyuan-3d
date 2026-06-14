import os
import time
import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# =========================
# Base paths
# =========================
BASE_DIR = os.path.expanduser("~/hunyuan3d")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================
# Cache setup
# =========================
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR

# =========================
# Load images
# =========================
images = {
    "front": os.path.join(ASSETS_DIR, "front.png"),
    "left": os.path.join(ASSETS_DIR, "left.png"),
    "back": os.path.join(ASSETS_DIR, "back.png")
}

rembg = BackgroundRemover()

for key in images:
    image = Image.open(images[key]).convert("RGBA")
    if image.mode == "RGB":
        image = rembg(image)
    images[key] = image

# =========================
# Load pipeline
# =========================
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv-turbo",
    variant="fp16"
)

pipeline.enable_flashvdm()

# =========================
# Run inference
# =========================
start_time = time.time()

mesh = pipeline(
    image=images,
    num_inference_steps=5,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type="trimesh"
)[0]

print(f"--- {time.time() - start_time:.2f} seconds ---")

# =========================
# Save output
# =========================
output_path = os.path.join(OUTPUT_DIR, "demo_mv3.glb")
mesh.export(output_path)

print(f"Saved to: {output_path}")