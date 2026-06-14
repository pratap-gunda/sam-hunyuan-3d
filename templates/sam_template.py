import os
import cv2
import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================
# PATHS
# ============================================

BASE_DIR = os.path.expanduser("~/hunyuan3d")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_PATH = os.path.join(ASSETS_DIR, "input.png")
ROUGH_MASK_PATH = os.path.join(ASSETS_DIR, "rough_mask.png")

OUTPUT_MASK_PATH = os.path.join(
    OUTPUT_DIR,
    "refined_mask.png"
)

CHECKPOINT = os.path.join(
    CACHE_DIR,
    "checkpoints",
    "sam2.1_hiera_large.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ============================================
# DEVICE
# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# LOAD MODEL
# ============================================

predictor = SAM2ImagePredictor(
    build_sam2(
        MODEL_CFG,
        CHECKPOINT,
        device=device
    )
)

# ============================================
# LOAD IMAGE
# ============================================

image = np.array(
    Image.open(IMAGE_PATH).convert("RGB")
)

predictor.set_image(image)

# ============================================
# LOAD ROUGH MASK
# ============================================

rough_mask = np.array(
    Image.open(ROUGH_MASK_PATH).convert("L")
)

rough_mask = (rough_mask > 127).astype(np.uint8)

# ============================================
# CREATE BOX FROM MASK
# ============================================

ys, xs = np.where(rough_mask > 0)

x1 = xs.min()
y1 = ys.min()

x2 = xs.max()
y2 = ys.max()

input_box = np.array(
    [x1, y1, x2, y2]
)

# ============================================
# REFINE
# ============================================

with torch.inference_mode():
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False
    )

# ============================================
# SAVE
# ============================================

final_mask = (
    masks[0].astype(np.uint8) * 255
)

Image.fromarray(final_mask).save(
    OUTPUT_MASK_PATH
)

print("Saved:", OUTPUT_MASK_PATH)

####
# ============================================
# CUT OBJECT FROM IMAGE
# ============================================

original = Image.open(IMAGE_PATH).convert("RGBA")

mask = Image.open(OUTPUT_MASK_PATH).convert("L")

# Apply mask as alpha
original.putalpha(mask)

object_rgba_path = os.path.join(
    OUTPUT_DIR,
    "object_rgba.png"
)

original.save(object_rgba_path)

print("Saved:", object_rgba_path)

# ============================================
# CROP TO OBJECT BOUNDS
# ============================================

mask_np = np.array(mask)

ys, xs = np.where(mask_np > 0)

if len(xs) > 0:

    x1 = xs.min()
    y1 = ys.min()

    x2 = xs.max()
    y2 = ys.max()

    cropped = original.crop(
        (x1, y1, x2 + 1, y2 + 1)
    )

    object_crop_path = os.path.join(
        OUTPUT_DIR,
        "object_crop.png"
    )

    cropped.save(object_crop_path)

    print("Saved:", object_crop_path)

# ============================================
# CREATE TRUE CUTOUT
# RGB OUTSIDE MASK = 0
# ALPHA OUTSIDE MASK = 0
# ============================================

original = np.array(
    Image.open(IMAGE_PATH).convert("RGBA")
)

mask = np.array(
    Image.open(OUTPUT_MASK_PATH).convert("L")
)

# Apply mask to RGB
original[:, :, 0] = (
    original[:, :, 0] * (mask / 255.0)
).astype(np.uint8)

original[:, :, 1] = (
    original[:, :, 1] * (mask / 255.0)
).astype(np.uint8)

original[:, :, 2] = (
    original[:, :, 2] * (mask / 255.0)
).astype(np.uint8)

# Alpha channel
original[:, :, 3] = mask

object_rgba = Image.fromarray(
    original,
    mode="RGBA"
)

object_rgba_path = os.path.join(
    OUTPUT_DIR,
    "premult.png"
)

object_rgba.save(
    object_rgba_path
)

print("Saved:", object_rgba_path)