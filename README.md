# SAM 2.1 + FLUX.1 Kontext + Hunyuan3D Client/Server 3D Reconstruction Pipeline

A client/server AI pipeline for converting a **single input image into a 3D asset** using:

* **SAM 2.1 Hiera Large** — object segmentation
* **FLUX.1 Kontext-dev** — multi-view image generation
* **Hunyuan3D** — 3D asset generation/reconstruction
* **FastAPI** — GPU inference server
* **PySide6** — Windows desktop client
* **Vast.ai** — GPU hosting

The heavy AI processing runs on the Vast.ai GPU server, while the Windows client provides an interactive interface for image selection, SAM point prompting, mask refinement, multi-view generation, and final 3D mesh download.

---

# 1. Pipeline Overview

```text
                         WINDOWS CLIENT
                              │
                              │ Upload Image
                              ▼
                    ┌─────────────────────┐
                    │     SAM 2.1         │
                    │   Hiera Large       │
                    │ Object Segmentation │
                    └──────────┬──────────┘
                               │
                               ▼
                        Object Crop
                        object_crop.png
                               │
                               ▼
                    ┌─────────────────────┐
                    │  FLUX.1 Kontext-dev │
                    │  Multi-view Image   │
                    │     Generation      │
                    └──────────┬──────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
               Front          Side         Back
             front.png      side.png      back.png
                  │            │            │
                  └────────────┼────────────┘
                               ▼
                    ┌─────────────────────┐
                    │      Hunyuan3D      │
                    │ 3D Reconstruction / │
                    │    Generation       │
                    └──────────┬──────────┘
                               │
                               ▼
                         FINAL 3D MESH
```

---

# 2. AI Models Used

## SAM 2.1 — Hiera Large

**Purpose:** Interactive object segmentation.

**Model:**

```text
SAM 2.1 Hiera Large
```

**Checkpoint:**

```text
sam2.1_hiera_large.pt
```

SAM 2.1 receives the original image together with user-defined point prompts.

### Point Prompt System

The Windows client supports interactive SAM-style prompting:

* **Left click** → Foreground point
* **Right click** → Background point
* Multiple points can be added
* Mask preview can be regenerated repeatedly

The client sends:

```text
image
points
labels
```

to the server.

SAM 2.1 generates the segmentation mask, which is then used to extract the selected object.

### Output

```text
object_crop.png
```

---

# 3. FLUX.1 Kontext-dev

**Purpose:** Generate consistent multi-view images of the segmented object.

**Model:**

```text
black-forest-labs/FLUX.1-Kontext-dev
```

FLUX Kontext receives:

```text
jobs/<job_id>/object_crop.png
```

and generates multiple views.

### Generated Views

```text
front.png
side.png
back.png
```

These views are passed to Hunyuan3D for 3D reconstruction.

### Multi-view Generation Goals

The FLUX stage is intended to provide:

* Front view
* Side view
* Back view
* Consistent object appearance
* Consistent materials
* Consistent visual characteristics
* Additional geometric information for 3D reconstruction

---

# 4. Hunyuan3D

**Purpose:** Generate the final 3D asset from the generated multi-view images.

Hunyuan3D receives:

```text
front.png
side.png
back.png
```

It does **not** directly consume:

```text
input.png
object_crop.png
```

The 3D reconstruction stage therefore operates only on the generated multi-view representation.

### Hunyuan3D Input

```text
front.png
side.png
back.png
```

### Hunyuan3D Output

The generated 3D asset/mesh is saved to the job output directory and downloaded by the Windows client.

---

# 5. Server Architecture

The server runs on a **Vast.ai GPU instance** using FastAPI.

```text
/root/server/
│
├── api.py
├── requirements.txt
│
├── templates/
│   ├── sam_template.py
│   └── hunyuan_template.py
│
├── cache/
│   └── checkpoints/
│       └── sam2.1_hiera_large.pt
│
├── jobs/
│   └── <job_id>/
│       ├── input.png
│       ├── points.json
│       ├── object_crop.png
│       ├── front.png
│       ├── side.png
│       ├── back.png
│       └── output/
│
└── logs/
    └── server.log
```

---

# 6. Server Setup — Vast.ai

Copy the `server/` directory to:

```text
/root/server/
```

Then install the dependencies:

```bash
cd /root/server
python3.12 -m pip install -r requirements.txt
```

---

# 7. Install SAM 2.1

Install SAM 2 into the **same Python environment** used to run the FastAPI server.

```bash
cd ~

git clone https://github.com/facebookresearch/sam2.git

cd sam2

python3.12 -m pip install -e .
```

---

# 8. Download SAM 2.1 Hiera Large Checkpoint

Create the checkpoint directory:

```bash
mkdir -p /root/server/cache/checkpoints
```

Download the checkpoint:

```bash
cd /root/server/cache/checkpoints

wget -O sam2.1_hiera_large.pt \
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Verify the checkpoint:

```bash
ls -lh /root/server/cache/checkpoints/sam2.1_hiera_large.pt
```

Expected file:

```text
/root/server/cache/checkpoints/sam2.1_hiera_large.pt
```

---

# 9. Start FastAPI Server

For a standard configuration:

```bash
cd /root/server

uvicorn api:app --host 0.0.0.0 --port 8000
```

The server will be available at:

```text
http://<VAST_IP>:8000
```

Make sure port **8000** is exposed by the Vast.ai instance.

---

# 10. Low-VRAM Configuration — 12 GB GPU

For GPUs with approximately **12 GB VRAM**, run the pipeline in low-VRAM mode.

SAM 2.1, FLUX, and Hunyuan3D should be loaded sequentially rather than remaining in GPU memory simultaneously.

```bash
cd /root/server

SERVER_LOW_VRAM=1 \
SERVER_MAX_WORKERS=1 \
FLUX_CPU_OFFLOAD=1 \
FLUX_MAX_SIZE=768 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Low-VRAM Strategy

```text
SAM 2.1
   │
   ▼
Unload SAM
   │
   ▼
FLUX Kontext
   │
   ▼
Unload FLUX
   │
   ▼
Hunyuan3D
   │
   ▼
Final Mesh
```

This reduces peak GPU memory usage.

---

# 11. 24 GB GPU — Aggressive Memory Saving

If FLUX still produces CUDA out-of-memory errors on a 24 GB GPU, use sequential CPU offloading and reduce the generated view size.

```bash
cd /root/server

SERVER_LOW_VRAM=1 \
SERVER_MAX_WORKERS=1 \
FLUX_SEQUENTIAL_OFFLOAD=1 \
FLUX_MAX_SIZE=512 \
FLUX_STEPS=16 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uvicorn api:app --host 0.0.0.0 --port 8000
```

This configuration prioritizes memory stability over generation speed.

---

# 12. High-VRAM Configuration

For larger GPUs, FLUX Kontext can be preloaded during server startup.

```bash
cd /root/server

SERVER_LOW_VRAM=0 \
SERVER_PRELOAD_FLUX=1 \
SERVER_MAX_WORKERS=1 \
uvicorn api:app --host 0.0.0.0 --port 8000
```

This reduces model-loading overhead but requires significantly more GPU memory.

---

# 13. Windows Client Setup

The client is a **PySide6 desktop application**.

Open PowerShell:

```powershell
cd client
```

Install dependencies:

```powershell
py -3.12 -m pip install -r requirements.txt
```

Start the application:

```powershell
py -3.12 main.py
```

---

# 14. Client Workflow

## Step 1 — Configure Server

Enter the Vast.ai endpoint into:

```text
Server URL
```

Example:

```text
http://<VAST_IP>:8000
```

---

## Step 2 — Select Input Image

The user selects an image containing the object that should be converted into a 3D asset.

Example:

```text
input.png
```

---

# 15. Step 3 — Add SAM 2.1 Points

The image viewer supports interactive point prompting.

```text
LEFT CLICK
     │
     ▼
Foreground Point
(Green)
```

```text
RIGHT CLICK
     │
     ▼
Background Point
(Red)
```

Multiple points can be added to improve segmentation accuracy.

---

# 16. Step 4 — Generate Mask Preview

Click:

```text
Generate Mask Preview
```

The client sends:

```text
image
points
labels
```

to the FastAPI server.

The server runs:

```text
SAM 2.1 Hiera Large
```

and returns the segmentation result.

The user can:

1. Add more foreground points
2. Add background points
3. Generate the mask again
4. Continue refining until the desired object is isolated

---

# 17. Step 5 — Generate Multi-Views

After confirming the mask, click:

```text
Generate Views
```

The server creates:

```text
object_crop.png
```

The FLUX stage then generates:

```text
front.png
side.png
back.png
```

using:

```text
black-forest-labs/FLUX.1-Kontext-dev
```

---

# 18. Step 6 — Generate 3D Mesh

After the three views have been generated, click:

```text
Generate Mesh
```

The server passes:

```text
front.png
side.png
back.png
```

to Hunyuan3D.

```text
Front View ──┐
             │
Side View ───┼──► Hunyuan3D ──► 3D Asset
             │
Back View ───┘
```

The resulting mesh is saved and made available for download.

---

# 19. Job Storage

Every request receives a unique:

```text
<job_id>
```

Job files are stored under:

```text
/root/server/jobs/<job_id>/
```

Example:

```text
/root/server/jobs/abc123/
├── input.png
├── points.json
├── object_crop.png
├── front.png
├── side.png
├── back.png
└── output/
    └── generated_mesh
```

The Windows client stores completed downloads under:

```text
client/downloads/<job_id>/
```

---

# 20. API Data Flow

The upload request contains the original image and SAM point prompts.

Example:

```json
{
    "image": "...",
    "points": [
        [320, 240],
        [450, 300],
        [200, 400]
    ],
    "labels": [
        1,
        1,
        0
    ]
}
```

Where:

```text
1 = foreground
0 = background
```

The server stores the prompts at:

```text
/root/server/jobs/<job_id>/points.json
```

---

# 21. Server Template Entry Points

The server separates the major AI stages into template modules.

### SAM 2.1

```python
templates.sam_template.run_sam2(...)
```

Responsible for:

* Loading SAM 2.1
* Processing point prompts
* Generating the segmentation mask
* Extracting the selected object
* Creating `object_crop.png`

### Hunyuan3D

```python
templates.hunyuan_template.run_hunyuan3d(...)
```

Responsible for:

* Loading Hunyuan3D
* Reading the generated multi-view images
* Running 3D reconstruction/generation
* Saving the final 3D asset

### FLUX Kontext

FLUX.1 Kontext-dev runs between the SAM and Hunyuan3D stages.

```text
SAM 2.1
   │
   ▼
object_crop.png
   │
   ▼
FLUX.1 Kontext-dev
   │
   ├──► front.png
   ├──► side.png
   └──► back.png
             │
             ▼
        Hunyuan3D
```

---

# 22. Complete Model Pipeline

```text
┌───────────────────────────┐
│       INPUT IMAGE         │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     SAM 2.1 Hiera Large   │
│                           │
│ Interactive Segmentation  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      OBJECT CROP          │
│     object_crop.png       │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    FLUX.1 Kontext-dev     │
│                           │
│   Multi-view Generation   │
└─────────────┬─────────────┘
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
    FRONT    SIDE    BACK
     PNG      PNG     PNG
       │      │      │
       └──────┼──────┘
              │
              ▼
┌───────────────────────────┐
│        Hunyuan3D          │
│                           │
│ 3D Reconstruction / Gen.  │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│       FINAL 3D ASSET      │
└───────────────────────────┘
```

---

# 23. Model Responsibility Summary

| Model                   | Purpose                      | Input                          | Output              |
| ----------------------- | ---------------------------- | ------------------------------ | ------------------- |
| **SAM 2.1 Hiera Large** | Object segmentation          | Original image + point prompts | Mask / object crop  |
| **FLUX.1 Kontext-dev**  | Multi-view generation        | Object crop                    | Front / Side / Back |
| **Hunyuan3D**           | 3D reconstruction/generation | Generated multi-view images    | 3D asset / mesh     |

---

# 24. Hardware Strategy

The pipeline is designed to support different GPU configurations.

### 12 GB GPU

```text
SAM 2.1
   ↓
Unload
   ↓
FLUX Kontext + CPU Offload
   ↓
Unload
   ↓
Hunyuan3D
```

Recommended settings:

```text
SERVER_LOW_VRAM=1
SERVER_MAX_WORKERS=1
FLUX_CPU_OFFLOAD=1
FLUX_MAX_SIZE=768
```

### 24 GB GPU

Use low-VRAM mode if necessary:

```text
SERVER_LOW_VRAM=1
SERVER_MAX_WORKERS=1
FLUX_SEQUENTIAL_OFFLOAD=1
FLUX_MAX_SIZE=512
FLUX_STEPS=16
```

### Higher-VRAM GPU

Models can be kept in memory more aggressively, reducing model-loading overhead.

```text
SERVER_LOW_VRAM=0
SERVER_PRELOAD_FLUX=1
SERVER_MAX_WORKERS=1
```

---

# 25. Design Philosophy

The system is intentionally modular.

```text
Segmentation
      ↓
Multi-view Generation
      ↓
3D Reconstruction
```

Each stage can be replaced independently.

For example:

```text
SAM 2.1
   ↓
Alternative Segmentation Model
```

or:

```text
FLUX.1 Kontext-dev
   ↓
Alternative Multi-view Model
```

or:

```text
Hunyuan3D
   ↓
Alternative 3D Generation Model
```

This architecture makes it easier to experiment with newer models without rewriting the entire application.

---

# 26. Final End-to-End Workflow

```text
Windows PySide6 Client
          │
          │ Upload Image
          ▼
      FastAPI Server
          │
          ▼
     SAM 2.1 Hiera Large
          │
          ▼
     Object Segmentation
          │
          ▼
       Object Crop
          │
          ▼
   FLUX.1 Kontext-dev
          │
          ▼
 ┌────────┼────────┐
 ▼        ▼        ▼
Front    Side     Back
 └────────┼────────┘
          │
          ▼
      Hunyuan3D
          │
          ▼
     Final 3D Asset
          │
          ▼
Windows Client Download
```

The resulting architecture provides a modular **image → segmentation → multi-view generation → 3D reconstruction** workflow, with the GPU-intensive inference isolated on Vast.ai and the user-facing workflow handled through the Windows PySide6 application.
