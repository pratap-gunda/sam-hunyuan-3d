# SAM2 + Hunyuan3D Client/Server Pipeline

## Server on Vast.ai

Copy the `server/` directory to `/root/server/`, install dependencies, then run FastAPI:

```bash
cd /root/server
python3.12 -m pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Install SAM2 into the same Python environment that runs the server:

```bash
cd ~
git clone https://github.com/facebookresearch/sam2.git
cd sam2
python3.12 -m pip install -e .
```

Download the SAM2.1 Hiera Large checkpoint into the server cache:

```bash
mkdir -p /root/server/cache/checkpoints
cd /root/server/cache/checkpoints
wget -O sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
ls -lh /root/server/cache/checkpoints/sam2.1_hiera_large.pt
```

The server writes job data to `/root/server/jobs/<job_id>/` and logs to `/root/server/logs/server.log`.

## Client on Windows

Install client dependencies and start the PySide6 UI:

```powershell
cd client
py -3.12 -m pip install -r requirements.txt
py -3.12 main.py
```

Set `Server URL` to the Vast.ai endpoint, browse for an input image, then add SAM-style point prompts in the image viewer. Left click adds a green foreground point, right click adds a red background point. Use `Generate Mask Preview` as many times as needed, refine the points, then click `Generate Mesh`. Completed downloads are stored under `client/downloads/<job_id>/`.

The upload request sends the input image plus `points` and `labels`; the server stores them in `jobs/<job_id>/points.json`.

## Template Entry Points

The server calls:

```python
templates.sam_template.run_sam2(...)
templates.hunyuan_template.run_hunyuan3d(...)
```

Hunyuan3D receives only `jobs/<job_id>/object_crop.png`; it never reads the original uploaded `input.png`.
