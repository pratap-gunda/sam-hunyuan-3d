from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


DOWNLOAD_FILES = (
    "refined_mask.png",
    "object_rgba.png",
    "object_crop.png",
    "result.glb",
)


class ApiClient:
    def __init__(self, server_url: str, timeout: int = 30) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def upload(self, image_path: str | Path, points: list[list[int]], labels: list[int]) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        self._validate_prompts(points, labels)

        with image_path.open("rb") as image_file:
            response = requests.post(
                f"{self.server_url}/upload",
                files={
                    "image": (image_path.name, image_file, "image/png"),
                },
                data={
                    "points": json.dumps(points),
                    "labels": json.dumps(labels),
                },
                timeout=self.timeout,
            )
        response.raise_for_status()
        return str(response.json()["job_id"])

    def update_points(self, job_id: str, points: list[list[int]], labels: list[int]) -> dict[str, Any]:
        self._validate_prompts(points, labels)
        response = requests.post(
            f"{self.server_url}/points/{job_id}",
            json={"points": points, "labels": labels},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return dict(response.json())

    def preview_mask(self, job_id: str, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with requests.post(
            f"{self.server_url}/preview_mask/{job_id}",
            timeout=max(self.timeout, 120),
            stream=True,
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        return destination

    def generate(self, job_id: str) -> dict[str, Any]:
        response = requests.post(f"{self.server_url}/generate/{job_id}", timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())

    def status(self, job_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.server_url}/status/{job_id}", timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())

    def download(self, job_id: str, filename: str, destination_dir: str | Path) -> Path:
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / filename

        with requests.get(
            f"{self.server_url}/download/{job_id}/{filename}",
            timeout=max(self.timeout, 120),
            stream=True,
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        return destination

    def delete_job(self, job_id: str) -> dict[str, Any]:
        response = requests.delete(f"{self.server_url}/job/{job_id}", timeout=self.timeout)
        response.raise_for_status()
        return dict(response.json())

    def _validate_prompts(self, points: list[list[int]], labels: list[int]) -> None:
        if not points:
            raise ValueError("Add at least one positive point before continuing")
        if len(points) != len(labels):
            raise ValueError("Point and label counts do not match")
        if 1 not in labels:
            raise ValueError("Add at least one positive point")


def upload(server_url: str, image_path: str | Path, points: list[list[int]], labels: list[int]) -> str:
    return ApiClient(server_url).upload(image_path, points, labels)


def update_points(server_url: str, job_id: str, points: list[list[int]], labels: list[int]) -> dict[str, Any]:
    return ApiClient(server_url).update_points(job_id, points, labels)


def preview_mask(server_url: str, job_id: str, destination: str | Path) -> Path:
    return ApiClient(server_url).preview_mask(job_id, destination)


def generate(server_url: str, job_id: str) -> dict[str, Any]:
    return ApiClient(server_url).generate(job_id)


def status(server_url: str, job_id: str) -> dict[str, Any]:
    return ApiClient(server_url).status(job_id)


def download(server_url: str, job_id: str, filename: str, destination_dir: str | Path) -> Path:
    return ApiClient(server_url).download(job_id, filename, destination_dir)


def delete_job(server_url: str, job_id: str) -> dict[str, Any]:
    return ApiClient(server_url).delete_job(job_id)
