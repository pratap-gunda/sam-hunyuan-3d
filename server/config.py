from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent
    max_workers: int = int(os.getenv("SERVER_MAX_WORKERS", "1"))
    low_vram: bool = os.getenv("SERVER_LOW_VRAM", "1") != "0"
    preload_flux: bool = os.getenv("SERVER_PRELOAD_FLUX", "0") == "1"

    @property
    def jobs_dir(self) -> Path:
        return self.base_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.base_dir / "cache"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"


settings = Settings()


def ensure_directories() -> None:
    for path in (
        settings.jobs_dir,
        settings.cache_dir,
        settings.outputs_dir,
        settings.logs_dir,
        settings.base_dir / "templates",
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging() -> None:
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(settings.logs_dir / "server.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
