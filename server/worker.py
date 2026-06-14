from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock

from config import settings
from job_manager import read_status, require_job_dir, write_status
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


class WorkerPool:
    def __init__(self, max_workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future[None]] = {}
        self._lock = RLock()

    def start(self, job_id: str) -> None:
        job_path = require_job_dir(job_id)
        status = read_status(job_id)
        if status.status == "completed":
            return
        if status.status == "running":
            return

        with self._lock:
            existing = self._futures.get(job_id)
            if existing and not existing.done():
                return
            write_status(job_id, "queued", 0, "Queued")
            future = self._executor.submit(run_pipeline, job_id, job_path)
            self._futures[job_id] = future
            future.add_done_callback(lambda completed: self._on_done(job_id, completed))

    def _on_done(self, job_id: str, future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            logger.error("Worker finished with failure for job %s", job_id)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)


worker_pool = WorkerPool(settings.max_workers)
