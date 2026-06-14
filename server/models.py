from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


JobStatusValue = Literal["queued", "running", "completed", "failed"]


class JobStatus(BaseModel):
    job_id: str
    status: JobStatusValue
    progress: int = Field(ge=0, le=100)
    message: str


class UploadResponse(BaseModel):
    job_id: str


class PointPrompts(BaseModel):
    points: list[list[int]]
    labels: list[int]

    @model_validator(mode="after")
    def validate_points(self) -> "PointPrompts":
        if not self.points:
            raise ValueError("At least one point is required")
        if len(self.points) != len(self.labels):
            raise ValueError("points and labels must have the same length")
        for point in self.points:
            if len(point) != 2:
                raise ValueError("Each point must contain [x, y]")
            if point[0] < 0 or point[1] < 0:
                raise ValueError("Point coordinates must be non-negative")
        for label in self.labels:
            if label not in {0, 1}:
                raise ValueError("Point labels must be 0 or 1")
        return self


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobSummary(BaseModel):
    job_id: str
    status: JobStatusValue
    progress: int
    message: str


class JobsResponse(BaseModel):
    jobs: list[JobSummary]
