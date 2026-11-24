"""Modelos de respuesta de la API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ImageInfo(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    pixels: int
    bytes: int
    sha256: str


class Parameters(BaseModel):
    filter: str
    mask_size: int
    threads_per_block: int
    grid_blocks: int


class Metrics(BaseModel):
    queue_wait_ms: float
    cpu_prep_ms: float
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    postprocess_ms: float
    server_total_ms: float
    cold_start: bool


class ConvolveResponse(BaseModel):
    request_id: str
    status: str
    image: ImageInfo
    parameters: Parameters
    metrics: Metrics
    gpu: dict[str, Any]
    result_url: str
    raw_url: str
    download_filename: str


class HealthResponse(BaseModel):
    status: str
    service: str
    source_commit: str
    git_commit: str
    gpu: dict[str, Any]
    queue: dict[str, int]
    limits: dict[str, int]
