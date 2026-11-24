"""Configuración compartida entre el motor PyCUDA, la API y los experimentos."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

SOURCE_COMMIT = "75fad23a526ff9dccea92040764e127e03751cb3"

BENCHMARK_COLUMNS = [
    "request_id",
    "execution_mode",
    "filter",
    "resolution",
    "mask_size",
    "threads_per_block",
    "grid_blocks",
    "repetition",
    "concurrency_level",
    "cold_start",
    "queue_wait_ms",
    "cpu_prep_ms",
    "h2d_ms",
    "kernel_ms",
    "d2h_ms",
    "postprocess_ms",
    "server_total_ms",
    "client_total_ms",
    "throughput_req_s",
    "status",
    "error_msg",
    "environment",
]


@dataclass(frozen=True)
class Settings:
    """Límites de entrada y parámetros operativos del servicio."""

    max_upload_mb: int
    max_pixels: int
    max_side: int
    queue_maxsize: int
    result_ttl_seconds: int
    threads_per_block: int
    gpu_headroom_mb: int
    cuda_cache_dir: Path


def _entero_positivo(nombre: str, valor_por_defecto: int) -> int:
    valor = int(os.getenv(nombre, str(valor_por_defecto)))
    if valor <= 0:
        raise ValueError(f"{nombre} debe ser positivo, se recibió {valor}")
    return valor


def load_settings() -> Settings:
    """Lee la configuraci��n desde variables de entorno con valores seguros."""
    hilos = _entero_positivo("APP_THREADS_PER_BLOCK", 256)
    if hilos not in (128, 256, 512):
        raise ValueError("APP_THREADS_PER_BLOCK debe ser 128, 256 o 512")
    return Settings(
        max_upload_mb=_entero_positivo("APP_MAX_UPLOAD_MB", 128),
        max_pixels=_entero_positivo("APP_MAX_PIXELS", 36_000_000),
        max_side=_entero_positivo("APP_MAX_SIDE", 8192),
        queue_maxsize=_entero_positivo("APP_QUEUE_MAXSIZE", 8),
        result_ttl_seconds=_entero_positivo("APP_RESULT_TTL_SECONDS", 900),
        threads_per_block=hilos,
        gpu_headroom_mb=_entero_positivo("APP_GPU_HEADROOM_MB", 256),
        cuda_cache_dir=Path(
            os.getenv(
                "APP_CUDA_CACHE_DIR",
                str(Path(tempfile.gettempdir()) / "pycuda-computo-paralelo-cache"),
            )
        ),
    )
