"""Paquete núcleo: procesamiento PyCUDA sin dependencias de la API."""

from .config import BENCHMARK_COLUMNS, SOURCE_COMMIT, Settings, load_settings
from .pycuda_engine import (
    CudaKernelError,
    CudaOutOfMemoryError,
    CudaProcessingError,
    CudaUnavailableError,
    EngineResult,
    GpuTimings,
    MEMORY_TYPE,
    PyCudaEngine,
)

__all__ = [
    "BENCHMARK_COLUMNS",
    "SOURCE_COMMIT",
    "Settings",
    "load_settings",
    "PyCudaEngine",
    "EngineResult",
    "GpuTimings",
    "MEMORY_TYPE",
    "CudaUnavailableError",
    "CudaOutOfMemoryError",
    "CudaKernelError",
    "CudaProcessingError",
]
