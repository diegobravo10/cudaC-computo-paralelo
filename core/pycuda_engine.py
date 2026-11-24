"""Motor PyCUDA desacoplado de la capa HTTP.

Conserva literalmente el kernel, los coeficientes, la precisión float32, la
extensión con ceros y la distribución 1D de un hilo por píxel de la Práctica 02.
El contexto CUDA se crea, se usa y se destruye en el mismo hilo; no existe
fallback silencioso a CPU.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.convolucion import KERNEL, _preparar_compilador_windows

LOGGER = logging.getLogger("core.pycuda_engine")
MEMORY_TYPE = "paginable (pageable);Sin pinned/host-mapped memory"


class CudaUnavailableError(RuntimeError):
    """No hay GPU NVIDIA, driver o contexto CUDA utilizable."""


class CudaOutOfMemoryError(RuntimeError):
    """La memoria libre del dispositivo no alcanza para la solicitud."""


class CudaKernelError(RuntimeError):
    """Falló la compilación, carga o lanzamiento del kernel propio."""


class CudaProcessingError(RuntimeError):
    """Fallo genérico durante la convolución en GPU."""


@dataclass(frozen=True)
class GpuTimings:
    h2d_ms: float
    kernel_ms: float
    d2h_ms: float
    gpu_total_ms: float
    init_ms: float
    reserve_ms: float


@dataclass(frozen=True)
class EngineResult:
    salida: np.ndarray
    bloques: int
    hilos: int
    tiempos: GpuTimings
    cold_start: bool
    device: str
    memoria_libre_mb: float
    memoria_total_mb: float


class PyCudaEngine:
    """Compila el kernel una vez y reserva/libera memoria en cada ejecución.

    La primera ejecución incluye creación de contexto y compilación
    (arranque en frío). Las siguientes reutilizan el contexto y el módulo.
    Las transferencias usan memoria paginable de NumPy; se declara
    explícitamente para no confundir H2D/D2H con memoria fijada.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.cuda: Any = None
        self.device: Any = None
        self.context: Any = None
        self.kernel: Any = None
        self.inicializado = False
        self.t_init_ms = 0.0

    def initialize(self) -> None:
        """Crea el contexto CUDA y compila/carga el SourceModule."""
        if self.inicializado:
            return

        inicio = time.perf_counter()
        _preparar_compilador_windows()
        try:
            import pycuda.driver as cuda
        except Exception as exc:  # pragma: no cover - depende del entorno
            raise CudaUnavailableError(f"PyCUDA no se pudo importar: {exc}") from exc

        try:
            cuda.init()
            if cuda.Device.count() == 0:
                raise CudaUnavailableError("No se detectaron dispositivos CUDA NVIDIA")
            self.device = cuda.Device(0)
            self.context = self.device.make_context()
        except CudaUnavailableError:
            raise
        except Exception as exc:
            raise CudaUnavailableError(f"No se pudo crear el contexto CUDA: {exc}") from exc

        try:
            from pycuda.compiler import SourceModule

            cache = Path(self.settings.cuda_cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            modulo = SourceModule(KERNEL, cache_dir=str(cache))
            self.kernel = modulo.get_function("convolucion")
        except Exception as exc:
            self.context.detach()
            self.context = None
            raise CudaKernelError(f"Falló SourceModule/KERNEL de la Práctica 02: {exc}") from exc

        self.cuda = cuda
        self.inicializado = True
        self.t_init_ms = (time.perf_counter() - inicio) * 1000.0
        LOGGER.info(
            "CUDA listo | dispositivo=%s | contexto+SourceModule=%.3f ms | %s",
            self.device.name(),
            self.t_init_ms,
            MEMORY_TYPE,
        )

    def execute(
        self,
        entrada: np.ndarray,
        mascara: np.ndarray,
        hilos: int,
    ) -> EngineResult:
        """Ejecuta la convolución y devuelve salida float32 y tiempos CUDA."""
        if hilos < 1:
            raise CudaProcessingError("hilos por bloque debe ser positivo")

        cold_start = not self.inicializado
        self.initialize()

        entrada = np.ascontiguousarray(entrada, dtype=np.float32)
        mascara = np.ascontiguousarray(mascara, dtype=np.float32)
        if entrada.ndim != 2 or mascara.ndim != 2 or mascara.shape[0] != mascara.shape[1]:
            raise CudaProcessingError("entrada/máscara deben ser 2D y la máscara cuadrada")

        alto, ancho = entrada.shape
        k = mascara.shape[0]
        nbytes = int(entrada.size * np.dtype(np.float32).itemsize)
        mask_bytes = int(mascara.size * np.dtype(np.float32).itemsize)
        necesario = 2 * nbytes + mask_bytes
        libre, total = self.cuda.mem_get_info()
        headroom = int(self.settings.gpu_headroom_mb * 2**20)
        if necesario + headroom > libre:
            raise CudaOutOfMemoryError(
                f"memoria insuficiente: se requieren {bytes_a_mib(necesario)} MiB "
                f"(+{self.settings.gpu_headroom_mb} MiB margen) y hay "
                f"{bytes_a_mib(libre)} MiB libres de {bytes_a_mib(total)} MiB"
            )

        salida = np.empty_like(entrada)
        bloques = (entrada.size + hilos - 1) // hilos
        d_entrada = None
        d_salida = None
        d_mascara = None
        reserva_inicio = time.perf_counter()

        try:
            d_entrada = self.cuda.mem_alloc(nbytes)
            d_salida = self.cuda.mem_alloc(nbytes)
            d_mascara = self.cuda.mem_alloc(mask_bytes)
            reserva_ms = (time.perf_counter() - reserva_inicio) * 1000.0
            LOGGER.info(
                "mem_alloc | entrada=%d B salida=%d B máscara=%d B | %.3f ms",
                nbytes,
                nbytes,
                mask_bytes,
                reserva_ms,
            )

            eventos = [self.cuda.Event() for _ in range(4)]
            eventos[0].record()
            self.cuda.memcpy_htod(d_entrada, entrada)
            self.cuda.memcpy_htod(d_mascara, mascara)
            eventos[1].record()

            self.kernel(
                d_entrada,
                d_salida,
                d_mascara,
                np.int32(ancho),
                np.int32(alto),
                np.int32(k),
                block=(hilos, 1, 1),
                grid=(bloques, 1, 1),
            )
            eventos[2].record()

            self.cuda.memcpy_dtoh(salida, d_salida)
            eventos[3].record()
            eventos[3].synchronize()

            h2d = eventos[0].time_till(eventos[1])
            kernel_ms = eventos[1].time_till(eventos[2])
            d2h = eventos[2].time_till(eventos[3])
            total_gpu = eventos[0].time_till(eventos[3])
            LOGGER.info(
                "H2D=%.3f ms | kernel=%.3f ms | D2H=%.3f ms | grid=%d block=%d | sync explícita",
                h2d,
                kernel_ms,
                d2h,
                bloques,
                hilos,
            )
        except Exception as exc:
            if isinstance(exc, CudaOutOfMemoryError):
                raise
            if _es_out_of_memory(exc, self.cuda):
                raise CudaOutOfMemoryError(f"CUDA rechazó la reserva/transferencia: {exc}") from exc
            if isinstance(exc, (CudaUnavailableError, CudaKernelError, CudaProcessingError)):
                raise
            raise CudaProcessingError(f"Error en ejecución PyCUDA: {exc}") from exc
        finally:
            for nombre, puntero in (
                ("entrada", d_entrada),
                ("salida", d_salida),
                ("mascara", d_mascara),
            ):
                if puntero is not None:
                    puntero.free()
                    LOGGER.info("free | %s", nombre)

        tiempos = GpuTimings(
            h2d_ms=h2d,
            kernel_ms=kernel_ms,
            d2h_ms=d2h,
            gpu_total_ms=total_gpu,
            init_ms=self.t_init_ms if cold_start else 0.0,
            reserve_ms=reserva_ms,
        )
        return EngineResult(
            salida=salida,
            bloques=bloques,
            hilos=hilos,
            tiempos=tiempos,
            cold_start=cold_start,
            device=self.device.name(),
            memoria_libre_mb=bytes_a_mib(libre),
            memoria_total_mb=bytes_a_mib(total),
        )

    def probe(self) -> dict[str, Any]:
        """Consulta GPU sin inicializar el contexto del worker."""
        try:
            import pycuda.driver as cuda
        except Exception as exc:
            raise CudaUnavailableError(f"PyCUDA no se pudo importar: {exc}") from exc

        try:
            cuda.init()
            if cuda.Device.count() == 0:
                raise CudaUnavailableError("No se detectaron dispositivos CUDA NVIDIA")
            dispositivo = cuda.Device(0)
            total = dispositivo.total_memory()
            libre = total
            contexto = None
            try:
                contexto = dispositivo.make_context()
                libre, total = cuda.mem_get_info()
            finally:
                if contexto is not None:
                    contexto.detach()
        except CudaUnavailableError:
            raise
        except Exception as exc:
            raise CudaUnavailableError(f"Diagnóstico CUDA fallido: {exc}") from exc

        import pycuda

        return {
            "disponible": True,
            "dispositivo": dispositivo.name(),
            "capacidad_computo": ".".join(str(x) for x in dispositivo.compute_capability()),
            "memoria_libre_mb": round(bytes_a_mib(libre), 2),
            "memoria_total_mb": round(bytes_a_mib(total), 2),
            "cuda_driver": _version_text(_llamar(cuda.get_driver_version)),
            "cuda_runtime": _version_text(_llamar(cuda.get_version)),
            "pycuda": getattr(pycuda, "VERSION_TEXT", "desconocida"),
            "tipo_memoria_transferencias": MEMORY_TYPE,
            "kernel": "KERNEL Práctica 02 (SourceModule)",
        }

    def close(self) -> None:
        """Libera el contexto CUDA desde el hilo que lo creó."""
        if self.context is not None:
            self.context.detach()
            self.context = None
            self.kernel = None
            self.inicializado = False
            LOGGER.info("Contexto CUDA liberado explícitamente")


def bytes_a_mib(bytes_: int) -> float:
    return round(bytes_ / 2**20, 2)


def _version_text(valor: Any) -> str:
    if isinstance(valor, int):
        return f"{valor // 1000}.{(valor % 1000) // 10}.{valor % 10}"
    if isinstance(valor, (tuple, list)):
        return ".".join(str(parte) for parte in valor)
    return str(valor)


def _llamar(func: Any) -> Any:
    try:
        return func()
    except Exception:
        return "no disponible"


def _es_out_of_memory(exc: Exception, cuda: Any) -> bool:
    mensaje = str(exc).lower()
    if "out of memory" in mensaje or "cuda_error_out_of_memory" in mensaje:
        return True
    clase = getattr(cuda, "MemoryError", None)
    return bool(clase and isinstance(exc, clase))
