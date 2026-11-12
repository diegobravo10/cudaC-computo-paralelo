"""Núcleo de la práctica: referencia CPU y convolución mediante PyCUDA."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np

TOLERANCIA = 1e-3


def crear_mascara(filtro: str, k: int) -> np.ndarray:
    if filtro == "media":
        if k < 1 or k % 2 == 0:
            raise ValueError("La media necesita k impar >= 1")
        return np.full((k, k), np.float32(1.0 / (k * k)), dtype=np.float32)
    if filtro == "nitidez":
        if k != 3:
            raise ValueError("Nitidez solo está definida para k = 3")
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    if filtro == "laplaciano":
        if k != 3:
            raise ValueError("Laplaciano solo está definido para k = 3")
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    raise ValueError(f"Filtro desconocido: {filtro}")


def convolucion_cpu(entrada: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Referencia NumPy con extensión de ceros y acumulación float32."""
    entrada = np.ascontiguousarray(entrada, dtype=np.float32)
    mascara = np.asarray(mascara, dtype=np.float32)
    alto, ancho = entrada.shape
    k = mascara.shape[0]
    r = k // 2
    salida = np.zeros_like(entrada)
    for i in range(k):
        dy = i - r
        y0, y1 = max(0, -dy), min(alto, alto - dy)
        if y1 <= y0:
            continue
        for j in range(k):
            dx = j - r
            x0, x1 = max(0, -dx), min(ancho, ancho - dx)
            if x1 <= x0:
                continue
            salida[y0:y1, x0:x1] += (
                entrada[y0 + dy:y1 + dy, x0 + dx:x1 + dx] * mascara[i, j]
            )
    return salida


@dataclass
class TiemposGPU:
    h2d: float
    kernel: float
    d2h: float
    total: float


KERNEL = r"""
__global__ void convolucion(const float *entrada, float *salida,
                            const float *mascara, int ancho, int alto, int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ancho * alto) return;
    const int fila = idx / ancho;
    const int col = idx % ancho;
    const int r = k / 2;
    float suma = 0.0f;
    for (int i = 0; i < k; ++i) {
        const int y = fila + i - r;
        if (y < 0 || y >= alto) continue;
        for (int j = 0; j < k; ++j) {
            const int x = col + j - r;
            if (x < 0 || x >= ancho) continue;
            suma += entrada[y * ancho + x] * mascara[i * k + j];
        }
    }
    salida[idx] = suma;
}
"""


class ConvolucionGPU:
    """Compila el kernel y reutiliza las reservas de memoria del dispositivo."""

    def __init__(self, ancho: int, alto: int, k: int):
        _preparar_compilador_windows()
        import pycuda.autoinit  # noqa: F401 - crea y destruye el contexto CUDA
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self.cuda = cuda
        self.ancho, self.alto, self.k = ancho, alto, k
        self.nbytes = ancho * alto * np.dtype(np.float32).itemsize
        self.d_entrada = cuda.mem_alloc(self.nbytes)
        self.d_salida = cuda.mem_alloc(self.nbytes)
        self.d_mascara = cuda.mem_alloc(k * k * np.dtype(np.float32).itemsize)
        cache = Path(tempfile.gettempdir()) / "pycuda-computo-paralelo-cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.kernel = SourceModule(KERNEL, cache_dir=str(cache)).get_function("convolucion")

    def ejecutar(self, entrada: np.ndarray, mascara: np.ndarray, hilos: int):
        cuda = self.cuda
        entrada = np.ascontiguousarray(entrada, dtype=np.float32)
        mascara = np.ascontiguousarray(mascara, dtype=np.float32)
        salida = np.empty_like(entrada)
        bloques = (entrada.size + hilos - 1) // hilos
        eventos = [cuda.Event() for _ in range(4)]

        eventos[0].record()
        cuda.memcpy_htod(self.d_entrada, entrada)
        cuda.memcpy_htod(self.d_mascara, mascara)
        eventos[1].record()
        self.kernel(self.d_entrada, self.d_salida, self.d_mascara,
                    np.int32(self.ancho), np.int32(self.alto), np.int32(self.k),
                    block=(hilos, 1, 1), grid=(bloques, 1, 1))
        eventos[2].record()
        cuda.memcpy_dtoh(salida, self.d_salida)
        eventos[3].record()
        eventos[3].synchronize()
        t = TiemposGPU(eventos[0].time_till(eventos[1]),
                       eventos[1].time_till(eventos[2]),
                       eventos[2].time_till(eventos[3]),
                       eventos[0].time_till(eventos[3]))
        return salida, bloques, t


def _preparar_compilador_windows() -> None:
    """Hace visible MSVC para NVCC cuando Python no nació en Developer PowerShell."""
    if os.name != "nt" or any((Path(p) / "cl.exe").exists() for p in os.environ["PATH"].split(os.pathsep)):
        return
    raiz = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio"
    candidatos = sorted(raiz.glob("*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"), reverse=True)
    if candidatos:
        os.environ["PATH"] = str(candidatos[0].parent) + os.pathsep + os.environ["PATH"]


def info_gpu() -> str:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    d = cuda.Context.get_device()
    cc = d.compute_capability()
    return "\n".join([
        f"GPU: {d.name()}",
        f"Capacidad de cómputo: {cc[0]}.{cc[1]}",
        f"Multiprocesadores (SM): {d.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)}",
        f"Memoria global: {d.total_memory() / 2**20:.0f} MiB",
        f"Máx. hilos por bloque: {d.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)}",
        f"PyCUDA: {getattr(__import__('pycuda'), 'VERSION_TEXT', 'instalado')}",
    ])
