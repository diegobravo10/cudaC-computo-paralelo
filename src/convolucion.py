from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import time

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


class TiempoSecuencialExcedido(RuntimeError):
    """La convolución secuencial superó el límite documentado."""

    def __init__(self, segundos: float):
        super().__init__(f"límite de {segundos:g} s excedido")
        self.segundos = segundos


def convolucion_secuencial(entrada: np.ndarray, mascara: np.ndarray,
                           limite_segundos: float | None = None) -> np.ndarray:
    """Convolución Python secuencial mediante recorridos explícitos.

    NumPy se usa únicamente para almacenar datos. El cálculo recorre píxeles y
    coeficientes sin vectorización, JIT, multiprocesamiento ni filtros externos.
    """
    entrada = np.ascontiguousarray(entrada, dtype=np.float32)
    mascara = np.ascontiguousarray(mascara, dtype=np.float32)
    alto, ancho = entrada.shape
    k = mascara.shape[0]
    r = k // 2
    salida = np.zeros_like(entrada)
    inicio = time.perf_counter()
    for fila in range(alto):
        if limite_segundos is not None and time.perf_counter() - inicio > limite_segundos:
            raise TiempoSecuencialExcedido(limite_segundos)
        for col in range(ancho):
            suma = np.float32(0.0)
            for i in range(k):
                y = fila + i - r
                if y < 0 or y >= alto:
                    continue
                for j in range(k):
                    x = col + j - r
                    if x < 0 or x >= ancho:
                        continue
                    producto = np.float32(entrada[y, x] * mascara[i, j])
                    suma = np.float32(suma + producto)
            salida[fila, col] = suma
    return salida


convolucion_cpu = convolucion_secuencial


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
        inicio_contexto = time.perf_counter()
        _preparar_compilador_windows()
        import pycuda.autoinit  # noqa: F401 - crea y destruye el contexto CUDA
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self.cuda = cuda
        self.ancho, self.alto, self.k = ancho, alto, k
        self.nbytes = ancho * alto * np.dtype(np.float32).itemsize
        cache = Path(tempfile.gettempdir()) / "pycuda-computo-paralelo-cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.kernel = SourceModule(KERNEL, cache_dir=str(cache)).get_function("convolucion")
        self.t_contexto_compilacion_ms = (time.perf_counter() - inicio_contexto) * 1000

        inicio_reserva = time.perf_counter()
        self.d_entrada = cuda.mem_alloc(self.nbytes)
        self.d_salida = cuda.mem_alloc(self.nbytes)
        self.d_mascara = cuda.mem_alloc(k * k * np.dtype(np.float32).itemsize)
        self.t_reserva_ms = (time.perf_counter() - inicio_reserva) * 1000

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
