"""CLI directo de PyCUDA: misma lógica que consume el servicio web."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import uuid
from pathlib import Path

from src.convolucion import crear_mascara
from src.imagen import leer_imagen

from core.config import BENCHMARK_COLUMNS, load_settings
from core.pycuda_engine import PyCudaEngine


def agregar_fila(ruta: Path, fila: dict[str, object]) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=BENCHMARK_COLUMNS)
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)


def fila_base(a: argparse.Namespace, ancho: int, alto: int) -> dict[str, object]:
    return {
        "request_id": "",
        "execution_mode": a.modo,
        "filter": a.filtro,
        "resolution": f"{ancho}x{alto}",
        "mask_size": a.k,
        "threads_per_block": a.hilos,
        "grid_blocks": 0,
        "repetition": 0,
        "concurrency_level": 1,
        "cold_start": False,
        "queue_wait_ms": 0.0,
        "cpu_prep_ms": 0.0,
        "h2d_ms": 0.0,
        "kernel_ms": 0.0,
        "d2h_ms": 0.0,
        "postprocess_ms": 0.0,
        "server_total_ms": 0.0,
        "client_total_ms": 0.0,
        "throughput_req_s": "",
        "status": "OK",
        "error_msg": "",
        "environment": a.entorno,
    }


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convolución PyCUDA directa (sin HTTP)")
    p.add_argument("--entrada", required=True)
    p.add_argument("--filtro", required=True, choices=("media", "nitidez", "laplaciano"))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--hilos", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--csv", default="results/benchmark_results.csv")
    p.add_argument("--modo", default="PYCUDA_DIRECT_HOST")
    p.add_argument("--entorno", default="host")
    return p.parse_args()


def main() -> int:
    a = argumentos()
    ajustes = load_settings()
    inicio_total = time.perf_counter()
    ancho = alto = 0
    try:
        inicio_prep = time.perf_counter()
        imagen = leer_imagen(a.entrada)
        alto, ancho = imagen.shape
        mascara = crear_mascara(a.filtro, a.k)
        cpu_prep_ms = (time.perf_counter() - inicio_prep) * 1000.0

        motor = PyCudaEngine(ajustes)
        fila = fila_base(a, ancho, alto)
        fila["cpu_prep_ms"] = round(cpu_prep_ms, 3)

        primera = motor.execute(imagen, mascara, a.hilos)
        fila["request_id"] = str(uuid.uuid4())
        fila["repetition"] = 0
        fila["cold_start"] = primera.cold_start
        fila["grid_blocks"] = primera.bloques
        fila["h2d_ms"] = round(primera.tiempos.h2d_ms + primera.tiempos.reserve_ms, 3)
        fila["kernel_ms"] = round(primera.tiempos.kernel_ms, 3)
        fila["d2h_ms"] = round(primera.tiempos.d2h_ms, 3)
        fila["server_total_ms"] = round(fila["h2d_ms"] + fila["kernel_ms"] + fila["d2h_ms"], 3)
        fila["client_total_ms"] = round((time.perf_counter() - inicio_total) * 1000.0, 3)
        agregar_fila(a.csv, fila)

        for _ in range(a.warmup):
            motor.execute(imagen, mascara, a.hilos)

        for rep in range(1, a.reps + 1):
            inicio = time.perf_counter()
            resultado = motor.execute(imagen, mascara, a.hilos)
            cliente_ms = (time.perf_counter() - inicio) * 1000.0
            fila = fila_base(a, ancho, alto)
            fila.update(
                {
                    "request_id": str(uuid.uuid4()),
                    "repetition": rep,
                    "cold_start": resultado.cold_start,
                    "grid_blocks": resultado.bloques,
                    "cpu_prep_ms": round(cpu_prep_ms, 3),
                    "h2d_ms": round(resultado.tiempos.h2d_ms + resultado.tiempos.reserve_ms, 3),
                    "kernel_ms": round(resultado.tiempos.kernel_ms, 3),
                    "d2h_ms": round(resultado.tiempos.d2h_ms, 3),
                    "server_total_ms": round(
                        fila["cpu_prep_ms"]
                        + resultado.tiempos.h2d_ms
                        + resultado.tiempos.reserve_ms
                        + resultado.tiempos.kernel_ms
                        + resultado.tiempos.d2h_ms,
                        3,
                    ),
                    "client_total_ms": round(cliente_ms, 3),
                }
            )
            agregar_fila(a.csv, fila)
        motor.close()
        return 0
    except Exception as exc:
        fila = fila_base(a, ancho, alto)
        fila.update(
            {
                "request_id": str(uuid.uuid4()),
                "status": "INCOMPLETE",
                "client_total_ms": round((time.perf_counter() - inicio_total) * 1000.0, 3),
                "error_msg": f"{type(exc).__name__}: {exc}",
            }
        )
        agregar_fila(a.csv, fila)
        print(f"ERROR: {fila['error_msg']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
