"""Programa de medición equivalente al original CUDA C, implementado con PyCUDA."""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from src.convolucion import (ConvolucionGPU, TOLERANCIA, convolucion_cpu,
                             crear_mascara, info_gpu)
from src.imagen import a_visualizacion, escribir_pgm, leer_imagen


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--entrada")
    p.add_argument("--filtro", choices=("media", "nitidez", "laplaciano"), default="media")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--hilos", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--csv", default="resultados/tiempos.csv")
    p.add_argument("--validacion", default="resultados/validacion.csv")
    p.add_argument("--salida")
    p.add_argument("--sin-cpu", action="store_true")
    p.add_argument("--info", action="store_true")
    return p.parse_args()


def agregar_csv(ruta, cabecera, fila):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(cabecera)
        w.writerow(fila)


def main():
    a = argumentos()
    if a.info:
        print(info_gpu())
        return 0
    if not a.entrada:
        print("Falta --entrada", file=sys.stderr)
        return 1
    if a.hilos < 1:
        raise ValueError("--hilos debe ser positivo")

    entrada = leer_imagen(a.entrada)
    alto, ancho = entrada.shape
    mascara = crear_mascara(a.filtro, a.k)
    gpu = ConvolucionGPU(ancho, alto, a.k)
    print(f"Imagen {ancho}x{alto} | filtro {a.filtro} {a.k}x{a.k} | "
          f"{a.hilos} hilos/bloque | {a.reps} repeticiones")
    cab = ["implementacion", "filtro", "ancho", "alto", "k", "hilos_por_bloque",
           "bloques", "repeticion", "t_h2d", "t_computo", "t_d2h", "t_total", "unidad"]

    salida_cpu = None
    if not a.sin_cpu:
        convolucion_cpu(entrada, mascara)
        for r in range(1, a.reps + 1):
            inicio = time.perf_counter()
            salida_cpu = convolucion_cpu(entrada, mascara)
            ms = (time.perf_counter() - inicio) * 1000
            agregar_csv(a.csv, cab, ["CPU", a.filtro, ancho, alto, a.k, 0, 0, r,
                                     0, ms, 0, ms, "ms"])
            print(f"  CPU  rep {r}: {ms:.3f} ms")

    for _ in range(3):
        salida_gpu, bloques, t = gpu.ejecutar(entrada, mascara, a.hilos)
    for r in range(1, a.reps + 1):
        salida_gpu, bloques, t = gpu.ejecutar(entrada, mascara, a.hilos)
        agregar_csv(a.csv, cab, ["CUDA", a.filtro, ancho, alto, a.k, a.hilos, bloques, r,
                                 t.h2d, t.kernel, t.d2h, t.total, "ms"])
        print(f"  CUDA rep {r}: h2d {t.h2d:.3f} | kernel {t.kernel:.3f} | "
              f"d2h {t.d2h:.3f} | total {t.total:.3f} ms")

    if salida_cpu is not None:
        diferencias = np.abs(salida_cpu.astype(np.float64) - salida_gpu.astype(np.float64))
        emax, emedio = float(diferencias.max()), float(diferencias.mean())
        ok = emax <= TOLERANCIA
        print(f"  Validación: error máx {emax:.3e} | error medio {emedio:.3e} | "
              f"tolerancia {TOLERANCIA:.0e} -> {'OK' if ok else 'FALLA'}")
        agregar_csv(a.validacion,
                    ["filtro", "ancho", "alto", "k", "hilos_por_bloque", "error_max",
                     "error_medio", "tolerancia", "resultado"],
                    [a.filtro, ancho, alto, a.k, a.hilos, emax, emedio, TOLERANCIA,
                     "OK" if ok else "FALLA"])
        if not ok:
            return 2
    if a.salida:
        escribir_pgm(a.salida, a_visualizacion(salida_gpu, a.filtro))
        print(f"  Salida guardada en {a.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
