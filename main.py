"""Medición de Python secuencial y PyCUDA con validación numérica."""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from src.convolucion import (ConvolucionGPU, TOLERANCIA,
                             TiempoSecuencialExcedido,
                             convolucion_secuencial, crear_mascara, info_gpu)
from src.imagen import a_visualizacion, escribir_pgm, leer_imagen

CABECERA_TIEMPOS = [
    "implementacion", "filtro", "ancho", "alto", "k", "hilos_por_bloque",
    "bloques", "repeticion", "t_h2d", "t_computo", "t_d2h", "t_total",
    "t_contexto_compilacion", "t_reserva", "unidad", "estado", "motivo",
]


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
    p.add_argument("--sin-secuencial", "--sin-cpu", action="store_true",
                   help="omite Python secuencial (útil para resoluciones grandes)")
    p.add_argument("--limite-secuencial", type=float,
                   help="segundos máximos por ejecución secuencial; registra NO_COMPLETADA")
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


def fila_tiempo(impl, a, ancho, alto, repeticion, *, bloques=0, hilos=0,
                 h2d="", computo="", d2h="", total="", contexto="", reserva="",
                 estado="OK", motivo=""):
    return [impl, a.filtro, ancho, alto, a.k, hilos, bloques, repeticion,
            h2d, computo, d2h, total, contexto, reserva, "ms", estado, motivo]


def main():
    a = argumentos()
    if a.info:
        print(info_gpu())
        return 0
    if not a.entrada:
        print("Falta --entrada", file=sys.stderr)
        return 1
    if a.hilos < 1 or a.reps < 1:
        raise ValueError("--hilos y --reps deben ser positivos")

    entrada = leer_imagen(a.entrada)
    alto, ancho = entrada.shape
    mascara = crear_mascara(a.filtro, a.k)
    gpu = ConvolucionGPU(ancho, alto, a.k)
    print(f"Imagen {ancho}x{alto} | filtro {a.filtro} {a.k}x{a.k} | "
          f"{a.hilos} hilos/bloque | {a.reps} repeticiones")
    print(f"  Preparación GPU (fuera de medición): contexto/compilación "
          f"{gpu.t_contexto_compilacion_ms:.3f} ms | reserva {gpu.t_reserva_ms:.3f} ms")

    salida_python = None
    secuencial_completada = False
    if not a.sin_secuencial:
        try:
            convolucion_secuencial(entrada, mascara, a.limite_secuencial)  # calentamiento
            for r in range(1, a.reps + 1):
                inicio = time.perf_counter()
                salida_python = convolucion_secuencial(entrada, mascara, a.limite_secuencial)
                ms = (time.perf_counter() - inicio) * 1000
                agregar_csv(a.csv, CABECERA_TIEMPOS,
                            fila_tiempo("PYTHON_SECUENCIAL", a, ancho, alto, r,
                                       computo=ms, total=ms))
                print(f"  Python secuencial rep {r}: {ms:.3f} ms")
            secuencial_completada = True
        except TiempoSecuencialExcedido as exc:
            motivo = f"Límite documentado de {exc.segundos:g} s excedido"
            agregar_csv(a.csv, CABECERA_TIEMPOS,
                        fila_tiempo("PYTHON_SECUENCIAL", a, ancho, alto, 0,
                                   estado="NO_COMPLETADA", motivo=motivo))
            print(f"  Python secuencial: NO COMPLETADA ({motivo})")

    for _ in range(3):
        salida_gpu, bloques, t = gpu.ejecutar(entrada, mascara, a.hilos)
    for r in range(1, a.reps + 1):
        salida_gpu, bloques, t = gpu.ejecutar(entrada, mascara, a.hilos)
        agregar_csv(a.csv, CABECERA_TIEMPOS,
                    fila_tiempo("PYCUDA", a, ancho, alto, r, bloques=bloques,
                               hilos=a.hilos, h2d=t.h2d, computo=t.kernel,
                               d2h=t.d2h, total=t.total,
                               contexto=gpu.t_contexto_compilacion_ms,
                               reserva=gpu.t_reserva_ms))
        print(f"  PyCUDA rep {r}: h2d {t.h2d:.3f} | kernel {t.kernel:.3f} | "
              f"d2h {t.d2h:.3f} | total {t.total:.3f} ms")

    if secuencial_completada:
        diferencias = np.abs(salida_python.astype(np.float64) - salida_gpu.astype(np.float64))
        emax, emedio = float(diferencias.max()), float(diferencias.mean())
        ok = emax <= TOLERANCIA
        print(f"  Validación Python secuencial vs PyCUDA: error máx {emax:.3e} | "
              f"error medio {emedio:.3e} | tolerancia {TOLERANCIA:.0e} -> "
              f"{'OK' if ok else 'FALLA'}")
        agregar_csv(a.validacion,
                    ["referencia", "comparada", "filtro", "ancho", "alto", "k",
                     "hilos_por_bloque", "error_max", "error_medio", "tolerancia", "resultado"],
                    ["PYTHON_SECUENCIAL", "PYCUDA", a.filtro, ancho, alto, a.k, a.hilos,
                     emax, emedio, TOLERANCIA, "OK" if ok else "FALLA"])
        if not ok:
            return 2

    if a.salida:
        escribir_pgm(a.salida, a_visualizacion(salida_gpu, a.filtro))
        print(f"  Salida guardada en {a.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
