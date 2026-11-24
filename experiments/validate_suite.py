"""Validación funcional, numérica, de errores y de aislamiento del servicio."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from core.config import load_settings
from core.pycuda_engine import PyCudaEngine
from experiments.common import (
    VALIDACION_COLUMNS,
    agregar_csv,
    cliente_convolucion,
    generar_hashes_imagenes,
    guardar_galeria_visual,
    imagen_solida,
    escribir_csv,
)
from src.convolucion import convolucion_secuencial, crear_mascara
from src.imagen import leer_imagen

MANUAL = {
    ("laplaciano", 3): np.array(
        [[4, 3, 2, 1, -6], [-5, 0, 0, 0, -11], [-10, 0, 0, 0, -16],
         [-15, 0, 0, 0, -21], [-46, -27, -28, -29, -56]], dtype=np.float32
    ),
    ("nitidez", 3): np.array(
        [[-3, -1, 1, 3, 11], [11, 7, 8, 9, 21], [21, 12, 13, 14, 31],
         [31, 17, 18, 19, 41], [67, 49, 51, 53, 81]], dtype=np.float32
    ),
}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Suite de validación del servicio PyCUDA")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--resultados-dir", default="results")
    p.add_argument("--entrada-pequena", default="data/256x170.jpg")
    p.add_argument("--imagen-cola", default="data/img_2000.pgm")
    p.add_argument("--hilos", type=int, default=256)
    p.add_argument("--tolerancia", type=float, default=1e-5)
    p.add_argument("--tolerancia-cpu", type=float, default=1e-4)
    p.add_argument("--cola-concurrencia", type=int, default=30)
    p.add_argument("--cola-mascara", type=int, default=21)
    p.add_argument("--omitir-cola", action="store_true")
    return p.parse_args()


def registrar(ruta: Path, prueba: str, configuracion: str, detalle: str,
              valor: object, tolerancia: object, resultado: str) -> None:
    agregar_csv(
        ruta,
        VALIDACION_COLUMNS,
        {
            "prueba": prueba,
            "configuracion": configuracion,
            "detalle": detalle,
            "valor": valor,
            "tolerancia": tolerancia,
            "resultado": resultado,
        },
    )


def error_absoluto(real: np.ndarray, esperado: np.ndarray) -> tuple[float, float]:
    diferencia = np.abs(real.astype(np.float64) - esperado.astype(np.float64))
    return float(diferencia.max()), float(diferencia.mean())


def estado_salud(base_url: str) -> bool:
    try:
        respuesta = requests.get(f"{base_url.rstrip('/')}/api/health", timeout=10)
        return respuesta.status_code == 200 and respuesta.json().get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    a = argumentos()
    resultados = Path(a.resultados_dir)
    resultados.mkdir(parents=True, exist_ok=True)
    validacion_csv = resultados / "validation.csv"
    validacion_csv.unlink(missing_ok=True)
    fallos = 0

    if not estado_salud(a.base_url):
        print(f"ERROR: el servicio no está disponible en {a.base_url}", file=sys.stderr)
        return 2

    hashes = generar_hashes_imagenes(Path(a.data_dir), resultados)
    print(f"Hashes registrados: {len(hashes)} imágenes")

    ajustes = load_settings()
    motor = PyCudaEngine(ajustes)
    try:
        for (filtro, k), esperado in MANUAL.items():
            imagen = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
            mascara = crear_mascara(filtro, k)
            gpu = motor.execute(imagen, mascara, a.hilos).salida
            emax, emedio = error_absoluto(gpu, esperado)
            ok = emax <= a.tolerancia
            fallos += 0 if ok else 1
            registrar(validacion_csv, "matriz_manual", f"{filtro}_{k}x{k}",
                      f"error_max={emax:.3e}; error_medio={emedio:.3e}",
                      emax, a.tolerancia, "OK" if ok else "FALLA")

        imagen_rara = np.random.default_rng(42).integers(
            0, 256, (17, 19)
        ).astype(np.float32)
        configuraciones = [
            ("media", 3), ("media", 9), ("media", 21),
            ("nitidez", 3), ("laplaciano", 3),
        ]
        for hilos in (128, 256):
            for filtro, k in configuraciones:
                mascara = crear_mascara(filtro, k)
                cpu = convolucion_secuencial(imagen_rara, mascara)
                resultado = motor.execute(imagen_rara, mascara, hilos)
                gpu = resultado.salida
                bloques = resultado.bloques
                esperado_bloques = (imagen_rara.size + hilos - 1) // hilos
                emax, emedio = error_absoluto(gpu, cpu)
                ok = emax <= a.tolerancia_cpu and bloques == esperado_bloques
                fallos += 0 if ok else 1
                registrar(validacion_csv, "dimensiones_no_multiplo",
                          f"17x19_{filtro}_{k}x{k}_hilos{hilos}",
                          f"bloques={bloques}/{esperado_bloques}; "
                          f"error_max={emax:.3e}; error_medio={emedio:.3e}; "
                          "cpu_float32_vs_gpu_float32",
                          emax, a.tolerancia_cpu, "OK" if ok else "FALLA")

        imagen_pequena = leer_imagen(a.entrada_pequena)
        for filtro, k in configuraciones:
            mascara = crear_mascara(filtro, k)
            directo = motor.execute(imagen_pequena, mascara, a.hilos).salida
            respuesta = cliente_convolucion(
                a.base_url,
                filtro=filtro,
                mask_size=k,
                image_path=Path(a.entrada_pequena),
                threads_per_block=a.hilos,
            )
            if respuesta["raw"] is None:
                emax = emedio = float("inf")
                detalle = f"HTTP {respuesta['status_code']}: {respuesta['body']}"
                ok = False
            else:
                emax, emedio = error_absoluto(directo, respuesta["raw"])
                detalle = f"error_max={emax:.3e}; error_medio={emedio:.3e}; antes_de_uint8"
                ok = emax <= a.tolerancia
            fallos += 0 if ok else 1
            registrar(validacion_csv, "directo_vs_servicio", f"{filtro}_{k}x{k}",
                      detalle, emax, a.tolerancia, "OK" if ok else "FALLA")
    finally:
        motor.close()

    max_bytes = ajustes.max_upload_mb * 2**20
    casos_invalidos = [
        ("archivo_no_imagen", b"esto no es una imagen", {}, "media", 3, 400),
        ("archivo_tamano", imagen_solida(128) + b"\x00" * (max_bytes + 1), {}, "media", 3, 413),
        ("filtro_desconocido", imagen_solida(128), {}, "desconocido", 3, 422),
        ("mascara_no_permitida", imagen_solida(128), {}, "media", 7, 422),
        ("mascara_laplaciano", imagen_solida(128), {}, "laplaciano", 9, 422),
    ]
    for nombre, payload, _, filtro, k, esperado in casos_invalidos:
        inicio = time.perf_counter()
        try:
            respuesta = requests.post(
                f"{a.base_url.rstrip('/')}/api/convolve",
                data={"filter": filtro, "mask_size": str(k)},
                files={"file": ("entrada.png", payload, "image/png")},
                timeout=60,
            )
            codigo = respuesta.status_code
            cuerpo = respuesta.json()
            mensaje = cuerpo.get("error", {}).get("message", "")
        except Exception as exc:
            codigo = -1
            mensaje = str(exc)
        transcurrido = (time.perf_counter() - inicio) * 1000.0
        ok = codigo == esperado
        fallos += 0 if ok else 1
        registrar(validacion_csv, "entrada_invalida", nombre,
                  f"HTTP={codigo}; esperado={esperado}; mensaje={mensaje}; {transcurrido:.1f} ms",
                  codigo, esperado, "OK" if ok else "FALLA")

    sano = estado_salud(a.base_url)
    fallos += 0 if sano else 1
    registrar(validacion_csv, "recuperacion", "health_tras_errores",
              "servicio responde 200 tras entradas inválidas", sano, True,
              "OK" if sano else "FALLA")

    payloads = [("cliente_A", imagen_solida(40)), ("cliente_B", imagen_solida(210))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ejecutor:
        futuros = [
            ejecutor.submit(
                cliente_convolucion, a.base_url,
                filtro="media", mask_size=3, payload=payload,
                filename=f"{nombre}.png", threads_per_block=a.hilos,
            )
            for nombre, payload in payloads
        ]
        respuestas = [f.result() for f in futuros]

    for (nombre, payload), respuesta in zip(payloads, respuestas):
        esperado = convolucion_secuencial(
            np.array(Image.open(io.BytesIO(payload)).convert("L"), dtype=np.float32),
            crear_mascara("media", 3),
        )
        if respuesta["raw"] is None:
            emax = float("inf")
            ok = False
            detalle = f"HTTP {respuesta['status_code']}"
        else:
            emax, emedio = error_absoluto(respuesta["raw"], esperado)
            detalle = f"error_max={emax:.3e}; error_medio={emedio:.3e}"
            ok = emax <= a.tolerancia
        fallos += 0 if ok else 1
        registrar(validacion_csv, "aislamiento", nombre, detalle, emax,
                  a.tolerancia, "OK" if ok else "FALLA")

    distintos = (
        respuestas[0]["raw"] is not None
        and respuestas[1]["raw"] is not None
        and not np.array_equal(respuestas[0]["raw"], respuestas[1]["raw"])
    )
    fallos += 0 if distintos else 1
    registrar(validacion_csv, "aislamiento", "sin_sobrescritura",
              "cada cliente recibió su propia imagen", distintos, True,
              "OK" if distintos else "FALLA")

    if not a.omitir_cola:
        inicio = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.cola_concurrencia) as ejecutor:
            futuros = [
                ejecutor.submit(
                    cliente_convolucion, a.base_url,
                    filtro="media", mask_size=a.cola_mascara,
                    image_path=Path(a.imagen_cola),
                    threads_per_block=a.hilos, timeout=600,
                )
                for _ in range(a.cola_concurrencia)
            ]
            codigos = [f.result()["status_code"] for f in futuros]
        transcurrido = (time.perf_counter() - inicio) * 1000.0
        rechazos = codigos.count(429)
        ok = rechazos > 0
        fallos += 0 if ok else 1
        registrar(validacion_csv, "cola", f"{a.cola_concurrencia}_solicitudes_media_{a.cola_mascara}x{a.cola_mascara}",
                  f"429={rechazos}; códigos={sorted(set(codigos))}; {transcurrido:.1f} ms",
                  rechazos, ">=1", "OK" if ok else "INCOMPLETA")
        if not ok:
            print("AVISO: la cola no se saturó; aumente --cola-concurrencia")

    sano = estado_salud(a.base_url)
    fallos += 0 if sano else 1
    registrar(validacion_csv, "recuperacion", "health_tras_concurrencia",
              "servicio responde 200 tras saturar la cola", sano, True,
              "OK" if sano else "FALLA")

    try:
        galeria = guardar_galeria_visual(
            base_url=a.base_url,
            imagen_path=Path(a.entrada_pequena),
            resultados_dir=resultados,
            hilos=a.hilos,
        )
        registrar(validacion_csv, "evidencia_visual", "galeria_filtros",
                  str(galeria), "original + 5 configuraciones", "mismo recorte", "OK")
    except Exception as exc:
        fallos += 1
        registrar(validacion_csv, "evidencia_visual", "galeria_filtros",
                  str(exc), "", "", "FALLA")

    with validacion_csv.open("r", encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    resumen = [
        {
            "prueba": prueba,
            "total": sum(1 for f in filas if f["prueba"] == prueba),
            "ok": sum(1 for f in filas if f["prueba"] == prueba and f["resultado"] == "OK"),
            "fallos": sum(1 for f in filas if f["prueba"] == prueba and f["resultado"] != "OK"),
        }
        for prueba in sorted({f["prueba"] for f in filas})
    ]
    escribir_csv(resultados / "validation_summary.csv",
                 ["prueba", "total", "ok", "fallos"], resumen)

    print(f"Validación: {len(filas)} comprobaciones, {fallos} fallos")
    print(f"CSV: {validacion_csv}")
    return 2 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
