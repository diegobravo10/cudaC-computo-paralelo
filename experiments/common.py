"""Utilidades compartidas por los scripts de validación y benchmark."""

from __future__ import annotations

import csv
import hashlib
import io
import mimetypes
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from PIL import Image

from core.config import load_settings
from core.pycuda_engine import PyCudaEngine
from src.convolucion import crear_mascara
from src.imagen import leer_imagen

VALIDACION_COLUMNS = [
    "prueba",
    "configuracion",
    "detalle",
    "valor",
    "tolerancia",
    "resultado",
]


def escribir_csv(ruta: Path, columnas: list[str], filas: Iterable[dict[str, Any]]) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({clave: fila.get(clave, "") for clave in columnas})


def agregar_csv(ruta: Path, columnas: list[str], fila: dict[str, Any]) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        if nuevo:
            escritor.writeheader()
        escritor.writerow({clave: fila.get(clave, "") for clave in columnas})


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(2**20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def dimensiones_imagen(ruta: Path) -> tuple[int, int]:
    with Image.open(ruta) as imagen:
        return imagen.size


def generar_hashes_imagenes(data_dir: Path, resultados_dir: Path) -> list[dict[str, Any]]:
    filas: list[dict[str, Any]] = []
    for lado in (2000, 4000, 6000):
        for extension in (".pgm", ".png"):
            ruta = Path(data_dir) / f"img_{lado}{extension}"
            if not ruta.exists():
                continue
            ancho, alto = dimensiones_imagen(ruta)
            filas.append(
                {
                    "archivo": ruta.name,
                    "ancho": ancho,
                    "alto": alto,
                    "bytes": ruta.stat().st_size,
                    "sha256": sha256_archivo(ruta),
                }
            )
    escribir_csv(Path(resultados_dir) / "image_hashes.csv",
                 ["archivo", "ancho", "alto", "bytes", "sha256"], filas)
    return filas


def imagen_solida(valor: int, tamaño: tuple[int, int] = (48, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", tamaño, color=valor).save(buffer, format="PNG")
    return buffer.getvalue()


def cliente_convolucion(
    base_url: str,
    *,
    filtro: str,
    mask_size: int,
    image_path: Path | None = None,
    payload: bytes | None = None,
    filename: str | None = None,
    threads_per_block: int | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Envía una solicitud y descarga el resultado completo antes de cronometrar."""
    if payload is None:
        if image_path is None:
            raise ValueError("image_path o payload son obligatorios")
        ruta = Path(image_path)
        payload = ruta.read_bytes()
        filename = filename or ruta.name
    filename = filename or "upload.png"
    tipo = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    datos = {"filter": filtro, "mask_size": str(mask_size)}
    if threads_per_block is not None:
        datos["threads_per_block"] = str(threads_per_block)

    inicio = time.perf_counter()
    respuesta = requests.post(
        f"{base_url.rstrip('/')}/api/convolve",
        data=datos,
        files={"file": (filename, payload, tipo)},
        timeout=timeout,
    )
    try:
        cuerpo = respuesta.json()
    except ValueError:
        cuerpo = {"error": {"message": respuesta.text[:500]}}

    salida: dict[str, Any] = {
        "status_code": respuesta.status_code,
        "body": cuerpo,
        "client_total_ms": 0.0,
        "raw": None,
        "png_bytes": None,
        "headers": dict(respuesta.headers),
    }

    if respuesta.ok:
        url_crudo = f"{base_url.rstrip('/')}{cuerpo['raw_url']}"
        url_png = f"{base_url.rstrip('/')}{cuerpo['result_url']}"
        crudo = requests.get(url_crudo, timeout=timeout)
        crudo.raise_for_status()
        png = requests.get(url_png, timeout=timeout)
        png.raise_for_status()
        ancho = int(cuerpo["image"]["width"])
        alto = int(cuerpo["image"]["height"])
        valores = np.frombuffer(crudo.content, dtype="<f4")
        if valores.size != ancho * alto:
            raise ValueError(f"raw de {valores.size} valores, esperado {ancho * alto}")
        salida["raw"] = valores.reshape((alto, ancho)).copy()
        salida["png_bytes"] = png.content

    salida["client_total_ms"] = (time.perf_counter() - inicio) * 1000.0
    return salida


def fila_directa(
    *,
    execution_mode: str,
    environment: str,
    filtro: str,
    resolution: str,
    mask_size: int,
    threads_per_block: int,
    grid_blocks: int,
    repetition: int,
    cold_start: bool,
    cpu_prep_ms: float,
    h2d_ms: float,
    kernel_ms: float,
    d2h_ms: float,
    server_total_ms: float,
    client_total_ms: float,
    status: str = "OK",
    error_msg: str = "",
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id or str(uuid.uuid4()),
        "execution_mode": execution_mode,
        "filter": filtro,
        "resolution": resolution,
        "mask_size": mask_size,
        "threads_per_block": threads_per_block,
        "grid_blocks": grid_blocks,
        "repetition": repetition,
        "concurrency_level": 1,
        "cold_start": cold_start,
        "queue_wait_ms": 0.0,
        "cpu_prep_ms": round(cpu_prep_ms, 3),
        "h2d_ms": round(h2d_ms, 3),
        "kernel_ms": round(kernel_ms, 3),
        "d2h_ms": round(d2h_ms, 3),
        "postprocess_ms": 0.0,
        "server_total_ms": round(server_total_ms, 3),
        "client_total_ms": round(client_total_ms, 3),
        "throughput_req_s": "",
        "status": status,
        "error_msg": error_msg,
        "environment": environment,
    }


def ejecutar_directo(
    *,
    image_path: Path,
    filtro: str,
    mask_size: int,
    hilos: int,
    reps: int,
    warmup: int,
    execution_mode: str,
    environment: str,
) -> list[dict[str, Any]]:
    """Ejecuta PyCUDA en este proceso y devuelve filas del CSV."""
    filas: list[dict[str, Any]] = []
    inicio = time.perf_counter()
    imagen = leer_imagen(image_path)
    alto, ancho = imagen.shape
    mascara = crear_mascara(filtro, mask_size)
    cpu_prep_ms = (time.perf_counter() - inicio) * 1000.0
    resolucion = f"{ancho}x{alto}"
    motor = PyCudaEngine(load_settings())

    def convertir(resultado: Any, repeticion: int, cliente_ms: float) -> dict[str, Any]:
        h2d = resultado.tiempos.h2d_ms + resultado.tiempos.reserve_ms
        return fila_directa(
            execution_mode=execution_mode,
            environment=environment,
            filtro=filtro,
            resolution=resolucion,
            mask_size=mask_size,
            threads_per_block=hilos,
            grid_blocks=resultado.bloques,
            repetition=repeticion,
            cold_start=resultado.cold_start,
            cpu_prep_ms=cpu_prep_ms,
            h2d_ms=h2d,
            kernel_ms=resultado.tiempos.kernel_ms,
            d2h_ms=resultado.tiempos.d2h_ms,
            server_total_ms=cpu_prep_ms + h2d + resultado.tiempos.kernel_ms + resultado.tiempos.d2h_ms,
            client_total_ms=cliente_ms,
        )

    try:
        primero = motor.execute(imagen, mascara, hilos)
        filas.append(convertir(primero, 0, (time.perf_counter() - inicio) * 1000.0))
        for _ in range(warmup):
            motor.execute(imagen, mascara, hilos)
        for rep in range(1, reps + 1):
            t0 = time.perf_counter()
            resultado = motor.execute(imagen, mascara, hilos)
            filas.append(convertir(resultado, rep, (time.perf_counter() - t0) * 1000.0))
    finally:
        motor.close()
    return filas


def fila_servicio(
    *,
    respuesta: dict[str, Any],
    execution_mode: str,
    environment: str,
    filtro: str,
    resolution: str,
    mask_size: int,
    hilos: int,
    repeticion: int,
    concurrency_level: int = 1,
    throughput_req_s: float | str = "",
    status: str | None = None,
    error_msg: str = "",
) -> dict[str, Any]:
    cuerpo = respuesta.get("body", {})
    metricas = cuerpo.get("metrics", {})
    parametros = cuerpo.get("parameters", {})
    estado = status or ("OK" if respuesta.get("status_code") == 200 else "ERROR")
    return {
        "request_id": cuerpo.get("request_id", str(uuid.uuid4())),
        "execution_mode": execution_mode,
        "filter": filtro,
        "resolution": resolution,
        "mask_size": mask_size,
        "threads_per_block": parametros.get("threads_per_block", hilos),
        "grid_blocks": parametros.get("grid_blocks", ""),
        "repetition": repeticion,
        "concurrency_level": concurrency_level,
        "cold_start": metricas.get("cold_start", ""),
        "queue_wait_ms": metricas.get("queue_wait_ms", ""),
        "cpu_prep_ms": metricas.get("cpu_prep_ms", ""),
        "h2d_ms": metricas.get("h2d_ms", ""),
        "kernel_ms": metricas.get("kernel_ms", ""),
        "d2h_ms": metricas.get("d2h_ms", ""),
        "postprocess_ms": metricas.get("postprocess_ms", ""),
        "server_total_ms": metricas.get("server_total_ms", ""),
        "client_total_ms": round(respuesta.get("client_total_ms", 0.0), 3),
        "throughput_req_s": round(throughput_req_s, 3) if isinstance(throughput_req_s, (int, float)) else throughput_req_s,
        "status": estado,
        "error_msg": error_msg or (cuerpo.get("error", {}).get("message", "") if isinstance(cuerpo.get("error"), dict) else ""),
        "environment": environment,
    }


METRICAS_RESUMEN = [
    "queue_wait_ms",
    "cpu_prep_ms",
    "h2d_ms",
    "kernel_ms",
    "d2h_ms",
    "postprocess_ms",
    "server_total_ms",
    "client_total_ms",
    "throughput_req_s",
]


def resumen_estadistico(
    experimentos: list[tuple[str, dict[str, Any]]],
    ruta: Path,
) -> list[dict[str, Any]]:
    grupos: dict[tuple[Any, ...], list[float]] = {}
    for experimento, fila in experimentos:
        if fila.get("status") != "OK":
            continue
        grupo_cold = "cold" if bool(fila.get("cold_start")) else "steady"
        clave = (
            experimento,
            fila.get("execution_mode"),
            fila.get("filter"),
            fila.get("resolution"),
            fila.get("mask_size"),
            fila.get("concurrency_level"),
            grupo_cold,
        )
        for metrica in METRICAS_RESUMEN:
            valor = fila.get(metrica)
            if valor in (None, ""):
                continue
            grupos.setdefault(clave + (metrica,), []).append(float(valor))

    filas: list[dict[str, Any]] = []
    for clave, valores in sorted(grupos.items(), key=lambda item: str(item[0])):
        (experimento, modo, filtro, resolucion, mascara, concurrencia, grupo, metrica) = clave
        median = statistics.median(valores)
        q1, q3 = (0.0, 0.0) if len(valores) == 1 else (
            float(np.percentile(valores, 25)),
            float(np.percentile(valores, 75)),
        )
        desviacion = statistics.stdev(valores) if len(valores) > 1 else 0.0
        filas.append(
            {
                "experimento": experimento,
                "execution_mode": modo,
                "filter": filtro,
                "resolution": resolucion,
                "mask_size": mascara,
                "concurrency_level": concurrencia,
                "grupo": grupo,
                "metrica": metrica,
                "n": len(valores),
                "mediana": round(median, 3),
                "iqr": round(q3 - q1, 3),
                "desviacion": round(desviacion, 3),
            }
        )

    escribir_csv(
        ruta,
        ["experimento", "execution_mode", "filter", "resolution", "mask_size",
         "concurrency_level", "grupo", "metrica", "n", "mediana", "iqr", "desviacion"],
        filas,
    )
    return filas


def guardar_galeria_visual(
    *,
    base_url: str,
    imagen_path: Path,
    resultados_dir: Path,
    hilos: int,
) -> Path:
    destino = Path(resultados_dir) / "visual"
    destino.mkdir(parents=True, exist_ok=True)
    configuraciones = [("media", 3), ("media", 9), ("media", 21),
                       ("nitidez", 3), ("laplaciano", 3)]
    original = Image.open(imagen_path).convert("L")
    lado = min(original.size)
    izquierda = (original.width - lado) // 2
    arriba = (original.height - lado) // 2
    caja = (izquierda, arriba, izquierda + lado, arriba + lado)
    recorte_original = original.crop(caja)

    piezas = [recorte_original]
    for filtro, k in configuraciones:
        respuesta = cliente_convolucion(
            base_url,
            filtro=filtro,
            mask_size=k,
            image_path=imagen_path,
            threads_per_block=hilos,
        )
        if respuesta["png_bytes"] is None:
            raise RuntimeError(f"no se obtuvo imagen para {filtro} {k}x{k}")
        nombre = f"{filtro}_{k}x{k}.png"
        (destino / nombre).write_bytes(respuesta["png_bytes"])
        with Image.open(io.BytesIO(respuesta["png_bytes"])) as procesada:
            recorte = procesada.convert("L").crop(caja)
        piezas.append(recorte)

    ancho_pieza, alto_pieza = piezas[0].size
    columnas = 3
    filas = 2
    lamina = Image.new("L", (ancho_pieza * columnas, alto_pieza * filas), color=255)
    for indice, pieza in enumerate(piezas):
        lamina.paste(pieza, ((indice % columnas) * ancho_pieza,
                             (indice // columnas) * alto_pieza))
    ruta = destino / "galeria_filtros.png"
    lamina.save(ruta)
    return ruta
