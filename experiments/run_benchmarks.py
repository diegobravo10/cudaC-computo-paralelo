"""Benchmarks directos, HTTP y de concurrencia con salidas trazables."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import requests

from core.config import BENCHMARK_COLUMNS, SOURCE_COMMIT, load_settings
from experiments.common import (
    cliente_convolucion,
    ejecutar_directo,
    escribir_csv,
    fila_servicio,
    generar_hashes_imagenes,
    resumen_estadistico,
)

RESOLUCIONES = (2000, 4000, 6000)
CONFIGURACIONES = (("media", 3), ("media", 9), ("media", 21),
                   ("nitidez", 3), ("laplaciano", 3))
NIVELES_CONCURRENCIA = (1, 2, 4, 8)


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ejecuta los benchmarks del servicio PyCUDA")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--resultados-dir", default="results")
    p.add_argument("--reps", type=int, default=5, help="repeticiones estables por configuración")
    p.add_argument("--warmup", type=int, default=1, help="calentamientos antes de medir")
    p.add_argument("--hilos", type=int, default=0, help="0 usa APP_THREADS_PER_BLOCK")
    p.add_argument("--environment", default="local")
    p.add_argument("--omitir-directo", action="store_true")
    p.add_argument("--omitir-servicio", action="store_true")
    p.add_argument("--omitir-concurrencia", action="store_true")
    return p.parse_args()


def ruta_entrada(data_dir: Path, lado: int) -> Path:
    for extension in (".pgm", ".png"):
        candidata = Path(data_dir) / f"img_{lado}{extension}"
        if candidata.exists():
            return candidata
    raise FileNotFoundError(f"No existe img_{lado}.pgm ni img_{lado}.png en {data_dir}")


def version(paquete: str) -> str:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return "no instalado"


def git_commit() -> str:
    externo = os.getenv("GIT_COMMIT", "").strip()
    if externo:
        return externo
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return salida.stdout.strip()
    except Exception:
        return "desconocido"


def nvidia_smi() -> str:
    ejecutable = shutil.which("nvidia-smi")
    if not ejecutable:
        return "nvidia-smi no disponible en PATH"
    try:
        salida = subprocess.run(
            [ejecutable,
             "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return salida.stdout.strip()
    except Exception as exc:
        return f"nvidia-smi falló: {exc}"


def guardar_entorno(resultados: Path, args: argparse.Namespace, ajustes: Any) -> dict[str, Any]:
    contenido = {
        "environment": args.environment,
        "source_commit": SOURCE_COMMIT,
        "git_commit": git_commit(),
        "sistema": {
            "plataforma": platform.platform(),
            "procesador": platform.processor(),
            "cpu_logicas": os.cpu_count(),
            "python": sys.version.replace("\n", " "),
        },
        "paquetes": {
            "numpy": version("numpy"),
            "Pillow": version("Pillow"),
            "pycuda": version("pycuda"),
            "fastapi": version("fastapi"),
            "uvicorn": version("uvicorn"),
            "requests": version("requests"),
        },
        "nvidia_smi": nvidia_smi(),
        "configuracion": {
            clave: str(valor) if isinstance(valor, Path) else valor
            for clave, valor in asdict(ajustes).items()
        },
        "base_url": args.base_url,
        "repeticiones_estables": args.reps,
        "calentamientos": args.warmup,
        "hilos_solicitados": args.hilos,
        "resoluciones": list(RESOLUCIONES),
        "configuraciones": [list(c) for c in CONFIGURACIONES],
        "niveles_concurrencia": list(NIVELES_CONCURRENCIA),
    }
    ruta = resultados / "environment.json"
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    return contenido


def estado_salud(base_url: str) -> bool:
    try:
        respuesta = requests.get(f"{base_url.rstrip('/')}/api/health", timeout=10)
        return respuesta.status_code == 200 and respuesta.json().get("status") == "ok"
    except Exception:
        return False


def celda(valor: Any) -> str:
    return str(valor).replace("|", "\\|")


def escribir_tabla_resumen(resumen: list[dict[str, Any]], ruta: Path) -> None:
    columnas = [
        ("experimento", "Experimento"),
        ("execution_mode", "Modo"),
        ("filter", "Filtro"),
        ("resolution", "Resolución"),
        ("mask_size", "Máscara"),
        ("concurrency_level", "Concurrencia"),
        ("grupo", "Grupo"),
        ("metrica", "Métrica"),
        ("n", "n"),
        ("mediana", "Mediana"),
        ("iqr", "IQR"),
        ("desviacion", "Desv. estándar"),
    ]
    lineas = [
        "# Resumen estadístico",
        "",
        "Tiempos en milisegundos; `throughput_req_s` en solicitudes/s.",
        "`cold` = repetición 0; `steady` = repeticiones posteriores al calentamiento.",
        "",
        "| " + " | ".join(titulo for _, titulo in columnas) + " |",
        "| " + " | ".join("---" for _ in columnas) + " |",
    ]
    for fila in resumen:
        lineas.append("| " + " | ".join(celda(fila.get(clave, "")) for clave, _ in columnas) + " |")
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> int:
    a = argumentos()
    resultados = Path(a.resultados_dir)
    resultados.mkdir(parents=True, exist_ok=True)
    csv_path = resultados / "benchmark_results.csv"
    csv_path.unlink(missing_ok=True)
    ajustes = load_settings()
    hilos = a.hilos if a.hilos > 0 else ajustes.threads_per_block

    guardar_entorno(resultados, a, ajustes)
    hashes = generar_hashes_imagenes(Path(a.data_dir), resultados)
    print(f"Entorno y {len(hashes)} hashes registrados")

    filas: list[dict[str, Any]] = []
    etiquetas: list[tuple[str, dict[str, Any]]] = []

    if not a.omitir_directo:
        for lado in RESOLUCIONES:
            try:
                ruta = ruta_entrada(Path(a.data_dir), lado)
            except FileNotFoundError as exc:
                print(f"AVISO: {exc}", file=sys.stderr)
                continue
            for filtro, k in CONFIGURACIONES:
                for fila in ejecutar_directo(
                    image_path=ruta,
                    filtro=filtro,
                    mask_size=k,
                    hilos=hilos,
                    reps=a.reps,
                    warmup=a.warmup,
                    execution_mode="directo_pycuda",
                    environment=a.environment,
                ):
                    filas.append(fila)
                    etiquetas.append((f"directo_{lado}", fila))
        print(f"Directo: {sum(1 for f in filas if f['execution_mode'] == 'directo_pycuda')} filas")

    if not a.omitir_servicio:
        if not estado_salud(a.base_url):
            print(f"ERROR: servicio no disponible en {a.base_url}", file=sys.stderr)
            return 2
        for lado in RESOLUCIONES:
            try:
                ruta = ruta_entrada(Path(a.data_dir), lado)
            except FileNotFoundError as exc:
                print(f"AVISO: {exc}", file=sys.stderr)
                continue
            for filtro, k in CONFIGURACIONES:
                for rep in range(a.reps + 1):
                    respuesta = cliente_convolucion(
                        a.base_url,
                        filtro=filtro,
                        mask_size=k,
                        image_path=ruta,
                        threads_per_block=hilos,
                        timeout=600,
                    )
                    cuerpo = respuesta["body"]
                    metadatos = cuerpo.get("image", {})
                    resolucion = (
                        f"{metadatos.get('width', '?')}x{metadatos.get('height', '?')}"
                        if metadatos else f"{lado}x{lado}"
                    )
                    fila = fila_servicio(
                        respuesta=respuesta,
                        execution_mode="http_secuencial",
                        environment=a.environment,
                        filtro=filtro,
                        resolution=resolucion,
                        mask_size=k,
                        hilos=hilos,
                        repeticion=rep,
                    )
                    filas.append(fila)
                    etiquetas.append((f"http_{lado}", fila))
        print(f"HTTP secuencial: {sum(1 for f in filas if f['execution_mode'] == 'http_secuencial')} filas")

    if not a.omitir_concurrencia:
        if not estado_salud(a.base_url):
            print(f"ERROR: servicio no disponible en {a.base_url}", file=sys.stderr)
            return 2
        try:
            ruta = ruta_entrada(Path(a.data_dir), 4000)
        except FileNotFoundError as exc:
            print(f"AVISO: {exc}", file=sys.stderr)
            ruta = None
        if ruta is not None:
            for nivel in NIVELES_CONCURRENCIA:
                total = max(nivel * 2, 4)
                inicio = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=nivel) as ejecutor:
                    respuestas = list(
                        ejecutor.map(
                            lambda _: cliente_convolucion(
                                a.base_url,
                                filtro="media",
                                mask_size=9,
                                image_path=ruta,
                                threads_per_block=hilos,
                                timeout=600,
                            ),
                            range(total),
                        )
                    )
                transcurrido = max(time.perf_counter() - inicio, 1e-9)
                correctas = sum(1 for r in respuestas if r["status_code"] == 200)
                throughput = correctas / transcurrido
                for indice, respuesta in enumerate(respuestas):
                    estado = None
                    if respuesta["status_code"] == 429:
                        estado = "RECHAZADA"
                    fila = fila_servicio(
                        respuesta=respuesta,
                        execution_mode="http_concurrente",
                        environment=a.environment,
                        filtro="media",
                        resolution="4000x4000",
                        mask_size=9,
                        hilos=hilos,
                        repeticion=indice,
                        concurrency_level=nivel,
                        throughput_req_s=throughput,
                        status=estado,
                    )
                    filas.append(fila)
                    etiquetas.append(("http_concurrencia_4000", fila))
                print(f"Concurrencia {nivel}: {correctas}/{total} OK, {throughput:.3f} req/s")
            sano = estado_salud(a.base_url)
            if not sano:
                print("AVISO: /api/health falló tras el experimento de concurrencia", file=sys.stderr)

    escribir_csv(csv_path, BENCHMARK_COLUMNS, filas)
    resumen = resumen_estadistico(etiquetas, resultados / "summary_stats.csv")
    escribir_tabla_resumen(resumen, resultados / "summary_table.md")

    print(f"Filas totales: {len(filas)}")
    print(f"CSV: {csv_path}")
    print(f"Resumen: {resultados / 'summary_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
