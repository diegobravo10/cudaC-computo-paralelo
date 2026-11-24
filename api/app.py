"""API HTTP y frontend del servicio de convolución PyCUDA."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from api.errors import ApiError
from api.gpu_worker import GpuWorker, QueueFullError
from api.result_store import ResultStore
from api.validation import decodificar_imagen, validar_hilos_por_bloque, validar_parametros
from core.config import SOURCE_COMMIT, load_settings
from core.pycuda_engine import (
    MEMORY_TYPE,
    CudaKernelError,
    CudaOutOfMemoryError,
    CudaProcessingError,
    CudaUnavailableError,
    PyCudaEngine,
)
from src.convolucion import crear_mascara
from src.imagen import a_visualizacion

LOGGER = logging.getLogger("api.app")
RAIZ = Path(__file__).resolve().parent.parent
STATIC_DIR = RAIZ / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ajustes = load_settings()
    motor = PyCudaEngine(ajustes)
    worker = GpuWorker(motor, ajustes.queue_maxsize)
    worker.start()
    resultados = ResultStore(
        Path(os.getenv("APP_RESULTS_DIR", str(RAIZ / "results"))),
        ajustes.result_ttl_seconds,
    )
    resultados.iniciar_limpieza()
    app.state.settings = ajustes
    app.state.engine = motor
    app.state.worker = worker
    app.state.results = resultados
    LOGGER.info(
        "Servicio iniciado | cola=%d | hilos/bloque=%d | commit=%s",
        ajustes.queue_maxsize,
        ajustes.threads_per_block,
        _git_commit(),
    )
    try:
        yield
    finally:
        worker.stop()
        resultados.detener_limpieza()


app = FastAPI(
    title="Servicio de convolución PyCUDA",
    description="API de convolución con kernels propios de la Práctica 02, Docker y GPU NVIDIA.",
    version="3.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(ApiError)
async def manejar_api_error(request: Request, exc: ApiError) -> JSONResponse:
    contenido: dict[str, Any] = {
        "error": {"code": exc.code, "message": exc.message},
        "request_id": getattr(request.state, "request_id", None),
    }
    contenido.update(exc.extra)
    return JSONResponse(status_code=exc.status_code, content=contenido)


@app.exception_handler(RequestValidationError)
async def manejar_validacion(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "PARAMETROS_INVALIDOS",
                "message": "parámetros de la solicitud inválidos",
                "details": str(exc.errors()),
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.get("/", include_in_schema=False)
async def indice() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/convolve")
async def convolucionar(
    request: Request,
    file: UploadFile = File(..., description="Imagen PNG/JPEG/BMP/TIFF/WEBP/PGM"),
    filter: str = Form("media", description="media | nitidez | laplaciano"),
    mask_size: int = Form(3, description="3 | 9 | 21 solo para media"),
    threads_per_block: int = Form(0, description="128 | 256 | 512; 0 usa el valor configurado"),
) -> JSONResponse:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    inicio_servidor = time.perf_counter()
    ajustes: Any = request.app.state.settings
    max_bytes = ajustes.max_upload_mb * 2**20

    try:
        filtro, k = validar_parametros(filter, mask_size)
        payload = await file.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ApiError(413, "ARCHIVO_TOO_LARGE", f"el archivo supera {ajustes.max_upload_mb} MiB")

        inicio_prep = time.perf_counter()
        imagen, metadatos = await run_in_threadpool(
            decodificar_imagen, payload, ajustes, file.filename or "upload"
        )
        mascara = await run_in_threadpool(crear_mascara, filtro, k)
        cpu_prep_ms = (time.perf_counter() - inicio_prep) * 1000.0

        hilos = threads_per_block if threads_per_block > 0 else ajustes.threads_per_block
        hilos = validar_hilos_por_bloque(hilos)

        futuro = request.app.state.worker.submit(request_id, imagen, mascara, hilos)
        desenlace = await run_in_threadpool(futuro.result)
        resultado = desenlace.resultado

        inicio_post = time.perf_counter()
        visual = a_visualizacion(resultado.salida, filtro)
        buffer = BytesIO()
        Image.fromarray(visual, mode="L").save(buffer, format="PNG")
        png = buffer.getvalue()
        crudo = np.ascontiguousarray(resultado.salida, dtype="<f4").tobytes()
        url_resultado, nombre_resultado = request.app.state.results.guardar(
            request_id, filtro, k, ".png", png
        )
        url_crudo, nombre_crudo = request.app.state.results.guardar(
            request_id, filtro, k, ".f32", crudo
        )
        postprocess_ms = (time.perf_counter() - inicio_post) * 1000.0
        total_ms = (time.perf_counter() - inicio_servidor) * 1000.0

        h2d_ms = resultado.tiempos.h2d_ms + resultado.tiempos.reserve_ms
        cabeceras = {
            "X-Request-ID": request_id,
            "X-Queue-Wait-Ms": f"{desenlace.queue_wait_ms:.3f}",
            "X-CPU-Prep-Ms": f"{cpu_prep_ms:.3f}",
            "X-H2D-Ms": f"{h2d_ms:.3f}",
            "X-Kernel-Ms": f"{resultado.tiempos.kernel_ms:.3f}",
            "X-D2H-Ms": f"{resultado.tiempos.d2h_ms:.3f}",
            "X-Postprocess-Ms": f"{postprocess_ms:.3f}",
            "X-Server-Total-Ms": f"{total_ms:.3f}",
            "X-Cold-Start": str(resultado.cold_start).lower(),
        }
        cuerpo = {
            "request_id": request_id,
            "status": "ok",
            "image": metadatos,
            "parameters": {
                "filter": filtro,
                "mask_size": k,
                "threads_per_block": hilos,
                "grid_blocks": resultado.bloques,
            },
            "metrics": {
                "queue_wait_ms": round(desenlace.queue_wait_ms, 3),
                "cpu_prep_ms": round(cpu_prep_ms, 3),
                "h2d_ms": round(h2d_ms, 3),
                "kernel_ms": round(resultado.tiempos.kernel_ms, 3),
                "d2h_ms": round(resultado.tiempos.d2h_ms, 3),
                "postprocess_ms": round(postprocess_ms, 3),
                "server_total_ms": round(total_ms, 3),
                "cold_start": resultado.cold_start,
            },
            "gpu": {
                "device": resultado.device,
                "memory_free_mb": resultado.memoria_libre_mb,
                "memory_total_mb": resultado.memoria_total_mb,
                "init_ms": round(resultado.tiempos.init_ms, 3),
                "reserve_ms": round(resultado.tiempos.reserve_ms, 3),
                "memory_type": MEMORY_TYPE,
                "event_synchronization": True,
            },
            "result_url": url_resultado,
            "raw_url": url_crudo,
            "download_filename": nombre_resultado,
        }
        return JSONResponse(status_code=200, content=cuerpo, headers=cabeceras)
    except ApiError:
        raise
    except QueueFullError as exc:
        raise ApiError(429, "COLA_LLENA", str(exc)) from exc
    except CudaUnavailableError as exc:
        raise ApiError(503, "GPU_NO_DISPONIBLE", str(exc)) from exc
    except CudaOutOfMemoryError as exc:
        raise ApiError(507, "MEMORIA_INSUFICIENTE", str(exc)) from exc
    except (CudaKernelError, CudaProcessingError) as exc:
        raise ApiError(503, "ERROR_GPU", str(exc)) from exc
    except ValueError as exc:
        raise ApiError(422, "MASCARA_INVALIDA", str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Fallo no controlado en /api/convolve")
        raise ApiError(500, "ERROR_INTERNO", f"fallo inesperado: {exc}") from exc
    finally:
        await file.close()


@app.get("/api/health")
async def salud(request: Request) -> JSONResponse:
    ajustes: Any = request.app.state.settings
    contenido: dict[str, Any] = {
        "status": "ok",
        "service": "convolucion-pycuda",
        "source_commit": SOURCE_COMMIT,
        "git_commit": _git_commit(),
        "queue": request.app.state.worker.estado(),
        "limits": {
            "max_upload_mb": ajustes.max_upload_mb,
            "max_pixels": ajustes.max_pixels,
            "max_side": ajustes.max_side,
            "threads_per_block": ajustes.threads_per_block,
        },
    }
    try:
        contenido["gpu"] = await run_in_threadpool(request.app.state.engine.probe)
    except CudaUnavailableError as exc:
        contenido.update(
            {
                "status": "unavailable",
                "error": {"code": "GPU_NO_DISPONIBLE", "message": str(exc)},
            }
        )
        return JSONResponse(status_code=503, content=contenido)
    return JSONResponse(status_code=200, content=contenido)


@app.get("/api/results/{nombre}")
async def obtener_resultado(nombre: str, request: Request) -> FileResponse:
    almacen = request.app.state.results
    return FileResponse(
        almacen.ruta(nombre),
        media_type=almacen.content_type(nombre),
        filename=nombre,
    )


def _git_commit() -> str:
    externo = os.getenv("GIT_COMMIT", "").strip()
    if externo:
        return externo
    try:
        salida = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RAIZ,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return salida.stdout.strip()
    except Exception:
        return "desconocido"
