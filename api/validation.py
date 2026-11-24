"""Validación estricta de parámetros e imágenes antes de usar la GPU."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

from core.config import Settings
from api.errors import ApiError

ImageFile.LOAD_TRUNCATED_IMAGES = False

FILTROS = ("media", "nitidez", "laplaciano")
MASKS_POR_FILTRO = {"media": (3, 9, 21), "nitidez": (3,), "laplaciano": (3,)}
HILOS_POR_BLOQUE = (128, 256, 512)
MAGICAS = {
    "PNG": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
    "JPEG": lambda b: b.startswith(b"\xff\xd8\xff"),
    "TIFF": lambda b: b.startswith((b"II*\x00", b"MM\x00*")),
    "BMP": lambda b: b.startswith(b"BM"),
    "WEBP": lambda b: b.startswith(b"RIFF") and b[8:12] == b"WEBP",
    "PGM/PPM": lambda b: len(b) >= 2 and b[0:1] == b"P" and b[1:2] in (b"1", b"2", b"3", b"4", b"5", b"6"),
}


def validar_parametros(filtro: str, mask_size: int) -> tuple[str, int]:
    filtro = filtro.strip().lower()
    if filtro not in FILTROS:
        raise ApiError(
            422,
            "FILTRO_INVALIDO",
            f"filtro '{filtro}' no permitido; use media, nitidez o laplaciano",
        )
    permitidas = MASKS_POR_FILTRO[filtro]
    if mask_size not in permitidas:
        raise ApiError(
            422,
            "MASCARA_INVALIDA",
            f"mǭscara {mask_size} no permitida para '{filtro}'; use {list(permitidas)}",
        )
    return filtro, mask_size


def validar_hilos_por_bloque(hilos: int) -> int:
    if hilos not in HILOS_POR_BLOQUE:
        raise ApiError(
            422,
            "HILOS_INVALIDOS",
            f"threads_per_block debe ser uno de {list(HILOS_POR_BLOQUE)}",
        )
    return hilos


def detectar_formato(payload: bytes) -> str | None:
    for nombre, detector in MAGICAS.items():
        if detector(payload):
            return nombre
    return None


def decodificar_imagen(payload: bytes, settings: Settings, filename: str) -> tuple[np.ndarray, dict[str, Any]]:
    max_bytes = settings.max_upload_mb * 2**20
    if len(payload) > max_bytes:
        raise ApiError(
            413,
            "ARCHIVO_TOO_LARGE",
            f"el archivo supera el límite de {settings.max_upload_mb} MiB",
        )

    formato_magico = detectar_formato(payload)
    if formato_magico is None:
        raise ApiError(400, "FORMATO_INVALIDO", "la cabecera no corresponde a una imagen soportada")

    try:
        with Image.open(BytesIO(payload)) as imagen:
            imagen.verify()
        with Image.open(BytesIO(payload)) as imagen:
            ancho, alto = imagen.size
            pixels = ancho * alto
            if ancho > settings.max_side or alto > settings.max_side:
                raise ApiError(
                    413,
                    "DIMENSIONES_EXCEDIDAS",
                    f"lado máximo permitido {settings.max_side} px; recibido {ancho}x{alto}",
                )
            if pixels > settings.max_pixels:
                raise ApiError(
                    413,
                    "PIXELS_EXCEDIDOS",
                    f"máximo {settings.max_pixels} píxeles; recibido {pixels}",
                )
            formato = imagen.format or formato_magico
            gris = np.array(imagen.convert("L"), dtype=np.float32, copy=True, order="C")
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ApiError(400, "IMAGEN_CORRUPTA", f"la imagen no se pudo decodificar: {exc}") from exc
    except Exception as exc:
        raise ApiError(400, "IMAGEN_INVALIDA", f"contenido de imagen inválido: {exc}") from exc

    metadatos = {
        "filename": Path(filename or "upload").name,
        "format": formato,
        "width": ancho,
        "height": alto,
        "pixels": pixels,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return gris, metadatos
