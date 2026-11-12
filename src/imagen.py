"""Lectura/escritura de imágenes y conversión para visualización."""

from pathlib import Path

import numpy as np
from PIL import Image


def leer_imagen(ruta) -> np.ndarray:
    return np.asarray(Image.open(ruta).convert("L"), dtype=np.float32)


def a_visualizacion(valores: np.ndarray, filtro: str) -> np.ndarray:
    valores = np.abs(valores) if filtro == "laplaciano" else valores
    return np.clip(np.rint(valores), 0, 255).astype(np.uint8)


def escribir_pgm(ruta, valores: np.ndarray) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(valores, mode="L").save(ruta)
