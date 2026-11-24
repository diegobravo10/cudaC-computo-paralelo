"""Almacenamiento temporal de resultados con nombres únicos y limpieza automática."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from api.errors import ApiError

EXTENSIONES = {".png": "image/png", ".f32": "application/octet-stream"}


class ResultStore:
    def __init__(self, root: Path, ttl_seconds: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._stop = threading.Event()
        self._hilo: threading.Thread | None = None

    def guardar(self, request_id: str, filtro: str, mask_size: int, extension: str, datos: bytes) -> tuple[str, str]:
        if extension not in EXTENSIONES:
            raise ValueError(f"extensión no soportada: {extension}")
        nombre = f"{request_id}_{filtro}_{mask_size}x{mask_size}{extension}"
        destino = self.root / nombre
        destino.write_bytes(datos)
        return f"/api/results/{nombre}", nombre

    def ruta(self, nombre: str) -> Path:
        seguro = Path(nombre).name
        if seguro != nombre or Path(nombre).suffix not in EXTENSIONES:
            raise ApiError(404, "RESULTADO_NO_ENCONTRADO", "resultado no disponible")
        destino = self.root / seguro
        if not destino.is_file():
            raise ApiError(404, "RESULTADO_NO_ENCONTRADO", "el resultado expiró o no existe")
        return destino

    def content_type(self, nombre: str) -> str:
        return EXTENSIONES[Path(nombre).suffix]

    def iniciar_limpieza(self) -> None:
        if self._hilo is not None:
            return
        self._stop.clear()
        self._hilo = threading.Thread(target=self._bucle, name="result-cleanup", daemon=True)
        self._hilo.start()

    def detener_limpieza(self) -> None:
        self._stop.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
            self._hilo = None
        self.limpiar()

    def limpiar(self) -> int:
        ahora = time.time()
        eliminados = 0
        for archivo in self.root.glob("*"):
            if archivo.is_file() and ahora - archivo.stat().st_mtime > self.ttl_seconds:
                archivo.unlink(missing_ok=True)
                eliminados += 1
        return eliminados

    def _bucle(self) -> None:
        while not self._stop.wait(60):
            self.limpiar()
