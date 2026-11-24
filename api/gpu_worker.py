"""Cola acotada y un único hilo propietario del contexto CUDA."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from core.pycuda_engine import PyCudaEngine


class QueueFullError(RuntimeError):
    """La cola de espera configurada está llena."""


@dataclass(frozen=True)
class GpuJob:
    request_id: str
    imagen: Any
    mascara: Any
    hilos: int
    encolado_en: float
    futuro: Future


@dataclass(frozen=True)
class WorkerOutcome:
    resultado: Any
    queue_wait_ms: float


class GpuWorker:
    """Serializa el acceso a GPU, mide espera en cola y rechaza con 429."""

    def __init__(self, engine: PyCudaEngine, maxsize: int) -> None:
        self.engine = engine
        self.cola: queue.Queue[GpuJob] = queue.Queue(maxsize=maxsize)
        self._finalizar = object()
        self._hilo = threading.Thread(target=self._bucle, name="cuda-worker", daemon=True)
        self._inicio = time.perf_counter()
        self._bloqueo = threading.Lock()
        self.en_proceso = 0
        self._activo = False

    def start(self) -> None:
        if not self._activo:
            self._hilo.start()
            self._activo = True

    def stop(self) -> None:
        if not self._activo:
            return
        self.cola.put(self._finalizar)
        self._hilo.join(timeout=30)
        self._activo = False
        self.engine.close()

    def submit(self, request_id: str, imagen: Any, mascara: Any, hilos: int) -> Future:
        futuro: Future = Future()
        trabajo = GpuJob(
            request_id=request_id,
            imagen=imagen,
            mascara=mascara,
            hilos=hilos,
            encolado_en=time.perf_counter(),
            futuro=futuro,
        )
        try:
            self.cola.put_nowait(trabajo)
        except queue.Full as exc:
            raise QueueFullError("cola GPU llena; reintente más tarde") from exc
        return futuro

    def estado(self) -> dict[str, int]:
        return {
            "queue_depth": self.cola.qsize(),
            "queue_maxsize": self.cola.maxsize,
            "in_flight": self.en_proceso,
        }

    def _bucle(self) -> None:
        while True:
            trabajo = self.cola.get()
            if trabajo is self._finalizar:
                self.cola.task_done()
                break
            with self._bloqueo:
                self.en_proceso += 1
            espera_ms = (time.perf_counter() - trabajo.encolado_en) * 1000.0
            try:
                resultado = self.engine.execute(trabajo.imagen, trabajo.mascara, trabajo.hilos)
                trabajo.futuro.set_result(WorkerOutcome(resultado, espera_ms))
            except BaseException as exc:
                trabajo.futuro.set_exception(exc)
            finally:
                with self._bloqueo:
                    self.en_proceso -= 1
                self.cola.task_done()
