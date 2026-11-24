# Práctica 03 — Servicio de convolución PyCUDA con Docker

Servicio HTTP/FastAPI que ejecuta los kernels propios de la Práctica 02 en GPU NVIDIA mediante PyCUDA. Conserva literalmente los coeficientes de los filtros `media`, `nitidez` y `laplaciano`, la extensión con ceros, la precisión `float32` y la distribución 1D de un hilo por píxel. No hay fallback a CPU: si la GPU falla, la API responde `503`/`507`.

## Trazabilidad

- Commit base de la Práctica 02: `75fad23a526ff9dccea92040764e127e03751cb3`.
- El mismo valor queda registrado en `core/config.py` (`SOURCE_COMMIT`) y se devuelve en `/api/health` como `source_commit`.
- El commit de la imagen se inyecta al construir con `--build-arg GIT_COMMIT=<hash>` y se devuelve como `git_commit`.
- Las entradas y salidas se identifican con SHA-256 en `results/image_hashes.csv` y con `request_id` en cada respuesta y fila de CSV.

## Estructura

```text
core/         configuración, motor PyCUDA y CLI directo
api/          FastAPI, validación, cola de GPU y almacén de resultados
web/static/   index.html, styles.css y app.js
experiments/  validate_suite.py y run_benchmarks.py
data/         imágenes de entrada 2000², 4000², 6000²
results/      artefactos generados (ignorado por Git)
tests/        pruebas numéricas de la Práctica 02
scripts/      scripts originales de la Práctica 02
```

## Requisitos

- Docker Engine y NVIDIA Container Toolkit en el host.
- GPU NVIDIA con driver compatible con CUDA 12.4.
- Imagen base: `nvidia/cuda:12.4.1-devel-ubuntu22.04`.
- Python 3.10 dentro del contenedor. En Windows, use `py` en el host (no `python`).

## Construcción limpia y despliegue

Desde la raíz del repositorio.

### 1. Construir contenedor

```powershell
git rev-parse HEAD
docker build --no-cache --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t computo-paralelo-p3:latest .
docker image inspect computo-paralelo-p3:latest --format '{{.Id}} {{.Created}} {{.Config.Cmd}}'
```

### 2. Ejecutar contenedor

```powershell
# Si el nombre ya existe:
docker rm -f convolucion-pycuda

docker run --rm --gpus all --name convolucion-pycuda -p 8000:8000 `
  -e APP_QUEUE_MAXSIZE=8 -e APP_RESULT_TTL_SECONDS=900 `
  computo-paralelo-p3:latest
```

### 3. Exponer a la nube con Cloudflare (opcional)

Con el contenedor corriendo en otra terminal:

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflared entrega una URL temporal `https://*.trycloudflare.com`; solo funciona mientras el contenedor y el túnel estén activos. La API no tiene autenticación, así que no compartas la URL públicamente.

Comprobaciones:

```powershell
curl.exe http://127.0.0.1:8000/api/health
docker logs convolucion-pycuda
docker stats --no-stream convolucion-pycuda
nvidia-smi
```

Apagado y limpieza:

```powershell
docker stop convolucion-pycuda
docker image rm computo-paralelo-p3:latest
```

Frontend: `http://127.0.0.1:8000/`. Documentación OpenAPI: `http://127.0.0.1:8000/docs`.

## API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Interfaz web. |
| `POST` | `/api/convolve` | Acepta `file`, `filter`, `mask_size` y `threads_per_block` (`128`, `256`, `512`). Devuelve URLs `.png` y `.f32`, `request_id`, par?metros, m?tricas e informaci?n de GPU. |
| `GET` | `/api/health` | Estado, GPU, cola, límites, commit y versiones. |
| `GET` | `/api/results/{nombre}` | Descarga del resultado `.png` o `.f32`. |

Ejemplo:

```powershell
curl.exe -Method Post -Uri http://127.0.0.1:8000/api/convolve `
  -Form file=@data/256x170.jpg -Form filter=media -Form mask_size=9 -Form threads_per_block=256
```

Errores:

| Código | Significado |
|---|---|
| `400` | Cabecera no soportada o imagen corrupta. |
| `413` | Archivo, lado o número de píxeles excedido. |
| `422` | Filtro, m?scara o `threads_per_block` inv?lido (`threads_per_block` permitido: `128`, `256`, `512`). |
| `429` | Cola de GPU llena; reintente con backoff. |
| `503` | GPU/contexto/kernel no disponible. |
| `507` | Memoria de GPU insuficiente. |

## Configuración

| Variable | Default | Uso |
|---|---:|---|
| `APP_MAX_UPLOAD_MB` | `128` | Tamaño máximo del archivo subido. |
| `APP_MAX_PIXELS` | `36000000` | Píxeles máximos por imagen. |
| `APP_MAX_SIDE` | `8192` | Lado máximo en píxeles. |
| `APP_QUEUE_MAXSIZE` | `8` | Solicitudes en espera; si se llena, `429`. |
| `APP_RESULT_TTL_SECONDS` | `900` | Limpieza automática de resultados. |
| `APP_THREADS_PER_BLOCK` | `256` | Hilos por bloque si no se envían en la petición. |
| `APP_GPU_HEADROOM_MB` | `256` | Margen de memoria antes de rechazar por `507`. |
| `APP_CUDA_CACHE_DIR` | temporal del sistema | Caché de `SourceModule`. |
| `APP_RESULTS_DIR` | `results/` | Destino de los resultados. |
| `GIT_COMMIT` | `git rev-parse HEAD` | Commit informado por `/api/health`. |

Un único worker posee el contexto CUDA y consume una cola acotada. La validación y preparación de imagen se ejecutan en `threadpool` para no bloquear el bucle de eventos; `queue_wait_ms` mide la espera real antes de entrar al motor. Las transferencias usan memoria paginable de NumPy, sin pinned/host-mapped memory; `h2d_ms` incluye la reserva (`reserve_ms`). Los tiempos CUDA se leen de `Event.time_till`, que ya devuelve milisegundos, y se sincroniza antes de consultarlos.

## Validación

```powershell
py -m experiments.validate_suite --base-url http://127.0.0.1:8000 --resultados-dir results
```

La suite comprueba:

- Matrices manuales pequeñas y entradas `17x19` en `128` y `256` hilos.
- Salida directa PyCUDA contra salida HTTP usando el buffer `.f32` **antes** del recorte/cuantización a `uint8`, con tolerancia `1e-5`.
- Comparación CPU `float32` vs GPU `float32` con tolerancia `1e-4`: el orden de acumulación difiere y el error absoluto máximo observado es `3.05e-5`, aproximadamente `1.2e-7` relativo para valores de 8 bits.
- Entradas inválidas: archivo corrupto, tamaño excedido, filtro desconocido y máscaras no permitidas.
- Recuperación de `/api/health` tras errores.
- Aislamiento entre dos clientes concurrentes con imágenes sólidas distintas.
- Saturación de la cola con `429`: por defecto 30 clientes sobre `img_2000.pgm` con media `21x21`; con `APP_QUEUE_MAXSIZE=8` se observan 9 aceptadas y 21 rechazadas.
- Galería visual con el mismo recorte central para original y todos los filtros.

Salidas: `results/validation.csv`, `results/validation_summary.csv` y `results/visual/galeria_filtros.png`. Ejecución de referencia: 29 comprobaciones, 0 fallos.

## Benchmarks

```powershell
py -m experiments.run_benchmarks --base-url http://127.0.0.1:8000 `
  --reps 5 --warmup 1 --resultados-dir results
```

Se miden las resoluciones `2000²`, `4000²` y `6000²` con las cinco configuraciones (`media` 3/9/21, `nitidez` 3 y `laplaciano` 3), en tres modos:

1. `directo_pycuda`: eventos CUDA dentro del motor; `repetition=0` es arranque en frío.
2. `http_secuencial`: incluye validación, subida, espera en cola, GPU, postproceso y descarga del `.png` y `.f32` en `client_total_ms`.
3. `http_concurrente`: niveles `1`, `2`, `4` y `8`, con `throughput_req_s` calculado sobre respuestas correctas.

Salidas:

- `results/benchmark_results.csv`: una fila por solicitud con las 22 columnas exactas (212 filas: 90 directas, 90 HTTP secuenciales y 32 concurrentes).
- `results/host_directo/benchmark_results.csv`: 90 ejecuciones directas en el anfitrión con `--environment host`.
- `results/integration_comparison.csv`: vista derivada de media `9x9`, `4000x4000`, 256 hilos, con `directo_host`, `directo_contenedor` y `servicio_contenedor` (18 filas).
- `results/summary_stats.csv` y `results/summary_table.md`: mediana, IQR y desviación estándar separando `cold` de `steady`.
- `results/environment.json`: CPU, RAM del sistema, GPU/`nvidia-smi`, SO, driver, Docker, Python y versiones de paquetes.
- `results/image_hashes.csv`: SHA-256 de las seis entradas.

Todas las columnas de tiempo terminan en `_ms`; `throughput_req_s` está en solicitudes/s. `client_total_ms` incluye la descarga completa exigida por la guía. Los tiempos CUDA son los valores crudos de `Event.time_till` en milisegundos tras sincronización explícita.

## Práctica 02 (conservada)

```powershell
py -m unittest discover -s tests -v
py main.py --info
py scripts\preparar_imagenes.py data\arbol.jpg
py main.py --entrada data\imagen_pequena.pgm --filtro media --k 9 --hilos 256 --reps 5 `
  --limite-secuencial 300 --csv resultados\tiempos.csv
py scripts\ejecutar_experimentos.py --reps 5 --entrada-pequena data\imagen_pequena.pgm
py scripts\graficar.py
py scripts\comparar_cuatro_versiones.py --entrada data\imagen_pequena.pgm --reps 5
```

## Despliegue

Modalidad prevista: **local** con GPU NVIDIA RTX 4070 Laptop accesible desde Docker Desktop/WSL2. La guía admite esta alternativa cuando no hay créditos o infraestructura de nube; el informe debe documentar esa limitación, el anfitrión, el acceso desde un cliente distinto cuando la red lo permita y las evidencias de `docker logs`, `docker stats`, `nvidia-smi` y `/api/health`.

## Checklist para el informe

Completar estos datos antes de entregar; el repositorio no los inventa:

| Apartado | Evidencia a adjuntar |
|---|---|
| Despliegue local o en nube | Justificación; proveedor/región/instancia si aplica; captura de `docker run`. |
| Inventario | CPU, RAM, GPU, SO, driver, Docker, Python, PyCUDA, CUDA, puertos y cliente distinto. |
| Build limpio | ID de imagen, imagen base, versiones fijas y commit desde `environment.json`. |
| Pruebas | `validation_summary.csv`, capturas de `/api/health` y respuestas `429`/`503`/`507`. |
| Rendimiento | `summary_table.md`, `benchmark_results.csv` y comentario sobre mediana/IQR/desviación. |
| Visual | `results/visual/galeria_filtros.png` con original y los cinco filtros. |
| Recursos | `docker stats`, `nvidia-smi`, fecha/hora de apagado o borrado de instancia. |
| Firmas | Nombre, código y firma de cada integrante. |

> Los directorios `results/` y `graphify-out/` están ignorados por Git. Copie las evidencias seleccionadas a la carpeta de entregable del grupo antes de hacer commit.
