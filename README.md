# Práctica 01 — Convolución de imágenes de alta resolución con CUDA C

Filtros de **media**, **nitidez** y **laplaciano** implementados en CPU (C++ secuencial) y en GPU
(CUDA C, bloques e hilos en X, un hilo por píxel, lectura desde memoria global, extensión con ceros).

## Dependencias

| Herramienta | Versión usada |
|---|---|
| CUDA Toolkit (`nvcc`) | 13.4 |
| Visual Studio / MSVC (`cl.exe`) | VS 18, MSVC 14.52 |
| Python | 3.12 + `numpy`, `pillow`, `matplotlib` |
| GPU | NVIDIA con capacidad de cómputo ≥ 5.0 |

```powershell
pip install -r requirements.txt
```

## Estructura

```
src/
  conv_cpu.cpp      convolución secuencial (referencia)
  conv_gpu.cu       kernel CUDA + medición con eventos CUDA
  imagen.cpp        lectura/escritura PGM y conversión a [0,255]
  mascaras.cpp      coeficientes de los filtros
  validacion.h      error absoluto máximo/medio y tolerancia
  main.cpp          programa de medición (CPU + CUDA + validación + CSV)
tests/
  pruebas.cpp       prueba manual 5x5, dimensiones no múltiplo, casos extremos
scripts/
  preparar_imagenes.py      recorte cuadrado y redimensionamiento a 2000/4000/6000
  info_equipo.py            registra CPU, RAM, GPU y versiones
  ejecutar_experimentos.py  corre los 4 experimentos obligatorios
  graficar.py               tablas resumen y gráficas
data/          imágenes de entrada (las .pgm se regeneran, no se suben)
resultados/    tiempos.csv, validacion.csv, resumen.csv, aceleraciones.csv, graficas/, imagenes/
```

## Compilación

```powershell
.\build.bat
```

Genera `bin\convolucion.exe` y `bin\pruebas.exe`. El script carga automáticamente el entorno de
Visual Studio si `cl.exe` no está en el PATH.

## Ejecución (flujo completo)

```powershell
# 1. Preparar las imágenes de entrada a partir de una foto de alta resolución
python scripts\preparar_imagenes.py ruta\a\foto.jpg

# 2. Pruebas de corrección
.\bin\pruebas.exe

# 3. Registrar el equipo
python scripts\info_equipo.py

# 4. Experimentos (1 calentamiento + 5 repeticiones por configuración)
python scripts\ejecutar_experimentos.py --reps 5

# 5. Tablas y gráficas
python scripts\graficar.py
```

### Ejecución manual de una configuración

```powershell
.\bin\convolucion.exe --entrada data\img_4000.pgm --filtro media --k 9 --hilos 256 --reps 5 `
    --csv resultados\tiempos.csv --salida resultados\imagenes\media9.pgm
.\bin\convolucion.exe --info     # datos de la GPU
```

Opciones: `--filtro media|nitidez|laplaciano`, `--k` (impar; nitidez y laplaciano solo 3),
`--hilos` (hilos por bloque), `--reps`, `--csv`, `--validacion`, `--salida`, `--sin-cpu`.

## Datos de entrada

Las imágenes `data/img_2000.pgm`, `img_4000.pgm` e `img_6000.pgm` no se incluyen por su tamaño
(4 MB, 16 MB y 36 MB). Se regeneran con `scripts/preparar_imagenes.py` a partir de la foto original;
el procedimiento aplicado queda documentado en `data/preparacion.txt`.

## Formato de los CSV

`resultados/tiempos.csv` (registros originales, una fila por repetición):

| columna | descripción |
|---|---|
| implementacion | CPU o CUDA |
| filtro, ancho, alto, k | configuración |
| hilos_por_bloque, bloques | 0 en CPU |
| repeticion | 1..reps (el calentamiento no se registra) |
| t_h2d, t_computo, t_d2h, t_total | tiempos; en CPU t_computo = t_total |
| unidad | ms |

`resultados/validacion.csv`: error absoluto máximo y medio CPU vs GPU por configuración.
