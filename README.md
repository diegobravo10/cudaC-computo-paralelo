# Práctica — Convolución de imágenes con PyCUDA

Migración a Python/PyCUDA del proyecto CUDA C. Conserva los filtros de media, nitidez y
laplaciano, extensión con ceros, distribución 1D (un hilo por píxel), validación CPU/GPU,
medición H2D/kernel/D2H, experimentos, CSV, imágenes y gráficas.

## Instalación y pruebas

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --info
```

## Ejecución

```powershell
python scripts\preparar_imagenes.py data\arbol.jpg
python main.py --entrada data\img_4000.pgm --filtro media --k 9 --hilos 256 --reps 5 `
  --csv resultados\tiempos.csv --salida resultados\imagenes\media9.pgm
python scripts\ejecutar_experimentos.py --reps 5
python scripts\graficar.py
```

Opciones: `--filtro media|nitidez|laplaciano`, `--k`, `--hilos`, `--reps`, `--csv`,
`--validacion`, `--salida`, `--sin-cpu` y `--info`.

La referencia CPU usa NumPy en `float32`; el cálculo GPU es un kernel CUDA compilado y
ejecutado por PyCUDA. Las reservas GPU se realizan fuera de las mediciones.
