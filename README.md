# Práctica — Convolución de imágenes con PyCUDA

Migración a Python/PyCUDA del proyecto CUDA C. Conserva los filtros de media, nitidez y
laplaciano, extensión con ceros, distribución 1D (un hilo por píxel), validación numérica,
medición H2D/kernel/D2H, experimentos, CSV, imágenes y gráficas.

Incluye dos implementaciones propias: Python secuencial puro (recorridos explícitos de
fila, columna y coeficientes) y PyCUDA. No se usa NumPy para realizar la convolución.

## Instalación y pruebas

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --info
```

## Ejecución

```powershell
python scripts\preparar_imagenes.py data\arbol.jpg
python main.py --entrada data\imagen_pequena.pgm --filtro media --k 9 --hilos 256 --reps 5 `
  --limite-secuencial 300 --csv resultados\tiempos.csv
python scripts\ejecutar_experimentos.py --reps 5 --entrada-pequena data\imagen_pequena.pgm
python scripts\graficar.py
```

Las resoluciones 2000², 4000² y 6000² se ejecutan en PyCUDA. Python secuencial se marca
como `NO_COMPLETADA` para esas entradas y se mide en una imagen pequeña común, tal como
permite la guía cuando las configuraciones grandes no son viables.

## Comparación de las cuatro versiones

Con el proyecto CUDA C compilado y una imagen pequeña:

```powershell
python scripts\comparar_cuatro_versiones.py --entrada data\imagen_pequena.pgm --reps 5
```

Genera `resultados_comparacion_cuatro/` con:

- C/C++ secuencial, CUDA C, Python secuencial y PyCUDA en el mismo CSV.
- Validaciones antes del recorte de visualización para cada pareja de referencia.
- Aceleraciones CUDA C/C++ y PyCUDA/Python.
- Tabla resumen y gráfica comparativa de las cuatro versiones.

Opciones: `--filtro media|nitidez|laplaciano`, `--k`, `--hilos`, `--reps`, `--csv`,
`--validacion`, `--salida`, `--sin-secuencial`, `--limite-secuencial` y `--info`.

Las cuatro versiones usan datos y acumulación `float32`. La lectura, escritura, contexto,
compilación y reserva quedan fuera del tiempo del kernel. El CSV registra por separado
contexto/compilación, reserva, H2D, kernel, D2H y total.
