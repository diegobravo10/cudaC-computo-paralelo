"""Convierte imágenes .pgm a .png para poder verlas en Windows.

Uso:
  python scripts/pgm_a_png.py resultados/imagenes/mi_foto_laplaciano.pgm
  python scripts/pgm_a_png.py resultados/imagenes        (todas las .pgm de la carpeta)
"""

import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

for arg in sys.argv[1:]:
    ruta = Path(arg)
    archivos = sorted(ruta.glob("*.pgm")) if ruta.is_dir() else [ruta]
    for pgm in archivos:
        Image.open(pgm).save(pgm.with_suffix(".png"))
        print(f"{pgm} -> {pgm.with_suffix('.png')}")
