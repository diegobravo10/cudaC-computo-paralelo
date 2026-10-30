"""Prepara las imágenes de entrada de la práctica.

A partir de UNA imagen original:
  1. Convierte a escala de grises (Pillow, modo "L": L = 0.299 R + 0.587 G + 0.114 B).
  2. Recorta el cuadrado central más grande posible (recorte común para todas).
  3. Redimensiona a 2000x2000, 4000x4000 y 6000x6000 con filtro LANCZOS.
  4. Guarda data/img_<N>.pgm (entrada del programa) y data/img_<N>.png (vista previa).
  5. Escribe data/preparacion.txt documentando lo realizado.

Uso:
  python scripts/preparar_imagenes.py ruta/a/foto.jpg
  python scripts/preparar_imagenes.py --sintetica     (imagen de prueba generada, sin foto)
  python scripts/preparar_imagenes.py foto.jpg --solo-convertir
      (solo pasa la foto a gris en data/<nombre>.pgm, con su tamaño original, para probar filtros)
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # permitir fotos muy grandes
TAMANOS = [2000, 4000, 6000]
DATA = Path(__file__).resolve().parent.parent / "data"


def imagen_sintetica(n=6000):
    """Patrón con gradientes, círculos y texturas para probar el flujo sin una foto."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    img = 120 * x + 60 * y
    img += 50 * np.sin(40 * np.pi * x) * np.cos(25 * np.pi * y)
    for cx, cy, r in [(0.3, 0.3, 0.15), (0.7, 0.6, 0.2), (0.5, 0.85, 0.08)]:
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = 230
    img += np.random.default_rng(0).normal(0, 8, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("imagen", nargs="?", help="foto original (jpg/png...)")
    p.add_argument("--sintetica", action="store_true")
    p.add_argument("--solo-convertir", action="store_true")
    a = p.parse_args()
    if not a.imagen and not a.sintetica:
        p.error("indique una imagen o use --sintetica")

    DATA.mkdir(exist_ok=True)
    if a.solo_convertir:
        gris = Image.open(a.imagen).convert("L")
        destino = DATA / (Path(a.imagen).stem + ".pgm")
        gris.save(destino)
        print(f"Generada {destino.relative_to(DATA.parent)} ({gris.width}x{gris.height})")
        return

    if a.sintetica:
        original = imagen_sintetica()
        origen = "imagen sintética generada por el script (6000x6000)"
    else:
        original = Image.open(a.imagen)
        origen = f"{Path(a.imagen).name} ({original.width}x{original.height}, modo {original.mode})"

    gris = original.convert("L")
    lado = min(gris.width, gris.height)
    izq = (gris.width - lado) // 2
    arr = (gris.height - lado) // 2
    caja = (izq, arr, izq + lado, arr + lado)
    recorte = gris.crop(caja)

    lineas = [
        f"Imagen original: {origen}",
        "Conversion a gris: Pillow convert('L')  ->  L = 0.299 R + 0.587 G + 0.114 B",
        f"Recorte cuadrado central: caja (izq, arriba, der, abajo) = {caja}, lado = {lado} px",
        "Redimensionamiento: Pillow resize(..., Image.LANCZOS)",
    ]
    for n in TAMANOS:
        img = recorte if n == lado else recorte.resize((n, n), Image.LANCZOS)
        img.save(DATA / f"img_{n}.pgm")
        img.save(DATA / f"img_{n}.png")
        accion = "reduccion" if n < lado else ("ampliacion" if n > lado else "sin cambio")
        lineas.append(f"  img_{n}.pgm: {lado}x{lado} -> {n}x{n} ({accion})")
        print(f"Generada data/img_{n}.pgm")

    if lado < max(TAMANOS):
        lineas.append(f"AVISO: el recorte ({lado}px) es menor que {max(TAMANOS)}px; "
                      "las versiones mayores son ampliaciones.")
    (DATA / "preparacion.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
