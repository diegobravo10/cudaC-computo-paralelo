import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TAMANOS = (2000, 4000, 6000)
DATA = Path(__file__).resolve().parent.parent / "data"


def sintetica(n=6000):
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    img = 120 * x + 60 * y + 50 * np.sin(40 * np.pi * x) * np.cos(25 * np.pi * y)
    for cx, cy, r in ((.3, .3, .15), (.7, .6, .2), (.5, .85, .08)):
        img[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = 230
    img += np.random.default_rng(0).normal(0, 8, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("imagen", nargs="?")
    p.add_argument("--sintetica", action="store_true")
    p.add_argument("--solo-convertir", action="store_true")
    a = p.parse_args()
    if not a.imagen and not a.sintetica:
        p.error("indique una imagen o use --sintetica")
    DATA.mkdir(exist_ok=True)
    if a.solo_convertir:
        gris = Image.open(a.imagen).convert("L")
        gris.save(DATA / f"{Path(a.imagen).stem}.pgm")
        return
    original = sintetica() if a.sintetica else Image.open(a.imagen)
    gris = original.convert("L")
    lado = min(gris.size)
    izq, arriba = (gris.width - lado) // 2, (gris.height - lado) // 2
    recorte = gris.crop((izq, arriba, izq + lado, arriba + lado))
    lineas = [f"Origen: {'imagen sintética' if a.sintetica else Path(a.imagen).name}",
              f"Recorte central: {(izq, arriba, izq + lado, arriba + lado)}",
              "Conversión: escala de grises; redimensionamiento: LANCZOS"]
    for n in TAMANOS:
        img = recorte if lado == n else recorte.resize((n, n), Image.Resampling.LANCZOS)
        img.save(DATA / f"img_{n}.pgm")
        img.save(DATA / f"img_{n}.png")
        print(f"Generada data/img_{n}.pgm")
    (DATA / "preparacion.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
