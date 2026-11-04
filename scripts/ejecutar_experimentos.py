"""Ejecuta los 4 experimentos obligatorios y guarda los tiempos en resultados/tiempos.csv.

Las configuraciones repetidas entre experimentos se ejecutan una sola vez
(la guía permite reutilizar mediciones en condiciones idénticas):
  - media 3x3, 4000, 256 hilos   -> experimentos "filtro" y "máscara"
  - media 9x9, 4000, 256 hilos   -> experimentos "resolución", "máscara" e "hilos"

Uso:  python scripts/ejecutar_experimentos.py [--reps 5]
"""

import argparse
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EXE = RAIZ / "bin" / "convolucion.exe"
RES = RAIZ / "resultados"
HILOS_FIJOS = 256

# (filtro, k, resolución, hilos por bloque)
EXPERIMENTOS = {
    "filtro":     [(f, 3, 4000, HILOS_FIJOS) for f in ("media", "nitidez", "laplaciano")],
    "resolucion": [("media", 9, n, HILOS_FIJOS) for n in (2000, 4000, 6000)],
    "mascara":    [("media", k, 4000, HILOS_FIJOS) for k in (3, 9, 21)],
    "hilos":      [("media", 9, 4000, h) for h in (128, 256, 512)],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()

    RES.mkdir(exist_ok=True)
    (RES / "imagenes").mkdir(exist_ok=True)
    for nombre in ("tiempos.csv", "validacion.csv"):
        (RES / nombre).unlink(missing_ok=True)  # empezar de cero

    configuraciones = []
    for lista in EXPERIMENTOS.values():
        for c in lista:
            if c not in configuraciones:
                configuraciones.append(c)

    for i, (filtro, k, n, hilos) in enumerate(configuraciones, 1):
        print(f"\n[{i}/{len(configuraciones)}] {filtro} {k}x{k} | {n}x{n} | {hilos} hilos")
        cmd = [str(EXE), "--entrada", str(RAIZ / "data" / f"img_{n}.pgm"),
               "--filtro", filtro, "--k", str(k), "--hilos", str(hilos), "--reps", str(a.reps),
               "--csv", str(RES / "tiempos.csv"), "--validacion", str(RES / "validacion.csv")]
        # Imágenes representativas para el informe: los filtros 3x3 y la media 21x21 en 4000.
        if n == 4000 and hilos == HILOS_FIJOS and k in (3, 21):
            cmd += ["--salida", str(RES / "imagenes" / f"{filtro}_{k}x{k}_{n}.pgm")]
        subprocess.run(cmd, check=True)

    print("\nListo: resultados/tiempos.csv y resultados/validacion.csv")


if __name__ == "__main__":
    main()
