"""Ejecuta PyCUDA en alta resolución y Python secuencial en una entrada viable."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
HILOS_FIJOS = 256
EXPERIMENTOS = {
    "filtro": [(f, 3, 4000, HILOS_FIJOS) for f in ("media", "nitidez", "laplaciano")],
    "resolucion": [("media", 9, n, HILOS_FIJOS) for n in (2000, 4000, 6000)],
    "mascara": [("media", k, 4000, HILOS_FIJOS) for k in (3, 9, 21)],
    "hilos": [("media", 9, 4000, h) for h in (128, 256, 512)],
}
CABECERA = ["implementacion", "filtro", "ancho", "alto", "k", "hilos_por_bloque",
            "bloques", "repeticion", "t_h2d", "t_computo", "t_d2h", "t_total",
            "t_contexto_compilacion", "t_reserva", "unidad", "estado", "motivo"]


def ejecutar_python(entrada, filtro, k, hilos, reps, res, sin_secuencial=False,
                     limite=None, salida=None):
    cmd = [sys.executable, str(RAIZ / "main.py"), "--entrada", str(entrada),
           "--filtro", filtro, "--k", str(k), "--hilos", str(hilos), "--reps", str(reps),
           "--csv", str(res / "tiempos.csv"), "--validacion", str(res / "validacion.csv")]
    if sin_secuencial:
        cmd.append("--sin-secuencial")
    if limite is not None:
        cmd += ["--limite-secuencial", str(limite)]
    if salida:
        cmd += ["--salida", str(salida)]
    subprocess.run(cmd, check=True, cwd=RAIZ)


def registrar_no_completadas(res):
    """Documenta configuraciones grandes omitidas, sin inventar tiempos."""
    casos = []
    for nombre in ("filtro", "resolucion", "mascara"):
        for filtro, k, n, hilos in EXPERIMENTOS[nombre]:
            caso = (filtro, k, n, hilos)
            if caso not in casos:
                casos.append(caso)
    ruta = res / "tiempos.csv"
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(CABECERA)
        for filtro, k, n, _ in casos:
            w.writerow(["PYTHON_SECUENCIAL", filtro, n, n, k, 0, 0, 0,
                        "", "", "", "", "", "", "ms", "NO_COMPLETADA",
                        "Resolución grande no viable; usar comparación común con --entrada-pequena"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--resultados", type=Path, default=RAIZ / "resultados")
    p.add_argument("--entrada-pequena", type=Path,
                   help="imagen común viable para Python secuencial y PyCUDA")
    p.add_argument("--limite-secuencial", type=float, default=300,
                   help="máximo por ejecución secuencial pequeña (segundos)")
    p.add_argument("--solo-pequena", action="store_true")
    a = p.parse_args()
    res = a.resultados.resolve()
    res.mkdir(parents=True, exist_ok=True)
    (res / "imagenes").mkdir(exist_ok=True)
    for nombre in ("tiempos.csv", "validacion.csv"):
        (res / nombre).unlink(missing_ok=True)

    if not a.solo_pequena:
        configuraciones = []
        for lista in EXPERIMENTOS.values():
            for c in lista:
                if c not in configuraciones:
                    configuraciones.append(c)
        for i, (filtro, k, n, hilos) in enumerate(configuraciones, 1):
            print(f"\n[{i}/{len(configuraciones)}] PyCUDA {filtro} {k}x{k} | {n}x{n} | {hilos} hilos")
            salida = None
            if n == 4000 and hilos == HILOS_FIJOS and k in (3, 21):
                salida = res / "imagenes" / f"{filtro}_{k}x{k}_{n}.pgm"
            ejecutar_python(RAIZ / "data" / f"img_{n}.pgm", filtro, k, hilos,
                             a.reps, res, sin_secuencial=True, salida=salida)
        registrar_no_completadas(res)

    if a.entrada_pequena:
        if not a.entrada_pequena.exists():
            p.error(f"no existe {a.entrada_pequena}")
        pequenas = [(f, 3) for f in ("media", "nitidez", "laplaciano")]
        pequenas += [("media", 9), ("media", 21)]
        print("\n== Comparación viable Python secuencial vs PyCUDA ==")
        for filtro, k in pequenas:
            ejecutar_python(a.entrada_pequena, filtro, k, HILOS_FIJOS, a.reps, res,
                             limite=a.limite_secuencial)
    else:
        print("\nAVISO: falta --entrada-pequena; no se ejecutó Python secuencial.")

    print(f"\nResultados: {res}")


if __name__ == "__main__":
    main()
