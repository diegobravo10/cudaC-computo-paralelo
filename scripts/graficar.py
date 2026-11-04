"""Genera tablas resumen y gráficas a partir de resultados/tiempos.csv.

Salidas:
  resultados/resumen.csv           mediana, IQR y desviación estándar por configuración
  resultados/aceleraciones.csv     aceleración de cómputo y con transferencias
  resultados/graficas/*.png        una gráfica por experimento + proporción de transferencias
  resultados/imagenes/*.png        conversión de las salidas .pgm a .png

Uso:  python scripts/graficar.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
GRAF = RES / "graficas"
HILOS_FIJOS = 256
TIEMPOS = ("t_h2d", "t_computo", "t_d2h", "t_total")


def cargar():
    """Agrupa las repeticiones por configuración y calcula estadísticas."""
    grupos = defaultdict(lambda: defaultdict(list))
    with open(RES / "tiempos.csv", newline="") as f:
        for fila in csv.DictReader(f):
            clave = (fila["implementacion"], fila["filtro"], int(fila["ancho"]),
                     int(fila["k"]), int(fila["hilos_por_bloque"]), int(fila["bloques"]))
            for t in TIEMPOS:
                grupos[clave][t].append(float(fila[t]))

    resumen = {}
    for clave, tiempos in grupos.items():
        est = {"reps": len(tiempos["t_total"])}
        for t, valores in tiempos.items():
            v = np.array(valores)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            est[t] = {"mediana": med, "iqr": q3 - q1, "std": v.std(ddof=1) if len(v) > 1 else 0.0,
                      "min": v.min(), "max": v.max()}
        resumen[clave] = est
    return resumen


def buscar(resumen, impl, filtro, n, k, hilos=HILOS_FIJOS):
    for (i, f, a, kk, h, _), est in resumen.items():
        if i == impl and f == filtro and a == n and kk == k and (impl == "CPU" or h == hilos):
            return est
    return None


def guardar_resumen(resumen):
    with open(RES / "resumen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementacion", "filtro", "resolucion", "k", "hilos_por_bloque", "bloques",
                    "reps", "tiempo", "mediana_ms", "iqr_ms", "std_ms", "min_ms", "max_ms"])
        for (impl, filtro, n, k, h, b), est in sorted(resumen.items()):
            for t in TIEMPOS:
                if impl == "CPU" and t in ("t_h2d", "t_d2h"):
                    continue
                e = est[t]
                w.writerow([impl, filtro, f"{n}x{n}", k, h, b, est["reps"], t,
                            f"{e['mediana']:.4f}", f"{e['iqr']:.4f}", f"{e['std']:.4f}",
                            f"{e['min']:.4f}", f"{e['max']:.4f}"])

    with open(RES / "aceleraciones.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filtro", "resolucion", "k", "hilos_por_bloque", "cpu_ms", "kernel_ms",
                    "gpu_total_ms", "pct_transferencias", "aceleracion_computo",
                    "aceleracion_con_transferencias"])
        for (impl, filtro, n, k, h, _), est in sorted(resumen.items()):
            if impl != "CUDA":
                continue
            cpu = buscar(resumen, "CPU", filtro, n, k)
            if cpu is None:
                continue
            c = cpu["t_computo"]["mediana"]
            kern = est["t_computo"]["mediana"]
            tot = est["t_total"]["mediana"]
            transf = est["t_h2d"]["mediana"] + est["t_d2h"]["mediana"]
            w.writerow([filtro, f"{n}x{n}", k, h, f"{c:.3f}", f"{kern:.3f}", f"{tot:.3f}",
                        f"{100 * transf / tot:.1f}", f"{c / kern:.2f}", f"{c / tot:.2f}"])


def barras_error(ax, x, est, t, **kw):
    med = [e[t]["mediana"] for e in est]
    err = [e[t]["iqr"] / 2 for e in est]
    ax.errorbar(x, med, yerr=err, marker="o", capsize=4, **kw)


def grafica_filtro(resumen):
    filtros = ["media", "nitidez", "laplaciano"]
    cpu = [buscar(resumen, "CPU", f, 4000, 3) for f in filtros]
    gpu = [buscar(resumen, "CUDA", f, 4000, 3) for f in filtros]
    x = np.arange(len(filtros))
    ancho = 0.27
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (etiqueta, datos, t) in enumerate([("CPU", cpu, "t_computo"),
                                               ("CUDA kernel", gpu, "t_computo"),
                                               ("CUDA total (con transferencias)", gpu, "t_total")]):
        ax.bar(x + (i - 1) * ancho, [d[t]["mediana"] for d in datos], ancho,
               yerr=[d[t]["iqr"] / 2 for d in datos], capsize=3, label=etiqueta)
    ax.set_xticks(x, ["Media 3x3", "Nitidez 3x3", "Laplaciano 3x3"])
    ax.set_yscale("log")
    ax.set_ylabel("Tiempo (ms, mediana, escala log)")
    ax.set_title("Experimento filtro: imagen 4000x4000, 256 hilos/bloque")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAF / "1_filtro.png", dpi=150)
    plt.close(fig)


def grafica_resolucion(resumen):
    ns = [2000, 4000, 6000]
    mp = [n * n / 1e6 for n in ns]
    cpu = [buscar(resumen, "CPU", "media", n, 9) for n in ns]
    gpu = [buscar(resumen, "CUDA", "media", n, 9) for n in ns]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    barras_error(axs[0], mp, cpu, "t_computo", label="CPU")
    axs[0].set_title("CPU")
    barras_error(axs[1], mp, gpu, "t_computo", label="Kernel")
    barras_error(axs[1], mp, gpu, "t_total", label="Total (con transferencias)")
    barras_error(axs[1], mp, gpu, "t_h2d", label="CPU→GPU")
    barras_error(axs[1], mp, gpu, "t_d2h", label="GPU→CPU")
    axs[1].set_title("CUDA (256 hilos/bloque)")
    for ax in axs:
        ax.set_xlabel("Megapíxeles (2000², 4000², 6000²)")
        ax.set_ylabel("Tiempo (ms, mediana)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Experimento resolución: media 9x9")
    fig.tight_layout()
    fig.savefig(GRAF / "2_resolucion.png", dpi=150)
    plt.close(fig)


def grafica_mascara(resumen):
    ks = [3, 9, 21]
    cpu = [buscar(resumen, "CPU", "media", 4000, k) for k in ks]
    gpu = [buscar(resumen, "CUDA", "media", 4000, k) for k in ks]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    barras_error(axs[0], ks, cpu, "t_computo", label="CPU")
    axs[0].set_title("CPU")
    barras_error(axs[1], ks, gpu, "t_computo", label="Kernel")
    barras_error(axs[1], ks, gpu, "t_total", label="Total (con transferencias)")
    axs[1].set_title("CUDA (256 hilos/bloque)")
    for ax in axs:
        ax.set_xticks(ks, [f"{k}x{k}\n({k*k} coef.)" for k in ks])
        ax.set_xlabel("Tamaño de la máscara")
        ax.set_ylabel("Tiempo (ms, mediana)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Experimento máscara: media, imagen 4000x4000")
    fig.tight_layout()
    fig.savefig(GRAF / "3_mascara.png", dpi=150)
    plt.close(fig)


def grafica_hilos(resumen):
    hs = [128, 256, 512]
    gpu = [buscar(resumen, "CUDA", "media", 4000, 9, h) for h in hs]
    hs, gpu = zip(*[(h, g) for h, g in zip(hs, gpu) if g is not None])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(hs))
    med = [g["t_computo"]["mediana"] for g in gpu]
    ax.bar(x, med, yerr=[g["t_computo"]["iqr"] / 2 for g in gpu], capsize=5, color="tab:orange")
    for xi, g in zip(x, gpu):
        e = g["t_computo"]
        ax.text(xi, e["mediana"], f"{e['mediana']:.2f} ms\nσ={e['std']:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"{h} hilos" for h in hs])
    ax.set_ylabel("Tiempo de kernel (ms, mediana)")
    ax.set_title("Experimento hilos por bloque: media 9x9, 4000x4000")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAF / "4_hilos.png", dpi=150)
    plt.close(fig)


def grafica_transferencias(resumen):
    """Barras apiladas: qué parte del tiempo total de GPU es copia y qué parte es kernel."""
    filas = sorted(((f, n, k, h), e) for (i, f, n, k, h, _), e in resumen.items() if i == "CUDA")
    etiquetas = [f"{f} {k}x{k}\n{n}² {h}h" for (f, n, k, h), _ in filas]
    partes = {t: np.array([e[t]["mediana"] for _, e in filas]) for t in ("t_h2d", "t_computo", "t_d2h")}
    total = sum(partes.values())
    fig, ax = plt.subplots(figsize=(12, 5))
    base = np.zeros(len(filas))
    for t, etiqueta in [("t_h2d", "CPU→GPU"), ("t_computo", "Kernel"), ("t_d2h", "GPU→CPU")]:
        pct = 100 * partes[t] / total
        ax.bar(etiquetas, pct, bottom=base, label=etiqueta)
        base += pct
    ax.set_ylabel("% del tiempo total en GPU")
    ax.set_title("Proporción de transferencias vs kernel")
    ax.legend(loc="upper right")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(GRAF / "5_transferencias.png", dpi=150)
    plt.close(fig)


def convertir_imagenes():
    for pgm in (RES / "imagenes").glob("*.pgm"):
        Image.open(pgm).save(pgm.with_suffix(".png"))


def main():
    GRAF.mkdir(parents=True, exist_ok=True)
    resumen = cargar()
    guardar_resumen(resumen)
    grafica_filtro(resumen)
    grafica_resolucion(resumen)
    grafica_mascara(resumen)
    grafica_hilos(resumen)
    grafica_transferencias(resumen)
    convertir_imagenes()
    print("Generados resumen.csv, aceleraciones.csv, graficas/*.png e imagenes/*.png")
    print((RES / "aceleraciones.csv").read_text())


if __name__ == "__main__":
    main()
