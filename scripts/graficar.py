import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
RES, HILOS = RAIZ / "resultados", 256
GRAF = RES / "graficas"
TS = ("t_h2d", "t_computo", "t_d2h", "t_total")


def cargar():
    grupos = defaultdict(lambda: defaultdict(list))
    with (RES / "tiempos.csv").open(newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            c = (fila["implementacion"], fila["filtro"], int(fila["ancho"]), int(fila["k"]),
                 int(fila["hilos_por_bloque"]), int(fila["bloques"]))
            for t in TS:
                grupos[c][t].append(float(fila[t]))
    resumen = {}
    for c, tiempos in grupos.items():
        est = {"reps": len(tiempos["t_total"])}
        for t, valores in tiempos.items():
            v = np.asarray(valores)
            q1, med, q3 = np.percentile(v, (25, 50, 75))
            est[t] = {"mediana": med, "iqr": q3 - q1,
                      "std": v.std(ddof=1) if len(v) > 1 else 0, "min": v.min(), "max": v.max()}
        resumen[c] = est
    return resumen


def buscar(r, impl, filtro, n, k, hilos=HILOS):
    return next((e for (i, f, a, kk, h, _), e in r.items()
                 if i == impl and f == filtro and a == n and kk == k and (i == "CPU" or h == hilos)), None)


def guardar_csv(r):
    with (RES / "resumen.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["implementacion", "filtro", "resolucion", "k", "hilos_por_bloque",
            "bloques", "reps", "tiempo", "mediana_ms", "iqr_ms", "std_ms", "min_ms", "max_ms"])
        for (impl, filtro, n, k, h, b), est in sorted(r.items()):
            for t in TS:
                if impl == "CPU" and t in ("t_h2d", "t_d2h"): continue
                e = est[t]; w.writerow([impl, filtro, f"{n}x{n}", k, h, b, est["reps"], t,
                    *[f"{e[x]:.4f}" for x in ("mediana", "iqr", "std", "min", "max")]])
    with (RES / "aceleraciones.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filtro", "resolucion", "k", "hilos_por_bloque", "cpu_ms",
            "kernel_ms", "gpu_total_ms", "pct_transferencias", "aceleracion_computo", "aceleracion_con_transferencias"])
        for (impl, filtro, n, k, h, _), est in sorted(r.items()):
            cpu = buscar(r, "CPU", filtro, n, k)
            if impl != "CUDA" or cpu is None: continue
            c, kern, tot = cpu["t_computo"]["mediana"], est["t_computo"]["mediana"], est["t_total"]["mediana"]
            tr = est["t_h2d"]["mediana"] + est["t_d2h"]["mediana"]
            w.writerow([filtro, f"{n}x{n}", k, h, f"{c:.3f}", f"{kern:.3f}", f"{tot:.3f}",
                        f"{100*tr/tot:.1f}", f"{c/kern:.2f}", f"{c/tot:.2f}"])


def linea(ax, x, datos, tiempo, etiqueta):
    ax.errorbar(x, [d[tiempo]["mediana"] for d in datos],
                yerr=[d[tiempo]["iqr"] / 2 for d in datos], marker="o", capsize=4, label=etiqueta)


def graficas(r):
    GRAF.mkdir(parents=True, exist_ok=True)
    filtros = ("media", "nitidez", "laplaciano"); x = np.arange(3); fig, ax = plt.subplots(figsize=(7, 4.5))
    cpu = [buscar(r, "CPU", f, 4000, 3) for f in filtros]; gpu = [buscar(r, "CUDA", f, 4000, 3) for f in filtros]
    for off, (nombre, d, t) in enumerate((("CPU", cpu, "t_computo"), ("PyCUDA kernel", gpu, "t_computo"), ("PyCUDA total", gpu, "t_total"))):
        ax.bar(x + (off-1)*.27, [e[t]["mediana"] for e in d], .27, yerr=[e[t]["iqr"]/2 for e in d], label=nombre)
    ax.set_xticks(x, ("Media", "Nitidez", "Laplaciano")); ax.set_yscale("log"); ax.set_ylabel("Tiempo (ms)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(GRAF / "1_filtro.png", dpi=150); plt.close(fig)

    ns = (2000, 4000, 6000); fig, axs = plt.subplots(1, 2, figsize=(11, 4.5)); xp = np.array(ns)**2/1e6
    linea(axs[0], xp, [buscar(r,"CPU","media",n,9) for n in ns], "t_computo", "CPU")
    g = [buscar(r,"CUDA","media",n,9) for n in ns]
    linea(axs[1], xp, g, "t_computo", "Kernel")
    linea(axs[1], xp, g, "t_h2d", "CPU→GPU")
    linea(axs[1], xp, g, "t_d2h", "GPU→CPU")
    linea(axs[1], xp, g, "t_total", "Total")
    axs[0].set_title("Procesamiento CPU")
    axs[1].set_title("PyCUDA: transferencias y kernel")
    for ax in axs: ax.set_xlabel("Megapíxeles"); ax.set_ylabel("Tiempo (ms)"); ax.legend(); ax.grid(alpha=.3)
    fig.suptitle("Resolución: media 9x9"); fig.tight_layout(); fig.savefig(GRAF/"2_resolucion.png", dpi=150); plt.close(fig)

    ks=(3,9,21); fig,axs=plt.subplots(1,2,figsize=(11,4.5)); linea(axs[0],ks,[buscar(r,"CPU","media",4000,k) for k in ks],"t_computo","CPU")
    g=[buscar(r,"CUDA","media",4000,k) for k in ks]; linea(axs[1],ks,g,"t_computo","Kernel"); linea(axs[1],ks,g,"t_total","Total")
    for ax in axs: ax.set_xlabel("Tamaño de máscara"); ax.set_ylabel("Tiempo (ms)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(GRAF/"3_mascara.png",dpi=150); plt.close(fig)

    hs=(128,256,512); g=[buscar(r,"CUDA","media",4000,9,h) for h in hs]; fig,ax=plt.subplots(figsize=(7,4.5))
    ax.bar(range(3),[e["t_computo"]["mediana"] for e in g],yerr=[e["t_computo"]["iqr"]/2 for e in g],capsize=5)
    ax.set_xticks(range(3),[f"{h} hilos" for h in hs]); ax.set_ylabel("Kernel (ms)"); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(GRAF/"4_hilos.png",dpi=150); plt.close(fig)

    filas=sorted(((f,n,k,h),e) for (i,f,n,k,h,_),e in r.items() if i=="CUDA"); etiquetas=[f"{f} {k}x{k}\n{n}² {h}h" for (f,n,k,h),_ in filas]
    partes={t:np.array([e[t]["mediana"] for _,e in filas]) for t in ("t_h2d","t_computo","t_d2h")}; total=sum(partes.values()); base=np.zeros(len(filas)); fig,ax=plt.subplots(figsize=(12,5))
    for t,nombre in (("t_h2d","CPU→GPU"),("t_computo","Kernel"),("t_d2h","GPU→CPU")):
        pct=100*partes[t]/total; ax.bar(etiquetas,pct,bottom=base,label=nombre); base+=pct
    ax.set_ylabel("% del tiempo GPU"); ax.legend(); fig.tight_layout(); fig.savefig(GRAF/"5_transferencias.png",dpi=150); plt.close(fig)


def main():
    r=cargar(); guardar_csv(r); graficas(r)
    for pgm in (RES/"imagenes").glob("*.pgm"): Image.open(pgm).save(pgm.with_suffix(".png"))
    print("Generados resumen.csv, aceleraciones.csv, graficas/*.png e imagenes/*.png")


if __name__ == "__main__": main()
