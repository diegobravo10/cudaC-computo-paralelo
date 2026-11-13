"""Resume CSV y genera gráficas de Python secuencial y PyCUDA."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
TS = ("t_h2d", "t_computo", "t_d2h", "t_total")


def cargar(res):
    grupos = defaultdict(lambda: defaultdict(list))
    no_completadas = []
    with (res / "tiempos.csv").open(newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            estado = fila.get("estado") or "OK"
            if estado != "OK":
                no_completadas.append(fila)
                continue
            impl = {"CPU": "PYTHON_SECUENCIAL", "CUDA": "PYCUDA"}.get(
                fila["implementacion"], fila["implementacion"])
            c = (impl, fila["filtro"], int(fila["ancho"]), int(fila["alto"]),
                 int(fila["k"]), int(fila["hilos_por_bloque"]), int(fila["bloques"]))
            for t in TS:
                if fila[t] != "":
                    grupos[c][t].append(float(fila[t]))
    resumen = {}
    for c, tiempos in grupos.items():
        est = {"reps": len(tiempos["t_total"])}
        for t, valores in tiempos.items():
            v = np.asarray(valores)
            q1, med, q3 = np.percentile(v, (25, 50, 75))
            est[t] = {"mediana": med, "iqr": q3-q1,
                      "std": v.std(ddof=1) if len(v)>1 else 0, "min": v.min(), "max": v.max()}
        resumen[c] = est
    return resumen, no_completadas


def buscar(r, impl, filtro, ancho, k, hilos=256, alto=None):
    alto = ancho if alto is None else alto
    return next((e for (i,f,a,al,kk,h,_),e in r.items()
                 if i==impl and f==filtro and a==ancho and al==alto and kk==k
                 and (impl=="PYTHON_SECUENCIAL" or h==hilos)), None)


def guardar_csv(res, r, no_completadas):
    with (res/"resumen.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["implementacion","filtro","ancho","alto","k","hilos_por_bloque",
            "bloques","reps","tiempo","mediana_ms","iqr_ms","std_ms","min_ms","max_ms","estado"])
        for (impl,filtro,a,al,k,h,b),est in sorted(r.items()):
            for t,e in est.items():
                if t=="reps": continue
                w.writerow([impl,filtro,a,al,k,h,b,est["reps"],t,*[f"{e[x]:.4f}" for x in ("mediana","iqr","std","min","max")],"OK"])
        for x in no_completadas:
            w.writerow([x["implementacion"],x["filtro"],x["ancho"],x["alto"],x["k"],
                        x["hilos_por_bloque"],x["bloques"],0,"","","","","","",x.get("estado","NO_COMPLETADA")])
    with (res/"aceleraciones.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["filtro","ancho","alto","k","hilos_por_bloque","python_ms",
            "kernel_ms","pycuda_total_ms","pct_transferencias","aceleracion_kernel","aceleracion_total"])
        for (impl,filtro,a,al,k,h,_),gpu in sorted(r.items()):
            if impl!="PYCUDA": continue
            py=buscar(r,"PYTHON_SECUENCIAL",filtro,a,k,alto=al)
            if py is None: continue
            sec=py["t_computo"]["mediana"]; kern=gpu["t_computo"]["mediana"]; total=gpu["t_total"]["mediana"]
            tr=gpu["t_h2d"]["mediana"]+gpu["t_d2h"]["mediana"]
            w.writerow([filtro,a,al,k,h,f"{sec:.4f}",f"{kern:.4f}",f"{total:.4f}",
                        f"{100*tr/total:.2f}",f"{sec/kern:.2f}",f"{sec/total:.2f}"])


def linea(ax,x,datos,t,etiqueta):
    pares=[(xx,d) for xx,d in zip(x,datos) if d is not None]
    if not pares: return
    xx,dd=zip(*pares); ax.errorbar(xx,[d[t]["mediana"] for d in dd],
        yerr=[d[t]["iqr"]/2 for d in dd],marker="o",capsize=4,label=etiqueta)


def graficas(res,r):
    graf=res/"graficas"; graf.mkdir(parents=True,exist_ok=True)
    filtros=("media","nitidez","laplaciano"); x=np.arange(3); fig,ax=plt.subplots(figsize=(8,4.8))
    gpu=[buscar(r,"PYCUDA",f,4000,3) for f in filtros]
    if all(gpu):
        for off,(nombre,t) in enumerate((("Kernel","t_computo"),("Total","t_total"))):
            ax.bar(x+(off-.5)*.34,[e[t]["mediana"] for e in gpu],.34,
                   yerr=[e[t]["iqr"]/2 for e in gpu],label=nombre)
        ax.set_xticks(x,[f.title() for f in filtros]); ax.set_ylabel("Tiempo (ms)"); ax.legend(); ax.grid(axis="y",alpha=.3)
        fig.tight_layout(); fig.savefig(graf/"1_filtro.png",dpi=150)
    plt.close(fig)

    ns=(2000,4000,6000); mp=np.array(ns)**2/1e6; g=[buscar(r,"PYCUDA","media",n,9) for n in ns]
    if all(g):
        fig,ax=plt.subplots(figsize=(8,4.8)); linea(ax,mp,g,"t_computo","Kernel"); linea(ax,mp,g,"t_h2d","CPU→GPU")
        linea(ax,mp,g,"t_d2h","GPU→CPU"); linea(ax,mp,g,"t_total","Total"); ax.set_xlabel("Megapíxeles")
        ax.set_ylabel("Tiempo (ms)"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(graf/"2_resolucion.png",dpi=150); plt.close(fig)

    ks=(3,9,21); g=[buscar(r,"PYCUDA","media",4000,k) for k in ks]
    if all(g):
        fig,ax=plt.subplots(figsize=(8,4.8)); linea(ax,ks,g,"t_computo","Kernel"); linea(ax,ks,g,"t_total","Total"); ax.set_xlabel("Tamaño de máscara")
        ax.set_ylabel("Tiempo (ms)"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(graf/"3_mascara.png",dpi=150); plt.close(fig)

    hs=(128,256,512); g=[buscar(r,"PYCUDA","media",4000,9,h) for h in hs]
    if all(g):
        fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(range(3),[e["t_computo"]["mediana"] for e in g],yerr=[e["t_computo"]["iqr"]/2 for e in g],capsize=5)
        ax.set_xticks(range(3),[f"{h} hilos" for h in hs]); ax.set_ylabel("Kernel (ms)"); ax.grid(axis="y",alpha=.3)
        fig.tight_layout(); fig.savefig(graf/"4_hilos.png",dpi=150); plt.close(fig)

    filas=sorted(((f,a,k,h),e) for (i,f,a,al,k,h,_),e in r.items() if i=="PYCUDA" and a==al and a>=2000)
    if filas:
        etiquetas=[f"{f} {k}×{k}\n{a}² {h}h" for (f,a,k,h),_ in filas]
        partes={t:np.array([e[t]["mediana"] for _,e in filas]) for t in ("t_h2d","t_computo","t_d2h")}
        total=sum(partes.values()); base=np.zeros(len(filas)); fig,ax=plt.subplots(figsize=(12,5))
        for t,nombre in (("t_h2d","CPU→GPU"),("t_computo","Kernel"),("t_d2h","GPU→CPU")):
            pct=100*partes[t]/total; ax.bar(etiquetas,pct,bottom=base,label=nombre); base+=pct
        ax.set_ylabel("% del tiempo GPU"); ax.legend(); fig.tight_layout(); fig.savefig(graf/"5_transferencias.png",dpi=150); plt.close(fig)

    comunes=[]
    for (impl,f,a,al,k,h,b),py in r.items():
        if impl!="PYTHON_SECUENCIAL": continue
        gpu=buscar(r,"PYCUDA",f,a,k,alto=al)
        if gpu: comunes.append((f,k,a,al,py,gpu))
    if comunes:
        labels=[f"{f} {k}×{k}\n{a}×{al}" for f,k,a,al,_,_ in comunes]; x=np.arange(len(comunes)); fig,ax=plt.subplots(figsize=(10,5))
        ax.bar(x-.2,[py["t_computo"]["mediana"] for *_,py,gpu in comunes],.4,label="Python secuencial")
        ax.bar(x+.2,[gpu["t_total"]["mediana"] for *_,py,gpu in comunes],.4,label="PyCUDA total")
        ax.set_xticks(x,labels); ax.set_yscale("log"); ax.set_ylabel("Tiempo (ms, escala log)"); ax.legend(); ax.grid(axis="y",alpha=.3)
        fig.tight_layout(); fig.savefig(graf/"6_python_vs_pycuda.png",dpi=150); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--resultados",type=Path,default=RAIZ/"resultados"); a=p.parse_args()
    res=a.resultados.resolve(); r,no=cargar(res); guardar_csv(res,r,no); graficas(res,r)
    for pgm in (res/"imagenes").glob("*.pgm"): Image.open(pgm).save(pgm.with_suffix(".png"))
    print(f"Generados CSV y gráficas en {res}")


if __name__=="__main__": main()
