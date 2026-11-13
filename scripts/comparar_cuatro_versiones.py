"""Ejecuta y compara C++ secuencial, CUDA C, Python secuencial y PyCUDA.

La entrada se convierte una sola vez a PGM y el mismo archivo se entrega a ambas
aplicaciones. Los resultados se guardan fuera de los proyectos originales.
"""

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
CUDA_C_PREDET = RAIZ.parent / "cudaC-computo-paralelo"
CONFIGS = [("media", 3), ("nitidez", 3), ("laplaciano", 3),
           ("media", 9), ("media", 21)]
COLUMNAS = ["implementacion", "filtro", "ancho", "alto", "k", "hilos_por_bloque",
            "bloques", "repeticion", "t_h2d", "t_computo", "t_d2h", "t_total",
            "unidad", "estado", "motivo"]


def preparar_pgm(origen, destino):
    imagen = Image.open(origen).convert("L")
    destino.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(destino)
    return imagen.size


def ejecutar(cmd, cwd):
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def combinar(c_csv, py_csv, destino):
    mapeo = {"CPU": "CPP_SECUENCIAL", "CUDA": "CUDA_C",
             "PYTHON_SECUENCIAL": "PYTHON_SECUENCIAL", "PYCUDA": "PYCUDA"}
    filas = []
    for ruta in (c_csv, py_csv):
        with ruta.open(newline="", encoding="utf-8") as f:
            for x in csv.DictReader(f):
                filas.append({c: x.get(c, "") for c in COLUMNAS} | {
                    "implementacion": mapeo[x["implementacion"]],
                    "estado": x.get("estado") or "OK",
                })
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS); w.writeheader(); w.writerows(filas)
    return filas


def combinar_validaciones(c_val, py_val, destino):
    columnas=["referencia","comparada","filtro","ancho","alto","k","hilos_por_bloque",
              "error_max","error_medio","tolerancia","resultado"]
    salida=[]
    with c_val.open(newline="",encoding="utf-8") as f:
        for x in csv.DictReader(f):
            salida.append({c:x.get(c,"") for c in columnas} | {
                "referencia":"CPP_SECUENCIAL","comparada":"CUDA_C"})
    with py_val.open(newline="",encoding="utf-8") as f:
        for x in csv.DictReader(f):
            salida.append({c:x.get(c,"") for c in columnas})
    with destino.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=columnas); w.writeheader(); w.writerows(salida)


def resumir(filas, res):
    grupos = defaultdict(list)
    for x in filas:
        if x["estado"] != "OK": continue
        clave=(x["implementacion"],x["filtro"],int(x["ancho"]),int(x["alto"]),
               int(x["k"]),int(x["hilos_por_bloque"]))
        grupos[clave].append(x)
    resumen={}
    for clave,rs in grupos.items():
        resumen[clave]={t:statistics.median(float(x[t] or 0) for x in rs)
                        for t in ("t_h2d","t_computo","t_d2h","t_total")}
    with (res/"resumen_cuatro_versiones.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["implementacion","filtro","ancho","alto","k","hilos",
                                     "mediana_h2d_ms","mediana_computo_ms","mediana_d2h_ms","mediana_total_ms"])
        for clave,t in sorted(resumen.items()): w.writerow([*clave,*[f"{t[x]:.6f}" for x in ("t_h2d","t_computo","t_d2h","t_total")]])
    with (res/"aceleraciones_cuatro_versiones.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["filtro","ancho","alto","k","hilos","cpp_ms","cuda_kernel_ms",
            "cuda_total_ms","speedup_cuda_kernel","speedup_cuda_total","python_ms","pycuda_kernel_ms",
            "pycuda_total_ms","speedup_pycuda_kernel","speedup_pycuda_total"])
        dims=sorted({(f,a,al,k) for i,f,a,al,k,h in resumen})
        for filtro,a,al,k in dims:
            def get(impl,h=0): return resumen.get((impl,filtro,a,al,k,h))
            cpp,cu,py,pg=get("CPP_SECUENCIAL"),get("CUDA_C",256),get("PYTHON_SECUENCIAL"),get("PYCUDA",256)
            if not all((cpp,cu,py,pg)): continue
            w.writerow([filtro,a,al,k,256,f"{cpp['t_total']:.6f}",f"{cu['t_computo']:.6f}",f"{cu['t_total']:.6f}",
                f"{cpp['t_total']/cu['t_computo']:.3f}",f"{cpp['t_total']/cu['t_total']:.3f}",f"{py['t_total']:.6f}",
                f"{pg['t_computo']:.6f}",f"{pg['t_total']:.6f}",f"{py['t_total']/pg['t_computo']:.3f}",f"{py['t_total']/pg['t_total']:.3f}"])
    return resumen


def graficar(resumen, res):
    configs=sorted({(f,a,al,k) for i,f,a,al,k,h in resumen})
    completos=[]
    for c in configs:
        f,a,al,k=c
        claves=[("CPP_SECUENCIAL",f,a,al,k,0),("CUDA_C",f,a,al,k,256),
                ("PYTHON_SECUENCIAL",f,a,al,k,0),("PYCUDA",f,a,al,k,256)]
        if all(x in resumen for x in claves): completos.append((c,claves))
    labels=[f"{f} {k}×{k}\n{a}×{al}" for (f,a,al,k),_ in completos]
    x=np.arange(len(completos)); width=.2; fig,ax=plt.subplots(figsize=(12,6))
    for pos,(nombre,idx,tiempo) in enumerate((("C++ secuencial",0,"t_total"),("CUDA C total",1,"t_total"),
                                               ("Python secuencial",2,"t_total"),("PyCUDA total",3,"t_total"))):
        valores=[resumen[claves[idx]][tiempo] for _,claves in completos]
        ax.bar(x+(pos-1.5)*width,valores,width,label=nombre)
    ax.set_xticks(x,labels); ax.set_yscale("log"); ax.set_ylabel("Tiempo mediano (ms, escala log)")
    ax.set_title("Comparación de las cuatro versiones sobre la misma imagen"); ax.legend(); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(res/"comparacion_cuatro_versiones.png",dpi=170); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--entrada",required=True,type=Path,help="imagen pequeña común")
    p.add_argument("--cuda-c-dir",type=Path,default=CUDA_C_PREDET)
    p.add_argument("--resultados",type=Path,default=RAIZ/"resultados_comparacion_cuatro")
    p.add_argument("--reps",type=int,default=5)
    p.add_argument("--hilos",type=int,default=256)
    p.add_argument("--limite-secuencial",type=float,default=300)
    a=p.parse_args(); res=a.resultados.resolve(); res.mkdir(parents=True,exist_ok=True)
    exe=(a.cuda_c_dir/"bin"/"convolucion.exe").resolve()
    if not exe.exists(): p.error(f"no existe {exe}; compile primero el proyecto CUDA C")
    if not a.entrada.exists(): p.error(f"no existe {a.entrada}")
    entrada=res/"entrada_comun.pgm"; ancho,alto=preparar_pgm(a.entrada,entrada)
    imagenes_c=res/"imagenes_cuda_c"; imagenes_py=res/"imagenes_pycuda"
    imagenes_c.mkdir(exist_ok=True); imagenes_py.mkdir(exist_ok=True)
    c_csv=res/"tiempos_cuda_c.csv"; py_csv=res/"tiempos_python_pycuda.csv"
    c_val=res/"validacion_cuda_c.csv"; py_val=res/"validacion_python_pycuda.csv"
    for ruta in (c_csv,py_csv,c_val,py_val): ruta.unlink(missing_ok=True)
    for filtro,k in CONFIGS:
        cmd_c=[str(exe),"--entrada",str(entrada),"--filtro",filtro,"--k",str(k),
            "--hilos",str(a.hilos),"--reps",str(a.reps),"--csv",str(c_csv),"--validacion",str(c_val)]
        cmd_py=[sys.executable,str(RAIZ/"main.py"),"--entrada",str(entrada),"--filtro",filtro,"--k",str(k),
            "--hilos",str(a.hilos),"--reps",str(a.reps),"--limite-secuencial",str(a.limite_secuencial),
            "--csv",str(py_csv),"--validacion",str(py_val)]
        if k==3:
            cmd_c += ["--salida",str(imagenes_c/f"{filtro}_3x3.pgm")]
            cmd_py += ["--salida",str(imagenes_py/f"{filtro}_3x3.pgm")]
        ejecutar(cmd_c,a.cuda_c_dir); ejecutar(cmd_py,RAIZ)
    filas=combinar(c_csv,py_csv,res/"tiempos_cuatro_versiones.csv")
    combinar_validaciones(c_val,py_val,res/"validacion_cuatro_versiones.csv")
    resumen=resumir(filas,res); graficar(resumen,res)
    Image.open(entrada).save(res/"entrada_comun.png")
    for carpeta in (imagenes_c,imagenes_py):
        for pgm in carpeta.glob("*.pgm"): Image.open(pgm).save(pgm.with_suffix(".png"))
    (res/"metodologia.txt").write_text(
        f"Entrada común: {entrada.name} ({ancho}x{alto})\nRepeticiones: {a.reps}\nHilos: {a.hilos}\n"
        "Calentamiento: 1 C++/Python y 3 CUDA C/PyCUDA.\n"
        "Validaciones antes del recorte: C++ vs CUDA C y Python secuencial vs PyCUDA.\n",
        encoding="utf-8")
    print(f"\nComparación de cuatro versiones generada en {res}")


if __name__=="__main__": main()
