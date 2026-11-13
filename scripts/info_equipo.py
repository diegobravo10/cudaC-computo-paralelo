"""Registra hardware, sistema y versiones exigidas por la guía."""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ=Path(__file__).resolve().parent.parent


def ejecutar(cmd):
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,check=False)
        return (r.stdout or r.stderr).strip()
    except Exception as exc:
        return f"(no disponible: {exc})"


def powershell(expr):
    return ejecutar(["powershell","-NoProfile","-Command",expr])


def main():
    p=argparse.ArgumentParser(); p.add_argument("--salida",type=Path,default=RAIZ/"resultados"/"equipo.txt"); a=p.parse_args()
    try:
        import pycuda
        pycuda_version=getattr(pycuda,"VERSION_TEXT",str(getattr(pycuda,"VERSION","desconocida")))
    except Exception as exc: pycuda_version=f"no disponible: {exc}"
    partes=["== Sistema ==",f"SO: {platform.platform()}",f"Arquitectura: {platform.machine()}","",
        "== CPU ==",powershell("Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"),"",
        "== RAM ==",powershell("'{0:N2} GiB' -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"),"",
        "== GPU y CUDA ==",ejecutar([sys.executable,str(RAIZ/"main.py"),"--info"]),"",
        "== Controlador ==",ejecutar(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"]),"",
        "== CUDA Toolkit ==",ejecutar(["nvcc","--version"]),"",
        "== Entorno Python ==",f"Python: {platform.python_version()}",f"NumPy: {np.__version__}",f"PyCUDA: {pycuda_version}"]
    texto="\n".join(partes)+"\n"; a.salida.parent.mkdir(parents=True,exist_ok=True)
    a.salida.write_text(texto,encoding="utf-8"); print(texto)


if __name__=="__main__": main()
