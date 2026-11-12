import platform
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def ejecutar(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except Exception as exc: return f"(no disponible: {exc})"


def main():
    partes=["== Sistema ==",f"SO: {platform.platform()}","","== GPU (PyCUDA) ==",
            ejecutar([sys.executable,str(RAIZ/"main.py"),"--info"]),"","== GPU (nvidia-smi) ==",
            ejecutar(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"]),
            "","== Herramientas ==",f"Python {platform.python_version()}"]
    salida=RAIZ/"resultados"/"equipo.txt"; salida.parent.mkdir(exist_ok=True)
    salida.write_text("\n".join(partes)+"\n",encoding="utf-8"); print("\n".join(partes))


if __name__ == "__main__": main()
