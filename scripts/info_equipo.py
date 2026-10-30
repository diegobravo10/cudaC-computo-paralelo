"""Registra el equipo y las versiones de herramientas en resultados/equipo.txt (Windows)."""

import platform
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def ejecutar(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(no disponible: {e})"


def powershell(expr):
    return ejecutar(f'powershell -NoProfile -Command "{expr}"')


def main():
    partes = [
        "== Sistema ==",
        f"SO: {platform.platform()}",
        "",
        "== CPU ==",
        powershell("Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,"
                   "NumberOfLogicalProcessors,MaxClockSpeed"),
        "",
        "== RAM ==",
        powershell("'{0:N1} GB' -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"),
        "",
        "== GPU (programa) ==",
        ejecutar(f'"{RAIZ / "bin" / "convolucion.exe"}" --info'),
        "",
        "== GPU (nvidia-smi) ==",
        ejecutar("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"),
        "",
        "== Herramientas ==",
        ejecutar("nvcc --version"),
        f"Python {platform.python_version()}",
    ]
    texto = "\n".join(partes) + "\n"
    salida = RAIZ / "resultados" / "equipo.txt"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
