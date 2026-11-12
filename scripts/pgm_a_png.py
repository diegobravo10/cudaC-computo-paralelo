import sys
from pathlib import Path

from PIL import Image

for arg in sys.argv[1:]:
    ruta = Path(arg)
    for pgm in (sorted(ruta.glob("*.pgm")) if ruta.is_dir() else [ruta]):
        Image.open(pgm).save(pgm.with_suffix(".png"))
        print(f"{pgm} -> {pgm.with_suffix('.png')}")
