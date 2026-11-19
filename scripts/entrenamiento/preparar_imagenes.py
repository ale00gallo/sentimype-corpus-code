"""Arma mvsa_images/ e imagenes_nuevas/ (con enlaces) a partir de la carpeta única imagenes/ de SentiMYPE_data_v2,
con los nombres que esperan los scripts de CLIP y de fusión:
  - mvsa_images/: archivos <review_id>_<n>.jpg
  - imagenes_nuevas/: archivos <review_id>.jpg (sin número)
Uso, parada en "entrenamiento v4":
  python3 preparar_imagenes.py data
"""
import os, re, sys, shutil
from pathlib import Path

origen = Path(sys.argv[1] if len(sys.argv) > 1 else "data").expanduser().resolve()
carpeta = origen / "imagenes" if (origen / "imagenes").is_dir() else origen
if not carpeta.is_dir():
    sys.exit(f"No encuentro la carpeta de imágenes en {origen}")
for c in ("mvsa_images", "imagenes_nuevas"):
    p = Path(c)
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    p.mkdir()
n = {"mvsa_images": 0, "imagenes_nuevas": 0}
for f in sorted(carpeta.iterdir()):
    if f.suffix.lower() != ".jpg":
        continue
    destino = "mvsa_images" if re.search(r"_\d+\.jpg$", f.name, re.I) else "imagenes_nuevas"
    os.symlink(f, Path(destino) / f.name)
    n[destino] += 1
print(f"mvsa_images: {n['mvsa_images']} | imagenes_nuevas: {n['imagenes_nuevas']} | total: {sum(n.values())}")
print("(esperado: 1,904 + 166 = 2,070)")
