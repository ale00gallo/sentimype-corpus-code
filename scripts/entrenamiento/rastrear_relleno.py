import os
"""Rastrea las estrellas rellenadas con 3.0 comparando las versiones del dataset unificado (Unified Dataset v2, v3, v6, Final...,
1_apify_*.csv) con el corpus actual. Una reseña se considera RELLENADA si en alguna versión anterior aparece con estrellas vacías
y en el corpus actual tiene 3.0. Cuenta cuántas caen en cada conjunto (entrenamiento, validación, tests y CLIP).
Además busca en los notebooks las líneas que podrían haber hecho el relleno.

Uso, en data"entrenamiento v4" (después de ronda2_paper_v15.py):
    python rastrear_relleno.py
    (opcional) --dir /otra/carpeta
Salida: resultados_v15_r2/RESULTADOS_paper_v15_ronda2_adenda.txt, rellenadas_por_conjunto.csv, rellenadas_lista.csv, estrellas_por_version.csv
y rehace resultados_v15_r2.zip.
"""
import json, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

V4 = Path(".").resolve()
OUT = V4 / "resultados_v15_r2"
OUT.mkdir(exist_ok=True)
TXT = []


def out(s=""):
    print(s); TXT.append(str(s))


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()[:80]


extra = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
_c = [p.resolve() for p in [V4.parent, Path(os.environ.get("DATA_DIR", "data")), Path(os.environ.get("DATA_DIR", "data"))] + extra if p.exists()]
RAICES = [p for p in dict.fromkeys(_c) if not any(q != p and q in p.parents for q in _c)]
SKIP = {"node_modules", "venv", ".venv", "clip_env", ".git", "models", "mvsa_images", "imagenes_nuevas", "__pycache__", "site-packages", "Library", "imagenes"}

fin = pd.read_csv(V4 / "Dataset_SentiMYPE_FINAL.csv", low_memory=False)
fin["r"] = pd.to_numeric(fin["rating_score"], errors="coerce")
for c in ("origen", "lote", "label_source"):
    if c not in fin.columns:
        fin[c] = "NO DISPONIBLE"
fin["k_texto"] = fin["raw_text"].map(norm)
base = fin[fin.r == 3].copy()
out("SentiMYPE, ronda 2: ADENDA a la tarea 5 (rastreo de estrellas rellenadas entre versiones del dataset)")
out(f"Corpus actual: {len(fin)} reseñas; con 3.0: {len(base)}. Carpetas revisadas: {', '.join(str(r) for r in RAICES)}")

# ------------------------------------------------------------------ versiones del dataset
EXCL = re.compile(r"^(train|validation|test|dataset_sentimype|sentimype_v\d|facebook_nuevo|resenas_nuevas|candidatas|grupo\d|clave_|figuras|resultados|puntajes|predicciones|velocimetro_|errores)", re.I)
COLR = ["rating_score", "rating", "stars", "estrellas"]
versiones = []
for raiz in RAICES:
    for dp, dn, fn in os.walk(raiz):
        dn[:] = [d for d in dn if d not in SKIP and not d.startswith(("resultados_", "splits_", "paquete", "SentiMYPE_resultados", "salida_verificacion", "figuras_", "neutras_nuevas"))]
        for f in fn:
            if not f.lower().endswith(".csv") or EXCL.match(f):
                continue
            ruta = Path(dp) / f
            try:
                if ruta.stat().st_size > 600_000_000:
                    continue
                cab = pd.read_csv(ruta, nrows=0).columns.tolist()
                low = {c.lower(): c for c in cab}
                cr = next((low[c] for c in COLR if c in low), None)
                ct = next((low[c] for c in ("raw_text", "text", "texto", "reviewtext") if c in low), None)
                ci = low.get("review_id")
                if cr is None or (ct is None and ci is None):
                    continue
                d = pd.read_csv(ruta, dtype=str, usecols=[c for c in (cr, ct, ci) if c], low_memory=False)
                d["r_ver"] = pd.to_numeric(d[cr], errors="coerce")
                if ci and d[ci].isin(set(fin.review_id)).mean() >= 0.3:
                    d["clave"], modo = d[ci], "review_id"
                elif ct:
                    d["clave"], modo = d[ct].map(norm), "texto"
                else:
                    continue
                d = d[["clave", "r_ver"]].drop_duplicates("clave", keep="last")
                versiones.append({"archivo": str(ruta), "modo": modo, "df": d, "mtime": ruta.stat().st_mtime})
            except Exception:
                continue
out(f"\nVersiones del dataset con estrellas y texto/review_id encontradas: {len(versiones)}")
filas_v = []
for v in sorted(versiones, key=lambda x: x["archivo"]):
    k = base.review_id if v["modo"] == "review_id" else base.k_texto
    m = pd.DataFrame({"clave": k.values}).merge(v["df"], on="clave", how="left", indicator=True)
    esta = m["_merge"] == "both"
    vac = esta & m.r_ver.isna()
    filas_v.append({"archivo": v["archivo"], "clave": v["modo"], "reseñas con 3.0 actual presentes": int(esta.sum()),
                    "de ellas con estrellas vacías": int(vac.sum()), "con 3.0": int((esta & (m.r_ver == 3)).sum()),
                    "con otro valor": int((esta & m.r_ver.notna() & (m.r_ver != 3)).sum())})
tv = pd.DataFrame(filas_v)
if len(tv):
    tv.to_csv(OUT / "estrellas_por_version.csv", index=False)
    out("\nReseñas con 3.0 en el corpus actual, según cada versión (vacías = sin estrellas en esa versión):")
    out("  " + tv.assign(archivo=tv.archivo.str.replace(str(Path.home()), "~")).to_string(index=False).replace("\n", "\n  "))

# ------------------------------------------------------------------ rellenadas
vac_en = {}
for v in versiones:
    k = base.review_id if v["modo"] == "review_id" else base.k_texto
    m = pd.DataFrame({"review_id": base.review_id.values, "clave": k.values}).merge(v["df"], on="clave", how="inner")
    for rid in m.review_id[m.r_ver.isna()]:
        vac_en.setdefault(rid, []).append(Path(v["archivo"]).name)
base["rellenada"] = base.review_id.isin(vac_en)
base["vacia_en"] = base.review_id.map(lambda r: "; ".join(sorted(set(vac_en.get(r, [])))))
out(f"\nReseñas con 3.0 que aparecen sin estrellas en alguna versión anterior (RELLENADAS, cota inferior): {int(base.rellenada.sum())} de {len(base)}")
base[base.rellenada][["review_id", "split_type", "origen", "label_source", "lote", "vacia_en", "raw_text"]].to_csv(OUT / "rellenadas_lista.csv", index=False)

sets = {"Entrenamiento (con estrellas)": fin[(fin.split_type == "train") & fin.r.notna()], "Validación (con estrellas)": fin[(fin.split_type == "validation") & fin.r.notna()],
        "Test humano": fin[fin.split_type == "test"], "Test multimodal": fin[fin.split_type == "test_multimodal"]}
try:
    sys.path.insert(0, str(V4))
    import entrenar_clip_velocimetro as mclip
    dfc, _, _ = mclip.preparar_datos()
    sets["CLIP (reseñas con imagen y estrellas)"] = fin[fin.review_id.isin(set(dfc.review_id))]
except Exception as e:
    out(f"  (CLIP: NO DISPONIBLE: {e})")
filas = []
out("\nRellenadas por conjunto:")
for nombre, d in sets.items():
    n3 = d[d.r == 3]
    rel = base[base.review_id.isin(n3.review_id) & base.rellenada]
    filas.append({"conjunto": nombre, "n con estrellas": len(d), "con 3.0": len(n3), "rellenadas (cota inferior)": len(rel),
                  "% del conjunto": round(100 * len(rel) / max(len(d), 1), 2), "por origen": str(rel.origen.value_counts().to_dict()),
                  "por label_source": str(rel.label_source.value_counts().to_dict())})
    out(f"  {nombre}: n = {len(d)} | con 3.0: {len(n3)} | rellenadas: {len(rel)} ({100 * len(rel) / max(len(d), 1):.2f} %) | origen {rel.origen.value_counts().to_dict()} | label_source {rel.label_source.value_counts().to_dict()}")
pd.DataFrame(filas).to_csv(OUT / "rellenadas_por_conjunto.csv", index=False)
out("\n  Importante: es una COTA INFERIOR. Solo cuenta reseñas que aparecen vacías en alguna versión que existe en tu Mac; "
    "las que ya entraron con 3.0 desde la primera versión, o las que no están en ninguna, no se pueden distinguir.")

# ------------------------------------------------------------------ código
out("\nLíneas de notebooks/scripts que tocan el rating y podrían haber hecho el relleno (revisar a mano):")
RX = re.compile(r"(rating|estrellas|stars)[^\n]{0,80}(fillna|combine_first|\.update\(|replace\(|where\(|= *3(\.0)?\s*$|\bor\s+3\b|\belse\s+3\b)|"
                r"(fillna|combine_first)[^\n]{0,60}(rating|estrellas|stars)", re.I)
hall = []
for raiz in RAICES:
    for dp, dn, fn in os.walk(raiz):
        dn[:] = [d for d in dn if d not in SKIP and not d.startswith(("resultados_", "splits_", "paquete", "SentiMYPE_resultados"))]
        for f in fn:
            if not f.endswith((".ipynb", ".py")) or f in ("rastrear_relleno.py", "ronda2_paper_v15.py", "recalcular_paper_v15.py"):
                continue
            ruta = Path(dp) / f
            try:
                if ruta.stat().st_size > 8_000_000:
                    continue
                txt = ruta.read_text(encoding="utf-8", errors="ignore")
                if f.endswith(".ipynb"):
                    lins = [(i, l) for c_ in json.loads(txt)["cells"] if c_.get("cell_type") == "code" for i, l in enumerate("".join(c_.get("source", [])).splitlines(), 1)]
                else:
                    lins = list(enumerate(txt.splitlines(), 1))
            except Exception:
                continue
            for n, l in lins:
                if RX.search(l) and "sentiment_label" not in l.split("fillna")[0][-60:]:
                    hall.append((str(ruta).replace(str(Path.home()), "~"), n, l.strip()[:170]))
vistos = set(); unicos = []
for a, n, l in hall:
    if (Path(a).name, l) not in vistos:
        vistos.add((Path(a).name, l)); unicos.append((a, n, l))
out(f"  {len(unicos)} líneas distintas (de {len(hall)} coincidencias; las copias del mismo notebook se agrupan):")
for a, n, l in unicos[:60]:
    out(f"    {a}:{n}: {l}")
pd.DataFrame(unicos, columns=["archivo", "linea", "codigo"]).to_csv(OUT / "5a_lineas_adenda.csv", index=False)

# ------------------------------------------------------------------ guardar
(OUT / "RESULTADOS_paper_v15_ronda2_adenda.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
shutil.copy(__file__, OUT / Path(__file__).name)
z = V4 / "resultados_v15_r2.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            zf.write(p, f"resultados_v15_r2/{p.relative_to(OUT)}")
print(f"\nListo: {OUT / 'RESULTADOS_paper_v15_ronda2_adenda.txt'} y {z}")
if sys.platform == "darwin":
    subprocess.run(["open", "-R", str(z)])
