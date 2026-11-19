"""Sensibilidad al relleno de estrellas SIN reentrenar: recalcula MAE y Pearson excluyendo las reseñas rellenadas que encontró
rastrear_relleno.py (cota inferior), y el Pearson de validación del que sale el peso por correlación.

Uso, en data"entrenamiento v4" (después de rastrear_relleno.py):
    python sensibilidad_relleno.py
Salida: resultados_v15_r2/RESULTADOS_paper_v15_ronda2_sensibilidad.txt, sensibilidad_relleno.csv y rehace resultados_v15_r2.zip.
"""
import glob, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr

V4 = Path(".").resolve()
OUT = V4 / "resultados_v15_r2"
TXT = []


def out(s=""):
    print(s); TXT.append(str(s))


def ultima(patron):
    c = sorted(glob.glob(str(V4 / patron)))
    return Path(c[-1]) if c else None


def mae_r(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs(real - pred))), float(pearsonr(real, pred)[0]), len(real)


rel = pd.read_csv(OUT / "rellenadas_lista.csv")
ids_rel = set(rel.review_id)
fin = pd.read_csv(V4 / "Dataset_SentiMYPE_FINAL.csv", low_memory=False, usecols=["review_id", "rating_score", "split_type"])
fin["r"] = pd.to_numeric(fin.rating_score, errors="coerce")
fin["ref"] = ((fin.r - 1) / 4 * 10).clip(0, 10)
filas = []
out("SentiMYPE, ronda 2: sensibilidad al relleno de estrellas (sin reentrenar)")
out(f"Reseñas rellenadas conocidas (cota inferior, de rastrear_relleno.py): {len(ids_rel)}")


def fila(conj, variante, m):
    filas.append({"conjunto": conj, "variante": variante, "MAE": round(m[0], 3), "Pearson": round(m[1], 3), "n": m[2]})
    out(f"  [{conj}] {variante}: MAE {m[0]:.3f} | Pearson {m[1]:.3f} | n = {m[2]}")


# 1. Velocímetro de texto, test humano
pv = ultima("resultados_roberta_v2_*/predicciones_test.csv")
out("\n1. Velocímetro de texto v4, test humano (referencia: estrellas reescaladas a 0-10)")
if pv is None:
    out("  NO DISPONIBLE: no encontré predicciones_test.csv del velocímetro v4.")
else:
    d = pd.read_csv(pv)
    d = d.merge(fin[["review_id", "r", "ref"]], on="review_id", how="left", suffixes=("", "_f"))
    d = d[d.ref.notna()]
    fila("Test humano", "todas las reseñas con estrellas", mae_r(d.ref, d.score_predicho))
    fila("Test humano", f"sin las {int(d.review_id.isin(ids_rel).sum())} rellenadas conocidas", mae_r(d[~d.review_id.isin(ids_rel)].ref, d[~d.review_id.isin(ids_rel)].score_predicho))
    fila("Test humano", "sin NINGUNA con 3.0 (referencia extrema; incluye 3.0 reales)", mae_r(d[d.r != 3].ref, d[d.r != 3].score_predicho))

# 2. Validación: Pearson del texto y peso por correlación
pval = ultima("resultados_cortes_*/puntajes_texto_validacion.csv")
out("\n2. Pearson de validación del texto (r_t) y peso por correlación w = r_v / (r_t + r_v)")
if pval is None:
    out("  NO DISPONIBLE: no encontré puntajes_texto_validacion.csv.")
else:
    v = pd.read_csv(pval).merge(fin[["review_id", "ref"]], on="review_id", how="left")
    v = v[v.ref.notna()]
    r_all = pearsonr(v.ref, v.score_texto)[0]
    r_sin = pearsonr(v[~v.review_id.isin(ids_rel)].ref, v[~v.review_id.isin(ids_rel)].score_texto)[0]
    r_v = 0.4147
    out(f"  r_t con todas ({len(v)} reseñas): {r_all:.4f} | sin las {int(v.review_id.isin(ids_rel).sum())} rellenadas en validación: {r_sin:.4f} | referencia v4 del entrenamiento: 0.8793")
    out(f"  r_v (imagen) = {r_v} (promedio de los 3 pliegues; NO se recalcula aquí: sale de los logs de CLIP)")
    out(f"  w con todas: {r_v / (r_all + r_v):.4f} | w sin rellenadas: {r_v / (r_sin + r_v):.4f}")
    filas.append({"conjunto": "Validación", "variante": "r_t todas / sin rellenadas", "MAE": None, "Pearson": f"{r_all:.4f} / {r_sin:.4f}", "n": len(v)})

# 3. CLIP fuera de muestra
pc = ultima("resultados_clip_velocimetro_v2_*/puntajes_imagen_oof.csv")
out("\n3. CLIP v4 fuera de muestra (todas las imágenes, validación cruzada por negocio)")
if pc is None:
    out("  NO DISPONIBLE: no encontré puntajes_imagen_oof.csv.")
else:
    c = pd.read_csv(pc)
    fila("CLIP, todas las imágenes del corpus", "todas las imágenes", mae_r(c.score, c.score_imagen_oof))
    m = ~c.review_id.isin(ids_rel)
    fila("CLIP, todas las imágenes del corpus", f"sin las imágenes de {int(c.review_id.isin(ids_rel).groupby(c.review_id).any().sum())} reseñas rellenadas conocidas", mae_r(c[m].score, c[m].score_imagen_oof))
    mm = c[c.test_multimodal == True] if "test_multimodal" in c else c.iloc[0:0]
    if len(mm):
        fila("CLIP, solo imágenes del test multimodal", "todas las imágenes", mae_r(mm.score, mm.score_imagen_oof))
        k = ~mm.review_id.isin(ids_rel)
        fila("CLIP, solo imágenes del test multimodal", "sin rellenadas conocidas", mae_r(mm[k].score, mm[k].score_imagen_oof))

out("\nQué NO cambia con el relleno: los F1 y MCC de clasificación (polaridad y 3 clases) usan las etiquetas adjudicadas, no las estrellas. "
    "Los clasificadores de RoBERTuito no usan estrellas; el velocímetro y CLIP sí, pero entrenaron con 3 y 3 reseñas rellenadas conocidas (0.06 % y 0.3 %).")
out("Límite: todo es COTA INFERIOR (solo las rellenadas detectables comparando versiones del dataset). No se reentrenó nada (5c).")

pd.DataFrame(filas).to_csv(OUT / "sensibilidad_relleno.csv", index=False)
(OUT / "RESULTADOS_paper_v15_ronda2_sensibilidad.txt").write_text("\n".join(TXT) + "\n", encoding="utf-8")
shutil.copy(__file__, OUT / Path(__file__).name)
z = V4 / "resultados_v15_r2.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            zf.write(p, f"resultados_v15_r2/{p.relative_to(OUT)}")
print(f"\nListo: {z}")
if sys.platform == "darwin":
    subprocess.run(["open", "-R", str(z)])
