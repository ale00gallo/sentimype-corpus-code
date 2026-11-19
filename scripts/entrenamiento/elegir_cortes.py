"""Elige los cortes del velocímetro (negativo < c1 <= neutro < c2 <= positivo) SOLO con la validación v4
y después los evalúa una sola vez en el test humano, junto a los cortes fijos 4.5 / 7.

Antes: python puntuar_validacion.py   (crea resultados_cortes_<fecha>/puntajes_texto_validacion.csv con el velocímetro v4)
Uso, en "entrenamiento v4":  python elegir_cortes.py
Guarda figuras/V4_cortes_velocimetro.csv y V4_cortes_top10.csv.
"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, matthews_corrcoef

FIJOS = (4.5, 7.0)
C = ["negativo", "neutro", "positivo"]


def ultima(p):
    c = sorted(glob.glob(p))
    return c[-1] if c else None


def clasificar(s, c1, c2):
    return np.where(s < c1, "negativo", np.where(s < c2, "neutro", "positivo"))


def mejores(y, s):
    filas = []
    for c1 in np.round(np.arange(3.0, 6.51, 0.1), 1):
        for c2 in np.round(np.arange(c1 + 0.5, 8.51, 0.1), 1):
            p = clasificar(s, c1, c2)
            filas.append((c1, c2, f1_score(y, p, average="macro"), matthews_corrcoef(y, p)))
    t = pd.DataFrame(filas, columns=["c1", "c2", "F1 macro val", "MCC val"])
    t["distancia a 4.5/7"] = np.hypot(t.c1 - FIJOS[0], t.c2 - FIJOS[1])
    return t.sort_values(["F1 macro val", "distancia a 4.5/7"], ascending=[False, True]).reset_index(drop=True)


vel = ultima("resultados_roberta_v2_*")
val_csv = ultima("resultados_cortes_*/puntajes_texto_validacion.csv")
if not val_csv or os.path.getmtime(val_csv) < os.path.getmtime(vel):
    raise SystemExit("ALTO: falta puntuar la validación con el velocímetro v4: corre antes puntuar_validacion.py")
va = pd.read_csv(val_csv)
va = va[va["sentiment_label"].isin(C)]
te = pd.read_csv(os.path.join(vel, "predicciones_test.csv"))
te = te[te["sentiment_label"].isin(C)]
col_s = "score_predicho" if "score_predicho" in te else "score"
print(f"Validación: {len(va)} reseñas {va.sentiment_label.value_counts().to_dict()} | origen: {va.label_source.value_counts().to_dict()}")
print(f"Test humano: {len(te)} reseñas (velocímetro de {vel})\n")

filas, top = [], []
for nombre, sub in (("toda la validación", va), ("validación sin etiquetas por estrellas", va[va.label_source != "rating"])):
    if sub.sentiment_label.nunique() < 3:
        continue
    t = mejores(sub.sentiment_label.values, sub.score_texto.values)
    top.append(t.head(10).assign(conjunto=nombre))
    c1, c2 = t.loc[0, "c1"], t.loc[0, "c2"]
    for etiqueta, (a, b) in (("fijos 4.5 / 7", FIJOS), (f"elegidos {c1} / {c2}", (c1, c2))):
        pv = clasificar(sub.score_texto.values, a, b)
        pt = clasificar(te[col_s].values, a, b)
        rep = pd.Series(pt).eq("neutro")
        filas.append({"Cortes elegidos con": nombre, "Regla": etiqueta, "n val": len(sub),
                      "F1 val": f1_score(sub.sentiment_label, pv, average="macro"),
                      "F1 test": f1_score(te.sentiment_label, pt, average="macro"), "MCC test": matthews_corrcoef(te.sentiment_label, pt),
                      "F1 neutro test": f1_score(te.sentiment_label, pt, labels=["neutro"], average="macro"),
                      "Predice neutro (test)": int(rep.sum())})
    # McNemar fijos vs elegidos en el test
    p_f, p_e = clasificar(te[col_s].values, *FIJOS), clasificar(te[col_s].values, c1, c2)
    ok_f, ok_e = p_f == te.sentiment_label.values, p_e == te.sentiment_label.values
    b, c = int((ok_f & ~ok_e).sum()), int((~ok_f & ok_e).sum())
    filas[-1]["p McNemar vs fijos"] = binomtest(c, b + c).pvalue if b + c else 1.0
    filas[-1]["Solo aciertan fijos / elegidos"] = f"{b} / {c}"

res = pd.DataFrame(filas)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
os.makedirs("figuras", exist_ok=True)
res.round(4).to_csv("figuras/V4_cortes_velocimetro.csv", index=False)
pd.concat(top).round(4).to_csv("figuras/V4_cortes_top10.csv", index=False)
print("\nGuardado: figuras/V4_cortes_velocimetro.csv y V4_cortes_top10.csv")
print("Los cortes se eligen SOLO con la validación; el test se mira una sola vez para reportar.")
