import os
import numpy as np
import pandas as pd

DATOS = os.path.join("sentimype_experimentos", "subir_a_drive", "datos")
OUT = os.path.join("sentimype_experimentos", "estadistica")
os.makedirs(OUT, exist_ok=True)
SEMILLA, B = 42, 2000

def conf(y, p, k): return np.bincount(y * k + p, minlength=k * k).reshape(k, k)
def f1m(C):
    tp = np.diag(C).astype(float); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    pres = (C.sum(0) + C.sum(1)) > 0; den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)[pres].mean())

def tost_bootstrap(y, p_nuevo, p_base, k, rng):
    n = len(y)
    difs = np.array([f1m(conf(y[i], p_nuevo[i], k)) - f1m(conf(y[i], p_base[i], k)) for i in (rng.integers(0, n, n) for _ in range(B))])
    punto = f1m(conf(y, p_nuevo, k)) - f1m(conf(y, p_base, k))
    ic90_l, ic90_u = np.percentile(difs, 5), np.percentile(difs, 95)
    ic95_l, ic95_u = np.percentile(difs, 2.5), np.percentile(difs, 97.5)
    return {
        "n": n, "dif_f1": punto,
        "ic95_inf": ic95_l, "ic95_sup": ic95_u,
        "ic90_inf": ic90_l, "ic90_sup": ic90_u,
        "equiv_delta_0.02": bool(ic90_l >= -0.02 and ic90_u <= 0.02),
        "equiv_delta_0.05": bool(ic90_l >= -0.05 and ic90_u <= 0.05),
    }

tb = pd.read_csv(os.path.join(DATOS, "test_b.csv"), dtype={"review_id": str})
tb["etq"] = tb["etiqueta"].str.strip().str.lower()
tb = tb[tb["etq"].isin(["negativo", "neutro", "positivo"]) & tb["s_v"].notna()].reset_index(drop=True)
rng = np.random.default_rng(SEMILLA)
filas = []

# 1. Fusión tardía w=0.10 y w=0.32 vs solo texto
for tarea, k in [("polaridad", 2), ("3 clases", 3)]:
    sub = tb[tb["etq"].isin(["negativo", "positivo"])].copy() if tarea == "polaridad" else tb.copy()
    y = (sub["etq"] == "positivo").astype(int).values if tarea == "polaridad" else sub["etq"].map({"negativo":0,"neutro":1,"positivo":2}).values
    st, sv = sub["s_t"].astype(float).values, sub["s_v"].astype(float).values
    pt = (st >= 5.0).astype(int) if tarea == "polaridad" else np.where(st < 3.2, 0, np.where(st < 7.3, 1, 2)).astype(int)
    for w in (0.10, 0.32):
        sf = (1 - w) * st + w * sv
        pf = (sf >= 5.0).astype(int) if tarea == "polaridad" else np.where(sf < 3.2, 0, np.where(sf < 7.3, 1, 2)).astype(int)
        r = tost_bootstrap(y, pf, pt, k, rng); r.update({"comparacion": f"Fusión w={w:.2f} vs solo texto", "tarea": tarea}); filas.append(r)

# 2. E3: LR-concat vs LR-texto
p_e3 = os.path.join("sentimype_experimentos", "e3", "e3_predicciones_test_b.csv")
if os.path.exists(p_e3):
    e3 = pd.read_csv(p_e3, dtype={"review_id": str})
    for tarea, k, col_cat, col_txt in [
        ("polaridad", 2, "pred_polaridad_LR_concat", "pred_polaridad_LR_texto"),
        ("3 clases", 3, "pred_3_clases_LR_concat", "pred_3_clases_LR_texto"),
    ]:
        sub = e3[e3[col_cat].notna()].copy()
        y = (sub["etq"] == "positivo").astype(int).values if tarea == "polaridad" else sub["etq"].map({"negativo":0,"neutro":1,"positivo":2}).values
        r = tost_bootstrap(y, sub[col_cat].astype(int).values, sub[col_txt].astype(int).values, k, rng)
        r.update({"comparacion": "E3 LR-concat vs LR-texto", "tarea": tarea}); filas.append(r)

# 3. E1: E1-c (texto+foto) vs E1-b (solo texto)
c_e1 = os.path.join("sentimype_experimentos", "e1")
if os.path.exists(os.path.join(c_e1, "llamadas_E1-b.csv")) and os.path.exists(os.path.join(c_e1, "llamadas_E1-c.csv")):
    b = pd.read_csv(os.path.join(c_e1, "llamadas_E1-b.csv"), dtype={"review_id": str})
    c = pd.read_csv(os.path.join(c_e1, "llamadas_E1-c.csv"), dtype={"review_id": str})
    m = b.merge(c, on="review_id", suffixes=("_b", "_c"))
    for tarea, k in [("polaridad", 2), ("3 clases", 3)]:
        if tarea == "polaridad":
            mm = m[m["etiqueta_real_b"].isin(["negativo", "positivo"])].copy()
            y = (mm["etiqueta_real_b"] == "positivo").astype(int).values
            pb = (pd.to_numeric(mm["pred_puntaje_b"], errors="coerce").fillna(0) >= 5.0).astype(int).values
            pc = (pd.to_numeric(mm["pred_puntaje_c"], errors="coerce").fillna(0) >= 5.0).astype(int).values
        else:
            y = m["etiqueta_real_b"].map({"negativo":0,"neutro":1,"positivo":2}).values
            pb = m["pred_etiqueta_b"].map({"negativo":0,"neutro":1,"positivo":2}).fillna((pd.Series(y)+1)%3).astype(int).values
            pc = m["pred_etiqueta_c"].map({"negativo":0,"neutro":1,"positivo":2}).fillna((pd.Series(y)+1)%3).astype(int).values
        r = tost_bootstrap(y, pc, pb, k, rng); r.update({"comparacion": "E1-c (texto+foto) vs E1-b (solo texto)", "tarea": tarea}); filas.append(r)

df_tost = pd.DataFrame(filas)
df_tost.round(4).to_csv(os.path.join(OUT, "tost_equivalencia.csv"), index=False, encoding="utf-8")
print(df_tost.round(4).to_string(index=False))
