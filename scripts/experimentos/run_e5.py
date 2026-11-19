import os, json, sys, datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

RAIZ = os.path.join("sentimype_experimentos", "e5")
DATOS = os.path.join("sentimype_experimentos", "subir_a_drive", "datos")
os.makedirs(RAIZ, exist_ok=True)
SEMILLA, B = 42, 2000

tb = pd.read_csv(os.path.join(DATOS, "test_b.csv"), dtype={"review_id": str})
if "raw_text" not in tb.columns and os.path.exists(os.path.join(DATOS, "test_multimodal.csv")):
    orig = pd.read_csv(os.path.join(DATOS, "test_multimodal.csv"), dtype={"review_id": str})
    tb = tb.merge(orig[["review_id", "raw_text", "source_platform"]], on="review_id", how="left", suffixes=("", "_orig"))

col_txt = "raw_text" if "raw_text" in tb.columns else "texto"
col_plat = "source_platform" if "source_platform" in tb.columns else "plataforma"
col_etq = "etiqueta" if "etiqueta" in tb.columns else "sentiment_label"

tb["etq"] = tb[col_etq].str.strip().str.lower()
tb = tb[tb["etq"].isin(["negativo", "neutro", "positivo"]) & tb["s_v"].notna()].copy()
tb["s_t"] = tb["s_t"].astype(float)
tb["s_v"] = tb["s_v"].astype(float)
tb["txt"] = tb[col_txt].fillna("").astype(str).str.strip()
tb["n_palabras"] = tb["txt"].apply(lambda x: 0 if x == "" else len(x.split()))

def grp_len(n):
    if n == 0: return "0 palabras (vacío)"
    if n <= 5: return "1 a 5 palabras"
    if n <= 20: return "6 a 20 palabras"
    return "Más de 20 palabras"

tb["grp_len"] = tb["n_palabras"].map(grp_len)
tb["grp_amb"] = np.where((tb["s_t"] >= 3.2) & (tb["s_t"] < 7.3), "Ambiguo (3.2 <= s_t < 7.3)", "No ambiguo")

# Conteo de reseñas con foto y sin texto en todo el corpus
foto_sin_txt = {"total_corpus": 0, "con_etiqueta": 0}
for ruta_corp in ("SentiMYPE_v3_completo.csv", "Dataset_SentiMYPE_FINAL.csv"):
    if os.path.exists(ruta_corp):
        corp = pd.read_csv(ruta_corp, dtype=str)
        ct = "raw_text" if "raw_text" in corp.columns else "clean_text"
        sin_t = corp[ct].fillna("").str.strip() == ""
        if os.path.exists("imagenes_index.csv"):
            con_f = corp["review_id"].isin(set(pd.read_csv("imagenes_index.csv", dtype=str)["review_id"]))
        else:
            con_f = corp.get("image_path", pd.Series("", index=corp.index)).fillna("") != ""
        sub_cf = corp[sin_t & con_f]
        foto_sin_txt["archivo_usado"] = ruta_corp
        foto_sin_txt["total_corpus"] = int(len(sub_cf))
        foto_sin_txt["con_etiqueta"] = int(sub_cf["sentiment_label"].fillna("").str.lower().isin(["negativo", "neutro", "positivo"]).sum())
        break

def conf(y, p, k): return np.bincount(y * k + p, minlength=k * k).reshape(k, k)
def f1m(C):
    tp = np.diag(C).astype(float); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    pres = (C.sum(0) + C.sum(1)) > 0; den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)[pres].mean())
def mccf(C):
    t = C.sum(1).astype(float); p = C.sum(0).astype(float); c = float(np.trace(C)); s = float(C.sum())
    den = np.sqrt((s ** 2 - (p * p).sum()) * (s ** 2 - (t * t).sum()))
    return float((c * s - (t * p).sum()) / den) if den > 0 else 0.0

def pred_clase(s, tarea):
    return (s >= 5.0).astype(int) if tarea == "polaridad" else np.where(s < 3.2, 0, np.where(s < 7.3, 1, 2)).astype(int)

rng = np.random.default_rng(SEMILLA)
filas = []
subgrupos = [("General Test-B", "Todos", tb)]
for col, nom in [("grp_len", "Longitud"), (col_plat, "Plataforma"), ("grp_amb", "Ambigüedad texto")]:
    for val, g in tb.groupby(col):
        subgrupos.append((nom, str(val), g))

for cat, sub, g in subgrupos:
    for tarea, k in [("polaridad", 2), ("3 clases", 3)]:
        gg = g[g["etq"].isin(["negativo", "positivo"])].copy() if tarea == "polaridad" else g.copy()
        n = len(gg)
        if n == 0: continue
        y = (gg["etq"] == "positivo").astype(int).values if tarea == "polaridad" else gg["etq"].map({"negativo":0,"neutro":1,"positivo":2}).values
        p_txt = pred_clase(gg["s_t"].values, tarea)
        C_t = conf(y, p_txt, k)
        f1_t, mcc_t = f1m(C_t), mccf(C_t)
        idx_boot = [rng.integers(0, n, n) for _ in range(B)]
        f1_t_b = np.array([f1m(conf(y[i], p_txt[i], k)) for i in idx_boot])

        for w in (0.05, 0.10, 0.32):
            s_fus = (1 - w) * gg["s_t"].values + w * gg["s_v"].values
            p_f = pred_clase(s_fus, tarea)
            C_f = conf(y, p_f, k)
            f1_f, mcc_f = f1m(C_f), mccf(C_f)
            f1_f_b = np.array([f1m(conf(y[i], p_f[i], k)) for i in idx_boot])
            dif_b = f1_f_b - f1_t_b
            c1, c2 = (p_f == y), (p_txt == y)
            b_, c_ = int((c2 & ~c1).sum()), int((~c2 & c1).sum())
            pval = 1.0 if (b_ + c_) == 0 else float(binomtest(min(b_, c_), b_ + c_, 0.5).pvalue)
            filas.append({
                "categoria": cat, "subgrupo": sub, "tarea": tarea, "n": n, "interpretable": n >= 30,
                "w_v": w, "f1_texto": f1_t, "f1_fus": f1_f, "mcc_texto": mcc_t, "mcc_fus": mcc_f,
                "f1_fus_ic95_inf": np.percentile(f1_f_b, 2.5), "f1_fus_ic95_sup": np.percentile(f1_f_b, 97.5),
                "dif_f1": f1_f - f1_t, "dif_ic95_inf": np.percentile(dif_b, 2.5), "dif_ic95_sup": np.percentile(dif_b, 97.5),
                "p_mcnemar": pval
            })

res = pd.DataFrame(filas)
res["p_holm"] = np.nan
for (tarea, w), idx in res[(res["interpretable"]) & (res["categoria"] != "General Test-B")].groupby(["tarea", "w_v"]).groups.items():
    pvals = res.loc[idx, "p_mcnemar"].values
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * pvals[order]))
    res.loc[idx[order], "p_holm"] = adj

res.round(4).to_csv(os.path.join(RAIZ, "e5_subgrupos.csv"), index=False, encoding="utf-8")
json.dump({"fecha": dt.datetime.now().isoformat(), "semilla": SEMILLA, "B": B, "foto_sin_texto_corpus": foto_sin_txt},
          open(os.path.join(RAIZ, "e5_meta.json"), "w"), indent=2)
print("Control rápido General Test-B (debe dar texto 0.982/0.777 y w=0.10 0.979/0.760):")
print(res[(res["categoria"]=="General Test-B") & (res["w_v"]==0.10)][["tarea","n","f1_texto","f1_fus"]].round(3).to_string(index=False))
print("\nFoto sin texto en corpus:", foto_sin_txt)
print("\nTabla E5 (w = 0.10 y 0.32):")
print(res[res["w_v"].isin([0.10, 0.32])][["categoria","subgrupo","tarea","n","f1_texto","f1_fus","dif_f1","dif_ic95_inf","dif_ic95_sup","p_mcnemar","p_holm"]].round(3).to_string(index=False))
