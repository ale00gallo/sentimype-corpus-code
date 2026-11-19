import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "entrenamiento"))
import os, sys, glob, shutil, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

OUT_BASE = os.path.join("sentimype_experimentos", "estadistica", "clip_semillas")
os.makedirs(OUT_BASE, exist_ok=True)

s42_dirs = sorted(glob.glob("resultados_clip_velocimetro_v2_2026-11-08*"))
if s42_dirs and not os.path.exists(os.path.join(OUT_BASE, "seed_42")):
    shutil.copytree(s42_dirs[-1], os.path.join(OUT_BASE, "seed_42"))
    print("Copiada semilla 42 existente desde:", s42_dirs[-1])

import entrenar_clip_velocimetro as mod
orig_armar = mod.armar_folds
orig_partir = mod.partir_val

for seed in (43, 44):
    dest = os.path.join(OUT_BASE, f"seed_{seed}")
    if os.path.exists(os.path.join(dest, "puntajes_imagen_oof.csv")):
        print(f"Semilla {seed} ya existe, saltando.")
        continue
    print(f"\n==================== INICIANDO CLIP SEMILLA {seed} (folds fijos en 42) ====================")
    antes = set(glob.glob("resultados_clip_velocimetro_v2_*"))
    llamadas_partir = [0]

    def armar_fijo(df, n_folds=3):
        mod.HIPER["seed"] = 42
        res = orig_armar(df, n_folds)
        mod.HIPER["seed"] = seed
        return res

    def partir_fijo(df):
        llamadas_partir[0] += 1
        if llamadas_partir[0] > 3:
            raise StopIteration("FIN_FOLDS")
        mod.HIPER["seed"] = 42
        res = orig_partir(df)
        mod.HIPER["seed"] = seed
        return res

    mod.HIPER["seed"] = seed
    mod.armar_folds = armar_fijo
    mod.partir_val = partir_fijo

    try:
        mod.main()
    except StopIteration as e:
        if str(e) != "FIN_FOLDS":
            raise

    nuevas = sorted(set(glob.glob("resultados_clip_velocimetro_v2_*")) - antes)
    if nuevas:
        shutil.move(nuevas[-1], dest)
        print(f"Semilla {seed} guardada en {dest}")

DATOS = os.path.join("sentimype_experimentos", "exportar_resultados", "datos")
tb = pd.read_csv(os.path.join(DATOS, "test_b.csv"), dtype={"review_id": str})
tb["etq"] = tb["etiqueta"].str.strip().str.lower()
tb = tb[tb["etq"].isin(["negativo", "neutro", "positivo"]) & tb["s_v"].notna()].reset_index(drop=True)

def conf(y, p, k): return np.bincount(y * k + p, minlength=k * k).reshape(k, k)
def f1m(C):
    tp = np.diag(C).astype(float); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    pres = (C.sum(0) + C.sum(1)) > 0; den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)[pres].mean())

filas = []
for seed in (42, 43, 44):
    p_oof = os.path.join(OUT_BASE, f"seed_{seed}", "puntajes_imagen_oof.csv")
    if not os.path.exists(p_oof): continue
    oof = pd.read_csv(p_oof, dtype={"review_id": str})
    oof_1 = oof.drop_duplicates(subset=["review_id"], keep="first").set_index("review_id")
    m = tb.copy()
    m["sv_seed"] = m["review_id"].map(oof_1["score_imagen_oof"]).astype(float)
    m = m[m["sv_seed"].notna()].copy()

    mae_oof = float(np.abs(oof["score"] - oof["score_imagen_oof"]).mean())
    r_oof = float(pearsonr(oof["score"], oof["score_imagen_oof"])[0])
    mp = m[m["etq"].isin(["negativo", "positivo"])]
    yp = (mp["etq"] == "positivo").astype(int).values
    y3 = m["etq"].map({"negativo":0,"neutro":1,"positivo":2}).values

    f1_img_pol = f1m(conf(yp, (mp["sv_seed"].values >= 5.0).astype(int), 2))
    f1_img_3c = f1m(conf(y3, np.where(m["sv_seed"].values < 3.2, 0, np.where(m["sv_seed"].values < 7.3, 1, 2)), 3))
    sf10_p = 0.9 * mp["s_t"].astype(float).values + 0.1 * mp["sv_seed"].values
    sf10_3 = 0.9 * m["s_t"].astype(float).values + 0.1 * m["sv_seed"].values
    f1_w10_pol = f1m(conf(yp, (sf10_p >= 5.0).astype(int), 2))
    f1_w10_3c = f1m(conf(y3, np.where(sf10_3 < 3.2, 0, np.where(sf10_3 < 7.3, 1, 2)), 3))
    sf32_p = 0.68 * mp["s_t"].astype(float).values + 0.32 * mp["sv_seed"].values
    sf32_3 = 0.68 * m["s_t"].astype(float).values + 0.32 * m["sv_seed"].values
    f1_w32_pol = f1m(conf(yp, (sf32_p >= 5.0).astype(int), 2))
    f1_w32_3c = f1m(conf(y3, np.where(sf32_3 < 3.2, 0, np.where(sf32_3 < 7.3, 1, 2)), 3))

    filas.append({
        "semilla": seed, "mae_oof": mae_oof, "pearson_oof": r_oof,
        "f1_img_pol": f1_img_pol, "f1_img_3c": f1_img_3c,
        "f1_w10_pol": f1_w10_pol, "f1_w10_3c": f1_w10_3c,
        "f1_w32_pol": f1_w32_pol, "f1_w32_3c": f1_w32_3c
    })

df_s = pd.DataFrame(filas)
df_s.round(4).to_csv(os.path.join(OUT_BASE, "clip_3_semillas_detalle.csv"), index=False)
resumen = pd.DataFrame({
    "metrica": [c for c in df_s.columns if c != "semilla"],
    "media": [df_s[c].mean() for c in df_s.columns if c != "semilla"],
    "de": [df_s[c].std(ddof=1) for c in df_s.columns if c != "semilla"],
})
resumen["media_pm_de"] = resumen.apply(lambda r: f"{r['media']:.3f} ± {r['de']:.3f}", axis=1)
resumen.round(4).to_csv(os.path.join(OUT_BASE, "clip_3_semillas_resumen.csv"), index=False)
print("\nRESUMEN 3 SEMILLAS DE CLIP:")
print(resumen[["metrica", "media_pm_de"]].to_string(index=False))
