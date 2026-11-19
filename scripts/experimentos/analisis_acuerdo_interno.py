import os, glob, zipfile, subprocess
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef

CLASES = ("negativo", "neutro", "positivo")
MAP = {c: i for i, c in enumerate(CLASES)}
SEMILLA, B = 42, 2000

BASE = "sentimype_experimentos"
OUT_DIR = os.path.join(BASE, "acuerdo_ciego")
FIG_DIR = os.path.join(BASE, "figuras")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def boot_ci(fn, arrays, B=B, seed=SEMILLA):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = fn(*arrays)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*(a[idx] for a in arrays))
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            pass
    return float(est), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def calc_alpha(M):
    return float(krippendorff.alpha(reliability_data=M.T, level_of_measurement="nominal"))

def calc_fleiss(M):
    tab, _ = aggregate_raters(M, n_cat=3)
    return float(fleiss_kappa(tab, method="fleiss"))

def f1_macro_2(y, p): return float(f1_score(y, p, average="macro", zero_division=0))
def mcc_val(y, p): return float(matthews_corrcoef(y, p))

def encontrar_col_llm(df):
    for cand in ["etiqueta_llm", "etiqueta_modelo", "pred_llm", "etiqueta_original_llm", "etiqueta_preasignada", "llm_etiqueta"]:
        if cand in df.columns:
            return cand
    return None

def procesar_grupo(nombre, codigo_grp, ruta_clave, patron_resp, ruta_datos_test, es_grupo_a=False):
    clave = pd.read_csv(ruta_clave, dtype=str)
    clave["nuestra_etiqueta"] = clave["etiqueta_adjudicada"].str.strip().str.lower()

    datos = pd.read_csv(ruta_datos_test, dtype={"review_id": str})
    col_llm = encontrar_col_llm(clave) or encontrar_col_llm(datos)
    cols_merge = ["review_id", "s_t"] + (["s_v"] if "s_v" in datos.columns else [])
    if col_llm and col_llm not in clave.columns:
        cols_merge.append(col_llm)

    df = clave[["review_id", "nuestra_etiqueta"] + ([col_llm] if col_llm in clave.columns else [])].merge(
        datos[cols_merge].drop_duplicates("review_id"), on="review_id", how="left"
    )
    if col_llm:
        df["etiqueta_llm"] = df[col_llm].str.strip().str.lower()

    archivos = sorted(glob.glob(patron_resp))
    cols_ext = []
    for f in archivos:
        r = pd.read_csv(f, dtype=str)
        col_e = [c for c in r.columns if "etiqueta" in c][0]
        r["lbl"] = r[col_e].str.strip().str.lower()
        r = r[r["lbl"].isin(CLASES)].drop_duplicates("review_id")
        if len(r) < 100:
            print(f"[{nombre}] Descartando {os.path.basename(f)} (solo completó {len(r)}/100)")
            continue
        nom = "anot_" + os.path.basename(f).replace("respuestas_ciegas_", "").replace("GrupoB_TestA_", "").replace(".csv", "")
        df = df.merge(r[["review_id", "lbl"]].rename(columns={"lbl": nom}), on="review_id", how="left")
        cols_ext.append(nom)

    n_anot = len(cols_ext)
    n_res = len(df)
    print(f"\n==================== {nombre}: {n_anot} anotadoras completas, {n_res} reseñas ====================")

    M = np.column_stack([df[c].map(MAP).values for c in cols_ext])
    alpha_est, alpha_l, alpha_u = boot_ci(calc_alpha, [M])
    fleiss_est, fleiss_l, fleiss_u = boot_ci(calc_fleiss, [M])
    print(f"1) Alfa de Krippendorff: {alpha_est:.3f} [IC95%: {alpha_l:.3f}, {alpha_u:.3f}]")
    print(f"   Kappa de Fleiss:      {fleiss_est:.3f} [IC95%: {fleiss_l:.3f}, {fleiss_u:.3f}]")

    mayoria = []
    for _, row in df[cols_ext].iterrows():
        cnt = Counter(row.values)
        top_lbl, top_n = cnt.most_common(1)[0]
        mayoria.append(top_lbl if top_n >= 3 else np.nan)
    df["mayoria_externa"] = mayoria
    n_empates = int(df["mayoria_externa"].isna().sum())
    n_valid_may = n_res - n_empates
    print(f"2) Mayoría externa (>=3 votos): {n_valid_may} válidas | {n_empates} empates sin etiqueta")

    sub_m = df[df["mayoria_externa"].notna()].copy()
    yn = sub_m["nuestra_etiqueta"].map(MAP).values
    ym = sub_m["mayoria_externa"].map(MAP).values
    kn_est, kn_l, kn_u = boot_ci(cohen_kappa_score, [yn, ym])
    print(f"3) Kappa nuestra vs mayoría externa (n={len(sub_m)}): {kn_est:.3f} [IC95%: {kn_l:.3f}, {kn_u:.3f}]")

    kllm_str = "N/A"
    if es_grupo_a and "etiqueta_llm" in sub_m.columns and sub_m["etiqueta_llm"].isin(CLASES).sum() > 0:
        sub_l = sub_m[sub_m["etiqueta_llm"].isin(CLASES)]
        kl_est, kl_l, kl_u = boot_ci(cohen_kappa_score, [sub_l["etiqueta_llm"].map(MAP).values, sub_l["mayoria_externa"].map(MAP).values])
        kllm_str = f"{kl_est:.3f} [{kl_l:.3f}, {kl_u:.3f}]"
        print(f"   Kappa LLM vs mayoría externa (n={len(sub_l)}):     {kllm_str}")
    elif es_grupo_a:
        print("   AVISO: No se encontró columna de etiqueta original del LLM en los CSVs.")

    k_ind = {}
    for c in cols_ext:
        ki = cohen_kappa_score(df["nuestra_etiqueta"].map(MAP).values, df[c].map(MAP).values)
        k_ind[c] = ki
    vals_k = list(k_ind.values())
    print(f"   Kappa nuestra vs cada anotadora: prom={np.mean(vals_k):.3f} | min={np.min(vals_k):.3f} | max={np.max(vals_k):.3f}")
    for c, v in k_ind.items():
        print(f"     - {c}: {v:.3f}")

    print("4) Desempeño del modelo (F1-Macro y MCC con IC 95%):")
    configs_score = [("Texto solo (w=0)", df["s_t"].astype(float).values)]
    if es_grupo_a and "s_v" in df.columns:
        configs_score.append(("Fusión (w=0.10)", 0.9 * df["s_t"].astype(float).values + 0.1 * df["s_v"].astype(float).values))

    res_modelos = []
    for nom_cfg, sc in configs_score:
        df[f"pred_pol_{nom_cfg}"] = np.where(sc >= 5.0, "positivo", "negativo")
        df[f"pred_3c_{nom_cfg}"]  = np.where(sc < 3.2, "negativo", np.where(sc < 7.3, "neutro", "positivo"))

        for ref_col, ref_nom in [("nuestra_etiqueta", "a) Nuestras etiquetas"), ("mayoria_externa", "b) Mayoría externa")]:
            s_ref = df[df[ref_col].notna()].copy()
            sc_sub = sc[df[ref_col].notna()]
            y3 = s_ref[ref_col].map(MAP).values
            p3 = np.where(sc_sub < 3.2, 0, np.where(sc_sub < 7.3, 1, 2))
            f1_3, f1_3l, f1_3u = boot_ci(f1_macro_2, [y3, p3])
            mc_3, mc_3l, mc_3u = boot_ci(mcc_val, [y3, p3])
            s_pol = s_ref[s_ref[ref_col].isin(["negativo", "positivo"])]
            sc_pol = sc_sub[s_ref[ref_col].isin(["negativo", "positivo"])]
            yp = (s_pol[ref_col] == "positivo").astype(int).values
            pp = (sc_pol >= 5.0).astype(int)
            f1_p, f1_pl, f1_pu = boot_ci(f1_macro_2, [yp, pp])
            mc_p, mc_pl, mc_pu = boot_ci(mcc_val, [yp, pp])
            print(f"   {nom_cfg} | {ref_nom} (n_3c={len(s_ref)}, n_pol={len(s_pol)}):")
            print(f"     · Polaridad: F1={f1_p:.3f} [{f1_pl:.3f}, {f1_pu:.3f}] | MCC={mc_p:.3f} [{mc_pl:.3f}, {mc_pu:.3f}]")
            print(f"     · 3 clases:  F1={f1_3:.3f} [{f1_3l:.3f}, {f1_3u:.3f}] | MCC={mc_3:.3f} [{mc_3l:.3f}, {mc_3u:.3f}]")
            res_modelos.append({
                "grupo": nombre, "modelo": nom_cfg, "referencia": ref_nom,
                "n_pol": len(s_pol), "f1_pol": f"{f1_p:.3f} [{f1_pl:.3f}, {f1_pu:.3f}]", "mcc_pol": f"{mc_p:.3f} [{mc_pl:.3f}, {mc_pu:.3f}]",
                "n_3c": len(s_ref), "f1_3c": f"{f1_3:.3f} [{f1_3l:.3f}, {f1_3u:.3f}]", "mcc_3c": f"{mc_3:.3f} [{mc_3l:.3f}, {mc_3u:.3f}]",
                "f1_3c_num": f1_3, "f1_pol_num": f1_p
            })

    df.to_csv(os.path.join(OUT_DIR, f"etiquetas_{codigo_grp}.csv"), index=False, encoding="utf-8")

    r_nuestra_3c = [x for x in res_modelos if x["modelo"] == "Texto solo (w=0)" and x["referencia"].startswith("a)")][0]
    r_ext_3c     = [x for x in res_modelos if x["modelo"] == "Texto solo (w=0)" and x["referencia"].startswith("b)")][0]

    fila_resumen = {
        "grupo": nombre,
        "n_anotadoras": n_anot,
        "n_resenas": n_res,
        "empates": n_empates,
        "alfa_krippendorff_ic95": f"{alpha_est:.3f} [{alpha_l:.3f}, {alpha_u:.3f}]",
        "kappa_fleiss_ic95": f"{fleiss_est:.3f} [{fleiss_l:.3f}, {fleiss_u:.3f}]",
        "kappa_nuestra_vs_mayoria_ic95": f"{kn_est:.3f} [{kn_l:.3f}, {kn_u:.3f}]",
        "kappa_llm_vs_mayoria_ic95": kllm_str,
        "kappa_nuestra_vs_indiv_prom_min_max": f"{np.mean(vals_k):.3f} ({np.min(vals_k):.3f}–{np.max(vals_k):.3f})",
        "f1_pol_nuestra_vs_externa": f"{r_nuestra_3c['f1_pol']} vs {r_ext_3c['f1_pol']}",
        "f1_3c_nuestra_vs_externa": f"{r_nuestra_3c['f1_3c']} vs {r_ext_3c['f1_3c']}",
    }
    return fila_resumen, pd.DataFrame(res_modelos), sub_m

res_a, mod_a, sub_a = procesar_grupo(
    "Grupo A (Test-B con foto)", "grupo_A_testB",
    "etiquetado_ciego_100/CLAVE_PRIVADA_NO_COMPARTIR.csv",
    "etiquetado_ciego_100/respuestas_*.csv",
    os.path.join(BASE, "subir_a_drive", "datos", "test_b.csv"),
    es_grupo_a=True
)

res_b, mod_b, sub_b = procesar_grupo(
    "Grupo B (Test-A solo texto)", "grupo_B_testA",
    "etiquetado_ciego_testA_100/CLAVE_PRIVADA_TEST_A_NO_COMPARTIR.csv",
    "etiquetado_ciego_testA_100/respuestas_*.csv",
    os.path.join(BASE, "subir_a_drive", "datos", "test_a.csv"),
    es_grupo_a=False
)

df_resumen = pd.DataFrame([res_a, res_b])
df_resumen.to_csv(os.path.join(OUT_DIR, "tabla_resumen_acuerdo.csv"), index=False, encoding="utf-8")
pd.concat([mod_a, mod_b], ignore_index=True).to_csv(os.path.join(OUT_DIR, "detalle_modelo_vs_externa.csv"), index=False, encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
etq_tit = ["Negativo", "Neutro", "Positivo"]

for ax, (sub, tit) in zip(axes, [
    (sub_a, f"Grupo A: Test-B con foto (n = {len(sub_a)})"),
    (sub_b, f"Grupo B: Test-A solo texto (n = {len(sub_b)})"),
]):
    yt = sub["nuestra_etiqueta"].map(MAP).values
    yp = sub["mayoria_externa"].map(MAP).values
    cm = np.bincount(yt * 3 + yp, minlength=9).reshape(3, 3)
    im = ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max() * 1.15)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(etq_tit)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(etq_tit)
    ax.set_xlabel("Mayoría externa (≥ 3 anotadoras)")
    ax.set_ylabel("Nuestra etiqueta adjudicada")
    ax.set_title(tit, fontweight="bold", pad=10)
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            col_txt = "white" if val > cm.max() * 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=col_txt, fontsize=12, fontweight="bold")

fig.suptitle("Concordancia por clase: etiqueta adjudicada vs. mayoría externa", fontsize=11.5, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_matriz_confusion_externa.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig3_matriz_confusion_externa.pdf"), bbox_inches="tight")
plt.close(fig)

ruta_zip = os.path.abspath("SentiMYPE_entregables_experimentos.zip")
with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint") and d != "best_model"]
        for fn in files:
            if fn.endswith((".pt", ".bin", ".safetensors")): continue
            fp = os.path.join(root, fn)
            z.write(fp, fp)
    for scr in ["run_e5.py", "run_e3.py", "run_tost.py", "run_clip_semillas.py", "analisis_acuerdo_interno.py", os.path.expanduser("python run_e1.py")]:
        if os.path.exists(scr):
            z.write(scr, f"{BASE}/scripts/{os.path.basename(scr)}")

print("\nListo. Tabla resumen guardada en:", os.path.join(OUT_DIR, "tabla_resumen_acuerdo.csv"))
print("Gráfico guardado en:", os.path.join(FIG_DIR, "fig3_matriz_confusion_externa.png"))
subprocess.run(["open", os.path.join(FIG_DIR, "fig3_matriz_confusion_externa.png")])
