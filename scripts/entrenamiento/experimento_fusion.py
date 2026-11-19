"""Experimento de fusion tardia SentiMYPE (texto + imagen) sobre el test multimodal.

Adaptado de SentiMYPE_Texto_vs_Imagen.ipynb para correr local y con los modelos v2:
  - puntaje de texto: RoBERTuito velocimetro (models/roberta-velocimetro/best_model)
  - puntaje de imagen: CLIP velocimetro, puntaje FUERA DE MUESTRA (cross-fitting k=3):
    cada imagen se puntua con un modelo que no la vio al entrenar
    (resultados_clip_velocimetro_v2_*/puntajes_imagen_oof.csv)
  - fusion: promedio ponderado con pesos por Pearson r de VALIDACION de cada modelo
    (nunca del test, para no inflar el resultado)
Evalua sobre las 362 resenas etiquetadas del test multimodal, que ninguno de los dos
modelos vio al entrenar.

Uso, parado en la carpeta "entrenamiento v2" (despues de entrenar el velocimetro de
RoBERTa y el de CLIP):

    python experimento_fusion.py

Deja resultados_fusion_v2_<fecha>/ con:
    reporte_fusion.html            todo el experimento (se abre solo en el navegador)
    metricas_test.json             numeros principales
    metricas_texto_imagen_fusion.csv, metricas_fusion_adaptativa_prueba.csv,
    grid_fusion_adaptativa_ajuste.csv, puntajes_por_resena.csv, contradicciones.csv,
    scatter_texto_vs_imagen.png, quien_acierta.png
"""
import base64
import glob
import html
import io
import json
import os
import subprocess
import sys
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEMILLA = 42
# Pesos que usa hoy la app (app/ml_models.py), como referencia
PESOS_APP = (0.7843, 0.2115)
COLORES = {"positivo": "#2F9BE6", "neutro": "#F3AE2B", "negativo": "#E8613F"}


def categoria(s):
    """Mismos cortes que la app (lib/format.js y reporte PDF)."""
    return "positivo" if s >= 7 else ("neutro" if s >= 4.5 else "negativo")


def polaridad(s):
    return "positivo" if s >= 5 else "negativo"


# ---------------------------------------------------------------- datos

def ruta_imagen(review_id, ref):
    candidatos = []
    ref = str(ref or "").strip()
    if ref and not ref.startswith("[") and not ref.startswith("http"):
        candidatos.append(os.path.basename(ref))
    candidatos += [f"{review_id}.jpg", f"{review_id}_0.jpg", f"{review_id}_1.jpg", f"{review_id}_2.jpg"]
    for carpeta in ("imagenes_nuevas", "mvsa_images"):
        for c in candidatos:
            p = os.path.join(carpeta, c)
            if os.path.exists(p):
                return p
    return None


def cargar_test_multimodal():
    d = pd.read_csv("test_multimodal.csv", encoding="utf-8", dtype=str)
    texto = d["clean_text"].fillna("").str.strip() if "clean_text" in d else pd.Series([""] * len(d))
    d["texto"] = texto.where(texto != "", d["raw_text"].fillna("").str.strip())
    d["label"] = d["sentiment_label"].str.strip().str.lower()
    d["imagen"] = [ruta_imagen(r, i) for r, i in zip(d["review_id"], d.get("imagenes", [""] * len(d)))]
    sin_img = int(d["imagen"].isna().sum())
    d = d[d["imagen"].notna() & d["label"].isin(COLORES)].reset_index(drop=True)
    return d[["review_id", "texto", "label", "imagen"]], sin_img


def pearson_validacion():
    """Pearson r de validacion de cada modelo, de sus carpetas de resultados v2."""
    r_texto = r_imagen = None
    car = sorted(glob.glob("resultados_roberta_v2_*"))
    if car and os.path.exists(os.path.join(car[-1], "log_entrenamiento.csv")):
        log = pd.read_csv(os.path.join(car[-1], "log_entrenamiento.csv"))
        ev = log.dropna(subset=["eval_mae"])
        if len(ev):
            r_texto = float(ev.loc[ev["eval_mae"].idxmin(), "eval_pearson_r"])
    car = sorted(glob.glob("resultados_clip_velocimetro_v2_*"))
    if car and os.path.exists(os.path.join(car[-1], "metricas_test.json")):
        v = json.load(open(os.path.join(car[-1], "metricas_test.json"))).get("pearson_r_validacion")
        if v is not None and not np.isnan(v):
            r_imagen = float(v)
    if r_imagen is None and car and os.path.exists(os.path.join(car[-1], "metricas_por_fold.csv")):
        pf = pd.read_csv(os.path.join(car[-1], "metricas_por_fold.csv"))
        if "pearson_r_validacion" in pf:
            r_imagen = float(pf["pearson_r_validacion"].mean())
    return r_texto, r_imagen


# ---------------------------------------------------------------- inferencia

def puntuar(df):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Dispositivo: {device}")

    tok = AutoTokenizer.from_pretrained("models/roberta-velocimetro/best_model")
    roberta = AutoModelForSequenceClassification.from_pretrained("models/roberta-velocimetro/best_model").to(device).eval()
    st = []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            enc = tok(df["texto"].iloc[i:i + 32].tolist(), truncation=True, max_length=128, padding="max_length",
                      return_tensors="pt").to(device)
            st += roberta(**enc).logits.squeeze(-1).float().cpu().tolist()
    del roberta

    car = sorted(glob.glob("resultados_clip_velocimetro_v2_*"))
    oof = pd.read_csv(os.path.join(car[-1], "puntajes_imagen_oof.csv"))
    por_imagen = dict(zip(oof["image_path"], oof["score_imagen_oof"]))
    por_resena = oof.groupby("review_id")["score_imagen_oof"].mean().to_dict()
    si = [por_imagen.get(p, por_resena.get(r, np.nan)) for p, r in zip(df["imagen"], df["review_id"])]
    out = df.copy()
    out["score_texto"] = np.clip(st, 0, 10).round(2)
    out["score_imagen"] = np.clip(si, 0, 10).round(2)
    faltan = int(out["score_imagen"].isna().sum())
    if faltan:
        print(f"(aviso) {faltan} resenas sin puntaje de imagen fuera de muestra, quedan fuera")
    return out.dropna(subset=["score_imagen"]).reset_index(drop=True)


# ---------------------------------------------------------------- analisis

def metricas(y_true, y_pred):
    from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
    return {"F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred), "Accuracy": accuracy_score(y_true, y_pred)}


def tabla_metricas(d, columnas):
    filas = {}
    pol = d[d.label.isin(["positivo", "negativo"])]
    for nombre, col in columnas.items():
        m3 = metricas(d.label, d[col].apply(categoria))
        m2 = metricas(pol.label, pol[col].apply(polaridad))
        filas[nombre] = {**{f"{k} (polaridad)": v for k, v in m2.items()}, **{f"{k} (3 clases)": v for k, v in m3.items()}}
    return pd.DataFrame(filas).T.round(4)


def fusion_adaptativa(st, si, w_base, tope, umbral):
    st, si = np.asarray(st, float), np.asarray(si, float)
    intensidad = np.abs(si - 5) / 5
    contradice = ((st >= 7) & (si < 4.5)) | ((st < 4.5) & (si >= 7))
    activa = contradice & (intensidad >= umbral)
    w = np.minimum(np.where(activa, w_base + (tope - w_base) * intensidad, w_base), tope)
    return (1 - w) * st + w * si


def f1_pol(d, s):
    from sklearn.metrics import f1_score
    m = d.label.isin(["positivo", "negativo"]).values
    return f1_score(d.label[m], pd.Series(np.asarray(s)[m]).apply(polaridad), average="macro", zero_division=0)


def f1_3(d, s):
    from sklearn.metrics import f1_score
    return f1_score(d.label, pd.Series(np.asarray(s)).apply(categoria), average="macro", zero_division=0)


def miniatura(path, lado=140):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((lado, lado))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def img_png(ruta):
    return "data:image/png;base64," + base64.b64encode(open(ruta, "rb").read()).decode()


def analizar(df, w_texto, w_imagen, origen_pesos, carpeta):
    from sklearn.model_selection import train_test_split
    df = df.copy()
    df["score_fusion"] = (w_texto * df.score_texto + w_imagen * df.score_imagen).round(2)
    wt_app = PESOS_APP[0] / sum(PESOS_APP)
    df["score_fusion_app"] = (wt_app * df.score_texto + (1 - wt_app) * df.score_imagen).round(2)
    df["cat_texto"] = df.score_texto.apply(categoria)
    df["cat_imagen"] = df.score_imagen.apply(categoria)
    df["diferencia"] = (df.score_texto - df.score_imagen).abs().round(2)

    def quien(f):
        t, i = f.cat_texto == f.label, f.cat_imagen == f.label
        return "Ambos" if t and i else "Texto" if t else "Imagen" if i else "Ninguno"
    df["acierta"] = df.apply(quien, axis=1)

    # 1. dispersion
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, g in df.groupby("label"):
        ax.scatter(g.score_texto, g.score_imagen, s=22, alpha=0.65, c=COLORES[lab], label=f"Etiqueta: {lab}")
    ax.plot([0, 10], [0, 10], ls="--", c="gray", lw=1)
    ax.set_xlabel("Puntaje del texto (RoBERTuito)")
    ax.set_ylabel("Puntaje de la imagen (CLIP)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Texto vs. imagen por resena (test multimodal)")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "scatter_texto_vs_imagen.png"), dpi=150)
    plt.close(fig)
    corr = float(df.score_texto.corr(df.score_imagen))
    concord = float((df.cat_texto == df.cat_imagen).mean())

    # 2. contradicciones
    tp_in = df[(df.score_texto >= 7) & (df.score_imagen < 4.5)].sort_values("diferencia", ascending=False)
    tn_ip = df[(df.score_texto < 4.5) & (df.score_imagen >= 7)].sort_values("diferencia", ascending=False)
    contra = pd.concat([tp_in, tn_ip])
    quien_tabla = contra["acierta"].value_counts().rename("resenas").to_frame()
    quien_tabla["%"] = (quien_tabla["resenas"] / max(1, len(contra)) * 100).round(1)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    if len(quien_tabla):
        quien_tabla["resenas"].plot.bar(ax=ax, color=["#2F9BE6", "#F3AE2B", "#E8613F", "#9AA5B1"][:len(quien_tabla)])
    ax.set_title("Cuando se contradicen, quien coincide con la etiqueta")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "quien_acierta.png"), dpi=150)
    plt.close(fig)

    # 3. metricas base
    base = tabla_metricas(df, {"Solo texto": "score_texto", "Solo imagen": "score_imagen",
                               f"Fusion v2 (pesos {w_texto:.2f}/{w_imagen:.2f})": "score_fusion",
                               f"Fusion con pesos actuales de la app ({wt_app:.2f}/{1 - wt_app:.2f})": "score_fusion_app"})

    # 4. fusion adaptativa: ajuste en una mitad, reporte en la otra
    ajuste, prueba = train_test_split(df, test_size=0.5, stratify=df.label, random_state=SEMILLA)
    ajuste, prueba = ajuste.reset_index(drop=True), prueba.reset_index(drop=True)
    grid = []
    for tope in (0.25, 0.30, 0.35, 0.40, 0.50):
        for umbral in (0.0, 0.5, 0.7, 0.9):
            s = fusion_adaptativa(ajuste.score_texto, ajuste.score_imagen, w_imagen, tope, umbral)
            grid.append({"tope": tope, "umbral_intensidad": umbral,
                         "F1 polaridad ajuste": f1_pol(ajuste, s), "F1 3 clases ajuste": f1_3(ajuste, s)})
    grid = pd.DataFrame(grid).sort_values(["F1 polaridad ajuste", "F1 3 clases ajuste"], ascending=False)
    mejor = grid.iloc[0]
    tope, umbral = float(mejor.tope), float(mejor.umbral_intensidad)
    prueba["score_adaptativa"] = fusion_adaptativa(prueba.score_texto, prueba.score_imagen, w_imagen, tope, umbral).round(2)
    comp = tabla_metricas(prueba, {"Solo texto": "score_texto", "Fusion v2": "score_fusion",
                                   f"Adaptativa (tope {tope:.0%}, umbral {umbral})": "score_adaptativa"})
    rng = np.random.default_rng(SEMILLA)
    difs_pol, difs_3 = [], []
    for _ in range(1000):
        m = prueba.sample(len(prueba), replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        difs_pol.append(f1_pol(m, m.score_adaptativa) - f1_pol(m, m.score_fusion))
        difs_3.append(f1_3(m, m.score_adaptativa) - f1_3(m, m.score_fusion))
    ic_pol = np.percentile(difs_pol, [2.5, 97.5])
    ic_3 = np.percentile(difs_3, [2.5, 97.5])

    # exportes
    cols = ["review_id", "label", "texto", "score_texto", "score_imagen", "score_fusion", "diferencia", "acierta", "imagen"]
    df[cols].to_csv(os.path.join(carpeta, "puntajes_por_resena.csv"), index=False, encoding="utf-8")
    contra[cols].to_csv(os.path.join(carpeta, "contradicciones.csv"), index=False, encoding="utf-8")
    base.to_csv(os.path.join(carpeta, "metricas_texto_imagen_fusion.csv"), encoding="utf-8")
    comp.to_csv(os.path.join(carpeta, "metricas_fusion_adaptativa_prueba.csv"), encoding="utf-8")
    grid.to_csv(os.path.join(carpeta, "grid_fusion_adaptativa_ajuste.csv"), index=False, encoding="utf-8")

    fila_fusion = base.iloc[2]
    resumen = {
        "n_resenas": int(len(df)), "n_polaridad": int(df.label.isin(["positivo", "negativo"]).sum()),
        "pesos": {"texto": round(w_texto, 4), "imagen": round(w_imagen, 4), "origen": origen_pesos},
        "f1_macro": float(fila_fusion["F1-macro (polaridad)"]), "mcc": float(fila_fusion["MCC (polaridad)"]),
        "f1_macro_3clases": float(fila_fusion["F1-macro (3 clases)"]),
        "solo_texto_f1_polaridad": float(base.iloc[0]["F1-macro (polaridad)"]),
        "solo_imagen_f1_polaridad": float(base.iloc[1]["F1-macro (polaridad)"]),
        "correlacion_texto_imagen": corr, "concordancia_categoria": concord,
        "contradicciones": int(len(contra)),
        "adaptativa": {"tope": tope, "umbral": umbral,
                       "dif_f1_polaridad_media": float(np.mean(difs_pol)), "ic95_polaridad": [float(x) for x in ic_pol],
                       "dif_f1_3clases_media": float(np.mean(difs_3)), "ic95_3clases": [float(x) for x in ic_3]},
    }
    with open(os.path.join(carpeta, "metricas_test.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # galeria
    def galeria(sub, titulo, n=12):
        filas = []
        for _, f in sub.head(n).iterrows():
            filas.append(
                f"<tr><td><img src='{miniatura(f.imagen)}' style='width:140px;border-radius:8px'></td>"
                f"<td style='max-width:420px'>{html.escape(f.texto[:400])}{'...' if len(f.texto) > 400 else ''}</td>"
                f"<td style='color:{COLORES[categoria(f.score_texto)]};font-weight:700'>{f.score_texto:.1f}</td>"
                f"<td style='color:{COLORES[categoria(f.score_imagen)]};font-weight:700'>{f.score_imagen:.1f}</td>"
                f"<td style='color:{COLORES[categoria(f.score_fusion)]};font-weight:700'>{f.score_fusion:.1f}</td>"
                f"<td>{f.label}</td><td><b>{f.acierta}</b></td></tr>")
        return (f"<h3>{titulo}: {len(sub)} resenas (mostrando {min(n, len(sub))})</h3>"
                "<table><tr><th>Imagen</th><th>Texto</th><th>Texto</th><th>Imagen</th><th>Fusion</th>"
                "<th>Etiqueta</th><th>Acierta</th></tr>" + "".join(filas) + "</table>")

    def veredicto(ic):
        return ("la mejora NO es estadisticamente significativa (el intervalo incluye el 0)"
                if ic[0] <= 0 <= ic[1] else ("mejora significativa" if ic[0] > 0 else "empeora de forma significativa"))

    partes = [
        "<!doctype html><html lang='es'><head><meta charset='utf-8'><title>SentiMYPE: fusion tardia</title>",
        "<style>body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1050px;margin:32px auto;padding:0 20px;"
        "color:#1e2e38}table{border-collapse:collapse;width:100%;margin:10px 0}td,th{border:1px solid #d0d7de;"
        "padding:6px 9px;text-align:left;vertical-align:top}th{background:#dcedf8}h2{margin-top:34px}"
        ".nota{color:#555;font-size:14px}</style></head><body>",
        "<h1>SentiMYPE: experimento de fusion tardia (texto + imagen)</h1>",
        f"<p class='nota'>Test multimodal: {resumen['n_resenas']} resenas con imagen ({resumen['n_polaridad']} en polaridad). "
        f"Ninguno de los dos modelos las vio al entrenar (imagen: puntaje fuera de muestra, k=3).<br>Pesos de la fusion v2: texto {w_texto:.4f}, imagen {w_imagen:.4f} "
        f"({html.escape(origen_pesos)}).<br>Cortes de categoria iguales a la app: positivo &ge; 7, neutro &ge; 4.5; polaridad: corte en 5.</p>",
        "<h2>1. Texto solo vs imagen sola vs fusion</h2>",
        base.to_html(escape=True),
        "<h2>2. Texto vs imagen por resena</h2>",
        f"<img src='{img_png(os.path.join(carpeta, 'scatter_texto_vs_imagen.png'))}' style='max-width:620px;width:100%'>",
        f"<p>Correlacion texto-imagen (Pearson): <b>{corr:.3f}</b>. Concordancia de categoria entre texto e imagen: "
        f"<b>{concord:.1%}</b>.</p>",
        "<h2>3. Contradicciones fuertes</h2>",
        f"<p>{len(contra)} de {len(df)} resenas ({len(contra) / max(1, len(df)):.1%}) tienen texto e imagen en categorias opuestas.</p>",
        quien_tabla.to_html(escape=True),
        f"<img src='{img_png(os.path.join(carpeta, 'quien_acierta.png'))}' style='max-width:520px;width:100%'>",
        galeria(tp_in, "Texto positivo, imagen negativa"),
        galeria(tn_ip, "Texto negativo, imagen positiva"),
        "<h2>4. Experimento: peso adaptativo de la imagen</h2>",
        "<p class='nota'>Cuando texto e imagen se contradicen y la imagen es extrema, se le da mas peso, con un tope. "
        "Los parametros se eligen en una mitad (ajuste) y se reportan en la otra (prueba), para no inflar el resultado.</p>",
        "<h3>Mejores configuraciones en ajuste</h3>", grid.head(8).round(4).to_html(index=False),
        f"<h3>Resultado en la mitad de prueba (tope {tope:.0%}, umbral {umbral})</h3>", comp.to_html(escape=True),
        f"<p>Diferencia de F1 polaridad (adaptativa menos fusion v2): <b>{np.mean(difs_pol):+.4f}</b>, IC95% "
        f"[{ic_pol[0]:+.4f}, {ic_pol[1]:+.4f}]: {veredicto(ic_pol)}.<br>"
        f"Diferencia de F1 3 clases: <b>{np.mean(difs_3):+.4f}</b>, IC95% [{ic_3[0]:+.4f}, {ic_3[1]:+.4f}]: {veredicto(ic_3)}.</p>",
        "</body></html>",
    ]
    ruta_html = os.path.join(carpeta, "reporte_fusion.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))
    return resumen, base, comp, ruta_html


def main():
    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    carpeta = f"resultados_fusion_v2_{fecha}"
    os.makedirs(carpeta, exist_ok=True)

    car_clip = sorted(glob.glob("resultados_clip_velocimetro_v2_*"))
    oof_ok = bool(car_clip) and os.path.exists(os.path.join(car_clip[-1], "puntajes_imagen_oof.csv"))
    for ruta in ("test_multimodal.csv", "models/roberta-velocimetro/best_model"):
        if not os.path.exists(ruta):
            raise SystemExit(f"Falta {ruta}. Entrena primero el velocimetro de RoBERTa (v2).")
    if not oof_ok:
        raise SystemExit("Falta puntajes_imagen_oof.csv: corre primero la version k=3 de entrenar_clip_velocimetro.py.")

    df, sin_img = cargar_test_multimodal()
    print(f"Test multimodal: {len(df)} resenas con imagen y etiqueta (sin imagen local: {sin_img})")

    r_t, r_i = pearson_validacion()
    if r_t is not None and r_i is not None and r_t > 0 and r_i > 0:
        w_texto, w_imagen = r_t / (r_t + r_i), r_i / (r_t + r_i)
        origen = f"Pearson r de validacion: texto {r_t:.4f}, imagen {r_i:.4f}"
    else:
        w_texto, w_imagen = PESOS_APP[0] / sum(PESOS_APP), PESOS_APP[1] / sum(PESOS_APP)
        origen = "pesos actuales de la app (no se encontraron los Pearson de validacion v2)"
    print(f"Pesos: texto {w_texto:.4f} | imagen {w_imagen:.4f} ({origen})")

    df = puntuar(df)
    resumen, base, comp, ruta_html = analizar(df, w_texto, w_imagen, origen, carpeta)

    print("\n" + "=" * 70)
    print(base.to_string())
    print(f"\nFusion v2 en polaridad: F1 {resumen['f1_macro']:.4f} | MCC {resumen['mcc']:.4f} "
          f"(solo texto: F1 {resumen['solo_texto_f1_polaridad']:.4f})")
    a = resumen["adaptativa"]
    print(f"Adaptativa vs fusion v2 (prueba): F1 polaridad {a['dif_f1_polaridad_media']:+.4f} "
          f"IC95% [{a['ic95_polaridad'][0]:+.4f}, {a['ic95_polaridad'][1]:+.4f}]")
    print(f"\nTodo guardado en {carpeta}/  (reporte_fusion.html se abre en el navegador)")
    if sys.platform == "darwin" and "--sin-abrir" not in sys.argv:
        subprocess.run(["open", ruta_html])


if __name__ == "__main__":
    main()
