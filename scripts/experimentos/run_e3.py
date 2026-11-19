import os, json, glob, sys, datetime as dt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from scipy.stats import binomtest

RAIZ = os.path.join("sentimype_experimentos", "e3")
DATOS = os.path.join("sentimype_experimentos", "exportar_resultados", "datos")
IMGS = os.path.join("sentimype_experimentos", "solo_en_la_laptop", "imagenes_test_b")
os.makedirs(RAIZ, exist_ok=True)
SEMILLA, B = 42, 2000
device = "mps" if torch.backends.mps.is_available() else "cpu"

tb = pd.read_csv(os.path.join(DATOS, "test_b.csv"), dtype={"review_id": str, "id_negocio": str})
col_etq = "etiqueta" if "etiqueta" in tb.columns else "sentiment_label"
tb["etq"] = tb[col_etq].str.strip().str.lower()
tb = tb[tb["etq"].isin(["negativo", "neutro", "positivo"]) & tb["s_v"].notna()].reset_index(drop=True)

# 1. Extraer embeddings de texto (768d)
rob_dir = "models/roberta-velocimetro/best_model"
if not os.path.exists(rob_dir):
    rob_dir = sorted(glob.glob("resultados_roberta_v2_*/best_model"))[-1]
tok = AutoTokenizer.from_pretrained(rob_dir)
rob = AutoModel.from_pretrained(rob_dir).to(device).eval()

txt_vecs = []
with torch.no_grad():
    for t in tb["texto"].fillna("").astype(str):
        enc = tok(t, return_tensors="pt", truncation=True, max_length=256).to(device)
        out = rob(**enc).last_hidden_state[:, 0, :].cpu().numpy()[0]
        norm = np.linalg.norm(out)
        txt_vecs.append(out / norm if norm > 0 else out)
X_txt = np.vstack(txt_vecs)

# 2. Extraer embeddings de imagen CLIP ViT-B/32 sin ajuste (512d)
clip_name = "openai/clip-vit-base-patch32"
cproc = CLIPProcessor.from_pretrained(clip_name)
cmod = CLIPModel.from_pretrained(clip_name).to(device).eval()

img_vecs = []
with torch.no_grad():
    for rid in tb["review_id"]:
        p = next((os.path.join(IMGS, f"{rid}{e}") for e in (".jpg",".jpeg",".png",".webp") if os.path.exists(os.path.join(IMGS, f"{rid}{e}"))), None)
        if not p:
            for d in ("mvsa_images", "imagenes_nuevas"):
                p = next((os.path.join(d, f"{rid}{e}") for e in (".jpg",".jpeg",".png",".webp") if os.path.exists(os.path.join(d, f"{rid}{e}"))), None)
                if p: break
        im = Image.open(p).convert("RGB")
        inp = cproc(images=im, return_tensors="pt").to(device)
        feat = cmod.get_image_features(**inp).cpu().numpy()[0]
        norm = np.linalg.norm(feat)
        img_vecs.append(feat / norm if norm > 0 else feat)
X_img = np.vstack(img_vecs)
np.savez_compressed(os.path.join(RAIZ, "embeddings_test_b.npz"), review_id=tb["review_id"].values, X_txt=X_txt, X_img=X_img)

def conf(y, p, k): return np.bincount(y * k + p, minlength=k * k).reshape(k, k)
def f1m(C):
    tp = np.diag(C).astype(float); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    pres = (C.sum(0) + C.sum(1)) > 0; den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)[pres].mean())
def mccf(C):
    t = C.sum(1).astype(float); p = C.sum(0).astype(float); c = float(np.trace(C)); s = float(C.sum())
    den = np.sqrt((s ** 2 - (p * p).sum()) * (s ** 2 - (t * t).sum()))
    return float((c * s - (t * p).sum()) / den) if den > 0 else 0.0

rng = np.random.default_rng(SEMILLA)
res_filas, preds_out = [], tb[["review_id", "id_negocio", "etq", "s_t"]].copy()

for tarea, k in [("polaridad", 2), ("3 clases", 3)]:
    mask = tb["etq"].isin(["negativo", "positivo"]).values if tarea == "polaridad" else np.ones(len(tb), dtype=bool)
    sub = tb[mask].reset_index(drop=True)
    y = (sub["etq"] == "positivo").astype(int).values if tarea == "polaridad" else sub["etq"].map({"negativo":0,"neutro":1,"positivo":2}).values
    groups = sub["id_negocio"].values
    Xt, Xi = X_txt[mask], X_img[mask]
    X_cat = np.hstack([Xt, Xi])

    p_ref = (sub["s_t"].astype(float).values >= 5.0).astype(int) if tarea == "polaridad" else np.where(sub["s_t"].astype(float).values < 3.2, 0, np.where(sub["s_t"].astype(float).values < 7.3, 1, 2)).astype(int)

    outer = StratifiedGroupKFold(n_splits=5)
    for variante, X_in, usar_pca in [
        ("LR-texto (768d)", Xt, False),
        ("LR-imagen (512d)", Xi, False),
        ("LR-concat (1280d)", X_cat, False),
        ("LR-texto PCA-64", Xt, True),
        ("LR-imagen PCA-64", Xi, True),
        ("LR-concat PCA-64+64", X_cat, True),
    ]:
        oof = np.zeros(len(y), dtype=int)
        for tr, va in outer.split(X_in, y, groups):
            if usar_pca and X_in.shape[1] == 1280:
                pt = PCA(n_components=min(64, len(tr)-1), random_state=SEMILLA).fit(X_in[tr, :768])
                pi = PCA(n_components=min(64, len(tr)-1), random_state=SEMILLA).fit(X_in[tr, 768:])
                Xtr = np.hstack([pt.transform(X_in[tr, :768]), pi.transform(X_in[tr, 768:])])
                Xva = np.hstack([pt.transform(X_in[va, :768]), pi.transform(X_in[va, 768:])])
            elif usar_pca:
                p_ = PCA(n_components=min(64, len(tr)-1), random_state=SEMILLA).fit(X_in[tr])
                Xtr, Xva = p_.transform(X_in[tr]), p_.transform(X_in[va])
            else:
                Xtr, Xva = X_in[tr], X_in[va]

            pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEMILLA))])
            inner = StratifiedGroupKFold(n_splits=3)
            gs = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1.0, 10.0]}, cv=inner, scoring="f1_macro")
            gs.fit(Xtr, y[tr], groups=groups[tr])
            oof[va] = gs.predict(Xva)

        col_pred = f"pred_{tarea.replace(' ','_')}_{variante.split()[0].replace('-','_')}" + ("_pca" if usar_pca else "")
        preds_out.loc[mask, col_pred] = oof

        C = conf(y, oof, k)
        f1, mcc = f1m(C), mccf(C)
        boots = [rng.integers(0, len(y), len(y)) for _ in range(B)]
        f1_b = np.array([f1m(conf(y[i], oof[i], k)) for i in boots])
        ref_b = np.array([f1m(conf(y[i], p_ref[i], k)) for i in boots])
        dif_ref = f1_b - ref_b
        c1, c2 = (oof == y), (p_ref == y)
        b_, c_ = int((c2 & ~c1).sum()), int((~c2 & c1).sum())
        p_ref_mc = 1.0 if (b_ + c_) == 0 else float(binomtest(min(b_, c_), b_ + c_, 0.5).pvalue)

        res_filas.append({
            "tarea": tarea, "modelo": variante, "n": len(y), "f1": f1,
            "f1_ic95_inf": np.percentile(f1_b, 2.5), "f1_ic95_sup": np.percentile(f1_b, 97.5),
            "mcc": mcc, "dif_vs_regresor_txt": f1 - f1m(conf(y, p_ref, k)),
            "dif_ic95_inf": np.percentile(dif_ref, 2.5), "dif_ic95_sup": np.percentile(dif_ref, 97.5),
            "p_mcnemar_vs_regresor": p_ref_mc
        })

df_res = pd.DataFrame(res_filas)
df_res.round(4).to_csv(os.path.join(RAIZ, "e3_resultados.csv"), index=False, encoding="utf-8")
preds_out.to_csv(os.path.join(RAIZ, "e3_predicciones_test_b.csv"), index=False, encoding="utf-8")
print(df_res[["tarea","modelo","n","f1","f1_ic95_inf","f1_ic95_sup","mcc","dif_vs_regresor_txt","p_mcnemar_vs_regresor"]].round(3).to_string(index=False))
