"""Arma las particiones v4 (sin fuga por negocio) a partir de SentiMYPE_data_v2."""
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("SentiMYPE_data_v2")
SAL = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("splits_v4")
SAL.mkdir(exist_ok=True)
SEMILLA, PROP_VAL = 42, 0.15
PLATAFORMAS = {"google_maps", "instagram", "tripadvisor", "facebook"}

d = pd.read_csv(BASE / "data/SentiMYPE_v2_completo.csv", low_memory=False)
th = pd.read_csv(BASE / "data/test_humano.csv", low_memory=False)
tm = pd.read_csv(BASE / "data/test_multimodal.csv", low_memory=False)
tr3 = pd.read_csv(BASE / "anterior/train.csv", low_memory=False)
va3 = pd.read_csv(BASE / "anterior/validation.csv", low_memory=False)
cols = list(tr3.columns)

# Grupo (negocio) para las filas sin mype_id: Instagram nuevo -> la publicación
sin_id = d["mype_id"].isna()
post = d["origen_dato"].astype(str).str.extract(r"instagram\.com/(?:p|reel)/([^/?#]+)")[0]
d.loc[sin_id & post.notna(), "mype_id"] = "ig_post_" + post[sin_id & post.notna()]
assert d["mype_id"].notna().all() or d.loc[d["mype_id"].isna(), "sentiment_label"].isna().all()

ids_test = set(th.review_id) | set(tm.review_id)
neg_test = set(th.mype_id.astype(str)) | set(tm.mype_id.astype(str))
neg_tr3, neg_va3 = set(tr3.mype_id.astype(str)), set(va3.mype_id.astype(str))
ids_tr3, ids_va3 = set(tr3.review_id), set(va3.review_id)

d["motivo_fuera"] = ""
d.loc[~d.source_platform.isin(PLATAFORMAS), "motivo_fuera"] = "fila mal formada"
d.loc[(d.motivo_fuera == "") & d.sentiment_label.isna(), "motivo_fuera"] = "sin etiqueta"
d.loc[(d.motivo_fuera == "") & (d.label_source == "descartada_no_pertinente") & ~d.review_id.isin(set(th.review_id) | set(tm.review_id)), "motivo_fuera"] = "descartada no pertinente"
d.loc[(d.motivo_fuera == "") & d.review_id.isin(ids_test), "motivo_fuera"] = "test"
d.loc[(d.motivo_fuera == "") & d.mype_id.astype(str).isin(neg_test), "motivo_fuera"] = "negocio de prueba"
nuevo = ~d.review_id.isin(ids_tr3 | ids_va3)
d.loc[(d.motivo_fuera == "") & nuevo & (d.idioma != "es"), "motivo_fuera"] = "no español (fila nueva)"

d["split_v4"] = ""
d.loc[d.review_id.isin(ids_tr3) & (d.motivo_fuera == ""), "split_v4"] = "train"
d.loc[d.review_id.isin(ids_va3) & (d.motivo_fuera == ""), "split_v4"] = "validation"
d.loc[d.review_id.isin(th.review_id), "split_v4"] = "test"
d.loc[d.review_id.isin(tm.review_id), "split_v4"] = "test_multimodal"

pend = (d.motivo_fuera == "") & (d.split_v4 == "")
# negocios ya repartidos conservan su lado
d.loc[pend & d.mype_id.astype(str).isin(neg_va3), "split_v4"] = "validation"
d.loc[pend & d.mype_id.astype(str).isin(neg_tr3) & (d.split_v4 == ""), "split_v4"] = "train"
pend = (d.motivo_fuera == "") & (d.split_v4 == "")
# negocios nuevos: 15 % a validación, estratificado por lote, agrupado por negocio
rng = np.random.default_rng(SEMILLA)
for lote, g in d[pend].groupby("lote"):
    negs = np.array(sorted(g.mype_id.astype(str).unique()))
    rng.shuffle(negs)
    n_val = int(round(len(negs) * PROP_VAL))
    val = set(negs[:n_val])
    idx = g.index
    d.loc[idx, "split_v4"] = np.where(g.mype_id.astype(str).isin(val), "validation", "train")

# chequeos
tr = d[d.split_v4 == "train"]; va = d[d.split_v4 == "validation"]
assert not set(tr.mype_id.astype(str)) & neg_test and not set(va.mype_id.astype(str)) & neg_test
comp = set(tr.mype_id.astype(str)) & set(va.mype_id.astype(str))
assert comp <= (neg_tr3 & neg_va3), "las filas nuevas agregaron negocios compartidos entre train y validation"
print(f"Negocios compartidos train/validation (heredados de la partición v3 del paper): {len(comp)}")
assert not set(tr.review_id) & ids_test and not set(va.review_id) & ids_test

def guardar(df, nombre, split):
    out = df.copy()
    out["split_type"] = split
    out[cols].to_csv(SAL / nombre, index=False)
guardar(tr, "train.csv", "train"); guardar(va, "validation.csv", "validation")
th[cols].to_csv(SAL / "test.csv", index=False); tm[cols].to_csv(SAL / "test_multimodal.csv", index=False)

final = d.copy()
final["split_type"] = np.where(final["split_v4"] != "", final["split_v4"], "fuera: " + final["motivo_fuera"])
final.drop(columns=["split_v4"]).to_csv(SAL / "Dataset_SentiMYPE_FINAL.csv", index=False)

# resumen
part = d[d.split_v4 != ""]
t = pd.crosstab([part.split_v4], part.sentiment_label).reindex(columns=["negativo", "neutro", "positivo"], fill_value=0)
t["total"] = t.sum(axis=1)
t = t.reindex(["train", "validation", "test", "test_multimodal"])
print(t); print()
print(pd.crosstab(part.split_v4, part.lote).reindex(["train", "validation", "test", "test_multimodal"]))
print("\nFuera de las particiones:", d.loc[d.split_v4 == "", "motivo_fuera"].value_counts().to_dict())
print("Negocios train / validation:", tr.mype_id.nunique(), "/", va.mype_id.nunique())
t.to_csv(SAL / "resumen_particiones_v4.csv")
