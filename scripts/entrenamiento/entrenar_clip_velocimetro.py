"""CLIP velocimetro (puntaje 0-10 por imagen) con validacion cruzada k=3 (cross-fitting).

Es el CLIP que usa la fusion tardia (texto + imagen).
Por que k-fold: el test multimodal concentra casi todas las resenas negativas con foto.
Si se excluyera del entrenamiento, el modelo aprenderia sin negativas. Con k=3 agrupado por
negocio, cada imagen recibe el puntaje de un modelo que NO la vio al entrenar (prediccion fuera
de muestra), y todos los modelos aprenden con negativas. Asi la fusion se evalua sin fuga.

Metodo de cada fold: CLIP-ViT-Base con las ultimas 2 capas descongeladas + cabeza de regresion,
perdida MSE, 25 epocas maximo con early stopping por MAE de validacion.
Al final entrena un modelo con todas las imagenes (el que usa la app).

Uso, parado en la carpeta "entrenamiento v2":

    caffeinate -i python entrenar_clip_velocimetro.py

Necesita: Dataset_SentiMYPE_FINAL.csv, mvsa_images/, imagenes_nuevas/.

Deja:
    resultados_clip_velocimetro_v2_<fecha>/  REPORTE.md, config.json, metricas_test.json,
        puntajes_imagen_oof.csv (puntaje fuera de muestra de cada imagen), metricas_por_fold.csv,
        curvas_entrenamiento.png, dispersion_test.png
    models/clip-velocimetro/best_model/      modelo final (todas las imagenes) para la app
    resultados_velocimetro_clip.json
"""
import glob
import json
import os
import platform
import sys
import time
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_MODEL_CLIP = "openai/clip-vit-base-patch32"
HIPER = {
    "n_folds": 3,
    "num_train_epochs": 25,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "early_stopping_patience": 6,
    "capas_descongeladas": 2,
    "validacion_interna": "11% de cada fold, agrupado por negocio",
    "seed": 42,
}
# Pearson r del CLIP velocimetro anterior (el que usa hoy la app para los pesos de fusion)
REFERENCIA_ANTERIOR = {"pearson_r": 0.2115}


def preparar_datos():
    from PIL import Image
    rating, split, negocio = {}, {}, {}
    for r in pd.read_csv("Dataset_SentiMYPE_FINAL.csv", encoding="utf-8", dtype=str).itertuples():
        try:
            rating[r.review_id] = float(r.rating_score)
        except (TypeError, ValueError):
            continue
        split[r.review_id] = str(r.split_type or "").strip()
        negocio[r.review_id] = str(getattr(r, "mype_id", "") or "").strip() or r.review_id  # v3: grupo = negocio
    registros, ilegibles, excluidas_mm = [], 0, 0
    for carpeta, con_sufijo in (("mvsa_images", True), ("imagenes_nuevas", False)):
        if not os.path.isdir(carpeta):
            print(f"(aviso) no esta la carpeta {carpeta}/, la salto")
            continue
        for path in sorted(glob.glob(os.path.join(carpeta, "*.jpg"))):
            fn = os.path.basename(path)
            review_id = fn.rsplit("_", 1)[0] if con_sufijo else fn[:-4]
            if review_id not in rating:
                continue
            es_mm = split.get(review_id, "").startswith("test_multimodal")
            excluidas_mm += int(es_mm)
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                ilegibles += 1
                continue
            registros.append({"review_id": review_id, "grupo": negocio.get(review_id, review_id),
                              "image_path": path, "test_multimodal": es_mm,
                              "score": max(0.0, min(10.0, (rating[review_id] - 1) / 4 * 10))})
    return pd.DataFrame(registros).drop_duplicates(subset=["image_path"]).reset_index(drop=True), ilegibles, excluidas_mm


def armar_folds(df, n_folds=3):
    """k folds agrupados por negocio y estratificados por clase del puntaje (asi todos tienen negativas)."""
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
    clase = np.where(df["score"] < 4.5, 0, np.where(df["score"] < 7, 1, 2))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=HIPER["seed"])
    folds = []
    for i_tv, i_te in sgkf.split(df, y=clase, groups=df["grupo"]):  # v3: pliegues por negocio
        tv, te = df.iloc[i_tv].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
        tr, va = partir_val(tv)
        assert not (set(tr.grupo) & set(te.grupo)) and not (set(va.grupo) & set(te.grupo))  # v3: sin negocios compartidos
        folds.append((tr, va, te))
    return folds


def partir_val(df):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1111, random_state=HIPER["seed"])
    i_tr, i_va = next(gss.split(df, groups=df["grupo"]))  # v3: por negocio
    tr, va = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_va].reset_index(drop=True)
    assert not (set(tr.review_id) & set(va.review_id))
    return tr, va


def guardar_graficas(historias, real, pred, carpeta, pearson):
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, hist in enumerate(historias, start=1):
        ev = [h for h in hist if "eval_mae" in h]
        ejes[0].plot([h["epoch"] for h in ev], [h["eval_mae"] for h in ev], "o-", label=f"fold {k}")
        ejes[1].plot([h["epoch"] for h in ev], [h["eval_pearson_r"] for h in ev], "o-", label=f"fold {k}")
    ejes[0].set_title("MAE de validacion por epoca")
    ejes[1].set_title("Pearson r de validacion por epoca")
    for ax in ejes:
        ax.set_xlabel("epoca")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "curvas_entrenamiento.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    jitter = np.random.default_rng(0).normal(0, 0.12, len(real))
    ax.scatter(real + jitter, pred, alpha=0.4, s=14)
    ax.plot([0, 10], [0, 10], "k--", linewidth=1)
    ax.set_xlabel("score real (estrellas llevadas a 0-10)")
    ax.set_ylabel("score predicho por la imagen (fuera de muestra)")
    ax.set_title(f"CLIP velocimetro, k=3 fuera de muestra (Pearson r = {pearson:.3f})")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "dispersion_test.png"), dpi=150)
    plt.close(fig)


def main():
    import torch
    import transformers
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
    from scipy.stats import pearsonr, spearmanr
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection, TrainingArguments, Trainer,
                              EarlyStoppingCallback, TrainerCallback)
    from transformers.modeling_outputs import SequenceClassifierOutput

    if torch.cuda.is_available():
        dispositivo = "cuda"
    elif torch.backends.mps.is_available():
        dispositivo = "mps"
    else:
        dispositivo = "cpu"
    print(f"Python: {sys.executable}")
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | dispositivo: {dispositivo}")
    if dispositivo == "cpu" and "--permitir-cpu" not in sys.argv:
        raise SystemExit("ALTO: este Python no ve ninguna GPU. Corre con el Python de clip_env.")
    torch.manual_seed(HIPER["seed"])

    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    carpeta = f"resultados_clip_velocimetro_v2_{fecha}"
    os.makedirs(carpeta, exist_ok=True)

    df, ilegibles, excluidas_mm = preparar_datos()
    folds = armar_folds(df, HIPER["n_folds"])
    print(f"Imagenes: {len(df)} de {df.review_id.nunique()} resenas | del test multimodal: {excluidas_mm} | "
          f"ilegibles: {ilegibles} | negativas (score < 4.5): {int((df.score < 4.5).sum())}")
    for k, (tr, va, te) in enumerate(folds, start=1):
        print(f"Fold {k}: train {len(tr)} | val {len(va)} | test {len(te)} | negativas en train {int((tr.score < 4.5).sum())}")

    config = {"fecha": fecha,
              "entorno": {"python": sys.executable, "version_python": platform.python_version(),
                          "torch": torch.__version__, "transformers": transformers.__version__,
                          "dispositivo": dispositivo, "maquina": platform.platform()},
              "modelo_base": BASE_MODEL_CLIP, "hiperparametros": HIPER,
              "datos": {"imagenes": int(len(df)), "resenas": int(df.review_id.nunique()),
                        "imagenes_test_multimodal": int(excluidas_mm), "ilegibles": int(ilegibles),
                        "negativas": int((df.score < 4.5).sum())}}
    with open(os.path.join(carpeta, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    processor = CLIPImageProcessor.from_pretrained(BASE_MODEL_CLIP)
    aumentar = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    class Imagenes(Dataset):
        def __init__(self, d, aumento=None):
            self.d, self.aumento = d.reset_index(drop=True), aumento

        def __len__(self):
            return len(self.d)

        def __getitem__(self, i):
            fila = self.d.iloc[i]
            img = Image.open(fila["image_path"]).convert("RGB")
            if self.aumento is not None:
                img = self.aumento(img)
            return {"pixel_values": processor(images=img, return_tensors="pt")["pixel_values"][0],
                    "labels": float(fila["score"])}

    def collate(batch):
        return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
                "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.float32)}

    class CLIPPartialFinetuneRegressor(torch.nn.Module):
        """Misma clase que usa el backend (app/ml_models.py) para cargar full_model.pt."""
        def __init__(self, base_model_name=BASE_MODEL_CLIP, n_unfrozen_layers=2):
            super().__init__()
            self.vision = CLIPVisionModelWithProjection.from_pretrained(base_model_name)
            for p in self.vision.parameters():
                p.requires_grad = False
            for capa in self.vision.vision_model.encoder.layers[-n_unfrozen_layers:]:
                for p in capa.parameters():
                    p.requires_grad = True
            for p in self.vision.visual_projection.parameters():
                p.requires_grad = True
            for p in self.vision.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(self.vision.config.projection_dim, 256), torch.nn.GELU(),
                torch.nn.Dropout(0.3), torch.nn.Linear(256, 1))

        def forward(self, pixel_values, labels=None):
            return SequenceClassifierOutput(logits=self.regressor(self.vision(pixel_values=pixel_values).image_embeds).squeeze(-1))

    class TrainerMSE(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.mse_loss(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    def metricas(eval_pred):
        preds, labels = eval_pred
        preds, labels = preds.reshape(-1), labels.reshape(-1)
        pr = pearsonr(labels, preds)[0] if np.std(preds) > 1e-8 else 0.0
        sr = spearmanr(labels, preds)[0] if np.std(preds) > 1e-8 else 0.0
        return {"mae": round(float(np.mean(np.abs(labels - preds))), 4),
                "rmse": round(float(np.sqrt(np.mean((labels - preds) ** 2))), 4),
                "pearson_r": round(float(pr), 4), "spearman_r": round(float(sr), 4)}

    class Limpiar(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if dispositivo == "mps":
                torch.mps.empty_cache()

    class Aviso(TrainerCallback):
        def __init__(self, nombre):
            self.nombre = nombre

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                print(f"\n>>> {self.nombre}, epoca {state.epoch:.0f}: MAE val {metrics.get('eval_mae', 0):.4f} | "
                      f"Pearson val {metrics.get('eval_pearson_r', 0):.4f}\n")

    def entrenar(tr, va, salida, nombre):
        model = CLIPPartialFinetuneRegressor()
        base_args = dict(
            output_dir=salida, num_train_epochs=HIPER["num_train_epochs"],
            per_device_train_batch_size=HIPER["per_device_train_batch_size"],
            per_device_eval_batch_size=HIPER["per_device_eval_batch_size"],
            learning_rate=HIPER["learning_rate"], weight_decay=HIPER["weight_decay"],
            warmup_ratio=HIPER["warmup_ratio"], lr_scheduler_type=HIPER["lr_scheduler_type"],
            save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False,
            logging_steps=50, fp16=(dispositivo == "cuda"), report_to="none", seed=HIPER["seed"],
            save_safetensors=False, dataloader_num_workers=0, save_total_limit=1,
        )
        try:
            args = TrainingArguments(eval_strategy="epoch", **base_args)
        except TypeError:
            args = TrainingArguments(evaluation_strategy="epoch", **base_args)
        trainer = TrainerMSE(model=model, args=args, train_dataset=Imagenes(tr, aumentar), eval_dataset=Imagenes(va),
                             data_collator=collate, compute_metrics=metricas,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=HIPER["early_stopping_patience"]),
                                        Aviso(nombre), Limpiar()])
        trainer.train()
        ev = [h for h in trainer.state.log_history if "eval_mae" in h]
        mejor = min(ev, key=lambda h: h["eval_mae"]) if ev else {}
        return trainer, model, mejor

    inicio = time.time()
    oof, por_fold, historias = [], [], []
    for k, (tr, va, te) in enumerate(folds, start=1):
        print(f"\n{'=' * 22} Fold {k}/{len(folds)} {'=' * 22}")
        trainer, model, mejor = entrenar(tr, va, f"models/clip-velocimetro-fold{k}", f"Fold {k}")
        historias.append(trainer.state.log_history)
        pred = np.clip(trainer.predict(Imagenes(te)).predictions.reshape(-1), 0, 10)
        out = te[["review_id", "image_path", "score", "test_multimodal"]].copy()
        out["score_imagen_oof"] = pred.round(3)
        out["fold"] = k
        oof.append(out)
        real = te["score"].values
        por_fold.append({"fold": k, "mae": float(np.mean(np.abs(real - pred))),
                         "pearson_r": float(pearsonr(real, pred)[0]) if np.std(pred) > 1e-8 else 0.0,
                         "pearson_r_validacion": float(mejor.get("eval_pearson_r", 0.0)),
                         "mejor_epoca": int(mejor.get("epoch", 0)), "n_test": int(len(te))})
        print(f"Fold {k}: MAE {por_fold[-1]['mae']:.4f} | Pearson {por_fold[-1]['pearson_r']:.4f} | "
              f"Pearson val {por_fold[-1]['pearson_r_validacion']:.4f}")
        del trainer, model
        if dispositivo == "mps":
            torch.mps.empty_cache()

    oof = pd.concat(oof, ignore_index=True)
    oof.to_csv(os.path.join(carpeta, "puntajes_imagen_oof.csv"), index=False, encoding="utf-8")
    pd.DataFrame(por_fold).to_csv(os.path.join(carpeta, "metricas_por_fold.csv"), index=False)
    real, pred = oof["score"].values, oof["score_imagen_oof"].values
    m = {"mae": float(np.mean(np.abs(real - pred))), "rmse": float(np.sqrt(np.mean((real - pred) ** 2))),
         "pearson_r": float(pearsonr(real, pred)[0]), "spearman_r": float(spearmanr(real, pred)[0]),
         "n_test": int(len(oof)), "pearson_r_validacion": float(np.mean([f["pearson_r_validacion"] for f in por_fold])),
         "por_fold": por_fold}
    guardar_graficas(historias, real, pred, carpeta, m["pearson_r"])
    # el resumen se guarda antes del modelo final, asi la fusion no depende de ese ultimo paso
    with open(os.path.join(carpeta, "metricas_test.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    with open("resultados_velocimetro_clip.json", "w") as f:
        json.dump({k: m[k] for k in ("mae", "rmse", "pearson_r", "spearman_r", "n_test")}, f, indent=2)

    print(f"\n{'=' * 22} Modelo final (todas las imagenes, para la app) {'=' * 22}")
    tr_fin, va_fin = partir_val(df)
    trainer, model, _ = entrenar(tr_fin, va_fin, "models/clip-velocimetro", "Modelo final")
    duracion_min = (time.time() - inicio) / 60
    destino = "models/clip-velocimetro/best_model"
    os.makedirs(destino, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(destino, "full_model.pt"))
    torch.save(model.regressor.state_dict(), os.path.join(destino, "regressor_head.pt"))
    processor.save_pretrained(destino)


    lineas = [
        "# Reporte: CLIP velocimetro v2 (k=3, prediccion fuera de muestra)", "",
        f"Fecha: {fecha} | Duracion: {duracion_min:.1f} min (3 folds + modelo final)  ",
        f"Dispositivo: {dispositivo} | torch {torch.__version__} | transformers {transformers.__version__}", "",
        f"Imagenes: {len(df)} de {df.review_id.nunique()} resenas; negativas (score < 4.5): {int((df.score < 4.5).sum())}; "
        f"del test multimodal: {excluidas_mm}; ilegibles: {ilegibles}.", "",
        "Cada imagen tiene el puntaje de un modelo que no vio ese negocio al entrenar (cross-fitting k=3, agrupado por negocio).", "",
        "| Metrica | Anterior | Esta corrida (fuera de muestra) |", "|---|---|---|",
        f"| Pearson r | {REFERENCIA_ANTERIOR['pearson_r']:.4f} | {m['pearson_r']:.4f} |",
        f"| MAE | n/d | {m['mae']:.4f} |", f"| RMSE | n/d | {m['rmse']:.4f} |",
        f"| Spearman r | n/d | {m['spearman_r']:.4f} |", "",
        f"Pearson r de validacion, promedio de folds (lo usa la fusion para fijar pesos): {m['pearson_r_validacion']:.4f}", "",
        "| Fold | MAE | Pearson | Pearson val | Mejor epoca | n |", "|---|---|---|---|---|---|",
    ] + [f"| {f['fold']} | {f['mae']:.4f} | {f['pearson_r']:.4f} | {f['pearson_r_validacion']:.4f} | "
         f"{f['mejor_epoca']} | {f['n_test']} |" for f in por_fold]
    with open(os.path.join(carpeta, "REPORTE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    print("\n" + "=" * 70)
    print(f"LISTO en {duracion_min:.1f} min. Todo guardado en: {carpeta}/")
    print(f"Fuera de muestra: MAE {m['mae']:.4f} | Pearson {m['pearson_r']:.4f} (anterior Pearson "
          f"{REFERENCIA_ANTERIOR['pearson_r']}) | Pearson val promedio {m['pearson_r_validacion']:.4f}")
    print("Siguiente: experimento_fusion.py")


if __name__ == "__main__":
    main()
