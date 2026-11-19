"""T8: RoBERTuito de POLARIDAD con otra semilla (mismo metodo que la semilla 42).

Uso, parado en la carpeta "entrenamiento v2":

    caffeinate -i python entrenar_polaridad_semilla.py 43
    caffeinate -i python entrenar_polaridad_semilla.py 44

Solo cambia la semilla; toda la configuracion es la de la corrida con semilla 42.
NO toca el modelo de la semilla 42 (models/roberta-polaridad) ni su carpeta de resultados:
    resultados_polaridad_semillas_<semilla>_<fecha>/   (metricas_test.json, REPORTE.md, graficas)
    models/roberta-polaridad-semilla<semilla>/best_model/
"""
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

BASE_MODEL = "pysentimiento/robertuito-sentiment-analysis"
L2 = {"positivo": 0, "negativo": 1}
ID2L = {v: k for k, v in L2.items()}
UMBRAL_F1 = 0.90
HIPER = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,    # x2 de acumulacion = 16 efectivo (igual que septiembre)
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_length": 128,
    "padding_fijo": True,                # memoria estable en la GPU del Mac
    "early_stopping_patience": 4,
    "metrica_seleccion": "f1_macro (validacion)",
    "seed": int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 43,
}
# Corrida de septiembre 2026 (numeros reportados en la)
REFERENCIA_SEPTIEMBRE = {"f1_macro": 0.9474, "mcc": 0.8959, "n_test": 476}


def cargar(split):
    d = pd.read_csv(f"{split}.csv", encoding="utf-8")
    texto = d["clean_text"].fillna("").astype(str).str.strip() if "clean_text" in d else pd.Series([""] * len(d))
    d["text"] = texto
    vacios = d["text"] == ""
    d.loc[vacios, "text"] = d.loc[vacios, "raw_text"].fillna("").astype(str).str.strip()
    d["sentiment_label"] = d["sentiment_label"].astype(str).str.strip().str.lower()
    d = d[(d["text"] != "") & d["sentiment_label"].isin(L2)].copy()
    d["label_id"] = d["sentiment_label"].map(L2)
    return d.reset_index(drop=True)


def guardar_curvas(log_history, ruta):
    train = [(h["epoch"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    evals = [h for h in log_history if "eval_loss" in h and "eval_f1_macro" in h]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    if train:
        ejes[0].plot([e for e, _ in train], [l for _, l in train], label="train loss", alpha=0.8)
    if evals:
        ejes[0].plot([h["epoch"] for h in evals], [h["eval_loss"] for h in evals], "o-", label="validacion loss")
        ejes[1].plot([h["epoch"] for h in evals], [h["eval_f1_macro"] for h in evals], "o-", label="F1 macro validacion")
        ejes[1].plot([h["epoch"] for h in evals], [h["eval_mcc"] for h in evals], "s-", label="MCC validacion")
    ejes[1].axhline(UMBRAL_F1, color="red", linestyle="--", linewidth=1, label=f"umbral estudio F1 {UMBRAL_F1}")
    ejes[0].set_title("Loss")
    ejes[0].set_xlabel("epoca")
    ejes[0].legend()
    ejes[1].set_title("Metricas de validacion por epoca")
    ejes[1].set_xlabel("epoca")
    ejes[1].set_ylim(0, 1.02)
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def analizar_test(df_test, logits, carpeta):
    from sklearn.metrics import (f1_score, matthews_corrcoef, accuracy_score, classification_report,
                                 confusion_matrix, ConfusionMatrixDisplay)
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    real = df_test["label_id"].values

    out = df_test[["review_id", "text", "sentiment_label"]].copy()
    out["prediccion"] = [ID2L[p] for p in pred]
    out["prob_positivo"] = probs[:, 0].round(4)
    out["prob_negativo"] = probs[:, 1].round(4)
    out["acierto"] = out["prediccion"] == out["sentiment_label"]
    out.to_csv(os.path.join(carpeta, "predicciones_test.csv"), index=False, encoding="utf-8")
    out[~out["acierto"]].to_csv(os.path.join(carpeta, "errores_test.csv"), index=False, encoding="utf-8")

    nombres = ["positivo", "negativo"]
    metricas = {
        "f1_macro": float(f1_score(real, pred, average="macro")),
        "mcc": float(matthews_corrcoef(real, pred)),
        "accuracy": float(accuracy_score(real, pred)),
        "n_test": int(len(real)),
        "n_errores": int((~out["acierto"]).sum()),
        "cumple_umbral_f1": bool(f1_score(real, pred, average="macro") >= UMBRAL_F1),
        "reporte_por_clase": classification_report(real, pred, target_names=nombres, output_dict=True,
                                                   zero_division=0),
    }
    cm = confusion_matrix(real, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=nombres).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Polaridad, test humano (n={len(real)})\nF1 macro {metricas['f1_macro']:.4f} | MCC {metricas['mcc']:.4f}")
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "matriz_confusion_polaridad.png"), dpi=150)
    plt.close(fig)
    return metricas


def escribir_reporte(carpeta, config, metricas, duracion_min, mejor_epoca):
    ref = REFERENCIA_SEPTIEMBRE
    estado = "CUMPLE" if metricas["cumple_umbral_f1"] else "NO CUMPLE"
    lineas = [
        f"# Reporte de entrenamiento: RoBERTuito polaridad v2, semilla {config['hiperparametros']['seed']}",
        "",
        f"Fecha: {config['fecha']}  ",
        f"Duracion: {duracion_min:.1f} minutos | Mejor epoca (validacion): {mejor_epoca}  ",
        f"Dispositivo: {config['entorno']['dispositivo']} | torch {config['entorno']['torch']} | "
        f"transformers {config['entorno']['transformers']}",
        "",
        "## Datos (solo positivo y negativo)",
        "",
        f"Train: {config['datos']['train']} | Validacion: {config['datos']['validation']} | "
        f"Test humano: {config['datos']['test']}  ",
        f"Distribucion train: {config['datos']['dist_train']}",
        "",
        f"## Resultado en test humano: {estado} el umbral (F1 macro >= {UMBRAL_F1})",
        "",
        "| Metrica | Septiembre  | Esta corrida |",
        "|---|---|---|",
        f"| F1-Score Macro | {ref['f1_macro']:.4f} | {metricas['f1_macro']:.4f} |",
        f"| MCC | {ref['mcc']:.4f} | {metricas['mcc']:.4f} |",
        f"| Accuracy | n/d | {metricas['accuracy']:.4f} |",
        f"| n test | {ref['n_test']} | {metricas['n_test']} |",
        "",
        "Nota: el test humano se depuro (filas sin etiqueta valida), por eso cambia el n;",
        "la comparacion con septiembre es orientativa, no un experimento pareado.",
        "",
        "| Clase | Precision | Recall | F1 | n |",
        "|---|---|---|---|---|",
    ]
    for c in ("positivo", "negativo"):
        r = metricas["reporte_por_clase"][c]
        lineas.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    lineas += [
        "",
        f"Errores en test: {metricas['n_errores']} (detalle en errores_test.csv, util para el analisis de errores).",
        "",
        "## Archivos",
        "",
        "curvas_entrenamiento.png, matriz_confusion_polaridad.png, predicciones_test.csv,",
        "errores_test.csv, log_entrenamiento.csv, metricas_test.json, config.json.",
    ]
    with open(os.path.join(carpeta, "REPORTE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def main():
    import torch
    import transformers
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                              Trainer, EarlyStoppingCallback, TrainerCallback)
    from sklearn.metrics import f1_score, matthews_corrcoef

    if torch.cuda.is_available():
        dispositivo = "cuda"
    elif torch.backends.mps.is_available():
        dispositivo = "mps"
    else:
        dispositivo = "cpu"
    print(f"Python: {sys.executable}")
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | dispositivo: {dispositivo}")
    print(f"Semilla: {HIPER['seed']}")
    if dispositivo == "cpu" and "--permitir-cpu" not in sys.argv:
        raise SystemExit("ALTO: este Python no ve ninguna GPU (mps o cuda). Corre con el Python de clip_env.")
    device = torch.device(dispositivo)
    torch.manual_seed(HIPER["seed"])

    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    carpeta = f"resultados_polaridad_semillas_{HIPER['seed']}_{fecha}"
    os.makedirs(carpeta, exist_ok=True)

    tr, va, te = cargar("train"), cargar("validation"), cargar("test")
    print(f"Datos: train {len(tr)} | validation {len(va)} | test humano {len(te)}")
    print(f"Train por clase: {tr['sentiment_label'].value_counts().to_dict()}")

    config = {
        "fecha": fecha,
        "entorno": {"python": sys.executable, "version_python": platform.python_version(),
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "dispositivo": dispositivo, "maquina": platform.platform()},
        "modelo_base": BASE_MODEL,
        "hiperparametros": HIPER,
        "datos": {"train": len(tr), "validation": len(va), "test": len(te),
                  "dist_train": tr["sentiment_label"].value_counts().to_dict()},
    }
    with open(os.path.join(carpeta, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def a_hf(d):
        ds = Dataset.from_dict({"text": d["text"].tolist(), "labels": d["label_id"].astype(int).tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=HIPER["max_length"],
                                          padding="max_length"), batched=True, remove_columns=["text"])

    ds_tr, ds_va, ds_te = a_hf(tr), a_hf(va), a_hf(te)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2L, label2id=L2, ignore_mismatched_sizes=True).to(device)

    def metricas_val(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1_macro": round(float(f1_score(labels, preds, average="macro")), 4),
                "mcc": round(float(matthews_corrcoef(labels, preds)), 4)}

    class LimpiarMemoriaGPU(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                if dispositivo == "mps":
                    torch.mps.empty_cache()
                elif dispositivo == "cuda":
                    torch.cuda.empty_cache()

    class AvisoEpoca(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                print(f"\n>>> Epoca {state.epoch:.0f}: F1 macro val {metrics.get('eval_f1_macro', 0):.4f} | "
                      f"MCC val {metrics.get('eval_mcc', 0):.4f}\n")

    base_args = dict(
        output_dir=f"models/roberta-polaridad-semilla{HIPER['seed']}",
        num_train_epochs=HIPER["num_train_epochs"],
        per_device_train_batch_size=HIPER["per_device_train_batch_size"],
        gradient_accumulation_steps=HIPER["gradient_accumulation_steps"],
        per_device_eval_batch_size=HIPER["per_device_eval_batch_size"],
        learning_rate=HIPER["learning_rate"], weight_decay=HIPER["weight_decay"],
        warmup_ratio=HIPER["warmup_ratio"], save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
        logging_steps=50, fp16=(dispositivo == "cuda"), report_to="none", seed=HIPER["seed"],
        save_safetensors=False, dataloader_num_workers=0, save_total_limit=2,
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **base_args)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **base_args)

    comunes = dict(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                   compute_metrics=metricas_val,
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=HIPER["early_stopping_patience"]),
                              AvisoEpoca(), LimpiarMemoriaGPU()])
    try:
        trainer = Trainer(processing_class=tokenizer, **comunes)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **comunes)

    inicio = time.time()
    trainer.train()
    duracion_min = (time.time() - inicio) / 60

    historia = trainer.state.log_history
    pd.DataFrame(historia).to_csv(os.path.join(carpeta, "log_entrenamiento.csv"), index=False)
    guardar_curvas(historia, os.path.join(carpeta, "curvas_entrenamiento.png"))
    evals = [h for h in historia if "eval_f1_macro" in h]
    mejor_epoca = int(max(evals, key=lambda h: h["eval_f1_macro"])["epoch"]) if evals else None

    logits = trainer.predict(ds_te).predictions
    metricas = analizar_test(te, logits, carpeta)
    metricas["mejor_epoca"] = mejor_epoca
    with open(os.path.join(carpeta, "metricas_test.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    trainer.save_model(f"models/roberta-polaridad-semilla{HIPER['seed']}/best_model")
    tokenizer.save_pretrained(f"models/roberta-polaridad-semilla{HIPER['seed']}/best_model")
    escribir_reporte(carpeta, config, metricas, duracion_min, mejor_epoca)

    print("\n" + "=" * 70)
    print(f"LISTO en {duracion_min:.1f} min. Todo guardado en: {carpeta}/")
    print(f"Test humano: F1 macro {metricas['f1_macro']:.4f} | MCC {metricas['mcc']:.4f} | "
          f"accuracy {metricas['accuracy']:.4f}  (septiembre: F1 {REFERENCIA_SEPTIEMBRE['f1_macro']} | "
          f"MCC {REFERENCIA_SEPTIEMBRE['mcc']})")
    print(f"Umbral de la estudio (F1 >= {UMBRAL_F1}): {'CUMPLE' if metricas['cumple_umbral_f1'] else 'NO CUMPLE'}")
    print("Abre REPORTE.md de esa carpeta para el resumen completo.")


if __name__ == "__main__":
    main()
