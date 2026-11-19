"""Entrenamiento RoBERTuito velocimetro (regresion 0-10) sobre el dataset final.

Uso, parado en la carpeta "entrenamiento v2":

    caffeinate -i python entrenar_roberta.py

Necesita en la misma carpeta: train.csv, validation.csv, test.csv.

Todo queda guardado en resultados_roberta_v2_<fecha>/ :
    REPORTE.md                  resumen legible de toda la corrida
    config.json                 entorno, versiones, hiperparametros, datos
    log_entrenamiento.csv       loss y metricas de cada paso y cada epoca
    curvas_entrenamiento.png    loss de train/validacion y MAE por epoca
    metricas_test.json          metricas finales en test
    predicciones_test.csv       cada resena de test con score real, predicho y error
    dispersion_test.png         predicho vs real
    matriz_confusion_3clases.png  el score llevado a positivo/neutro/negativo
    ejemplos.txt                predicciones sobre frases de prueba
Y ademas, para que el notebook de Fusion siga funcionando:
    models/roberta-velocimetro/best_model/
    resultados_velocimetro_roberta.json
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
HIPER = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,   # x2 de acumulacion = 16 efectivo
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 16,
    "group_by_length": False,
    "padding_fijo": True,                # todos los lotes a 128 tokens: la GPU del Mac
                                         # compila una sola vez y la memoria no crece
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_length": 128,
    "early_stopping_patience": 4,
    "seed": 42,
}
# Corrida de referencia con el dataset anterior (25 oct 2026, mismo modelo y config)
REFERENCIA_ANTERIOR = {"mae": 1.8792, "rmse": 2.8791, "pearson_r": 0.7843,
                       "spearman_r": 0.8322, "n_test": 592}
UMBRAL_NEG, UMBRAL_POS = 3.5, 6.5


def a_clase(score):
    if score <= UMBRAL_NEG:
        return "negativo"
    if score >= UMBRAL_POS:
        return "positivo"
    return "neutro"


def cargar_datos():
    frames = []
    for split in ("train", "validation", "test"):
        d = pd.read_csv(f"{split}.csv", encoding="utf-8")
        d["split"] = split
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    texto = df["clean_text"].fillna("").astype(str).str.strip() if "clean_text" in df else pd.Series([""] * len(df))
    df["text"] = texto
    vacios = df["text"] == ""
    df.loc[vacios, "text"] = df.loc[vacios, "raw_text"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating_score"], errors="coerce")
    df = df[(df["text"] != "") & df["rating"].notna()].copy()
    df["score"] = ((df["rating"] - 1) / 4 * 10).clip(0, 10)
    return df[["review_id", "text", "rating", "score", "split", "sentiment_label"]]


def guardar_curvas(log_history, ruta):
    train = [(h["epoch"], h["loss"]) for h in log_history if "loss" in h and "eval_loss" not in h]
    evals = [h for h in log_history if "eval_loss" in h and "eval_mae" in h]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    if train:
        ejes[0].plot([e for e, _ in train], [l for _, l in train], label="train loss", alpha=0.8)
    if evals:
        ejes[0].plot([h["epoch"] for h in evals], [h["eval_loss"] for h in evals], "o-", label="validacion loss")
        ejes[1].plot([h["epoch"] for h in evals], [h["eval_mae"] for h in evals], "o-", color="tab:orange", label="MAE validacion")
        ejes[1].plot([h["epoch"] for h in evals], [h["eval_pearson_r"] for h in evals], "s-", color="tab:green", label="Pearson validacion")
    ejes[0].set_title("Loss (MSE)")
    ejes[0].set_xlabel("epoca")
    ejes[0].set_yscale("log")
    ejes[0].legend()
    ejes[1].set_title("Metricas de validacion por epoca")
    ejes[1].set_xlabel("epoca")
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def analizar_test(df_test, preds, carpeta):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (mean_absolute_error, mean_squared_error, accuracy_score,
                                 f1_score, classification_report, confusion_matrix,
                                 ConfusionMatrixDisplay)
    out = df_test.copy()
    out["score_predicho"] = np.clip(preds, 0, 10)
    out["error_abs"] = (out["score_predicho"] - out["score"]).abs()
    out["clase_predicha"] = out["score_predicho"].map(a_clase)
    out.sort_values("error_abs", ascending=False).to_csv(
        os.path.join(carpeta, "predicciones_test.csv"), index=False, encoding="utf-8")

    real, pred = out["score"].values, out["score_predicho"].values
    metricas = {
        "mae": float(mean_absolute_error(real, pred)),
        "rmse": float(mean_squared_error(real, pred) ** 0.5),
        "pearson_r": float(pearsonr(real, pred)[0]),
        "spearman_r": float(spearmanr(real, pred)[0]),
        "n_test": int(len(out)),
    }

    con_label = out[out["sentiment_label"].isin(["positivo", "neutro", "negativo"])]
    clases = ["negativo", "neutro", "positivo"]
    if len(con_label):
        metricas["accuracy_3clases"] = float(accuracy_score(con_label["sentiment_label"], con_label["clase_predicha"]))
        metricas["f1_macro_3clases"] = float(f1_score(con_label["sentiment_label"], con_label["clase_predicha"],
                                                       labels=clases, average="macro", zero_division=0))
        metricas["reporte_por_clase"] = classification_report(
            con_label["sentiment_label"], con_label["clase_predicha"], labels=clases,
            output_dict=True, zero_division=0)
        cm = confusion_matrix(con_label["sentiment_label"], con_label["clase_predicha"], labels=clases)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay(cm, display_labels=clases).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Test: score llevado a 3 clases\n(neg <= {UMBRAL_NEG} < neutro < {UMBRAL_POS} <= pos)")
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
        fig.tight_layout()
        fig.savefig(os.path.join(carpeta, "matriz_confusion_3clases.png"), dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    jitter = np.random.default_rng(0).normal(0, 0.12, len(real))
    ax.scatter(real + jitter, pred, alpha=0.35, s=14)
    ax.plot([0, 10], [0, 10], "k--", linewidth=1, label="prediccion perfecta")
    ax.set_xlabel("score real (estrellas llevadas a 0-10)")
    ax.set_ylabel("score predicho")
    ax.set_title(f"Test: predicho vs real (Pearson r = {metricas['pearson_r']:.3f})")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "dispersion_test.png"), dpi=150)
    plt.close(fig)
    return metricas


def escribir_reporte(carpeta, config, metricas, ejemplos, duracion_min):
    ref = REFERENCIA_ANTERIOR
    lineas = [
        "# Reporte de entrenamiento: RoBERTuito velocimetro v2",
        "",
        f"Fecha: {config['fecha']}  ",
        f"Duracion: {duracion_min:.1f} minutos  ",
        f"Dispositivo: {config['entorno']['dispositivo']}  ",
        f"Python: {config['entorno']['python']}  ",
        f"torch {config['entorno']['torch']}, transformers {config['entorno']['transformers']}",
        "",
        "## Datos",
        "",
        f"Train: {config['datos']['train']} | Validacion: {config['datos']['validation']} | Test: {config['datos']['test']}  ",
        "Fuente: train.csv / validation.csv / test.csv generados por preparar_splits.py desde Dataset_SentiMYPE_FINAL.csv.",
        "Solo entran resenas con estrellas (la regresion necesita rating).",
        "",
        "## Resultado en test vs corrida anterior (dataset previo)",
        "",
        "| Metrica | Anterior | Esta corrida | Mejor si |",
        "|---|---|---|---|",
        f"| MAE | {ref['mae']:.4f} | {metricas['mae']:.4f} | baja |",
        f"| RMSE | {ref['rmse']:.4f} | {metricas['rmse']:.4f} | baja |",
        f"| Pearson r | {ref['pearson_r']:.4f} | {metricas['pearson_r']:.4f} | sube |",
        f"| Spearman r | {ref['spearman_r']:.4f} | {metricas['spearman_r']:.4f} | sube |",
        f"| n test | {ref['n_test']} | {metricas['n_test']} | |",
        "",
        "Nota: el test cambio de tamano (se depuraron filas sin etiqueta valida), asi que la",
        "comparacion es orientativa, no un experimento pareado.",
        "",
    ]
    if "accuracy_3clases" in metricas:
        lineas += [
            "## Score llevado a 3 clases",
            "",
            f"Umbrales: negativo <= {UMBRAL_NEG} < neutro < {UMBRAL_POS} <= positivo  ",
            f"Accuracy: {metricas['accuracy_3clases']:.4f} | F1 macro: {metricas['f1_macro_3clases']:.4f}",
            "",
            "| Clase | Precision | Recall | F1 | n |",
            "|---|---|---|---|---|",
        ]
        for c in ("negativo", "neutro", "positivo"):
            r = metricas["reporte_por_clase"].get(c, {})
            lineas.append(f"| {c} | {r.get('precision', 0):.3f} | {r.get('recall', 0):.3f} | {r.get('f1-score', 0):.3f} | {int(r.get('support', 0))} |")
        lineas.append("")
    lineas += ["## Ejemplos", ""] + [f"- [{s:.1f}/10] {t}" for t, s in ejemplos] + [
        "",
        "## Archivos de esta carpeta",
        "",
        "curvas_entrenamiento.png, dispersion_test.png, matriz_confusion_3clases.png,",
        "predicciones_test.csv (ordenado de mayor a menor error, util para analisis de errores),",
        "log_entrenamiento.csv, metricas_test.json, config.json, ejemplos.txt.",
    ]
    with open(os.path.join(carpeta, "REPORTE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def main():
    import torch
    import transformers
    from datasets import Dataset, DatasetDict
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                              Trainer, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback)

    permitir_cpu = "--permitir-cpu" in sys.argv
    if torch.cuda.is_available():
        dispositivo = "cuda"   # Colab u otra GPU NVIDIA
    elif torch.backends.mps.is_available():
        dispositivo = "mps"    # Mac M1/M2
    else:
        dispositivo = "cpu"
    print(f"Python: {sys.executable}")
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | dispositivo: {dispositivo}")
    if dispositivo == "cpu" and not permitir_cpu:
        raise SystemExit("ALTO: este Python no ve ninguna GPU (mps o cuda). En CPU tardaria horas. "
                         "Corre con el Python de clip_env, o agrega --permitir-cpu si de verdad quieres.")
    device = torch.device(dispositivo)
    torch.manual_seed(HIPER["seed"])

    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    carpeta = f"resultados_roberta_v2_{fecha}"
    os.makedirs(carpeta, exist_ok=True)

    df = cargar_datos()
    conteo = df["split"].value_counts().to_dict()
    print(f"Datos: train {conteo.get('train', 0)} | validation {conteo.get('validation', 0)} | test {conteo.get('test', 0)}")

    config = {
        "fecha": fecha,
        "entorno": {"python": sys.executable, "version_python": platform.python_version(),
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "dispositivo": dispositivo, "maquina": platform.platform()},
        "modelo_base": BASE_MODEL,
        "hiperparametros": HIPER,
        "datos": {k: int(conteo.get(k, 0)) for k in ("train", "validation", "test")},
        "escala": "score = (estrellas - 1) / 4 * 10",
    }
    with open(os.path.join(carpeta, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def a_hf(d):
        return Dataset.from_dict({"text": d["text"].tolist(), "labels": d["score"].astype("float32").tolist()})

    ds = DatasetDict({s: a_hf(df[df["split"] == s]) for s in ("train", "validation", "test")})
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=HIPER["max_length"],
                                    padding="max_length"),
                batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, problem_type="regression", ignore_mismatched_sizes=True).to(device)

    def metricas_val(eval_pred):
        from scipy.stats import pearsonr, spearmanr
        preds, labels = eval_pred
        preds, labels = preds.reshape(-1), labels.reshape(-1)
        return {"mae": round(float(np.mean(np.abs(labels - preds))), 4),
                "rmse": round(float(np.sqrt(np.mean((labels - preds) ** 2))), 4),
                "pearson_r": round(float(pearsonr(labels, preds)[0]), 4),
                "spearman_r": round(float(spearmanr(labels, preds)[0]), 4)}

    class LimpiarMemoriaGPU(TrainerCallback):
        """Libera la cache de la GPU del M1 cada 50 pasos y en cada evaluacion,
        para que la memoria no se acumule a lo largo de la corrida."""
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                if dispositivo == "mps":
                    torch.mps.empty_cache()
                elif dispositivo == "cuda":
                    torch.cuda.empty_cache()

        def on_evaluate(self, args, state, control, **kwargs):
            if dispositivo == "mps":
                torch.mps.empty_cache()

    class AvisoEpoca(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                print(f"\n>>> Epoca {state.epoch:.0f}: MAE val {metrics.get('eval_mae', 0):.4f} | "
                      f"Pearson val {metrics.get('eval_pearson_r', 0):.4f}\n")

    base_args = dict(
        output_dir="models/roberta-velocimetro",
        num_train_epochs=HIPER["num_train_epochs"],
        per_device_train_batch_size=HIPER["per_device_train_batch_size"],
        per_device_eval_batch_size=HIPER["per_device_eval_batch_size"],
        gradient_accumulation_steps=HIPER["gradient_accumulation_steps"],
        group_by_length=HIPER["group_by_length"],
        learning_rate=HIPER["learning_rate"], weight_decay=HIPER["weight_decay"],
        warmup_ratio=HIPER["warmup_ratio"], save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False,
        logging_steps=50, fp16=(dispositivo == "cuda"), report_to="none", seed=HIPER["seed"],
        save_safetensors=False, dataloader_num_workers=0, save_total_limit=2,
    )
    try:
        args = TrainingArguments(eval_strategy="epoch", **base_args)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **base_args)

    comunes = dict(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                   data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=metricas_val,
                   callbacks=[EarlyStoppingCallback(early_stopping_patience=HIPER["early_stopping_patience"]),
                              AvisoEpoca(), LimpiarMemoriaGPU()])
    try:
        trainer = Trainer(processing_class=tokenizer, **comunes)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **comunes)

    inicio = time.time()
    trainer.train()
    duracion_min = (time.time() - inicio) / 60

    pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(carpeta, "log_entrenamiento.csv"), index=False)
    guardar_curvas(trainer.state.log_history, os.path.join(carpeta, "curvas_entrenamiento.png"))

    pred_test = trainer.predict(ds["test"]).predictions.reshape(-1)
    metricas = analizar_test(df[df["split"] == "test"].reset_index(drop=True), pred_test, carpeta)
    with open(os.path.join(carpeta, "metricas_test.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    trainer.save_model("models/roberta-velocimetro/best_model")
    tokenizer.save_pretrained("models/roberta-velocimetro/best_model")
    with open("resultados_velocimetro_roberta.json", "w") as f:
        json.dump({k: metricas[k] for k in ("mae", "rmse", "pearson_r", "spearman_r", "n_test")}, f, indent=2)

    frases = [
        "La comida estuvo excelente y el servicio muy rápido, totalmente recomendado",
        "Pésimo servicio, la comida fría y tardaron una hora en atendernos",
        "Estuvo bien la comida, nada del otro mundo, precio normal",
        "Muy rico el plato principal pero el servicio fue bastante lento",
    ]
    enc = tokenizer(frases, truncation=True, max_length=HIPER["max_length"], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        scores = trainer.model(**enc).logits.squeeze(-1).cpu().numpy()
    ejemplos = [(t, float(np.clip(s, 0, 10))) for t, s in zip(frases, scores)]
    with open(os.path.join(carpeta, "ejemplos.txt"), "w", encoding="utf-8") as f:
        for t, s in ejemplos:
            f.write(f"[{s:.1f}/10] {t}\n")

    escribir_reporte(carpeta, config, metricas, ejemplos, duracion_min)

    print("\n" + "=" * 70)
    print(f"LISTO en {duracion_min:.1f} min. Todo guardado en: {carpeta}/")
    print(f"Test: MAE {metricas['mae']:.4f} | Pearson {metricas['pearson_r']:.4f} "
          f"(antes: MAE {REFERENCIA_ANTERIOR['mae']} | Pearson {REFERENCIA_ANTERIOR['pearson_r']})")
    if "accuracy_3clases" in metricas:
        print(f"Como 3 clases: accuracy {metricas['accuracy_3clases']:.4f} | F1 macro {metricas['f1_macro_3clases']:.4f}")
    for t, s in ejemplos:
        print(f"  [{s:.1f}/10] {t}")
    print("Abre REPORTE.md de esa carpeta para el resumen completo.")


if __name__ == "__main__":
    main()
