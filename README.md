# SentiMYPE: Multimodal Sentiment Corpus & Experimental Code

Anonymous repository accompanying the double-blind submission of **SentiMYPE**.

## 1. Repository Contents
- `data/Dataset_SentiMYPE_FINAL.csv`: Anonymized metadata, engineered linguistic features, labels, and partition assignments for **14,112** records (**10,154** active reviews across train, validation, and test splits, plus **3,958** excluded records documented in `motivo_fuera`; 1 malformed record with corrupted columns was omitted from the public CSV).
- `scripts/entrenamiento/`: Data splitting, threshold calibration, unimodal text/image regressors, classifiers, and late-fusion scripts.
- `scripts/experimentos/`: Inter-annotator agreement (`analisis_acuerdo_interno.py`), early fusion (`run_e3.py`), zero-shot LLM baseline (`run_e1.py` + `prompt_fijado.txt`), subgroup Holm-Bonferroni tests (`run_e5.py`), and TOST equivalence tests (`run_tost.py`).
- `requirements.txt`: Exact Python package versions and random seeds (`42, 43, 44`).

> **Data Privacy & Execution Scope:** Raw review texts and original customer images are excluded in accordance with data privacy and platform terms of service. Consequently, scripts that read raw text or image files (`entrenar_roberta.py`, `entrenar_clip_velocimetro.py`, `run_e1.py`, `run_e3.py`) require access to the private local media/text store (`DATA_DIR`), whereas split verification, agreement metrics, subgroup statistics, and equivalence tests can be inspected directly against the tabular metadata. Weak/initial pre-labeling of non-rated reviews was performed using Claude Opus 5 and 5.5 followed by manual verification.

## 2. Dataset Dictionary (`data/Dataset_SentiMYPE_FINAL.csv`)

| Column | Description | Allowed / Observed Values |
| :--- | :--- | :--- |
| `review_id` | Unique anonymized review identifier | UUID / alphanumeric ID |
| `mype_id` | Anonymized business or post identifier | `gm_*` (1,608 businesses), `ig_post_*` (46 public Instagram posts without account ID) |
| `source_platform` | Origin platform | `google_maps`, `instagram`, `facebook` |
| `rubro` | Normalized business sector (lowercase) | `restaurante`, `cafeteria`, `pasteleria`, `polleria`, etc. |
| `rating_score` | Customer star rating (when available) | `1` to `5` (empty if unrated) |
| `sentiment_label` | Final ground-truth sentiment label | `negativo`, `neutro`, `positivo` (empty if excluded) |
| `sentiment_label_debil` | Weak label derived from star rating | `negativo` (1–2), `neutro` (3), `positivo` (4–5) |
| `label_source` | Annotation provenance | `rating`, `llm_asistido`, `humano`, `descartada_no_pertinente` |
| `split_type` | Dataset partition (`test` = Test-A; `test_multimodal` = Test-B; `validation` = paper validation set) | `train` (8,179), `validation` (1,087), `test` (526), `test_multimodal` (362), or `fuera:*` (3,958) |
| `motivo_fuera` | Exclusion reason for the 3,958 non-active records (empty for active splits including Test-A and Test-B) | `negocio_prueba`, `no_espanol`, `sin_etiqueta`, or empty |
| `review_date` | Review publication date | ISO `YYYY-MM-DD` or relative platform string (`Hace 4 meses`, `Hace un año`, etc., preserved as scraped) |
| `idioma` | Detected language code | `es` (active corpus) |
| `n_caracteres` | Character count of cleaned text | Integer >= 0 |
| `n_palabras` | Word count of cleaned text | Integer >= 0 |
| `n_emojis` | Emoji count in review | Integer >= 0 |
| `n_imagenes_locales` | Number of associated local images | Integer >= 0 (`346` reviews with image in `test_multimodal` / Test-B) |

*Note on partition overlap:* `test` (Test-A) and `test_multimodal` (Test-B) share 87 businesses with each other, and share **0 businesses** with `train` or `validation`.

## 3. Reproducibility Map (Paper Results -> Scripts)

| Paper Result | Script |
| :--- | :--- |
| **Table II** (Dataset partitions) | `scripts/entrenamiento/armar_splits.py` |
| **Table III** (External human validation) | `scripts/experimentos/analisis_acuerdo_interno.py` |
| **Table IV** (Text models: RoBERTuito regressor & classifiers) | `scripts/entrenamiento/entrenar_roberta.py`, `entrenar_polaridad.py`, `entrenar_3clases.py`, `elegir_cortes.py` |
| **Table IV** (Image regressor & 3-seed stability) | `scripts/entrenamiento/entrenar_clip_velocimetro.py`, `scripts/experimentos/run_clip_semillas.py` |
| **Table V** (Late fusion) | `scripts/entrenamiento/experimento_fusion.py` |
| **Table V** (Early fusion, E3) | `scripts/experimentos/run_e3.py` |
| **Zero-shot LLM baseline (E1) & token cost** | `scripts/experimentos/run_e1.py` (`gemini-3.5-flash-lite`, `temperature=0`, `thinking_budget=-1`, `scripts/experimentos/prompt_fijado.txt`) |
| **Subgroup analysis & Holm correction (E5)** | `scripts/experimentos/run_e5.py` |
| **Fig. 3** (TOST equivalence test) | `scripts/experimentos/run_tost.py` |

## 4. Hyperparameters & Seeds
- **Text regressor (RoBERTuito):** effective batch size 16 (8 x 2 gradient accumulation), up to 10 epochs, early stopping patience 4, learning rate `2e-5`.
- **Visual regressor (CLIP ViT-B/32):** batch size 16, up to 25 epochs, early stopping patience 6, learning rate `2e-5`, cosine scheduler.
- **Random seeds:** `42` (main split & primary runs), `43`, and `44` (polarity classifier and visual regressor stability).
