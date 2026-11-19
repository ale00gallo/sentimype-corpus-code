# SentiMYPE: Multimodal Sentiment Corpus & Experimental Code

Anonymous repository accompanying the double-blind submission of **SentiMYPE**.

## 1. Repository Contents
- `data/Dataset_SentiMYPE_FINAL.csv`: Anonymized metadata, engineered linguistic features, ground-truth labels, and partition assignments for **14,112** records (**10,154** active reviews across `train`, `validation`, `test` [Test-A], and `test_multimodal` [Test-B], plus **3,958** excluded records; 1 malformed record with shifted columns was omitted from the public CSV).
- `scripts/entrenamiento/`: Data splitting, threshold calibration, unimodal text/image regressors, classifiers, and late-fusion scripts.
- `scripts/experimentos/`: Inter-annotator agreement (`analisis_acuerdo_interno.py`), early fusion (`run_e3.py`), zero-shot LLM baseline (`run_e1.py` + `prompt_fijado.txt` + `prompt_fijado.sha256` + `modelo_fijado.json`), subgroup Holm-Bonferroni tests (`run_e5.py`), and TOST equivalence tests (`run_tost.py`).
- `requirements.txt`: Python package versions and random seeds (`42, 43, 44`).

> **Data Privacy & Execution Scope:** Raw review texts, native platform review IDs, and original customer images are omitted or replaced with opaque UUIDv5 identifiers in accordance with data privacy and platform terms of service. Consequently, scripts that consume raw text, media files, or intermediate prediction logs (`entrenar_roberta.py`, `entrenar_clip_velocimetro.py`, `run_e1.py`, `run_e3.py`, `run_e5.py`, `run_tost.py`, `analisis_acuerdo_interno.py`) expect the private local media/prediction store (`DATA_DIR` / `sentimype_experimentos/`), whereas partition structure, class distributions, and feature summaries can be verified directly against `data/Dataset_SentiMYPE_FINAL.csv`. Initial pre-labeling of non-rated reviews was performed with Claude Opus 5 and 5.5 followed by manual verification.

## 2. Dataset Dictionary (`data/Dataset_SentiMYPE_FINAL.csv`)

| Column | Description | Observed Values in CSV |
| :--- | :--- | :--- |
| `review_id` | Unique opaque UUIDv5 review identifier (all native TripAdvisor/Instagram IDs replaced by opaque hashes) | 14,112 unique UUIDv5 strings |
| `mype_id` | Anonymized business/source identifier (`1,608` unique groups) | `gm_*` (831), `biz_*` (730), `fb_*` (47) |
| `source_platform` | Origin platform | `google_maps` (9,563), `instagram` (2,677), `tripadvisor` (1,232), `facebook` (640) |
| `rubro` | Normalized business sector (lowercase, ASCII, empty when unavailable in source) | 114 normalized categories (`restaurante`, `cafeteria`, `pasteleria`, `barberia`, `ferreteria`, etc.; 5,092 empty) |
| `rating_score` | Customer star rating (when present in platform export) | `1` to `5` (including 5 TripAdvisor business-level aggregate ratings in active splits; empty if unrated) |
| `sentiment_label` | Final ground-truth sentiment label | `negativo`, `neutro`, `positivo` (empty in 101 excluded rows without label) |
| `sentiment_label_debil` | Weak label derived from star rating (`1–2` neg, `3` neu, `4–5` pos) | `negativo`, `neutro`, `positivo` (empty when unrated or in relative-date rows; present in 2 unrated rows) |
| `label_source` | Annotation provenance | `rating`, `llm_asistido`, `humano`, `descartada_no_pertinente` (19 empty in excluded rows) |
| `split_type` | Dataset partition (`test` = Test-A; `test_multimodal` = Test-B; `validation` = paper validation set) | `train` (8,179), `validation` (1,087), `test` (526), `test_multimodal` (362), `fuera:*` (3,958) |
| `motivo_fuera` | Exclusion reason for non-active records (empty for all active splits including Test-A and Test-B) | `negocio de prueba`, `no español (fila nueva)`, `sin etiqueta`, or empty |
| `review_date` | Review publication date preserved from scraping (`2008-07-31` to `2026-11-07` in ISO format; scraping reference date: `2026-11-07`) | 10,160 ISO `YYYY-MM-DD`, 2,301 relative strings (`Hace 4 meses`, `Fecha de edición: Hace...`, 161 in Test-B), 1,651 empty |
| `idioma` | Detected language code | `es` (active corpus; includes 3 `pt` rows in `train`) |
| `n_caracteres` | Character count of cleaned text | Float/Integer >= 0 (1,634 empty in legacy rows without character count) |
| `n_palabras` | Word count of cleaned text | Float/Integer >= 0 (1,634 empty in legacy rows without word count) |
| `n_emojis` | Emoji count in review | Float/Integer >= 0 (1,634 empty in legacy rows without emoji count) |
| `n_imagenes_locales` | Number of associated local images | Integer >= 0 (`346` reviews with image in `test_multimodal` / Test-B) |

*Note on partition overlap:* `test` (Test-A) and `test_multimodal` (Test-B) share **87** businesses with each other, and share **0** businesses with `train` or `validation` (as evaluated in the paper's unseen-business protocol; `train` and `validation` share 352 businesses, and 1 review from `gm_9c94f091` remains in `test_multimodal` as part of the fixed 362-review benchmark).

## 3. Reproducibility Map (Paper Results -> Scripts)

| Paper Result | Script |
| :--- | :--- |
| **Table II** (Dataset partitions) | `scripts/entrenamiento/armar_splits.py` |
| **Table III** (External human validation) | `scripts/experimentos/analisis_acuerdo_interno.py` |
| **Table IV** (Text models: RoBERTuito regressor & classifiers) | `scripts/entrenamiento/entrenar_roberta.py`, `entrenar_polaridad.py`, `entrenar_3clases.py`, `elegir_cortes.py` |
| **Table IV** (Image regressor & 3-seed stability) | `scripts/entrenamiento/entrenar_clip_velocimetro.py`, `scripts/experimentos/run_clip_semillas.py` |
| **Table V** (Late fusion) | `scripts/entrenamiento/experimento_fusion.py` |
| **Table V** (Early fusion, E3) | `scripts/experimentos/run_e3.py` |
| **Zero-shot LLM baseline (E1) & token cost** | `scripts/experimentos/run_e1.py` (`gemini-3.5-flash-lite`, `temperature=0`, `pensamiento=-1` [omits `thinkingConfig`, i.e. default model behavior], `scripts/experimentos/prompt_fijado.txt`, `scripts/experimentos/modelo_fijado.json`) |
| **Subgroup analysis & Holm correction (E5)** | `scripts/experimentos/run_e5.py` |
| **Fig. 3** (TOST equivalence test) | `scripts/experimentos/run_tost.py` |

## 4. Hyperparameters & Seeds
- **Text regressor (RoBERTuito):** effective batch size 16 (8 x 2 gradient accumulation), up to 10 epochs, early stopping patience 4, learning rate `2e-5`.
- **Visual regressor (CLIP ViT-B/32):** batch size 16, up to 25 epochs, early stopping patience 6, learning rate `2e-5`, cosine scheduler.
- **Random seeds:** `42` (main split & primary runs), `43`, and `44` (polarity classifier and visual regressor stability).
