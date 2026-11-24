# Product Requirements Document (PRD)
### Track B — Supervised Predictive Modeling
**Project Title:** Decoding Tax Liability: A Supervised Machine Learning and Explainable AI Approach to RZLT Classifications
**Version:** 1.0 | **Date:** November 2026 | **Status:** In Development

---
## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Background & Context](#2-background--context)
3. [Objectives](#3-objectives)
4. [Scope](#4-scope)
5. [Data Architecture](#5-data-architecture)
6. [System Architecture](#6-system-architecture)
7. [Input–Process–Output (IPO) Workflow](#7-inputprocessoutput-ipo-workflow)
8. [Feature Specification](#8-feature-specification)
9. [Model Specification](#9-model-specification)
10. [Evaluation Criteria](#10-evaluation-criteria)
11. [Explainability Requirements](#11-explainability-requirements)
12. [File & Folder Structure](#12-file--folder-structure)
13. [Notebook Pipeline Reference](#13-notebook-pipeline-reference)
14. [Constraints & Assumptions](#14-constraints--assumptions)
15. [Glossary](#15-glossary)

---

## 1. Project Overview

This project builds a **binary supervised classification system** that predicts whether a land parcel in Ireland is designated as a **Residential Zoned Land Tax (RZLT) live site** — meaning it is zoned for residential development and subject to immediate taxation — or whether it is a **non-taxed zoned parcel** that falls outside the current tax inclusion map.

The system combines:
- **RZLT Annual Draft Map data** (official government tax parcel records)
- **Generalised Zoning Typology (GZT)** data (standardised national zoning classifications)
- **GTFS static transit feeds** from the National Transport Authority (NTA)

The result is an interpretable, geospatially-aware machine learning model that quantifies **which factors drive a parcel's tax inclusion status**, supporting policy research, urban planning analysis, and planning authority transparency.

---

## 2. Background & Context

### What is RZLT?
The **Residential Zoned Land Tax (RZLT)** is an Irish government tax introduced under the Finance Act 2021 (amended in 2022 and 2023). It imposes an annual levy of **3% of the market value** of land that is:
- Zoned for residential use, and
- Serviced with the infrastructure necessary for residential development

The tax is designed to **incentivise landowners to activate underdeveloped zoned land** and ease Ireland's housing supply shortage.

Each year, local authorities publish a **draft RZLT map** identifying parcels designated as live sites (`LIVE_SITE = "Y"`). Owners can appeal to have parcels removed from the map (`LIVE_SITE = "N"` after removal or exemption). The annual cycle creates a dataset where both included and excluded parcels can be observed.

### What is GZT?
The **Generalised Zoning Typology (GZT)** is a national standardised zoning classification system maintained by the Office of the Planning Regulator (OPR) in Ireland. It harmonises the diverse, inconsistent zoning codes used across 31 local authorities into a common set of zone type codes (e.g., `R1`, `R2`, `M2`, `C6`), enabling cross-authority spatial analysis.

### Why Machine Learning?
The RZLT tax mapping process involves complex, multi-factor decisions by planning authorities. By training tree-based classifiers on historical parcel attributes, this project uncovers the **latent decision rules** embedded in the mapping framework — going beyond simple rule-based filtering to quantify the marginal effect of each spatial, zoning, and governance factor.

---

## 3. Objectives

| #   | Objective                                                                                       | Priority         |
| --- | ----------------------------------------------------------------------------------------------- | ---------------- |
| 1   | Train a binary classifier to predict `LIVE_SITE` status (Y/N) with high ROC-AUC                 | **Must Have**    |
| 2   | Generate SHAP feature importance explanations for all predictions                               | **Must Have**    |
| 3   | Integrate GTFS transit proximity as an engineered spatial feature                               | **Must Have**    |
| 4   | Support per-local-authority model runs (e.g., CWLA, SDLA, CCLA)                                 | **Must Have**    |
| 5   | Produce a merged, balanced training dataset combining RZLT positives with GZT-derived negatives | **Must Have**    |
| 6   | Save trained model artefacts for reproducibility and future inference                           | **Should Have**  |
| 7   | Extend to all 28 Irish local authorities in the RZLT dataset                                    | **Nice to Have** |

---

## 4. Scope

### In Scope
- Binary classification: `LIVE_SITE = "Y"` (class 1) vs `LIVE_SITE = "N"` (class 0)
- Feature engineering from geospatial data (area, transit proximity, zoning)
- One-hot encoding of categorical zoning and authority fields
- Model training with Random Forest and XGBoost
- Cross-validated performance benchmarking
- SHAP explainability analysis
- Dataset balancing via GZT spatial overlay differencing

### Out of Scope
- Real-time or live API-based predictions
- Regression modelling (e.g., predicting tax liability amount)
- UI / dashboard for end users
- Automated scheduling or cloud deployment

---

## 5. Data Architecture

### 5.1 Primary Datasets

| Dataset               | File                          | Format  | Size    | Description                                               |
| --------------------- | ----------------------------- | ------- | ------- | --------------------------------------------------------- |
| RZLT Annual Draft Map | `RZLT-Dataset.geojson`        | GeoJSON | ~385 MB | 260,463 live-site parcel polygons across 28 LAs           |
| GZT Current Plan      | `GZT_Current_Plan_*.geojson`  | GeoJSON | ~263 MB | 86,686 zoning polygons with standardised `ZONE_GZT` codes |
| GTFS Transit Stops    | `Datasets/GTFS_All/stops.txt` | CSV     | —       | National Transport Authority stop coordinates (lat/lon)   |

### 5.2 Derived / Intermediate Datasets

| Dataset                 | File                                    | Description                                        |
| ----------------------- | --------------------------------------- | -------------------------------------------------- |
| LA-filtered RZLT sample | `RZLT_Samples/rzlt_sample_CWLA.geojson` | RZLT parcels filtered to Carlow (all `Y`)          |
| LA-filtered GZT sample  | `GZT_Samples/CWLA_GZT_Sample.geojson`   | GZT parcels filtered to Carlow                     |
| Merged training dataset | `RZLT_GZT_Merged/CWLA_Merged.geojson`   | Balanced dataset: 1,341 `Y` + 728 `N` = 2,069 rows |

### 5.3 Key Columns

| Column                      | Type                  | Source     | Description                                       |
| --------------------------- | --------------------- | ---------- | ------------------------------------------------- |
| `LIVE_SITE`                 | Categorical (`Y`/`N`) | RZLT       | **Target variable** — tax inclusion status        |
| `SITE_AREA`                 | Float (hectares)      | Computed   | Polygon area in hectares via EPSG:2157 geometry   |
| `ZONE_GZT`                  | Categorical           | GZT        | Standardised zoning code (e.g., `R1`, `R2`, `M2`) |
| `LOCAL_AUTHORITY`           | Categorical           | RZLT / GZT | Local authority code (e.g., `CWLA`, `SDLA`)       |
| `LOCAL_AUTHORITY_NAME`      | String                | RZLT       | Full name of local authority                      |
| `DISTANCE_TO_TRANSIT`       | Float (metres)        | Engineered | Nearest GTFS bus/rail stop distance via cKDTree   |
| `centroid_x` / `centroid_y` | Float                 | Computed   | ITM projected centroid coordinates (EPSG:2157)    |
| `geometry`                  | Polygon               | RZLT / GZT | Parcel polygon geometry                           |

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│                                                                      │
│  RZLT-Dataset.geojson    GZT_Current_Plan.geojson    GTFS stops.txt  │
└──────────┬────────────────────────┬──────────────────────┬──────────┘
           │                        │                      │
           ▼                        ▼                      │
┌──────────────────┐    ┌───────────────────────┐         │
│ RZLT - Data-     │    │  GZT - Data-Sampling  │         │
│ Sampling.ipynb   │    │  .ipynb               │         │
│                  │    │                       │         │
│ Filter by LA     │    │ Filter by LA          │         │
│ → rzlt_sample_   │    │ Remove slivers <0.01ha│         │
│   CWLA.geojson   │    │ → CWLA_GZT_Sample     │         │
│   (all Y)        │    │   .geojson            │         │
└────────┬─────────┘    └──────────┬────────────┘         │
         │                         │                      │
         └────────────┬────────────┘                      │
                      ▼                                   │
         ┌────────────────────────────┐                   │
         │   GZT&RZLT Merge.ipynb     │                   │
         │                            │                   │
         │ Spatial overlay difference │                   │
         │ GZT zones MINUS RZLT areas │                   │
         │ → Non-taxed N candidates   │                   │
         │ Balance Y:N ratio          │                   │
         │ → CWLA_Merged.geojson      │                   │
         └────────────┬───────────────┘                   │
                      │                                   │
                      ▼                                   ▼
         ┌─────────────────────────────────────────────────────┐
         │                Training Model.ipynb                  │
         │                                                      │
         │  Feature_Extractor.py                                │
         │   ├─ Load merged GeoJSON                            │
         │   ├─ Compute DISTANCE_TO_TRANSIT (cKDTree + GTFS)◄──┘
         │   ├─ Compute SITE_AREA                              │
         │   └─ One-Hot Encode ZONE_GZT, LOCAL_AUTHORITY       │
         │                                                      │
         │  Stratified Train/Test Split (75/25)                │
         │                                                      │
         │  ┌─────────────────┐  ┌──────────────────────────┐  │
         │  │  Random Forest  │  │        XGBoost           │  │
         │  │  n=300 trees    │  │  n=300, lr=0.05          │  │
         │  │  balanced class │  │  scale_pos_weight        │  │
         │  └────────┬────────┘  └────────────┬─────────────┘  │
         │           └──────────────┬──────────┘               │
         │                          ▼                           │
         │        Evaluation (ROC-AUC, Confusion Matrix,        │
         │        Classification Report, 5-Fold CV)             │
         │                          │                           │
         │                          ▼                           │
         │        SHAP Explainability (TreeExplainer)           │
         │          Beeswarm | Bar Chart | Dependence Plots     │
         │                          │                           │
         │                          ▼                           │
         │        Save Models → /Models/*.pkl                   │
         └─────────────────────────────────────────────────────┘
```

---

## 7. Input–Process–Output (IPO) Workflow

### INPUTS

| Input                         | Format               | Description                                                  |
| ----------------------------- | -------------------- | ------------------------------------------------------------ |
| `RZLT-Dataset.geojson`        | GeoJSON polygon      | Full national RZLT live-site parcel dataset (all `Y`)        |
| `GZT_Current_Plan_*.geojson`  | GeoJSON polygon      | National standardised zoning dataset                         |
| `GTFS_All/stops.txt`          | CSV                  | NTA GTFS static feed — all public transport stop coordinates |
| Local Authority Code (config) | String e.g. `"CWLA"` | Selects which local authority's parcels to process           |

---

### PROCESS

#### Stage 1 — LA Filtering (RZLT & GZT separately)

**RZLT - Data-Sampling.ipynb**
1. Load full `RZLT-Dataset.geojson` (260,463 rows, all `LIVE_SITE = "Y"`)
2. Filter rows matching local authority code or name (e.g., `CWLA` → Carlow, 1,341 rows)
3. Export filtered sample → `RZLT_Samples/rzlt_sample_CWLA.geojson`

**GZT - Data-Sampling.ipynb**
1. Load full GZT dataset (86,686 features)
2. Filter to same local authority (3,402 raw → 3,310 after sliver removal)
3. Standardise `ZONE_GZT` codes, compute `SITE_AREA`, extract centroids
4. Export all eligible GZT parcels → `GZT_Samples/CWLA_GZT_Sample.geojson`

> **Important:** The GZT sample must **not** be capped. All 3,310 GZT parcels should be exported so the merge stage has sufficient negative candidates. The original 200-row cap was a development artefact and has been removed.

---

#### Stage 2 — Merge & Negative Class Generation

**GZT&RZLT Merge.ipynb**
1. Load RZLT sample (class `Y`) and GZT sample
2. Reproject both to **EPSG:2157** (Irish Transverse Mercator)
3. Filter GZT to taxable zone codes only: `R1`, `R2`, `C6`, `M2`, `M3`, `C2.1`
4. **Spatial overlay difference**: subtract RZLT polygons from GZT zones → remaining land = non-taxed zoned land (`LIVE_SITE = "N"`)
5. Remove geometry slivers < 0.01 ha
6. Compute `SITE_AREA` for negative parcels
7. **Downsample negatives** to max 2× the number of positives (1:2 Y:N ratio)
8. Align schemas, concatenate, encode binary target (`Y=1`, `N=0`)
9. Export → `RZLT_GZT_Merged/CWLA_Merged.geojson`

**Output after merge (CWLA):** 1,341 `Y` + 728 `N` = **2,069 total rows**

---

#### Stage 3 — Feature Engineering

**Feature_Extractor.py / Training Model.ipynb Cell 3**

| Feature               | Method                                                                       |
| --------------------- | ---------------------------------------------------------------------------- |
| `SITE_AREA`           | `geometry.area / 10000` (m² → ha) in EPSG:2157                               |
| `DISTANCE_TO_TRANSIT` | `cKDTree(stop_coords).query(parcel_centroids)` → nearest GTFS stop in metres |
| `ZONE_GZT`            | `OneHotEncoder` — one binary column per zoning type                          |
| `LOCAL_AUTHORITY`     | `OneHotEncoder` — one binary column per LA code                              |

**Final feature matrix `X`:** `[SITE_AREA, DISTANCE_TO_TRANSIT, ZONE_GZT_R1, ZONE_GZT_R2, ..., LOCAL_AUTHORITY_CWLA, ...]`

---

#### Stage 4 — Model Training

**Training Model.ipynb**

| Step               | Detail                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| Split              | Stratified 75% train / 25% test (`random_state=42`)                                              |
| Imbalance handling | RF: `class_weight='balanced'` · XGB: `scale_pos_weight = neg/pos`                                |
| Random Forest      | `n_estimators=300`, `max_depth=None`, `n_jobs=-1`                                                |
| XGBoost            | `n_estimators=300`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8` |

---

### OUTPUTS

| Output                    | Format              | Description                                              |
| ------------------------- | ------------------- | -------------------------------------------------------- |
| **Classification Report** | Console / Notebook  | Precision, Recall, F1 per class                          |
| **ROC-AUC Score**         | Float               | Area under the ROC curve for both models                 |
| **Confusion Matrix**      | Matplotlib plot     | TP/TN/FP/FN breakdown                                    |
| **ROC Curve Plot**        | Matplotlib plot     | Comparative RF vs XGBoost ROC curves                     |
| **5-Fold CV ROC-AUC**     | Console + bar chart | Cross-validated AUC with standard deviation              |
| **SHAP Summary Plot**     | Matplotlib plot     | Global beeswarm feature impact across test set           |
| **SHAP Bar Chart**        | Matplotlib plot     | Mean absolute SHAP value per feature                     |
| **SHAP Dependence Plots** | Matplotlib plot     | Marginal effect of `DISTANCE_TO_TRANSIT` and `SITE_AREA` |
| **`rf_classifier.pkl`**   | Joblib binary       | Serialised Random Forest model                           |
| **`xgb_classifier.pkl`**  | Joblib binary       | Serialised XGBoost model                                 |
| **`ohe_encoder.pkl`**     | Joblib binary       | Fitted OneHotEncoder for inference on new data           |
| **`feature_names.pkl`**   | Joblib binary       | Ordered feature name list for SHAP alignment             |

---

## 8. Feature Specification

### 8.1 `SITE_AREA` — Parcel Size (Hectares)

- **Type:** Continuous numerical
- **Computation:** `geometry.area / 10000.0` after reprojecting to EPSG:2157
- **Why it matters:** Larger zoned parcels are more likely to be included on the RZLT map, as the tax targets significant underdeveloped landholdings. Tiny slivers (< 0.01 ha) are excluded as geometry artefacts.
- **Range (CWLA):** 0.01 ha – ~6.74 ha, mean ~0.83 ha

### 8.2 `DISTANCE_TO_TRANSIT` — Transit Proximity (Metres)

- **Type:** Continuous numerical, engineered
- **Computation:** `cKDTree` nearest-neighbour lookup from parcel centroid (EPSG:2157) to all NTA GTFS bus/rail stop coordinates (reprojected from WGS84 to ITM)
- **Why it matters:** RZLT's eligibility criteria includes infrastructure servicing. Land close to public transport stops is more likely to meet the RZLT servicing test, making transit proximity a plausible predictor of tax inclusion.
- **Source:** NTA GTFS static feed (`stops.txt` — `stop_lat`, `stop_lon`)

### 8.3 `ZONE_GZT` — Generalised Zoning Typology Code

- **Type:** Categorical → One-Hot Encoded
- **Encoding:** One binary column per zone code present in the LA sample
- **Why it matters:** RZLT applies specifically to **residentially zoned** land. Zone type is the most direct determinant — codes like `R1` (Low-Density Residential), `R2` (Medium-Density Residential), `M2` (Mixed Use), and `C6` (City/Town Centre) are eligible, while `G1` (Green Infrastructure) and `O1` (Open Space) are not.
- **Common CWLA values:** `R2` (dominant), `M2`, `C2.1`, `R1`, `M3`, `C6`

### 8.4 `LOCAL_AUTHORITY` — Governance Unit

- **Type:** Categorical → One-Hot Encoded
- **Values:** 28 Irish local authority codes (e.g., `CWLA`, `SDLA`, `CCLA`, `WDLA`)
- **Why it matters:** Each local authority has discretion in applying RZLT mapping criteria, creating authority-level systematic variation. Including this feature allows the model to capture inter-authority policy differences.

---

## 9. Model Specification

### 9.1 Random Forest Classifier

| Parameter      | Value        | Rationale                                            |
| -------------- | ------------ | ---------------------------------------------------- |
| `n_estimators` | 300          | Sufficient ensemble size for stable variance         |
| `max_depth`    | `None`       | Full tree growth; regularised via ensemble averaging |
| `class_weight` | `'balanced'` | Inversely scales weights by class frequency          |
| `random_state` | 42           | Reproducibility                                      |
| `n_jobs`       | -1           | Parallelise across all CPU cores                     |

### 9.2 XGBoost Classifier

| Parameter          | Value       | Rationale                                                        |
| ------------------ | ----------- | ---------------------------------------------------------------- |
| `n_estimators`     | 300         | Gradient boosting with 300 sequential trees                      |
| `learning_rate`    | 0.05        | Conservative shrinkage to prevent overfitting                    |
| `max_depth`        | 5           | Moderate tree depth to balance expressiveness and generalisation |
| `subsample`        | 0.8         | Row subsampling for stochastic regularisation                    |
| `colsample_bytree` | 0.8         | Feature subsampling per tree                                     |
| `scale_pos_weight` | `neg/pos`   | Computed dynamically from training label distribution            |
| `eval_metric`      | `'logloss'` | Appropriate for binary classification                            |
| `random_state`     | 42          | Reproducibility                                                  |

### 9.3 Why Two Models?

Random Forest and XGBoost are both tree-based ensembles but differ structurally:
- **Random Forest** builds trees in parallel (bagging) — high variance reduction, robust to noise
- **XGBoost** builds trees sequentially (boosting) — high bias reduction, better on structured tabular data

Running both allows direct comparison. SHAP explainability is applied to XGBoost as it provides the richest gradient-based attributions via `TreeExplainer`.

---

## 10. Evaluation Criteria

### Primary Metric: ROC-AUC
The area under the ROC curve is used as the primary evaluation metric because:
- The dataset is **moderately imbalanced** (Y:N ≈ 1,341:728 for CWLA)
- Accuracy alone is misleading with imbalanced classes
- ROC-AUC measures the classifier's ability to **rank** positive parcels above negative ones regardless of threshold

### Secondary Metrics

| Metric                  | Purpose                                                     |
| ----------------------- | ----------------------------------------------------------- |
| **Precision**           | Of parcels predicted as `Y`, how many are truly live sites? |
| **Recall**              | Of all true live sites, how many did we correctly identify? |
| **F1 Score**            | Harmonic mean of precision and recall (class-level)         |
| **Confusion Matrix**    | TP/TN/FP/FN breakdown to assess error types                 |
| **5-Fold CV AUC ± std** | Generalisability beyond a single train/test split           |

### Minimum Acceptable Thresholds

| Model         | ROC-AUC Target |
| ------------- | -------------- |
| Random Forest | ≥ 0.80         |
| XGBoost       | ≥ 0.82         |

---

## 11. Explainability Requirements

SHAP (SHapley Additive exPlanations) is applied to the trained XGBoost model using `shap.TreeExplainer`. Three output plots are required:

| Plot                                       | Type                   | Purpose                                                                                                                     |
| ------------------------------------------ | ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Beeswarm (Summary)**                     | `shap.summary_plot`    | Shows distribution of SHAP values per feature across the test set — both direction (positive/negative impact) and magnitude |
| **Bar Chart**                              | `shap.plots.bar`       | Ranks features by mean absolute SHAP value — overall global importance                                                      |
| **Dependence Plot: `DISTANCE_TO_TRANSIT`** | `shap.dependence_plot` | Shows how increasing transit distance changes the probability of tax inclusion — expected negative relationship             |
| **Dependence Plot: `SITE_AREA`**           | `shap.dependence_plot` | Shows how parcel size affects tax inclusion probability — expected positive relationship                                    |

> **Note:** SHAP values are computed on the **test set** only (unseen data), ensuring the explainability reflects generalised model behaviour, not training memorisation.

---

## 12. File & Folder Structure

```
RZLT - Project/
│
├── Datasets/
│   ├── RZLT-Dataset.geojson                    ← Full national RZLT dataset (all Y)
│   ├── GZT_Current_Plan_*.geojson              ← Full national GZT zoning dataset
│   ├── GTFS_All/
│   │   └── stops.txt                           ← NTA GTFS transit stop coordinates
│   ├── RZLT_Samples/
│   │   ├── rzlt_sample_CWLA.geojson            ← LA-filtered RZLT parcels
│   │   ├── rzlt_sample_zone_R1.geojson         ← Zone-filtered sample (prototype)
│   │   └── Feature_Extractor.py                ← Reusable feature extraction script
│   ├── GZT_Samples/
│   │   └── CWLA_GZT_Sample.geojson             ← LA-filtered GZT parcels
│   └── RZLT_GZT_Merged/
│       └── CWLA_Merged.geojson                 ← Merged training dataset (Y+N)
│
├── Models/                                     ← Saved model artefacts (post-training)
│   ├── rf_classifier.pkl
│   ├── xgb_classifier.pkl
│   ├── ohe_encoder.pkl
│   └── feature_names.pkl
│
├── RZLT - Data-Sampling.ipynb                  ← Stage 1A: Filter RZLT by LA
├── GZT - Data-Sampling.ipynb                   ← Stage 1B: Filter GZT by LA
├── GZT&RZLT Merge.ipynb                        ← Stage 2: Spatial merge & class generation
├── Training Model.ipynb                        ← Stage 3: Feature eng. + training + eval
├── RZLT - Prototype.ipynb                      ← Experimental prototype (archived)
└── PRD.md                                      ← This document
```

---

## 13. Notebook Pipeline Reference

The full pipeline runs across **4 notebooks** in sequence:

```
[RZLT - Data-Sampling.ipynb]      [GZT - Data-Sampling.ipynb]
  Filter RZLT parcels by LA          Filter GZT parcels by LA
  → rzlt_sample_CWLA.geojson         → CWLA_GZT_Sample.geojson
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
              [GZT&RZLT Merge.ipynb]
               Spatial overlay difference
               Balance Y and N classes
               → CWLA_Merged.geojson
                         │
                         ▼
              [Training Model.ipynb]
               Feature engineering
               RF + XGBoost training
               SHAP explainability
               → Models/*.pkl
```

### To run for a different Local Authority
1. In `RZLT - Data-Sampling.ipynb`: change `'CWLA'` to the new LA code
2. In `GZT - Data-Sampling.ipynb`: change `'CWLA'` to the new LA code
3. In `GZT&RZLT Merge.ipynb`: update `rzlt_path`, `gzt_path`, `LOCAL_AUTHORITY`, and `output_path`
4. In `Training Model.ipynb`: update `INPUT_FILE` to point to the new merged file

---

## 14. Constraints & Assumptions

| Constraint / Assumption                                       | Detail                                                                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **RZLT dataset contains only `Y` records**                    | The official RZLT map only lists live sites. Negative class (`N`) must be synthetically generated from GZT spatial differencing.                 |
| **GZT spatial overlay produces valid negatives**              | Land within taxable GZT zones (R1, R2, M2, C6, M3, C2.1) that does not overlap with RZLT polygons is assumed to be non-taxed eligible land.      |
| **GTFS stops are a valid proxy for infrastructure servicing** | NTA GTFS stop proximity is used as an approximation for the "serviced land" criterion in RZLT eligibility.                                       |
| **EPSG:2157 (ITM) for all spatial operations**                | All distance and area calculations require a metre-based projected CRS. All datasets are reprojected before spatial operations.                  |
| **Single LA per model run**                                   | Each training run targets one local authority. Cross-authority models are out of scope for v1.0.                                                 |
| **Class imbalance handled via weighting**                     | No oversampling (e.g., SMOTE) is applied. Imbalance is addressed through `class_weight='balanced'` and `scale_pos_weight`.                       |
| **GZT 200-cap removed**                                       | The original 200-row cap in `GZT - Data-Sampling.ipynb` was a development artefact and has been removed to maximise negative candidate coverage. |
| **Python 3.13.5, standard data science stack**                | GeoPandas, XGBoost, SHAP, Scikit-learn, Scipy, Matplotlib, Joblib                                                                                |

---

## 15. Glossary

| Term                           | Definition                                                                                                                              |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **RZLT**                       | Residential Zoned Land Tax — Irish annual levy on underdeveloped serviced zoned land                                                    |
| **GZT**                        | Generalised Zoning Typology — OPR's national harmonised zoning classification system                                                    |
| **LIVE_SITE**                  | Target column; `"Y"` = parcel is on the RZLT active tax map; `"N"` = not included                                                       |
| **ZONE_GZT**                   | Standardised zone code from GZT (e.g., `R1`=Low-Density Residential, `R2`=Medium-Density Residential, `M2`=Mixed Use, `C6`=Town Centre) |
| **LOCAL_AUTHORITY**            | Irish local authority code (e.g., `CWLA`=Carlow, `SDLA`=South Dublin, `CCLA`=Cork City)                                                 |
| **GTFS**                       | General Transit Feed Specification — standardised public transit data format                                                            |
| **NTA**                        | National Transport Authority — Ireland's transport planning body                                                                        |
| **OPR**                        | Office of the Planning Regulator — Ireland's planning oversight body                                                                    |
| **EPSG:2157**                  | Irish Transverse Mercator (ITM) — Ireland's official metric projected CRS                                                               |
| **cKDTree**                    | Scipy's optimised k-dimensional tree for fast nearest-neighbour spatial queries                                                         |
| **SHAP**                       | SHapley Additive exPlanations — game-theory-based model explainability method                                                           |
| **ROC-AUC**                    | Receiver Operating Characteristic — Area Under Curve; primary model performance metric                                                  |
| **OHE**                        | One-Hot Encoding — converts categorical variables into binary indicator columns                                                         |
| **Spatial Overlay Difference** | GIS operation: `A MINUS B` — returns parts of geometry A not covered by geometry B                                                      |
| **ITM**                        | Irish Transverse Mercator — equivalent to EPSG:2157                                                                                     |
| **LA**                         | Local Authority — one of 31 Irish county/city councils                                                                                  |

