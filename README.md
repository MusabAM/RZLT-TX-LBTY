# Ireland RZLT Statutory Outcome Prediction Pipeline

An end-to-end spatial machine learning and explainable AI (XAI) pipeline for predicting **Residential Zoned Land Tax (RZLT)** statutory map outcomes across Ireland—classifying land parcels into **Included / Liable** vs. **Proposed Exclusion / Removal** using spatial geometry, statutory zoning codes, and multi-modal public transport accessibility.

---

## 1. Architectural Pivot: From External GZT Spatial Subtraction to `ZONE_SYMBOLOGY`

### Initial Approach (Deprecated: External GZT Overlay)

The initial prototype attempted to manufacture negative (non-taxed) training samples by performing geometric subtraction (`gpd.overlay`) between the National RZLT layer and the external Generalised Zoning Types (GZT) layer. This introduced two critical bottlenecks:

- **Computational & Memory Overhead:** Loading and intersecting two multi-gigabyte polygon layers caused severe RAM saturation and Jupyter kernel hangs.
- **Artificial Negatives:** Unmapped GZT parcels included land never actively evaluated for RZLT boundary revisions, diluting policy interpretability.

### Current Approach: Direct `ZONE_SYMBOLOGY` Classification

Auditing the national RZLT schema (`~260,463` parcels) revealed that statutory outcomes are already encoded directly inside the `ZONE_SYMBOLOGY` attribute:

| `ZONE_SYMBOLOGY` Code | Statutory Definition             | National Count | Target Class             |
| :-------------------- | :------------------------------- | :------------- | :----------------------- |
| **`1`**               | Retained Residential             | 253,487        | `1` (Included / Liable)  |
| **`2`**               | Retained Mixed-Use               | 3,234          | `1` (Included / Liable)  |
| **`5`**               | Proposed Inclusion (Residential) | 2,869          | `1` (Included / Liable)  |
| **`6`**               | Proposed Inclusion (Mixed-Use)   | 285            | `1` (Included / Liable)  |
| **`3`**               | Proposed Exclusion (Residential) | 312            | `0` (Excluded / Removed) |
| **`4`**               | Proposed Exclusion (Mixed-Use)   | 276            | `0` (Excluded / Removed) |

**Key Impacts of the Pivot:**

- **Policy Ground-Truth Validity:** Every training instance represents an official local authority determination rather than a synthetic spatial negative.
- **Sub-Second Execution & Lightweight Storage:** Eliminating external GZT polygon overlays reduced preprocessing time from minutes to seconds and resolved kernel memory bottlenecks.
- **Zero Target Leakage:** `ZONE_SYMBOLOGY` is strictly excluded from model predictors ($X$), forcing models to learn physical, geometric, and transit drivers.

---

## 2. Dataset Progression & Sampling Strategy

To handle the severe natural class imbalance (`259,875` Included vs. `588` Excluded, ~`442:1`) while preserving representation across all six symbology codes and filtering sliver geometries (`SITE_AREA < 0.005 ha`), three tiered datasets were constructed in **Irish Transverse Mercator (`EPSG:2157`)**:

1. **`RZLT_Samples/RZLT_AllCodes_Balanced_1to1.geojson` (`~1,176` rows):** 1:1 balanced sample for rapid prototyping and unbiased baseline checks.
2. **`RZLT_Samples/RZLT_AllCodes_Stratified_10to1.geojson` (`~6,468` rows raw / `6,303` cleaned):** Keeps 100% of Excluded parcels (`3, 4`) alongside a stratified 10:1 sample of Included/Liable parcels (`1, 2, 5, 6`).
3. **`RZLT_Samples/RZLT_AllCodes_10to1_WithGTFS.geojson` (`6,303` rows — `5,766` Included vs. `537` Excluded):** Production feature-enriched GeoJSON integrating real National Transport Authority (NTA) GTFS transit metrics and polygon geometry features.

---

## 3. Feature Engineering (`EPSG:2157`)

### Multi-Modal NTA GTFS Transit Accessibility (`cKDTree`)

Using the official Irish NTA GTFS `stops.txt` feed projected to `EPSG:2157`, Euclidean distances and catchment densities were computed across `6,303` cleaned parcels:

| Metric                                               | Mean      | Median (50%) | Min    | Max         |
| :--------------------------------------------------- | :-------- | :----------- | :----- | :---------- |
| **`DIST_TO_BUS_M`** (Nearest Bus Stop)               | 267.6 m   | 194.4 m      | 5.6 m  | 4,742.5 m   |
| **`DIST_TO_RAIL_LUAS_M`** (Nearest Rail/DART/Luas)   | 6,590.4 m | 2,531.3 m    | 82.3 m | 105,920.3 m |
| **`DIST_TO_ANY_TRANSIT_M`** (Nearest Mode)           | 262.9 m   | 194.2 m      | 5.6 m  | 4,566.7 m   |
| **`BUS_STOPS_WITHIN_500M`** (500m Walking Catchment) | 7.0 stops | 6.0 stops    | 0      | 77          |

- **Median Transit Density Contrast:** Included/Liable parcels sit near a median of **7.0 bus stops within 500m**, compared to **4.0 bus stops within 500m** for Excluded parcels.
- Log-transformed distance features (`LOG_DIST_BUS`, `LOG_DIST_RAIL_LUAS`) stabilize right-skew across rural and urban-edge sites.

### Geometry & Spatial Clustering Features

- **`SHAPE_COMPACTNESS` (Polsby-Popper Index):** $\frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$ — identifies narrow road verges, access strips, and irregular slivers frequently removed by local authorities.
- **`PERIM_TO_AREA`:** Perimeter-to-area ratio capturing boundary irregularity.
- **`PARCELS_WITHIN_500M`:** Neighboring RZLT parcel count within a $500\text{m}$ radius via `cKDTree.query_ball_point`.
- **`SITE_AREA` & `LOG_SITE_AREA`:** Standardized parcel area in hectares.
- **Categorical Predictors:** `ZONE_GZT` (zoning code) and `LOCAL_AUTHORITY` (evaluated both with and without county fixed effects).

---

## 4. Model Evaluation & Benchmarking (5-Fold Stratified CV)

All models were trained using **XGBoost (`XGBClassifier`)** with leak-free inside-fold `OneHotEncoder`, dynamic `scale_pos_weight` ($\approx 0.093$ for the 10:1 split), and **PR-AUC (`average_precision_score`)** tracked separately for both the majority **Inclusion** class and minority **Exclusion** class.

### Model A: Full Model (With `LOCAL_AUTHORITY` + GTFS + Geometry)

| Fold     | ROC-AUC    | PR-AUC Inclusion (`1`) | PR-AUC Exclusion (`0`) |
| :------- | :--------- | :--------------------- | :--------------------- |
| Fold 1   | 0.9849     | 0.9986                 | 0.8669                 |
| Fold 2   | 0.9802     | 0.9972                 | 0.8693                 |
| Fold 3   | 0.9804     | 0.9981                 | 0.8792                 |
| Fold 4   | 0.9684     | 0.9963                 | 0.8424                 |
| Fold 5   | 0.9852     | 0.9986                 | 0.8804                 |
| **Mean** | **0.9798** | **0.9977**             | **0.8676**             |

**Out-of-Fold Classification Report (Default `0.50` Threshold):**

| Class                       | Precision | Recall   | F1-Score | Support             |
| :-------------------------- | :-------- | :------- | :------- | :------------------ |
| **Excluded (`3, 4`)**       | 0.67      | 0.90     | 0.77     | 537                 |
| **Included (`1, 2, 5, 6`)** | 0.99      | 0.96     | 0.97     | 5,766               |
| **Macro Avg / Accuracy**    | **0.83**  | **0.93** | **0.87** | **6,303 (95% Acc)** |

---

### Model B: Policy-Only Model (Excluding `LOCAL_AUTHORITY`)

To eliminate county-level administrative fixed effects (`CCLA`, `COLA`, `KELA`) and isolate universal statutory and physical criteria, `LOCAL_AUTHORITY` was removed and the decision threshold was tuned on out-of-fold probabilities (`Optimal Inclusion Threshold = 0.2763`):

| Fold     | ROC-AUC    | PR-AUC Inclusion (`1`) | PR-AUC Exclusion (`0`) |
| :------- | :--------- | :--------------------- | :--------------------- |
| Fold 1   | 0.9558     | 0.9952                 | 0.7345                 |
| Fold 2   | 0.9609     | 0.9960                 | 0.7095                 |
| Fold 3   | 0.9553     | 0.9953                 | 0.7585                 |
| Fold 4   | 0.9395     | 0.9914                 | 0.7392                 |
| Fold 5   | 0.9611     | 0.9962                 | 0.7483                 |
| **Mean** | **0.9545** | **0.9948**             | **0.7380**             |

**Out-of-Fold Classification Report (Threshold-Tuned at `0.2763`):**

| Class                       | Precision | Recall   | F1-Score | Support             |
| :-------------------------- | :-------- | :------- | :------- | :------------------ |
| **Excluded (`3, 4`)**       | 0.71      | 0.70     | 0.71     | 537                 |
| **Included (`1, 2, 5, 6`)** | 0.97      | 0.97     | 0.97     | 5,766               |
| **Macro Avg / Accuracy**    | **0.84**  | **0.84** | **0.84** | **6,303 (95% Acc)** |

---

## 5. Explainable AI (SHAP) Insights

SHAP (`TreeExplainer`) beeswarm analyses across both models confirm four core statutory dynamics:

1. **Spatial Clustering (`PARCELS_WITHIN_500M`):** The #1 predictor in both models—dense urban/suburban parcel clusters strongly drive Inclusion, while isolated parcels push toward Exclusion[cite: 1, 2].
2. **County-Level Administrative Divergence:** When included, `LOCAL_AUTHORITY` reveals stark regional variance—Cork County (`CCLA`) and Cork City (`COLA`) strongly favor Inclusion, whereas Kildare (`KELA`), Offaly (`OYLA`), and Monaghan (`MOLA`) show significantly higher exclusion rates[cite: 1].
3. **Parcel Geometry (`SHAPE_COMPACTNESS` & `PERIM_TO_AREA`):** Irregular, elongated polygons (low compactness / high perimeter-to-area ratio) push strongly toward Exclusion, reflecting routine removal of narrow road verges and unbuildable edge strips[cite: 1, 2].
4. **Statutory Zoning (`ZONE_GZT`) & Transit Proximity:** Mixed-use zones (`M2`, `M1`, `M4`) push toward Exclusion (consistent with commercial premises exemptions), while residential zones (`R2`, `RS`, `R3`) and multi-modal transit proximity (`DIST_TO_RAIL_LUAS_M`, `DIST_TO_BUS_M`, `BUS_STOPS_WITHIN_500M`) reinforce Inclusion[cite: 1, 2].

---

## 6. Repository Structure & Next Steps

```text
├── RZLT_Samples/
│   ├── RZLT_AllCodes_Balanced_1to1.geojson       # 1:1 balanced prototype sample
│   ├── RZLT_AllCodes_Stratified_10to1.geojson    # 10:1 stratified national sample
│   └── RZLT_AllCodes_10to1_WithGTFS.geojson      # Final GTFS + geometry enriched dataset
├── gtfs/
│   └── stops.txt                                 # Irish NTA GTFS static stops feed
└── README.md
```
