# Ireland RZLT Statutory Outcome Prediction Pipeline

An end-to-end spatial machine learning and explainable AI (XAI) pipeline for predicting **Residential Zoned Land Tax (RZLT)** statutory map outcomes across Ireland—classifying land parcels into **Included / Liable** vs. **Proposed Exclusion / Removal** using spatial geometry, statutory zoning codes, and multi-modal public transport accessibility.

---

## Interactive Streamlit Dashboard

The repository includes `app.py`, an interactive Streamlit dashboard for exploring RZLT predictions, comparing model results with statutory outcomes, explaining individual predictions, and testing hypothetical parcel characteristics.

### Dashboard Features

- **Geospatial Parcel Map & Outliers:** Displays filtered parcels on a map and provides parcel details when hovering over a parcel. Red indicates a higher exclusion probability and green indicates a higher inclusion probability.
- **Model vs. Statutory Comparison:** Shows the displayed parcel count, actual exclusions, predicted exclusions, and policy agreement rate.
- **Filtered Parcel Table:** Lists key parcel attributes, statutory status, model probabilities, and whether each parcel is a model/statutory disagreement.
- **Parcel SHAP Explanation:** Generates a local SHAP waterfall plot showing the features that increase or decrease inclusion probability for a selected parcel.
- **What-If Policy Simulator:** Allows users to adjust zoning, geometry, density, and public transport attributes and immediately see the predicted outcome.

### Requirements

Run the dashboard with Python 3.9 or later. Install the required packages in the project environment:

```powershell
python -m pip install streamlit geopandas pandas numpy scipy shapely joblib pydeck shap matplotlib xgboost
```

The application expects the following files at the repository paths shown below:

```text
Models/RZLT_PolicyOnly_FinalModel.joblib
Datasets/RZLT_Samples/RZLT_AllCodes_10to1_FullFeatures.geojson
Datasets/RZLT-Dataset.geojson
Datasets/GTFS_All/stops.txt
```

The model and 10:1 stratified feature dataset are sufficient to explore the sample mode. The full national mode additionally requires the national RZLT GeoJSON and GTFS stops file.

### Start the Dashboard

From the repository root, run:

```powershell
streamlit run app.py
```

Streamlit will open the dashboard in a browser automatically. If it does not, open the Local URL printed in the terminal, normally `http://localhost:8501`.

To stop the dashboard, return to the terminal and press `Ctrl+C`.

### How to Use It

1. **Choose a dataset scope** in the sidebar:
   - **10:1 Stratified Sample:** Displays the 6,303-parcel audit sample. A Local Authority filter can be applied for a smaller review area.
   - **Full National Map:** Displays 100% of parcels in a selected Local Authority. The full national data is loaded once and filtered by county before feature enrichment and scoring.

2. **Apply policy and spatial filters**:
   - Select one or more `ZONE_GZT` zoning codes.
   - Select one or more actual statutory statuses.
   - Adjust the **Inclusion Probability Cutoff**. The default is the model's tuned inclusion threshold (`0.2758`).
   - Enable **Show Statutory Outliers Only** to isolate parcels where the model prediction differs from the recorded statutory outcome.

3. **Review the KPIs and map**:
   - Hover over a parcel or point to view its parcel ID, authority, zoning, actual status, probability values, density, and accessibility features.
   - Use the **Filtered Parcel & Statutory Outlier Table** to sort and inspect matching records.
   - Full and sample views cap map rendering at 15,000 parcels for browser performance. They prioritize parcels with lower inclusion probabilities, but the complete filtered table remains available below the map.

4. **Explain an individual parcel**:
   - Open the **Parcel SHAP Explanation** tab.
   - Select a parcel ID from the filtered records.
   - Review the SHAP waterfall plot. Blue indicates features pushing the prediction lower toward exclusion; red indicates features pushing it higher toward inclusion.

5. **Run a what-if scenario**:
   - Open the **What-If Policy Simulator** tab.
   - Change the zoning, site area, shape compactness, perimeter-to-area ratio, parcel density, bus distance, rail/Luas distance, or nearby bus stop count.
   - Compare the simulated statutory prediction with both the sample inclusion probability and the calibrated national probability.

### Probability Values

- **Sample Inclusion Probability (`PRED_INCL_PROB`)** is the model probability produced from the training sample distribution.
- **Calibrated National Probability (`NAT_CALIBRATED_PROB`)** adjusts that probability toward the estimated national inclusion prevalence.
- **Dynamic predictions** are calculated using the cutoff selected in the sidebar. This cutoff controls the displayed predicted class but does not change the underlying model or probabilities.

These outputs are decision-support estimates and should not replace formal statutory assessment or local authority determinations.

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

## 2. Dataset Inventory & File Naming Conventions

All spatial datasets follow the convention `RZLT_<Scope>_<SamplingRatio>_<EnrichmentLevel>.geojson` and are projected in **Irish Transverse Mercator (`EPSG:2157`)** with sliver geometries (`SITE_AREA < 0.005 ha`) removed:

| File Path                                               | Rows      | Role & Experiments Used In                                                                                                                                                                |
| :------------------------------------------------------ | :-------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `National_RZLT.geojson`                                 | `260,463` | Raw national source layer; used for stratified sampling and computing true physical neighbor counts (`cKDTree`) across Ireland.                                                           |
| `gtfs/stops.txt`                                        | —         | Official Irish NTA GTFS static stops feed; used to compute multi-modal Bus and Rail/DART/Luas distances.                                                                                  |
| `RZLT_Samples/RZLT_AllCodes_Balanced_1to1.geojson`      | `~1,176`  | `1:1` balanced prototype sample across all six `ZONE_SYMBOLOGY` codes for fast baseline checks.                                                                                           |
| `RZLT_Samples/RZLT_AllCodes_Stratified_10to1.geojson`   | `6,303`   | `10:1` stratified sample (`5,766` Included vs. `537` Excluded) preserving sub-type proportions.                                                                                           |
| `RZLT_Samples/RZLT_AllCodes_10to1_WithGTFS.geojson`     | `6,303`   | Adds NTA GTFS transit metrics & shape features. Used to benchmark **County-Aware (`With LOCAL_AUTHORITY`)** vs. initial **Policy-Only (`Without LOCAL_AUTHORITY`)**.                      |
| `RZLT_Samples/RZLT_AllCodes_10to1_FullFeatures.geojson` | `6,303`   | Replaces sampled neighbor counts with true national `500m` & `1000m` parcel density. Used for **Final Policy-Only CV, `StratifiedGroupKFold`, Hyperparameter Tuning, and Outlier Audit**. |
| `Models/RZLT_PolicyOnly_FinalModel.joblib`              | —         | Serialized production artifact (`XGBClassifier`, `OneHotEncoder`, feature column list, and optimal probability threshold `0.2758`).                                                       |

---

## 3. Feature Engineering (`EPSG:2157`)

### True National Neighbor Density (`cKDTree` on All 260,463 Parcels)

Computing neighbor counts against the full national dataset eliminated sampling-rate distortion and revealed a **~19x to 22x physical density separation** between Excluded and Included/Liable land:

| Density Metric                  | Mean    | Median (Overall) | Median (`Excluded`) | Median (`Included_Liable`) |
| :------------------------------ | :------ | :--------------- | :------------------ | :------------------------- |
| **`TRUE_PARCELS_WITHIN_500M`**  | 669.5   | 657.0            | **37.0**            | **717.0**                  |
| **`TRUE_PARCELS_WITHIN_1000M`** | 2,059.1 | 1,948.0          | **96.0**            | **2,150.5**                |

### Multi-Modal NTA GTFS Transit Accessibility (`cKDTree`)

- **`DIST_TO_BUS_M`:** Nearest NTA bus stop (`Mean: 267.6m`, `Median: 194.4m`).
- **`DIST_TO_RAIL_LUAS_M`:** Nearest Irish Rail, DART, or Luas station (`Mean: 6,590.4m`, `Median: 2,531.3m`).
- **`DIST_TO_ANY_TRANSIT_M`:** Minimum distance across all public transport modes (`Median: 194.2m`).
- **`BUS_STOPS_WITHIN_500M`:** Bus stops inside a $500\text{m}$ walking catchment (`Median: 7.0` stops for Included vs. `4.0` stops for Excluded).
- **Log Transforms:** `LOG_DIST_BUS`, `LOG_DIST_RAIL_LUAS`, `LOG_TRUE_PARCELS_500M`, and `LOG_SITE_AREA`.

### Polygon Geometry & Statutory Zoning

- **`SHAPE_COMPACTNESS` (Polsby-Popper Index):** $\frac{4\pi \times \text{Area}}{\text{Perimeter}^2}$ — identifies narrow road verges and irregular slivers routinely excluded by local authorities.
- **`PERIM_TO_AREA`:** Boundary-to-area ratio capturing polygon irregularity.
- **`ZONE_GZT`:** Statutory Generalised Zoning Type (`R2`, `RS`, `R3`, `M1`, `M2`, `M4`, etc.).

---

## 4. Model Evolution & Evaluation Benchmarks

All models use `XGBClassifier` with leak-free inside-fold `OneHotEncoder` and dynamic `scale_pos_weight`.

| Model Experiment                                   | Input Dataset                              | Validation Strategy             | ROC-AUC    | PR-AUC Inclusion (`1`) | PR-AUC Exclusion (`0`) | Max Exclusion F1 (Cutoff) |
| :------------------------------------------------- | :----------------------------------------- | :------------------------------ | :--------- | :--------------------- | :--------------------- | :------------------------ |
| **1. County-Aware (`With LOCAL_AUTHORITY`)**       | `RZLT_AllCodes_10to1_WithGTFS.geojson`     | Stratified 5-Fold CV            | **0.9798** | **0.9977**             | **0.8676**             | `0.77` (`0.5000`)         |
| **2. Policy-Only Baseline (`No LOCAL_AUTHORITY`)** | `RZLT_AllCodes_10to1_WithGTFS.geojson`     | Stratified 5-Fold CV            | **0.9545** | **0.9948**             | **0.7380**             | `0.7065` (`0.2763`)       |
| **3. Policy-Only + True National Density**         | `RZLT_AllCodes_10to1_FullFeatures.geojson` | Stratified 5-Fold CV            | **0.9715** | **0.9967**             | **0.8143**             | `0.7787` (`0.2158`)       |
| **4. Tuned Policy-Only (In-Sample Regions)**       | `RZLT_AllCodes_10to1_FullFeatures.geojson` | Stratified 5-Fold CV            | **0.9716** | **0.9968**             | **0.8063**             | `0.7779` (`0.2758`)       |
| **5. Tuned Policy-Only (Unseen Counties)**         | `RZLT_AllCodes_10to1_FullFeatures.geojson` | `StratifiedGroupKFold` (5-Fold) | **0.9198** | **0.9866**             | **0.7037**             | `0.7035` (`0.1594`)       |

### Tuned Production Hyperparameters (`RandomizedSearchCV`)

- `max_depth=4`, `n_estimators=300`, `learning_rate=0.05`, `subsample=0.95`, `colsample_bytree=1.0`, `min_child_weight=1`, `reg_alpha=0.0`, `reg_lambda=1.0`.

---

## 5. Explainable AI (SHAP) & Statutory Outlier Insights

1. **Multi-Scale Urban Density (`TRUE_PARCELS_WITHIN_500M` & `1000M`):** The top two drivers in the Policy-Only model—dense urban/suburban residential clusters strongly drive Inclusion, while isolated fringe parcels push toward Exclusion[cite: 3].
2. **Geometry & Sliver Filtering (`SHAPE_COMPACTNESS` & `PERIM_TO_AREA`):** Low-compactness strips and awkward road verges push strongly toward Exclusion[cite: 3].
3. **Mixed-Use vs. Residential Zoning (`ZONE_GZT`):** Mixed-use zones (`M2`, `M1`, `M4`) push toward Exclusion (reflecting commercial premises exemptions), while residential zones (`R2`, `RS`, `R3`) push toward Inclusion[cite: 3].
4. **Statutory Outlier Audit:** High-confidence false inclusions (`SDLA00113781`, `SDLA00037177`) are physically compact, high-density suburban `R2` plots whose removal reflects non-spatial legal exemptions (e.g., existing LPT-liable dwellings or utility infrastructure) rather than map geometry.

---

## 6. Recommended Notebook & Repository Structure

```text
├── 01_RZLT_Data_Sampling_And_Symbology_Prep.ipynb       # National 260k filtering & 1:1 / 10:1 stratified GeoJSON creation
├── 02_NTA_GTFS_And_National_Spatial_Features.ipynb      # GTFS transit distances, Polsby-Popper geometry & true 260k density
├── 03_Model_Benchmarking_And_SHAP_Explainability.ipynb  # County-Aware vs. Policy-Only 5-Fold CV & SHAP beeswarm plots
├── 04_Hyperparameter_Tuning_SpatialCV_And_ErrorAudit.ipynb # RandomizedSearchCV, StratifiedGroupKFold, joblib export & outliers
├── RZLT_Samples/
│   ├── RZLT_AllCodes_Balanced_1to1.geojson
│   ├── RZLT_AllCodes_Stratified_10to1.geojson
│   ├── RZLT_AllCodes_10to1_WithGTFS.geojson
│   └── RZLT_AllCodes_10to1_FullFeatures.geojson
├── Models/
│   └── RZLT_PolicyOnly_FinalModel.joblib
├── gtfs/
│   └── stops.txt
└── README.md
```
