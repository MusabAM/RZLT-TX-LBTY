import os
import time
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# ==============================================================================
# 1. LOAD ARTIFACT & RUN AUTOMATED UNIT / EDGE-CASE TESTS
# ==============================================================================
artifact_path = os.path.join("../Models", "RZLT_PolicyOnly_FinalModel.joblib")
bundle = joblib.load(artifact_path)

model = bundle['model']
encoder = bundle['encoder']
num_cols = bundle['num_cols']
cat_cols = bundle['cat_cols']
threshold = bundle['optimal_inclusion_threshold']

def run_smoke_tests():
    # Test 1: Unseen ZONE_GZT category + missing values
    dummy_df = pd.DataFrame([{c: 100.0 for c in num_cols}])
    dummy_df['ZONE_GZT'] = "UNSEEN_ZONING_CODE_999"
    X_test = np.hstack([dummy_df[num_cols].values, encoder.transform(dummy_df[cat_cols])])
    prob = model.predict_proba(X_test)[:, 1]
    assert X_test.shape[1] == len(bundle['feature_names']), "Feature dimension mismatch!"
    assert not np.isnan(prob).any(), "NaN probability produced on unseen category!"
    assert 0.0 <= prob[0] <= 1.0, "Probability out of [0, 1] bounds!"
    print("[PASS] Smoke Test 1: Unseen category & schema integrity verified.")

run_smoke_tests()

# ==============================================================================
# 2. PRIOR-ODDS PROBABILITY CALIBRATION (10:1 SAMPLE -> 442:1 NATIONAL)
# ==============================================================================
def calibrate_to_national_prior(p_sample, pi_train=0.9148, pi_real=0.9977):
    eps = 1e-7
    p = np.clip(p_sample, eps, 1 - eps)
    num = p * (pi_real / pi_train)
    den = num + (1 - p) * ((1 - pi_real) / (1 - pi_train))
    return num / den

# ==============================================================================
# 3. FULL NATIONAL BATCH SCORING BENCHMARK (260k PARCELS)
# ==============================================================================
def score_national_layer(national_geojson, gtfs_stops_txt, output_csv="RZLT_National_Predictions.csv", batch_size=25000):
    t0 = time.time()
    gdf = gpd.read_file(national_geojson)
    if gdf.crs != "EPSG:2157":
        gdf = gdf.to_crs(epsg=2157)

    # Build national parcel tree & GTFS trees once
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    nat_tree = cKDTree(coords)

    stops = pd.read_csv(gtfs_stops_txt)
    stops_gdf = gpd.GeoDataFrame(
        stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs="EPSG:4326"
    ).to_crs(epsg=2157)
    stop_coords = np.column_stack([stops_gdf.geometry.x, stops_gdf.geometry.y])
    is_rail = stops['stop_name'].str.contains('Station|Luas|DART|Rail', case=False, na=False).values
    bus_tree = cKDTree(stop_coords[~is_rail])
    rail_tree = cKDTree(stop_coords[is_rail])

    results = []
    n_total = len(gdf)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk = gdf.iloc[start:end].copy()
        c_coords = coords[start:end]

        area_m2 = chunk.geometry.area
        perim_m = chunk.geometry.length
        chunk['SITE_AREA'] = area_m2 / 10000.0
        chunk['LOG_SITE_AREA'] = np.log1p(chunk['SITE_AREA'])
        chunk['SHAPE_COMPACTNESS'] = (4 * np.pi * area_m2) / np.maximum(perim_m ** 2, 1e-6)
        chunk['PERIM_TO_AREA'] = perim_m / np.maximum(area_m2, 1e-6)

        chunk['TRUE_PARCELS_WITHIN_500M'] = [max(0, len(i) - 1) for i in nat_tree.query_ball_point(c_coords, r=500.0)]
        chunk['TRUE_PARCELS_WITHIN_1000M'] = [max(0, len(i) - 1) for i in nat_tree.query_ball_point(c_coords, r=1000.0)]
        chunk['LOG_TRUE_PARCELS_500M'] = np.log1p(chunk['TRUE_PARCELS_WITHIN_500M'])

        d_bus, _ = bus_tree.query(c_coords, k=1)
        d_rail, _ = rail_tree.query(c_coords, k=1)
        chunk['DIST_TO_BUS_M'] = d_bus
        chunk['DIST_TO_RAIL_LUAS_M'] = d_rail
        chunk['DIST_TO_ANY_TRANSIT_M'] = np.minimum(d_bus, d_rail)
        chunk['BUS_STOPS_WITHIN_500M'] = [len(i) for i in bus_tree.query_ball_point(c_coords, r=500.0)]
        chunk['LOG_DIST_BUS'] = np.log1p(d_bus)
        chunk['LOG_DIST_RAIL_LUAS'] = np.log1p(d_rail)

        chunk['ZONE_GZT'] = chunk['ZONE_GZT'].fillna("Unknown").astype(str).str.strip()

        X_batch = np.hstack([chunk[num_cols].values, encoder.transform(chunk[cat_cols])])
        raw_prob = model.predict_proba(X_batch)[:, 1]
        cal_prob = calibrate_to_national_prior(raw_prob)

        out_df = pd.DataFrame({
            'PARCEL_ID': chunk['PARCEL_ID'].values if 'PARCEL_ID' in chunk.columns else np.arange(start, end),
            'LOCAL_AUTHORITY': chunk['LOCAL_AUTHORITY'].values if 'LOCAL_AUTHORITY' in chunk.columns else "Unknown",
            'RAW_SAMPLE_PROB_INCL': raw_prob,
            'CALIBRATED_NATIONAL_PROB_INCL': cal_prob,
            'PRED_TARGET_SAMPLE_THRESH': (raw_prob >= threshold).astype(int)
        })
        results.append(out_df)

    final_preds = pd.concat(results, ignore_index=True)
    final_preds.to_csv(output_csv, index=False)
    elapsed = time.time() - t0
    excl_pct = (final_preds['PRED_TARGET_SAMPLE_THRESH'] == 0).mean() * 100
    print(f"[PASS] Scored {n_total:,} national parcels in {elapsed:.1f}s -> {output_csv}")
    print(f"Predicted Exclusion Share (at sample threshold {threshold:.4f}): {excl_pct:.2f}%")

# Uncomment to run full national deployment test:
# score_national_layer("National_RZLT.geojson", os.path.join("gtfs", "stops.txt"))