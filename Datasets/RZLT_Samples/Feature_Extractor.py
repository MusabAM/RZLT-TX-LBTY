import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import OneHotEncoder

# ── 1. LOAD YOUR FILE ──────────────────────────────────────────────────────────
INPUT_FILE = "Datasets/RZLT_GZT_Merged/CWLA_Merged.geojson"  # ← change this

gdf = gpd.read_file(INPUT_FILE).to_crs(epsg=2157)

# ── 2. DETECT WHETHER GZT-MERGED OR RZLT-ONLY ─────────────────────────────────
is_merged = gdf['LIVE_SITE'].nunique() > 1  # True if both Y and N exist
print(f"File type: {'GZT-Merged (Y+N)' if is_merged else 'RZLT-only (Y only)'}")
print(gdf['LIVE_SITE'].value_counts())

if not is_merged:
    print("⚠️  Warning: Only one class found. Classification requires both Y and N.")

# ── 3. TRANSIT DISTANCE FEATURE ───────────────────────────────────────────────
stops_df = pd.read_csv("Datasets/GTFS_All/stops.txt")
stops_gdf = gpd.GeoDataFrame(
    stops_df,
    geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
    crs="EPSG:4326"
).to_crs(epsg=2157)

stop_coords   = np.column_stack([stops_gdf.geometry.x, stops_gdf.geometry.y])
parcel_coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])

distances, _ = cKDTree(stop_coords).query(parcel_coords)
gdf['DISTANCE_TO_TRANSIT'] = distances

# ── 4. SITE AREA (ensure it exists) ───────────────────────────────────────────
if 'SITE_AREA' not in gdf.columns or gdf['SITE_AREA'].isnull().any():
    gdf['SITE_AREA'] = gdf.geometry.area / 10000.0

# ── 5. ENCODE FEATURES ────────────────────────────────────────────────────────
num_cols = ['SITE_AREA', 'DISTANCE_TO_TRANSIT']
cat_cols = ['ZONE_GZT', 'LOCAL_AUTHORITY']

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded      = ohe.fit_transform(gdf[cat_cols])
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))

X = np.hstack([gdf[num_cols].values, cat_encoded])
y = (gdf['LIVE_SITE'] == 'Y').astype(int).values
feature_names = num_cols + cat_feature_names

print(f"\nFeature matrix shape: {X.shape}")
print(f"Class distribution  → Y: {y.sum()}, N: {(y==0).sum()}")