import geopandas as gpd
import pandas as pd

# Load your RZLT dataset
gdf_rzlt = gpd.read_file("Datasets/RZLT-Dataset.geojson")

# 1. Check unique values and counts of ZONE_SYMBOLOGY
print("=== ZONE_SYMBOLOGY Value Counts ===")
print(gdf_rzlt['ZONE_SYMBOLOGY'].value_counts(dropna=False))

# 2. Cross-tabulate ZONE_SYMBOLOGY against ZONE_GZT and STATUS
print("\n=== Cross-Tab: ZONE_SYMBOLOGY vs. ZONE_GZT ===")
print(pd.crosstab(gdf_rzlt['ZONE_SYMBOLOGY'], gdf_rzlt['ZONE_GZT'], dropna=False))

if 'STATUS' in gdf_rzlt.columns:
    print("\n=== Cross-Tab: ZONE_SYMBOLOGY vs. STATUS ===")
    print(pd.crosstab(gdf_rzlt['ZONE_SYMBOLOGY'], gdf_rzlt['STATUS'], dropna=False))

# 3. Inspect sample rows for each ZONE_SYMBOLOGY category
print("\n=== Sample Attributes per Symbology Code ===")
cols_to_check = [c for c in ['PARCEL_ID', 'ZONE_SYMBOLOGY', 'ZONE_GZT', 'ZONE_ORIG', 'STATUS', 'LIVE_SITE', 'SITE_AREA'] if c in gdf_rzlt.columns]
print(gdf_rzlt.groupby('ZONE_SYMBOLOGY')[cols_to_check].head(2))