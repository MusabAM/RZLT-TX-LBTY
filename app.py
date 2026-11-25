import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely import force_2d
import streamlit as st
import pydeck as pdk
import shap
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PAGE CONFIG & HEADER
# ==============================================================================
st.set_page_config(
    page_title="Ireland RZLT Statutory Outcome Explorer",
    page_icon="🇮🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇮🇪 Ireland RZLT Statutory Outcome & Explainability Dashboard")
st.caption(
    "Policy-Only XGBoost Classifier | Switch between the 10:1 Stratified Audit Sample (6,303 parcels) "
    "and the Full National RZLT Layer (~260,463 parcels)."
)

# ==============================================================================
# 2. CACHED MODEL & SPATIAL INDEX LOADERS
# ==============================================================================
@st.cache_resource
def load_model_bundle():
    model_path = os.path.join("Models", "RZLT_PolicyOnly_FinalModel.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model artifact not found at `{model_path}`. Run Notebook 04 first.")
        st.stop()
    bundle = joblib.load(model_path)
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer

def calibrate_to_national_prior(p_sample, pi_train=0.9148, pi_real=0.9977):
    eps = 1e-7
    p = np.clip(p_sample, eps, 1.0 - eps)
    num = p * (pi_real / pi_train)
    den = num + (1.0 - p) * ((1.0 - pi_real) / (1.0 - pi_train))
    return num / den

@st.cache_data(show_spinner="Loading 10:1 Stratified Feature Dataset...")
def load_sample_dataset(sample_path):
    gdf = gpd.read_file(sample_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=2157)
    elif gdf.crs.to_epsg() != 2157:
        gdf = gdf.to_crs(epsg=2157)
    gdf["geometry"] = force_2d(gdf.geometry)
    return gdf

@st.cache_data(show_spinner="Loading Full National RZLT Layer (~260k parcels) & Spatial Trees...")
def load_full_national_base(national_path, gtfs_path):
    gdf = gpd.read_file(national_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=2157)
    elif gdf.crs.to_epsg() != 2157:
        gdf = gdf.to_crs(epsg=2157)

    gdf["geometry"] = force_2d(gdf.geometry)
    gdf["SITE_AREA"] = gdf.geometry.area / 10000.0
    gdf = gdf[gdf["SITE_AREA"] >= 0.005].copy().reset_index(drop=True)

    # Map statutory target from ZONE_SYMBOLOGY
    symb = gdf["ZONE_SYMBOLOGY"].astype(str).str.strip()
    gdf["target"] = np.where(symb.isin(["3", "4"]), 0, 1)
    gdf["DECISION_STATUS"] = np.where(gdf["target"] == 0, "Excluded", "Included_Liable")

    # Build national centroid array for true 500m & 1000m density lookups
    centroids = gdf.geometry.centroid
    all_coords = np.column_stack([centroids.x, centroids.y])

    # Load GTFS stops in EPSG:2157
    stops = pd.read_csv(gtfs_path)
    stops_gdf = gpd.GeoDataFrame(
        stops, geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326"
    ).to_crs(epsg=2157)
    stop_coords = np.column_stack([stops_gdf.geometry.x, stops_gdf.geometry.y])
    is_rail = stops["stop_name"].str.contains("Station|Luas|DART|Rail", case=False, na=False).values

    return gdf, all_coords, stop_coords[~is_rail], stop_coords[is_rail]

def enrich_and_score_subset(gdf_subset, bundle, all_coords=None, bus_coords=None, rail_coords=None):
    df = gdf_subset.copy().reset_index(drop=True)

    # Compute features on the fly if running on raw National_RZLT subset
    if "TRUE_PARCELS_WITHIN_500M" not in df.columns:
        area_m2 = df.geometry.area
        perim_m = df.geometry.length
        df["SITE_AREA"] = area_m2 / 10000.0
        df["LOG_SITE_AREA"] = np.log1p(df["SITE_AREA"])
        df["SHAPE_COMPACTNESS"] = (4 * np.pi * area_m2) / np.maximum(perim_m ** 2, 1e-6)
        df["PERIM_TO_AREA"] = perim_m / np.maximum(area_m2, 1e-6)

        sub_cent = df.geometry.centroid
        sub_coords = np.column_stack([sub_cent.x, sub_cent.y])

        nat_tree = cKDTree(all_coords)
        df["TRUE_PARCELS_WITHIN_500M"] = [max(0, len(i) - 1) for i in nat_tree.query_ball_point(sub_coords, r=500.0)]
        df["TRUE_PARCELS_WITHIN_1000M"] = [max(0, len(i) - 1) for i in nat_tree.query_ball_point(sub_coords, r=1000.0)]
        df["LOG_TRUE_PARCELS_500M"] = np.log1p(df["TRUE_PARCELS_WITHIN_500M"])

        bus_tree = cKDTree(bus_coords)
        rail_tree = cKDTree(rail_coords)
        d_bus, _ = bus_tree.query(sub_coords, k=1)
        d_rail, _ = rail_tree.query(sub_coords, k=1)
        df["DIST_TO_BUS_M"] = d_bus
        df["DIST_TO_RAIL_LUAS_M"] = d_rail
        df["DIST_TO_ANY_TRANSIT_M"] = np.minimum(d_bus, d_rail)
        df["BUS_STOPS_WITHIN_500M"] = [len(i) for i in bus_tree.query_ball_point(sub_coords, r=500.0)]
        df["LOG_DIST_BUS"] = np.log1p(d_bus)
        df["LOG_DIST_RAIL_LUAS"] = np.log1p(d_rail)

    df["ZONE_GZT"] = df["ZONE_GZT"].fillna("Unknown").astype(str).str.strip()
    if "LOCAL_AUTHORITY" in df.columns:
        df["LOCAL_AUTHORITY"] = df["LOCAL_AUTHORITY"].fillna("Unknown").astype(str).str.strip()
    else:
        df["LOCAL_AUTHORITY"] = "Unknown"

    if "PARCEL_ID" not in df.columns:
        df["PARCEL_ID"] = [f"PARCEL_{i:06d}" for i in range(len(df))]

    num_cols = bundle["num_cols"]
    cat_cols = bundle["cat_cols"]
    X_mat = np.hstack([df[num_cols].values, bundle["encoder"].transform(df[cat_cols])])

    raw_probs = bundle["model"].predict_proba(X_mat)[:, 1]
    cal_probs = calibrate_to_national_prior(raw_probs)

    df["PRED_INCL_PROB"] = np.round(raw_probs, 4)
    df["NAT_CALIBRATED_PROB"] = np.round(cal_probs, 4)

    # Project to EPSG:4326 for PyDeck
    df_4326 = df.to_crs(epsg=4326)
    cent_4326 = df_4326.geometry.centroid
    df_4326["lon"] = cent_4326.x
    df_4326["lat"] = cent_4326.y

    r = ((1.0 - raw_probs) * 235 + raw_probs * 34).astype(int)
    g = ((1.0 - raw_probs) * 64 + raw_probs * 197).astype(int)
    b = ((1.0 - raw_probs) * 52 + raw_probs * 94).astype(int)
    df_4326["fill_color"] = [[int(rv), int(gv), int(bv), 210] for rv, gv, bv in zip(r, g, b)]

    return df_4326, X_mat

bundle, explainer = load_model_bundle()

# ==============================================================================
# 3. SIDEBAR: DATASET SCOPE SELECTOR & FILTERS
# ==============================================================================
st.sidebar.header("📂 Dataset Scope")
dataset_mode = st.sidebar.radio(
    "Select Dataset Resolution",
    [
        "10:1 Stratified Sample (6,303 parcels)",
        "Full National Map (~260k parcels by County)"
    ],
    help="Switch to Full National Map to view 100% of parcels inside a selected Local Authority."
)

sample_path = os.path.join("Datasets/RZLT_Samples", "RZLT_AllCodes_10to1_FullFeatures.geojson")
if not os.path.exists(sample_path):
    sample_path = "RZLT_AllCodes_10to1_FullFeatures.geojson"

national_path = "Datasets/RZLT-Dataset.geojson"
gtfs_path = os.path.join("Datasets/GTFS_All", "stops.txt")

st.sidebar.header("🔍 Spatial & Policy Filters")

if dataset_mode == "10:1 Stratified Sample (6,303 parcels)":
    raw_sample = load_sample_dataset(sample_path)
    raw_sample["LOCAL_AUTHORITY"] = raw_sample["LOCAL_AUTHORITY"].fillna("Unknown").astype(str).str.strip()
    authorities = ["All Local Authorities"] + sorted(raw_sample["LOCAL_AUTHORITY"].unique().tolist())
    selected_la = st.sidebar.selectbox("Local Authority (Evaluation Filter Only)", authorities)

    if selected_la != "All Local Authorities":
        subset_gdf = raw_sample[raw_sample["LOCAL_AUTHORITY"] == selected_la]
    else:
        subset_gdf = raw_sample

    scored_gdf, X_active = enrich_and_score_subset(subset_gdf, bundle)

else:
    if not os.path.exists(national_path) or not os.path.exists(gtfs_path):
        st.sidebar.error("`Datasets/RZLT-Dataset.geojson` or `Datasets/GTFS_All/stops.txt` not found in root directory.")
        st.stop()

    nat_gdf, all_coords, bus_coords, rail_coords = load_full_national_base(national_path, gtfs_path)
    nat_gdf["LOCAL_AUTHORITY"] = nat_gdf["LOCAL_AUTHORITY"].fillna("Unknown").astype(str).str.strip()
    county_list = sorted(nat_gdf["LOCAL_AUTHORITY"].unique().tolist())
    default_idx = county_list.index("KELA") if "KELA" in county_list else 0

    selected_la = st.sidebar.selectbox(
        "Select Local Authority (100% Full County Layer)",
        county_list,
        index=default_idx,
        help="Full 260k mode filters by Local Authority first so 100% of county parcels render smoothly."
    )
    subset_gdf = nat_gdf[nat_gdf["LOCAL_AUTHORITY"] == selected_la]
    scored_gdf, X_active = enrich_and_score_subset(subset_gdf, bundle, all_coords, bus_coords, rail_coords)

zoning_options = sorted(scored_gdf["ZONE_GZT"].unique().tolist())
selected_zones = st.sidebar.multiselect("Statutory Zoning (`ZONE_GZT`)", zoning_options, default=zoning_options)

status_options = sorted(scored_gdf["DECISION_STATUS"].unique().tolist())
selected_status = st.sidebar.multiselect("Actual Statutory Status", status_options, default=status_options)

custom_thresh = st.sidebar.slider(
    "Inclusion Probability Cutoff",
    min_value=0.05,
    max_value=0.95,
    value=float(round(bundle["optimal_inclusion_threshold"], 4)),
    step=0.01
)

only_outliers = st.sidebar.checkbox("Show Statutory Outliers Only (Model vs. Council Disagreements)", value=False)

# Apply attribute filters while keeping positional alignment for SHAP
mask = scored_gdf["ZONE_GZT"].isin(selected_zones) & scored_gdf["DECISION_STATUS"].isin(selected_status)
filtered_gdf = scored_gdf[mask].copy()
filtered_gdf["DYNAMIC_PRED"] = np.where(
    filtered_gdf["PRED_INCL_PROB"] >= custom_thresh, "Predicted Included", "Predicted Excluded"
)
filtered_gdf["DYNAMIC_OUTLIER"] = (
    ((filtered_gdf["target"] == 0) & (filtered_gdf["PRED_INCL_PROB"] >= custom_thresh)) |
    ((filtered_gdf["target"] == 1) & (filtered_gdf["PRED_INCL_PROB"] < custom_thresh))
)

if only_outliers:
    filtered_gdf = filtered_gdf[filtered_gdf["DYNAMIC_OUTLIER"]]

# ==============================================================================
# 4. KPI SUMMARY METRICS
# ==============================================================================
k1, k2, k3, k4 = st.columns(4)
total_shown = len(filtered_gdf)
pred_excl_cnt = int((filtered_gdf["DYNAMIC_PRED"] == "Predicted Excluded").sum())
actual_excl_cnt = int((filtered_gdf["target"] == 0).sum())
agreement_rate = (
    ((filtered_gdf["target"] == 1) & (filtered_gdf["DYNAMIC_PRED"] == "Predicted Included") |
     (filtered_gdf["target"] == 0) & (filtered_gdf["DYNAMIC_PRED"] == "Predicted Excluded")).mean() * 100
    if total_shown > 0 else 0.0
)

k1.metric("Displayed Parcels", f"{total_shown:,}")
k2.metric("Actual Excluded (Statutory)", f"{actual_excl_cnt:,}")
k3.metric("Model Predicted Excluded", f"{pred_excl_cnt:,}")
k4.metric("Policy Agreement Rate", f"{agreement_rate:.1f}%")

# ==============================================================================
# 5. MAIN TABS: MAP VIEW | PARCEL SHAP INSPECTOR | WHAT-IF SIMULATOR
# ==============================================================================
tab_map, tab_shap, tab_sim = st.tabs([
    "🗺️ Geospatial Parcel Map & Outliers",
    "🔬 Parcel SHAP Explanation",
    "🎛️ What-If Policy Simulator"
])

with tab_map:
    if total_shown == 0:
        st.warning("No parcels match the active sidebar filters.")
    else:
        center_lat = float(filtered_gdf["lat"].mean())
        center_lon = float(filtered_gdf["lon"].mean())
        zoom_level = 10.5 if selected_la != "All Local Authorities" else 6.6

        # Prioritize Excluded/Outlier parcels first if count exceeds 15,000 WebGL cap
        map_gdf = filtered_gdf.sort_values(by="PRED_INCL_PROB", ascending=True).head(15000).copy()
        if len(filtered_gdf) > 15000:
            st.info(f"Rendering top 15,000 parcels (prioritizing lowest inclusion probabilities) out of {total_shown:,} for smooth browser performance.")

        geojson_data = map_gdf[[
            "PARCEL_ID", "LOCAL_AUTHORITY", "ZONE_GZT", "DECISION_STATUS",
            "PRED_INCL_PROB", "NAT_CALIBRATED_PROB", "TRUE_PARCELS_WITHIN_500M",
            "SHAPE_COMPACTNESS", "DIST_TO_BUS_M", "fill_color", "geometry"
        ]].__geo_interface__

        poly_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_data,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color="properties.fill_color",
            line_width_min_pixels=2,
        )

        point_df = map_gdf.drop(columns=["geometry"]).copy()
        point_layer = pdk.Layer(
            "ScatterplotLayer",
            data=point_df,
            get_position=["lon", "lat"],
            get_fill_color="fill_color",
            get_radius=45,
            radius_min_pixels=3,
            radius_max_pixels=10,
            pickable=True,
        )

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom_level, pitch=0)
        tooltip = {
            "html": (
                "<b>Parcel ID:</b> {PARCEL_ID}<br/>"
                "<b>County:</b> {LOCAL_AUTHORITY} | <b>Zoning:</b> {ZONE_GZT}<br/>"
                "<b>Actual Status:</b> {DECISION_STATUS}<br/>"
                "<b>Sample Incl. Prob:</b> {PRED_INCL_PROB}<br/>"
                "<b>Calibrated National Prob:</b> {NAT_CALIBRATED_PROB}<br/>"
                "<b>Parcels within 500m:</b> {TRUE_PARCELS_WITHIN_500M}<br/>"
                "<b>Compactness:</b> {SHAPE_COMPACTNESS} | <b>Bus (m):</b> {DIST_TO_BUS_M}"
            ),
            "style": {"backgroundColor": "#1e293b", "color": "white", "fontSize": "12px"}
        }

        st.pydeck_chart(pdk.Deck(layers=[poly_layer, point_layer], initial_view_state=view_state, tooltip=tooltip))
        st.caption("Color Legend: Red = High Exclusion Probability | Green = High Inclusion Probability")

        st.subheader("📋 Filtered Parcel & Statutory Outlier Table")
        display_cols = [
            "PARCEL_ID", "LOCAL_AUTHORITY", "ZONE_GZT", "SITE_AREA",
            "SHAPE_COMPACTNESS", "TRUE_PARCELS_WITHIN_500M", "DIST_TO_BUS_M",
            "DECISION_STATUS", "PRED_INCL_PROB", "NAT_CALIBRATED_PROB", "DYNAMIC_OUTLIER"
        ]
        display_cols = [c for c in display_cols if c in filtered_gdf.columns]
        st.dataframe(
            filtered_gdf[display_cols].sort_values(by="PRED_INCL_PROB", ascending=True),
            use_container_width=True,
            height=320
        )

with tab_shap:
    st.subheader("Local Feature Attribution (SHAP Waterfall)")
    if total_shown == 0:
        st.warning("Adjust filters to select a parcel.")
    else:
        parcel_list = filtered_gdf["PARCEL_ID"].tolist()
        chosen_id = st.selectbox("Select Parcel ID to Explain", parcel_list)

        row_pos = scored_gdf.index[scored_gdf["PARCEL_ID"] == chosen_id][0]
        row_info = scored_gdf.loc[row_pos]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual Statutory Status", row_info["DECISION_STATUS"])
        c2.metric("Model Inclusion Prob", f"{row_info['PRED_INCL_PROB']:.4f}")
        c3.metric("National Calibrated Prob", f"{row_info['NAT_CALIBRATED_PROB']:.4f}")
        c4.metric("True 500m Neighbors", f"{int(row_info['TRUE_PARCELS_WITHIN_500M']):,}")

        x_single = X_active[row_pos:row_pos + 1]
        shap_exp = explainer(x_single)
        shap_exp.feature_names = bundle["feature_names"]

        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
        plt.title(f"SHAP Drivers for {chosen_id} ({row_info['LOCAL_AUTHORITY']} - {row_info['ZONE_GZT']})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tab_sim:
    st.subheader("Simulate Statutory Outcome Under Hypothetical Parcel Characteristics")
    col_a, col_b = st.columns(2)
    with col_a:
        sim_zone = st.selectbox("Zoning Code (`ZONE_GZT`)", zoning_options, index=0)
        sim_area = st.slider("Site Area (Hectares)", 0.01, 5.00, 0.25, step=0.01)
        sim_compact = st.slider("Polsby-Popper Shape Compactness (0–1)", 0.05, 0.95, 0.65, step=0.01)
        sim_perim_area = st.slider("Perimeter-to-Area Ratio", 0.01, 0.50, 0.08, step=0.01)

    with col_b:
        sim_p500 = st.slider("True Parcels Within 500m", 0, 2500, 650, step=10)
        sim_p1000 = st.slider("True Parcels Within 1000m", 0, 7000, 1900, step=25)
        sim_bus = st.slider("Distance to Nearest Bus Stop (m)", 5, 4000, 200, step=10)
        sim_rail = st.slider("Distance to Nearest Rail/Luas (m)", 50, 50000, 2500, step=50)
        sim_stops500 = st.slider("Bus Stops Within 500m", 0, 50, 6, step=1)

    sim_dict = {
        "SITE_AREA": sim_area,
        "LOG_SITE_AREA": np.log1p(sim_area),
        "SHAPE_COMPACTNESS": sim_compact,
        "PERIM_TO_AREA": sim_perim_area,
        "TRUE_PARCELS_WITHIN_500M": sim_p500,
        "TRUE_PARCELS_WITHIN_1000M": sim_p1000,
        "LOG_TRUE_PARCELS_500M": np.log1p(sim_p500),
        "DIST_TO_BUS_M": sim_bus,
        "DIST_TO_RAIL_LUAS_M": sim_rail,
        "DIST_TO_ANY_TRANSIT_M": min(sim_bus, sim_rail),
        "BUS_STOPS_WITHIN_500M": sim_stops500,
        "LOG_DIST_BUS": np.log1p(sim_bus),
        "LOG_DIST_RAIL_LUAS": np.log1p(sim_rail),
        "ZONE_GZT": sim_zone
    }

    sim_df = pd.DataFrame([sim_dict])
    X_sim = np.hstack([
        sim_df[bundle["num_cols"]].values,
        bundle["encoder"].transform(sim_df[bundle["cat_cols"]])
    ])

    sim_raw_prob = float(bundle["model"].predict_proba(X_sim)[0, 1])
    sim_nat_prob = float(calibrate_to_national_prior(sim_raw_prob))
    sim_decision = "INCLUDED / LIABLE" if sim_raw_prob >= custom_thresh else "EXCLUDED / REMOVED"

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Simulated Statutory Prediction", sim_decision)
    r2.metric("Sample Inclusion Probability", f"{sim_raw_prob:.4f}")
    r3.metric("Calibrated National Probability", f"{sim_nat_prob:.4f}")