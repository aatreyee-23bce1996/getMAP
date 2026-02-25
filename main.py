import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from model import NO2DownscalingModel
from utils import (
    load_satellite_data,
    load_ground_data,
    create_no2_map,
    create_comparison_plot,
    calculate_metrics,
    handle_missing_data,
    save_satellite_data,
    save_ground_measurements,
    generate_demo_data,
)
from database import init_db, get_db

# ── DB init ────────────────────────────────────────────────────────────────────
init_db()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="getMAP · Software Engineering Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #c8d6e8;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d1224 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #a8bcd4 !important; }

/* ---- Hero header ---- */
.hero {
    background: linear-gradient(135deg, #0d1224 0%, #0f2340 50%, #0a0e1a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,140,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero .subtitle {
    font-size: 1rem;
    color: #6b8aad;
    font-weight: 400;
}
.hero .badge {
    display: inline-block;
    background: rgba(0,140,255,0.15);
    border: 1px solid rgba(0,140,255,0.3);
    color: #4da6ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 8px;
}

/* ---- Cards ---- */
.card {
    background: #0d1224;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #4da6ff;
    margin-bottom: 0.8rem;
}

/* ---- Metric tiles ---- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.metric-tile {
    background: linear-gradient(135deg, #0d1a2e, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}
.metric-tile .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #4da6ff;
    line-height: 1.1;
}
.metric-tile .label {
    font-size: 0.72rem;
    color: #6b8aad;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ---- Status pill ---- */
.status-ok   { color: #39e87a; }
.status-warn { color: #f0b429; }
.status-info { color: #4da6ff; }

/* ---- Divider ---- */
hr { border-color: #1e2d4a !important; }

/* ---- Streamlit overrides ---- */
.stButton>button {
    background: linear-gradient(135deg, #0050a0, #0070cc);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.55rem 1.5rem;
    transition: all 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #0060bb, #0088ee);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,120,255,0.3);
}
[data-testid="stFileUploader"] {
    border: 2px dashed #1e3a5f;
    border-radius: 10px;
    background: #090d1a;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: #a8bcd4; }

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0050a0, #00aaff) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <span class="badge">🛰️ TROPOMI / Sentinel-5P</span>
    <span class="badge">🤖 ML Downscaling</span>
    <span class="badge">🌏 India Air Quality</span>
  </div>
  <h1>NO₂ Satellite Downscaling</h1>
  <p class="subtitle">
    AI-powered spatial resolution enhancement of tropospheric NO₂ maps  ·  Software Engineering Lab · SIH 2024 Problem Statement
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    algorithm = st.selectbox(
        "Algorithm",
        ["Random Forest", "XGBoost", "Gradient Boosting"],
        help="XGBoost usually performs best; falls back to RF if not installed."
    )
    scale_factor = st.slider("Upscaling factor", min_value=2, max_value=8, value=4, step=2,
                             help="Output resolution = input × factor")
    fill_method  = st.radio("Gap-fill method", ["Interpolate", "Mean fill"],
                            help="How to handle cloudy pixels (NaN)")
    st.markdown("---")
    use_demo = st.checkbox("🔬 Use demo data", value=False,
                           help="Generate synthetic NO₂ data to try without uploading files")
    st.markdown("---")
    st.markdown("""
<div style="font-size:0.78rem; color:#4a6480; line-height:1.6">
<b style="color:#6b8aad">Data sources</b><br>
• TROPOMI/Sentinel-5P (ESA/NASA)<br>
• OMI/Aura (NASA GES DISC)<br>
• CPCB ground stations<br><br>
<b style="color:#6b8aad">Validation</b><br>
80/20 train-test split on spatial data
</div>
""", unsafe_allow_html=True)

# ── File upload ────────────────────────────────────────────────────────────────
st.markdown('<div class="card-title">📂 DATA INPUT</div>', unsafe_allow_html=True)
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.markdown("**Satellite Data** · GeoTIFF (.tif / .tiff)")
    satellite_file = st.file_uploader("Upload satellite NO₂ GeoTIFF", type=['tif', 'tiff'],
                                      label_visibility="collapsed")

with col_up2:
    st.markdown("**Ground Station Data** · CSV")
    ground_file = st.file_uploader("Upload CPCB ground station CSV", type=['csv'],
                                   label_visibility="collapsed")
    st.caption("Expected columns: `latitude`, `longitude`, `no2_value`, `station_name`")

# ── Main logic ─────────────────────────────────────────────────────────────────
algo_map = {
    "Random Forest":    "random_forest",
    "XGBoost":          "xgboost",
    "Gradient Boosting":"gradient_boosting",
}

data       = None
transform  = None
crs        = None

if use_demo:
    st.info("🔬 Demo mode: using synthetic NO₂ data (64×64 grid).")
    data = generate_demo_data(64, 64)
elif satellite_file is not None:
    with st.spinner("Loading satellite data…"):
        data, transform, crs = load_satellite_data(satellite_file)
    if data is None:
        st.error("❌ Could not read GeoTIFF. Make sure it's a valid single-band rasterio-compatible file.")

if data is not None:
    # ── Input preview ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-title">🗺️ INPUT DATA</div>', unsafe_allow_html=True)

    nan_pct = np.isnan(data).mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid size",   f"{data.shape[0]}×{data.shape[1]}")
    c2.metric("NaN / cloudy", f"{nan_pct:.1f}%")
    c3.metric("Min NO₂",     f"{np.nanmin(data):.2e}")
    c4.metric("Max NO₂",     f"{np.nanmax(data):.2e}")

    fill = 'interpolate' if fill_method == "Interpolate" else 'mean'
    processed = handle_missing_data(data, method=fill)

    st.plotly_chart(create_no2_map(processed, "Original Coarse-Resolution NO₂"),
                    use_container_width=True)

    # ── Ground data ────────────────────────────────────────────────────────────
    if ground_file is not None:
        gdf = load_ground_data(ground_file)
        if gdf is not None:
            db = next(get_db())
            if transform:
                save_satellite_data(db, processed, transform)
            save_ground_measurements(db, gdf)
            st.success(f"✅ Saved {len(gdf)} ground station records to database.")
            with st.expander("📋 Ground station preview"):
                st.dataframe(gdf.head(20), use_container_width=True)

    # ── Train + predict ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="card-title">🤖 MODEL TRAINING & DOWNSCALING</div>',
                unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Downscaling", type="primary")

    if run_btn:
        prog = st.progress(0, text="Initialising model…")

        model = NO2DownscalingModel(algorithm=algo_map[algorithm])
        prog.progress(15, text="Preparing features…")

        with st.spinner(f"Training {algorithm} on {processed.shape[0]*processed.shape[1]:,} pixels…"):
            X_val, y_val = model.train(processed)
        prog.progress(60, text="Generating high-resolution predictions…")

        downscaled = model.predict(processed, scale_factor=scale_factor)
        prog.progress(90, text="Computing metrics…")

        y_pred_val = model.model.predict(X_val)
        metrics    = calculate_metrics(y_val, y_pred_val)
        prog.progress(100, text="Done ✅")

        # ── Metrics ────────────────────────────────────────────────────────────
        st.markdown(f"""
<div class="metric-grid">
  <div class="metric-tile">
    <div class="value">{metrics['R2']:.4f}</div>
    <div class="label">R² Score</div>
  </div>
  <div class="metric-tile">
    <div class="value">{metrics['RMSE']:.2e}</div>
    <div class="label">RMSE</div>
  </div>
  <div class="metric-tile">
    <div class="value">{metrics['MAE']:.2e}</div>
    <div class="label">MAE</div>
  </div>
  <div class="metric-tile">
    <div class="value">{metrics['Bias']:.2e}</div>
    <div class="label">Bias</div>
  </div>
</div>
""", unsafe_allow_html=True)

        if metrics['R2'] >= 0.85:
            st.success(f"🟢 Excellent model fit  (R² = {metrics['R2']:.4f})")
        elif metrics['R2'] >= 0.65:
            st.warning(f"🟡 Acceptable model fit  (R² = {metrics['R2']:.4f}) — try XGBoost or increase data size")
        else:
            st.error(f"🔴 Poor model fit (R² = {metrics['R2']:.4f}) — consider more training data")

        # ── Feature importance ─────────────────────────────────────────────────
        fi = model.get_feature_importance()
        if fi:
            with st.expander("📊 Feature importance"):
                fi_df = pd.DataFrame.from_dict(fi, orient='index', columns=['Importance'])
                fi_df = fi_df.sort_values('Importance', ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df['Importance'], y=fi_df.index,
                    orientation='h',
                    marker_color='#4da6ff',
                    marker_line_width=0
                ))
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c8d6e8'),
                    xaxis=dict(title='Importance', gridcolor='#1e2d4a'),
                    yaxis=dict(gridcolor='#1e2d4a'),
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=280,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        # ── Comparison ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title">🔬 RESOLUTION COMPARISON</div>',
                    unsafe_allow_html=True)

        st.info(f"Resolution enhanced: **{processed.shape[0]}×{processed.shape[1]}**  →  "
                f"**{downscaled.shape[0]}×{downscaled.shape[1]}** (×{scale_factor})")

        st.plotly_chart(create_comparison_plot(processed, downscaled),
                        use_container_width=True)

        st.plotly_chart(create_no2_map(downscaled,
                        f"Downscaled NO₂ Map (×{scale_factor} resolution)"),
                        use_container_width=True)

        # ── Download ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title">⬇️ EXPORT</div>', unsafe_allow_html=True)
        d_col1, d_col2 = st.columns(2)

        csv_data = pd.DataFrame(downscaled).to_csv(index=False).encode('utf-8')
        d_col1.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="no2_downscaled.csv",
            mime="text/csv"
        )

        # Metrics export
        metrics_csv = pd.DataFrame([metrics]).to_csv(index=False).encode('utf-8')
        d_col2.download_button(
            "📥 Download Metrics",
            data=metrics_csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; padding: 4rem 2rem; color: #4a6480;">
  <div style="font-size: 4rem; margin-bottom: 1rem;">🛰️</div>
  <div style="font-size: 1.2rem; font-weight: 600; color: #6b8aad; margin-bottom: 0.5rem;">
    Upload satellite data or enable demo mode to begin
  </div>
  <div style="font-size: 0.9rem;">
    Supports GeoTIFF from TROPOMI/Sentinel-5P, OMI/Aura, or Google Earth Engine exports
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.8rem; color:#4a6480; padding: 0.5rem 0">
  Software Engineering Lab · SIH 2024 PROBLEM STATEMENT · Data: TROPOMI/Sentinel-5P · CPCB Ground Stations
</div>
""", unsafe_allow_html=True)