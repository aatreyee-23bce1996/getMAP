# getMAP 🛰️

> **AI-powered spatial downscaling of tropospheric NO₂ satellite maps**  
> Software Engineering Lab · BCSE301P · SIH Problem Statement

[![Python](https://img.shields.io/badge/Python-3.11+-3670A0?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## What is getMAP?

getMAP is a full-stack ML application that takes **coarse-resolution satellite NO₂ data** (3.5 km from TROPOMI/Sentinel-5P or 13 km from OMI/Aura) and generates **fine-resolution air quality maps** using machine learning — up to 8× sharper than the raw input.

It addresses a real gap: while individual tools exist for satellite processing and ML modelling, no comprehensive, validated, end-to-end solution exists for NO₂ downscaling. getMAP closes that gap.

**Problem statement:** SIH — Downscaling of Satellite-based Air Quality Maps using AI/ML  
**Desired output:** Fine spatial resolution tropospheric NO₂ map of India, validated against CPCB ground station data.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration (.env)](#configuration)
6. [Running the App](#running-the-app)
7. [App Walkthrough](#app-walkthrough)
8. [Data Sources](#data-sources)
9. [ML Algorithms](#ml-algorithms)
10. [Project Structure](#project-structure)
11. [Validation & Metrics](#validation--metrics)
12. [Export & Results](#export--results)

---

## Features

- **Three ML algorithms** — Random Forest, XGBoost, Gradient Boosting; switchable from the UI
- **Cloudy-pixel gap filling** — spatial interpolation or mean-fill before training
- **7-dimensional feature engineering** — spatial coordinates, local mean/std (3×3 window), row & column gradients
- **Up to 8× resolution enhancement** — bicubic pre-upsampling + ML refinement
- **Interactive Plotly maps** — zoom, pan, hover tooltips on both original and downscaled grids
- **Side-by-side comparison** — original coarse vs. downscaled fine resolution on the same colour scale
- **Feature importance chart** — understand which spatial features drive predictions
- **CPCB ground truth validation** — upload CSV from CPCB Advanced Search; records saved to database
- **Metrics dashboard** — MSE, RMSE, MAE, R², Bias with colour-coded quality indicator
- **One-click CSV export** — download the downscaled map and metrics for your report
- **Demo mode** — try everything without uploading any data

---

## Architecture

```
getMAP/
├── main.py          ← Streamlit frontend + orchestration
├── model.py         ← ML model (RF / XGBoost / GBM) with feature engineering
├── utils.py         ← Data I/O, visualisation, metrics
├── database.py      ← SQLAlchemy ORM (SQLite by default)
├── index.html       ← Standalone presentation/demo landing page
├── styles.css       ← Streamlit custom CSS (auto-loaded)
├── pyproject.toml   ← Dependency manifest
└── .env             ← Your local config (not committed to git)
```

**Pipeline flow:**

```
GeoTIFF upload          CPCB CSV upload
      │                       │
      ▼                       ▼
 load_satellite_data    load_ground_data
      │                       │
      ▼                       ▼
 handle_missing_data    save_ground_measurements
 (gap fill NaN/clouds)
      │
      ▼
 NO2DownscalingModel.train()
 ├── prepare_features()  → 7-dim feature vectors
 ├── StandardScaler      → normalise features
 └── RF / XGB / GBM fit on 80% of valid pixels
      │
      ▼
 NO2DownscalingModel.predict(scale_factor)
 ├── bicubic zoom to target resolution
 └── ML refinement on fine-resolution grid
      │
      ▼
 calculate_metrics()    create_comparison_plot()
      │                       │
      ▼                       ▼
  Metrics display        Plotly heatmaps
      │
      ▼
  CSV download
```

---

## Prerequisites

- Python **3.11 or newer**
- pip (comes with Python)
- ~500 MB disk space (for dependencies)
- Internet connection for first install

Optional but recommended: [Google Earth Engine account](https://earthengine.google.com/) for downloading TROPOMI GeoTIFFs directly.

---

## Installation

**Do not run `pip install .`** — the flat project layout causes setuptools to error. Install dependencies directly instead.

### Step 1 — Clone / download the project

```bash
# If using git
git clone https://github.com/your-username/getmap.git
cd getmap

# Or just unzip and navigate to the folder
cd C:\Users\nanda\Downloads\swelab
```

### Step 2 — (Optional but recommended) Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install streamlit numpy pandas plotly scikit-learn xgboost scipy rasterio sqlalchemy python-dotenv matplotlib Pillow
```

This installs everything getMAP needs. The full list is also in `pyproject.toml` for reference.

---

## Configuration

Create a file named **`.env`** in the root project folder (same folder as `main.py`):

```
DATABASE_URL=sqlite:///./getmap.db
```

That's the only line you need. SQLite creates the database file automatically on first run.

---

## Running the App

From inside the project folder with your virtual environment active:

```bash
streamlit run main.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser. The app is ready.

> **Presentation page:** Open `index.html` directly in any browser (double-click the file) for a standalone landing page that describes the project — useful for demos and lab presentations without needing the Streamlit server running.

---

## App Walkthrough

### 1. Sidebar — Configure the model

On the left side of the app you'll find three controls:

| Control | What it does |
|---|---|
| **Algorithm** | Choose between Random Forest, XGBoost (recommended), or Gradient Boosting |
| **Upscaling factor** | How much finer the output is. 4× means a 64×64 grid becomes 256×256 |
| **Gap-fill method** | How to handle cloudy pixels (NaN). Interpolate is usually better |
| **Use demo data** | Generates synthetic NO₂ data so you can try everything without uploading |

---

### 2. Upload your data

The main area has two upload boxes:

**Left box — Satellite Data (GeoTIFF)**  
Upload a `.tif` or `.tiff` file exported from TROPOMI/Sentinel-5P or OMI/Aura. Single-band NO₂ column density. See [Data Sources](#data-sources) for where to get this.

**Right box — Ground Station Data (CSV)**  
Upload a CSV downloaded from the CPCB Advanced Search. The file needs these columns (case-insensitive): `latitude`, `longitude`, `no2_value`, `station_name`. Extra columns are ignored.

> **No data yet?** Tick **"Use demo data"** in the sidebar to generate a synthetic 64×64 NO₂ grid and see the full pipeline in action.

---

### 3. Input data preview

Once data loads, you'll see:

- Four stat tiles showing **grid size**, **% cloudy pixels**, **min NO₂**, and **max NO₂**
- An interactive **Plotly heatmap** of the original coarse-resolution input
- If you uploaded ground station CSV, a confirmation message and a preview table

---

### 4. Run downscaling

Click the **"🚀 Run Downscaling"** button. A progress bar tracks the stages:

1. Model initialisation
2. Feature preparation (7 spatial features per valid pixel)
3. Training on 80% of pixels (the held-out 20% is used for validation)
4. High-resolution prediction on the upscaled grid
5. Metric computation

---

### 5. Metrics dashboard

After training, four metric tiles appear:

| Metric | Good range | Meaning |
|---|---|---|
| **R² Score** | ≥ 0.85 | Fraction of variance explained. Closer to 1.0 is better |
| **RMSE** | As low as possible | Root mean squared error in NO₂ units |
| **MAE** | As low as possible | Mean absolute error — less sensitive to outliers than RMSE |
| **Bias** | Near 0 | Systematic over/under-prediction |

A colour-coded banner tells you at a glance: 🟢 Excellent (R² ≥ 0.85), 🟡 Acceptable (≥ 0.65), 🔴 Poor (< 0.65).

---

### 6. Feature importance

Click the **"📊 Feature importance"** expander to see which of the 7 input features the model relied on most. For NO₂ data, `NO2 Value` and `Local Mean` typically dominate, which makes physical sense — nearby pixel values are the strongest predictor of a pixel's fine-resolution value.

---

### 7. Resolution comparison

A side-by-side Plotly figure shows the **original coarse** grid (left) next to the **downscaled fine** grid (right) on identical colour scales. Both are interactive — zoom into an urban area to see the sharpening clearly.

Below it, a full-width view of the downscaled map with the chosen upscaling factor in the title.

---

### 8. Export results

Two download buttons appear at the bottom:

- **📥 Download CSV** — the downscaled NO₂ grid as a CSV matrix (rows × columns)
- **📥 Download Metrics** — MSE, RMSE, MAE, R², Bias in a single-row CSV for your lab report

---

## Data Sources

### Satellite NO₂ (pick one)

| Source | Resolution | Format | Link |
|---|---|---|---|
| TROPOMI/Sentinel-5P (NASA Earthdata) | 3.5 km | HDF5 swath | [Earthdata search](https://search.earthdata.nasa.gov/search/granules?p=C2089270961-GES_DISC&pg[0][v]=f&pg[0][gsk]=-start_date&q=tropomi%20no2&tl=1726635700.002!3!!) |
| TROPOMI/Sentinel-5P (Google Earth Engine) | 3.5 km | GeoTIFF ✅ | [GEE catalogue](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2#description) |
| OMI/Aura (NASA Earthdata) | 13 km | HDF4 gridded | [Earthdata search](https://search.earthdata.nasa.gov/search/granules?p=C1266136111-GES_DISC&pg[0][v]=f&pg[0][gsk]=start_date&q=omi%20tropospheric%20no2&tl=1726635700.002!3!!) |
| OMI/Aura MINDS (direct download) | 13 km | NetCDF gridded | [GES DISC](https://measures.gesdisc.eosdis.nasa.gov/data/MINDS/OMI_MINDS_NO2d.1.1/2024/) |

### Ground station NO₂ (CPCB)

1. Go to [CPCB CCMS](https://app.cpcbccr.com/ccr/#/caaqm-dashboard-all/caaqmlanding)
2. Click **Advanced Search**
3. Select parameter: **NO2**, frequency: **Daily**
4. Choose your date range and stations
5. Download as CSV
6. Rename columns to `latitude`, `longitude`, `no2_value`, `station_name` if needed

---

## ML Algorithms

getMAP implements three algorithms, all selectable from the sidebar:

### Random Forest (default fallback)
- 200 trees, max depth 12
- Robust to satellite data outliers and cloud-gap noise
- Provides feature importance out of the box
- No extra installation needed

### XGBoost ⭐ Recommended
- 200 estimators, max depth 6, learning rate 0.1
- Sub-sampling regularisation reduces overfitting
- Consistently best R² on spatial regression tasks
- Requires `xgboost` package (included in install command above)

### Gradient Boosting
- 150 estimators, max depth 5, learning rate 0.1
- Sequential tree building reduces model bias
- Good middle ground — slower to train but no extra dependency

### Features used by all models

| Feature | Description |
|---|---|
| Normalised row | Pixel's y-position as fraction of grid height |
| Normalised col | Pixel's x-position as fraction of grid width |
| NO₂ value | Raw input NO₂ column density |
| Local mean | Mean of 3×3 neighbourhood window |
| Local std | Standard deviation of 3×3 neighbourhood |
| Row gradient | Spatial rate of change in y direction |
| Col gradient | Spatial rate of change in x direction |

---

## Project Structure

```
getmap/
│
├── main.py              # Streamlit app — UI, upload handling, orchestration
├── model.py             # NO2DownscalingModel class
│   ├── __init__         # Initialise chosen algorithm + StandardScaler
│   ├── prepare_features # Build 7-dim feature matrix from 2D NO₂ array
│   ├── train            # 80/20 split → fit → return val set
│   ├── predict          # Bicubic upsample → ML refine → return fine grid
│   └── get_feature_importance
│
├── utils.py             # Helper functions
│   ├── load_satellite_data    # rasterio GeoTIFF reader
│   ├── load_ground_data       # CPCB CSV reader with column normalisation
│   ├── handle_missing_data    # NaN gap-fill (interpolate or mean)
│   ├── save_satellite_data    # Sampled pixel persist to DB
│   ├── save_ground_measurements
│   ├── create_no2_map         # Plotly heatmap
│   ├── create_comparison_plot # Side-by-side Plotly figure
│   ├── calculate_metrics      # MSE, RMSE, MAE, R², Bias
│   └── generate_demo_data     # Synthetic NO₂ for testing
│
├── database.py          # SQLAlchemy models + session factory
│   ├── SatelliteData    # Table: sampled satellite pixels
│   └── GroundMeasurement # Table: CPCB station readings
│
├── index.html           # Standalone landing page (open in browser directly)
├── styles.css           # Custom Streamlit CSS (dark theme)
├── pyproject.toml       # Project metadata and dependency list
└── .env                 # Your local environment variables (create this yourself)
```

---

## Validation & Metrics

getMAP uses a strict **80/20 spatial train-test split**. The 20% held-out pixels are never seen during training — this satisfies the SIH requirement of validating on "unseen independent data."

**Interpreting R²:**

| R² | Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent — model captures spatial structure well |
| 0.75 – 0.90 | Good — suitable for most research applications |
| 0.65 – 0.75 | Acceptable — consider more training data or XGBoost |
| < 0.65 | Poor — likely insufficient spatial variation in input data |

---

## Export & Results

After running downscaling, two files can be downloaded from the app:

**`no2_downscaled.csv`** — The high-resolution NO₂ grid as a matrix. Each row is a latitude slice, each column is a longitude slice. Values are in mol/m² (same units as input).

**`model_metrics.csv`** — A single row with columns: `MSE`, `RMSE`, `MAE`, `R2`, `Bias`. Paste this directly into your lab report.

Both files are also auto-saved to the SQLite database (`getmap.db`) during each session.

---

## Team

**SWELAB · BCSE301P** — VIT  
Built for Software Engineering Lab, problem statement taken from SIH 2024: *Downscaling of Satellite-based Air Quality Maps using AI/ML*

---

*Data: ESA/NASA TROPOMI Sentinel-5P · OMI/Aura GES DISC · CPCB CCMS India*