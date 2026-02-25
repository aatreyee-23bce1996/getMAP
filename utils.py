import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy.orm import Session
from database import SatelliteData, GroundMeasurement

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_satellite_data(file_obj):
    """Load satellite GeoTIFF data. Returns (data_array, transform, crs)."""
    if not RASTERIO_AVAILABLE:
        return None, None, None
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        with rasterio.open(tmp_path) as src:
            data = src.read(1).astype(np.float32)
            data[data <= -9999] = np.nan   # common nodata sentinel
            transform = src.transform
            crs = src.crs
        os.unlink(tmp_path)
        return data, transform, crs
    except Exception as e:
        print(f"Error loading satellite data: {e}")
        return None, None, None


def load_ground_data(file_obj):
    """Load CPCB ground station CSV. Expects columns: latitude, longitude, no2_value, station_name."""
    try:
        df = pd.read_csv(file_obj)
        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        required = {'latitude', 'longitude', 'no2_value'}
        missing = required - set(df.columns)
        if missing:
            # Try common alternatives
            rename_map = {
                'lat': 'latitude', 'lon': 'longitude', 'lng': 'longitude',
                'no2': 'no2_value', 'no2_concentration': 'no2_value',
                'station': 'station_name'
            }
            df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        print(f"Error loading ground data: {e}")
        return None


def save_satellite_data(db: Session, data, transform, timestamp=None):
    """Persist satellite pixels to DB (samples to avoid huge writes)."""
    if transform is None:
        return
    if timestamp is None:
        timestamp = datetime.utcnow()

    rows, cols = data.shape
    resolution = abs(transform[0]) if transform else 1.0

    # Sample every Nth pixel to keep DB size reasonable
    step = max(1, min(rows, cols) // 50)
    entries = []
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            if not np.isnan(data[i, j]):
                lon = transform[2] + j * transform[0]
                lat = transform[5] + i * transform[4]
                entries.append(SatelliteData(
                    timestamp=timestamp,
                    latitude=float(lat),
                    longitude=float(lon),
                    no2_value=float(data[i, j]),
                    resolution=float(resolution),
                    source='TROPOMI/Sentinel-5P'
                ))
    db.bulk_save_objects(entries)
    db.commit()


def save_ground_measurements(db: Session, data: pd.DataFrame):
    """Persist ground station rows to DB."""
    entries = []
    for _, row in data.iterrows():
        try:
            entries.append(GroundMeasurement(
                timestamp=datetime.utcnow(),
                latitude=float(row.get('latitude', 0)),
                longitude=float(row.get('longitude', 0)),
                no2_value=float(row.get('no2_value', 0)),
                station_name=str(row.get('station_name', 'Unknown'))
            ))
        except Exception:
            continue
    db.bulk_save_objects(entries)
    db.commit()


def handle_missing_data(data, method='interpolate'):
    """Fill NaN gaps via linear interpolation or mean substitution."""
    df = pd.DataFrame(data)
    if method == 'interpolate':
        filled = df.interpolate(method='linear', axis=0).interpolate(method='linear', axis=1)
        result = filled.values
    else:
        result = np.nan_to_num(data, nan=float(np.nanmean(data)))
    return result.astype(np.float32)


def create_no2_map(data, title="NO2 Concentration Map", colorscale="RdYlBu_r"):
    """Interactive heatmap via Plotly."""
    fig = px.imshow(
        data,
        labels=dict(color="NO₂ (mol/m²)"),
        title=title,
        color_continuous_scale=colorscale,
        aspect='auto'
    )
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=16),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0')
    )
    return fig


def create_comparison_plot(original, downscaled):
    """Side-by-side comparison figure."""
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original (Coarse)", "Downscaled (Fine)"),
        horizontal_spacing=0.05
    )
    fig.add_trace(
        go.Heatmap(z=original, colorscale='RdYlBu_r', showscale=True,
                   colorbar=dict(x=0.45, title="mol/m²")),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=downscaled, colorscale='RdYlBu_r', showscale=True,
                   colorbar=dict(x=1.01, title="mol/m²")),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Resolution Comparison",
        title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def calculate_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MSE': float(mse),
        'RMSE': float(np.sqrt(mse)),
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(np.mean(np.abs(y_true - y_pred))),
        'Bias': float(np.mean(y_pred - y_true)),
    }


def generate_demo_data(rows=64, cols=64):
    """Generate synthetic NO2-like data for demo/testing."""
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, cols)
    y = np.linspace(0, 4 * np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    base = (np.sin(xx) * np.cos(yy) + 1) * 3e-4
    noise = np.random.normal(0, 1e-5, (rows, cols))
    # Add some urban hotspots
    for _ in range(5):
        cx, cy = np.random.randint(10, rows-10), np.random.randint(10, cols-10)
        r = np.sqrt((np.arange(rows)[:, None] - cx)**2 + (np.arange(cols)[None, :] - cy)**2)
        base += np.exp(-r**2 / 50) * 2e-4
    return (base + noise).astype(np.float32)