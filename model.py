import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class NO2DownscalingModel:
    def __init__(self, algorithm='random_forest'):
        """
        algorithm: 'random_forest', 'xgboost', 'gradient_boosting'
        """
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}

        if algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        else:
            # Default: Random Forest
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

    def prepare_features(self, data):
        """
        Prepare spatial + spectral features for ML model.
        Features: normalized lat, normalized lon, NO2 value,
                  local mean, local std (3x3 window), row gradient, col gradient
        """
        rows, cols = data.shape

        # Compute local statistics using sliding window approximation
        from scipy.ndimage import uniform_filter, generic_filter
        local_mean = uniform_filter(np.nan_to_num(data, nan=np.nanmean(data)), size=3)
        local_std  = generic_filter(np.nan_to_num(data, nan=np.nanmean(data)),
                                    np.std, size=3)

        # Gradients
        grad_row, grad_col = np.gradient(np.nan_to_num(data, nan=np.nanmean(data)))

        X, y = [], []
        for i in range(rows):
            for j in range(cols):
                if not np.isnan(data[i, j]):
                    X.append([
                        i / rows,
                        j / cols,
                        data[i, j],
                        local_mean[i, j],
                        local_std[i, j],
                        grad_row[i, j],
                        grad_col[i, j],
                    ])
                    y.append(data[i, j])

        return np.array(X), np.array(y)

    def train(self, data):
        """Train the model on input data. Returns (X_val, y_val) for metrics."""
        X, y = self.prepare_features(data)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Training metrics
        y_pred_val = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        self.training_metrics = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2': r2_score(y_val, y_pred_val),
        }

        return X_val, y_val

    def predict(self, data, scale_factor=4):
        """
        Generate high-resolution predictions at scale_factor x original resolution.
        Uses bicubic upsampling for initial grid, then ML refinement.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        rows, cols = data.shape
        new_rows = rows * scale_factor
        new_cols = cols * scale_factor

        # Initial upscaling via bicubic interpolation
        upscaled = zoom(np.nan_to_num(data, nan=np.nanmean(data)),
                        scale_factor, order=3)

        # Prepare features on the high-res grid
        from scipy.ndimage import uniform_filter, generic_filter
        local_mean = uniform_filter(upscaled, size=3)
        local_std  = generic_filter(upscaled, np.std, size=3)
        grad_row, grad_col = np.gradient(upscaled)

        grid_i, grid_j = np.meshgrid(
            np.linspace(0, 1, new_rows),
            np.linspace(0, 1, new_cols),
            indexing='ij'
        )

        X_pred = np.column_stack([
            grid_i.ravel(),
            grid_j.ravel(),
            upscaled.ravel(),
            local_mean.ravel(),
            local_std.ravel(),
            grad_row.ravel(),
            grad_col.ravel(),
        ])

        X_pred_scaled = self.scaler.transform(X_pred)
        predictions = self.model.predict(X_pred_scaled)

        return predictions.reshape(new_rows, new_cols)

    def get_feature_importance(self):
        """Return feature importance if supported by the model."""
        feature_names = [
            'Norm. Row', 'Norm. Col', 'NO2 Value',
            'Local Mean', 'Local Std', 'Row Gradient', 'Col Gradient'
        ]
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))
        return {}