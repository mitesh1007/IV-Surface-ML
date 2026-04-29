"""
gp_surface.py
Gaussian Process Regression for IV surface fitting.
Kernel: ConstantKernel * RBF(anisotropic) + WhiteKernel
Features: (log(K/S), T)
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class IVSurfaceGP:
    def __init__(self, n_restarts=5):
        self.kernel = (
            C(1.0, (1e-3, 1e3)) *
            RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 10.0)) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=n_restarts,
            normalize_y=True, alpha=1e-6
        )
        self.scaler_X = StandardScaler()
        self.is_fitted = False

    def _features(self, df):
        underlying_map = {"SPY": 0.0, "QQQ": 0.5, "IWM": 1.0}
        if "underlying" in df.columns:
            u = df["underlying"].map(underlying_map).fillna(0.5).values
        else:
            u = np.full(len(df), 0.0)  # default to SPY for grid prediction
        return np.column_stack([np.log(df["moneyness"].values), df["tte"].values, u])

    def fit(self, df):
        X, y = self._features(df), df["iv"].values
        if len(X) > 500:
            idx = np.random.choice(len(X), 500, replace=False)
            X, y = X[idx], y[idx]
            print(f"[GP] Subsampled to 500 points")
        X_s = self.scaler_X.fit_transform(X)
        print(f"[GP] Fitting on {len(X)} points...")
        self.gp.fit(X_s, y)
        self.is_fitted = True
        print(f"[GP] Train R²: {self.gp.score(X_s, y):.4f}")
        return self

    def predict(self, df, return_std=True):
        assert self.is_fitted
        X_s = self.scaler_X.transform(self._features(df))
        if return_std:
            iv, std = self.gp.predict(X_s, return_std=True)
            return np.clip(iv, 0.01, 5.0), std
        return np.clip(self.gp.predict(X_s), 0.01, 5.0)

    def predict_grid(self, moneyness_grid, tte_grid):
        assert self.is_fitted
        M, T = np.meshgrid(moneyness_grid, tte_grid, indexing="ij")
        grid_df = pd.DataFrame({"moneyness": M.ravel(), "tte": T.ravel()})
        iv, std = self.predict(grid_df, return_std=True)
        return iv.reshape(M.shape), std.reshape(M.shape)

    def evaluate(self, df):
        iv_pred = self.predict(df, return_std=False)
        iv_true = df["iv"].values
        mae  = np.mean(np.abs(iv_pred - iv_true))
        rmse = np.sqrt(np.mean((iv_pred - iv_true)**2))
        r2   = 1 - np.sum((iv_true - iv_pred)**2) / np.sum((iv_true - iv_true.mean())**2)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
