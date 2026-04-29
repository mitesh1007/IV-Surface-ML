"""
setup_project.py
----------------
Run this ONCE in your project folder:
    python setup_project.py

Creates all directories and writes all source files automatically.
"""

import os

# ── Create directories ───────────────────────────────────────────────────────
dirs = ["data", "iv", "models", "arbitrage", "viz", "outputs"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "__init__.py"), "w").close()
print("✓ Directories created")

# ── Write each file ──────────────────────────────────────────────────────────

files = {}

# ════════════════════════════════════════════════════════════════════════════
files["data/fetch_options.py"] = '''"""
fetch_options.py
Pulls SPY options chain from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def fetch_spy_options(ticker: str = "SPY"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    S = float(hist["Close"].iloc[-1])
    print(f"[Data] {ticker} spot price: ${S:.2f}")

    try:
        tbill = yf.Ticker("^IRX")
        r = float(tbill.history(period="1d")["Close"].iloc[-1]) / 100.0
    except Exception:
        r = 0.05
    print(f"[Data] Risk-free rate: {r*100:.2f}%")

    expirations = stock.options
    today = datetime.today()
    records = []

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        tte = (exp_date - today).days / 365.0
        if tte < 7 / 365 or tte > 1.0:
            continue
        try:
            chain = stock.option_chain(exp_str)
        except Exception:
            continue
        for option_type, df_opt in [("call", chain.calls), ("put", chain.puts)]:
            df_opt = df_opt.copy()
            df_opt["option_type"] = option_type
            df_opt["expiry"] = exp_str
            df_opt["tte"] = tte
            records.append(df_opt)

    if not records:
        raise RuntimeError("No options data fetched.")

    df = pd.concat(records, ignore_index=True)
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df = df[df["volume"] > 0]
    df = df[df["bid"] > 0]
    df = df[df["mid_price"] > 0.05]
    df["moneyness"] = df["strike"] / S
    df = df[(df["moneyness"] >= 0.8) & (df["moneyness"] <= 1.2)]
    df = df[["strike","expiry","tte","mid_price","option_type",
             "moneyness","volume","impliedVolatility"]].copy()
    df.rename(columns={"impliedVolatility": "yf_iv"}, inplace=True)
    df = df.dropna(subset=["strike","tte","mid_price"]).reset_index(drop=True)

    print(f"[Data] Contracts after filtering: {len(df)}")
    print(f"[Data] Expirations: {df[\'expiry\'].nunique()}")
    return S, r, df


if __name__ == "__main__":
    S, r, df = fetch_spy_options()
    print(df.head())
'''

# ════════════════════════════════════════════════════════════════════════════
files["iv/extractor.py"] = '''"""
extractor.py
Black-Scholes IV extraction via Brent\'s root-finding method.

BS Call: C = S*N(d1) - K*e^(-rT)*N(d2)
d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
d2 = d1 - σ*√T
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from tqdm import tqdm


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def bs_price(S, K, T, r, sigma, option_type):
    return bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)


def extract_iv(market_price, S, K, T, r, option_type,
               sigma_low=1e-4, sigma_high=10.0):
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "call" \
                else max(K * np.exp(-r * T) - S, 0.0)
    if market_price <= intrinsic:
        return None
    objective = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
    try:
        if objective(sigma_low) * objective(sigma_high) > 0:
            return None
        iv = brentq(objective, sigma_low, sigma_high, xtol=1e-6, maxiter=500)
        return iv if 0.01 <= iv <= 3.0 else None
    except (ValueError, RuntimeError):
        return None


def extract_iv_surface(df, S, r):
    ivs = []
    print(f"[IV] Extracting IV for {len(df)} contracts...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Brent root-finding"):
        ivs.append(extract_iv(row["mid_price"], S, row["strike"],
                               row["tte"], r, row["option_type"]))
    df = df.copy()
    df["iv"] = ivs
    before = len(df)
    df = df.dropna(subset=["iv"])
    print(f"[IV] Success: {len(df)}/{before} | Range: {df[\'iv\'].min():.1%}–{df[\'iv\'].max():.1%}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    true_sigma = 0.20
    price = bs_call(S, K, T, r, true_sigma)
    recovered = extract_iv(price, S, K, T, r, "call")
    print(f"True σ: {true_sigma:.4f} | Recovered σ: {recovered:.4f}")
    assert abs(recovered - true_sigma) < 1e-5
    print("Sanity check passed ✓")
'''

# ════════════════════════════════════════════════════════════════════════════
files["models/gp_surface.py"] = '''"""
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
            RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 10.0)) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=n_restarts,
            normalize_y=True, alpha=1e-6
        )
        self.scaler_X = StandardScaler()
        self.is_fitted = False

    def _features(self, df):
        return np.column_stack([np.log(df["moneyness"].values), df["tte"].values])

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
'''

# ════════════════════════════════════════════════════════════════════════════
files["models/mlp_surface.py"] = '''"""
mlp_surface.py
PyTorch MLP for IV surface fitting.
Architecture: Input(2) -> 64 -> 128 -> 64 -> Softplus(1)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class IVNet(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 1),   nn.Softplus()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class IVSurfaceMLP:
    def __init__(self, epochs=300, batch_size=64, lr=1e-3, dropout=0.1, patience=30):
        self.epochs, self.batch_size = epochs, batch_size
        self.lr, self.patience = lr, patience
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = IVNet(dropout).to(self.device)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False

    def _features(self, df):
        return np.column_stack([np.log(df["moneyness"].values), df["tte"].values])

    def fit(self, df):
        X = self._features(df)
        y = df["iv"].values.reshape(-1, 1)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        X_tr_s  = self.scaler_X.fit_transform(X_tr)
        X_val_s = self.scaler_X.transform(X_val)
        y_tr_s  = self.scaler_y.fit_transform(y_tr).ravel()
        y_val_s = self.scaler_y.transform(y_val).ravel()

        t = lambda a: torch.FloatTensor(a).to(self.device)
        train_dl = DataLoader(TensorDataset(t(X_tr_s), t(y_tr_s)), batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(TensorDataset(t(X_val_s), t(y_val_s)), batch_size=256)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()
        best_val, patience_count, best_state = np.inf, 0, None

        print(f"[MLP] Training on {self.device} | {len(X_tr)} train, {len(X_val)} val")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = sum(
                (lambda loss: (loss.backward(), optimizer.step(), optimizer.zero_grad(), loss.item() * len(xb))[3])
                (criterion(self.model(xb), yb))
                for xb, yb in train_dl
            ) / len(train_dl.dataset)

            self.model.eval()
            with torch.no_grad():
                val_loss = sum(criterion(self.model(xb), yb).item() * len(xb)
                               for xb, yb in val_dl) / len(val_dl.dataset)
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= self.patience:
                print(f"[MLP] Early stop at epoch {epoch} | best val: {best_val:.6f}")
                break
            if epoch % 50 == 0 or epoch == 1:
                print(f"[MLP] Epoch {epoch:>3} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        if best_state:
            self.model.load_state_dict(best_state)
        self.is_fitted = True
        return self

    def predict(self, df):
        assert self.is_fitted
        X_s = self.scaler_X.transform(self._features(df))
        self.model.eval()
        with torch.no_grad():
            y_s = self.model(torch.FloatTensor(X_s).to(self.device)).cpu().numpy().reshape(-1,1)
        return np.clip(self.scaler_y.inverse_transform(y_s).ravel(), 0.01, 5.0)

    def predict_grid(self, moneyness_grid, tte_grid):
        assert self.is_fitted
        M, T = np.meshgrid(moneyness_grid, tte_grid, indexing="ij")
        grid_df = pd.DataFrame({"moneyness": M.ravel(), "tte": T.ravel()})
        return self.predict(grid_df).reshape(M.shape)

    def evaluate(self, df):
        iv_pred, iv_true = self.predict(df), df["iv"].values
        mae  = np.mean(np.abs(iv_pred - iv_true))
        rmse = np.sqrt(np.mean((iv_pred - iv_true)**2))
        r2   = 1 - np.sum((iv_true - iv_pred)**2) / np.sum((iv_true - iv_true.mean())**2)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}
'''

# ════════════════════════════════════════════════════════════════════════════
files["arbitrage/checks.py"] = '''"""
checks.py
No-arbitrage conditions for IV surface.

Calendar: total variance w(K,T) = σ²*T must be non-decreasing in T.
Butterfly: w(x,T) must be convex in log-moneyness x = log(K/S).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ArbitrageReport:
    calendar_violations: int
    butterfly_violations: int
    calendar_violation_rate: float
    butterfly_violation_rate: float
    calendar_details: pd.DataFrame
    butterfly_details: pd.DataFrame

    def summary(self):
        lines = [
            "═══════════════════════════════════",
            "   ARBITRAGE CHECK REPORT",
            "═══════════════════════════════════",
            f"Calendar  violations: {self.calendar_violations}  ({self.calendar_violation_rate:.1%})",
            f"Butterfly violations: {self.butterfly_violations} ({self.butterfly_violation_rate:.1%})",
            "✓ Arbitrage-free" if self.calendar_violations == 0 and self.butterfly_violations == 0
                               else "⚠ Violations detected",
            "═══════════════════════════════════",
        ]
        return "\\n".join(lines)


def check_calendar_arbitrage(iv_surface, moneyness_grid, tte_grid, tol=1e-4):
    violations = []
    tv = iv_surface**2 * tte_grid[np.newaxis, :]
    for i, m in enumerate(moneyness_grid):
        for j in range(len(tte_grid) - 1):
            if tv[i, j] - tv[i, j+1] > tol:
                violations.append({"moneyness": m, "T1": tte_grid[j],
                                    "T2": tte_grid[j+1], "size": tv[i,j] - tv[i,j+1]})
    return pd.DataFrame(violations)


def check_butterfly_arbitrage(iv_surface, moneyness_grid, tte_grid, tol=1e-4):
    violations = []
    lm = np.log(moneyness_grid)
    tv = iv_surface**2 * tte_grid[np.newaxis, :]
    for j, T in enumerate(tte_grid):
        for i in range(1, len(moneyness_grid) - 1):
            dx1 = lm[i] - lm[i-1]
            dx2 = lm[i+1] - lm[i]
            d2  = tv[i+1,j]/dx2 - tv[i,j]*(1/dx1+1/dx2) + tv[i-1,j]/dx1
            if d2 < -tol:
                violations.append({"moneyness": moneyness_grid[i], "tte": T, "size": abs(d2)})
    return pd.DataFrame(violations)


def run_arbitrage_checks(iv_surface, moneyness_grid, tte_grid):
    print("[Arbitrage] Calendar check...")
    cal_df = check_calendar_arbitrage(iv_surface, moneyness_grid, tte_grid)
    print("[Arbitrage] Butterfly check...")
    but_df = check_butterfly_arbitrage(iv_surface, moneyness_grid, tte_grid)
    total  = len(moneyness_grid) * len(tte_grid)
    report = ArbitrageReport(
        calendar_violations=len(cal_df),
        butterfly_violations=len(but_df),
        calendar_violation_rate=len(cal_df)/(total+1e-9),
        butterfly_violation_rate=len(but_df)/(total+1e-9),
        calendar_details=cal_df,
        butterfly_details=but_df
    )
    print(report.summary())
    return report
'''

# ════════════════════════════════════════════════════════════════════════════
files["viz/plotter.py"] = '''"""
plotter.py
Plotly visualizations: 3D surface, IV smiles, model comparison, GP uncertainty.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

COLORS = {"gp": "#636EFA", "mlp": "#EF553B", "scatter": "#00CC96", "atm": "#FFA15A"}


def plot_iv_surface_3d(iv_gp, iv_mlp, moneyness_grid, tte_grid,
                       save_path="outputs/iv_surface_3d.html"):
    M, T = np.meshgrid(moneyness_grid, tte_grid, indexing="ij")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"surface"},{"type":"surface"}]],
                        subplot_titles=["Gaussian Process","MLP"])
    fig.add_trace(go.Surface(x=M, y=T*365, z=iv_gp*100, colorscale="Viridis",
                             colorbar=dict(x=0.46, title="IV %")), row=1, col=1)
    fig.add_trace(go.Surface(x=M, y=T*365, z=iv_mlp*100, colorscale="Plasma",
                             colorbar=dict(x=1.02, title="IV %")), row=1, col=2)
    fig.update_layout(title="SPY IV Surface — GP vs MLP", height=600, template="plotly_dark",
                      scene=dict(xaxis_title="Moneyness",yaxis_title="Days",zaxis_title="IV %"),
                      scene2=dict(xaxis_title="Moneyness",yaxis_title="Days",zaxis_title="IV %"))
    fig.write_html(save_path)
    print(f"[Plot] Saved → {save_path}")
    return fig


def plot_iv_smiles(df, iv_gp_pred, iv_mlp_pred, n_slices=4,
                   save_path="outputs/iv_smiles.html"):
    df = df.copy()
    df["gp_iv"] = iv_gp_pred
    df["mlp_iv"] = iv_mlp_pred
    expiries = sorted(df["expiry"].unique())
    selected = expiries[::max(1, len(expiries)//n_slices)][:n_slices]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"T={e}" for e in selected])
    for idx, exp in enumerate(selected):
        r, c = idx//2+1, idx%2+1
        sub = df[df["expiry"]==exp].sort_values("moneyness")
        fig.add_trace(go.Scatter(x=sub["moneyness"], y=sub["iv"]*100, mode="markers",
                                 marker=dict(color=COLORS["scatter"],size=6),
                                 name="Market", showlegend=(idx==0)), row=r, col=c)
        fig.add_trace(go.Scatter(x=sub["moneyness"], y=sub["gp_iv"]*100, mode="lines",
                                 line=dict(color=COLORS["gp"],width=2),
                                 name="GP", showlegend=(idx==0)), row=r, col=c)
        fig.add_trace(go.Scatter(x=sub["moneyness"], y=sub["mlp_iv"]*100, mode="lines",
                                 line=dict(color=COLORS["mlp"],width=2,dash="dash"),
                                 name="MLP", showlegend=(idx==0)), row=r, col=c)
    fig.update_layout(title="IV Smiles — Market vs GP vs MLP", height=600, template="plotly_dark")
    fig.write_html(save_path)
    print(f"[Plot] Saved → {save_path}")
    return fig


def plot_model_comparison(df, iv_gp, iv_mlp, save_path="outputs/model_comparison.html"):
    iv_true = df["iv"].values * 100
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Gaussian Process","MLP"])
    for col, (name, pred) in enumerate([("GP", iv_gp*100), ("MLP", iv_mlp*100)], 1):
        color = COLORS["gp"] if name=="GP" else COLORS["mlp"]
        lim = [min(iv_true.min(), pred.min()), max(iv_true.max(), pred.max())]
        fig.add_trace(go.Scatter(x=iv_true, y=pred, mode="markers",
                                 marker=dict(color=color,size=4,opacity=0.5),
                                 showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                 line=dict(color="white",width=1,dash="dot"),
                                 showlegend=False), row=1, col=col)
    fig.update_layout(title="Predicted vs Actual IV", height=450, template="plotly_dark")
    fig.write_html(save_path)
    print(f"[Plot] Saved → {save_path}")
    return fig


def plot_gp_uncertainty(iv_gp, iv_gp_std, moneyness_grid, tte_grid,
                        fixed_tte_idx=2, save_path="outputs/gp_uncertainty.html"):
    T_val = tte_grid[fixed_tte_idx]
    iv  = iv_gp[:, fixed_tte_idx] * 100
    std = iv_gp_std[:, fixed_tte_idx] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moneyness_grid, y=iv+std, mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=moneyness_grid, y=iv-std, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(99,110,250,0.2)", name="±1σ"))
    fig.add_trace(go.Scatter(x=moneyness_grid, y=iv, mode="lines",
                             line=dict(color=COLORS["gp"],width=2.5), name="GP mean"))
    fig.add_vline(x=1.0, line_dash="dot", line_color=COLORS["atm"],
                  annotation_text="ATM")
    fig.update_layout(title=f"GP Uncertainty — T={T_val*365:.0f}d",
                      xaxis_title="Moneyness", yaxis_title="IV (%)",
                      template="plotly_dark", height=450)
    fig.write_html(save_path)
    print(f"[Plot] Saved → {save_path}")
    return fig
'''

# ════════════════════════════════════════════════════════════════════════════
files["main.py"] = '''"""
main.py — Full pipeline runner
Run: python main.py
"""
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(__file__))

from data.fetch_options import fetch_spy_options
from iv.extractor       import extract_iv_surface
from models.gp_surface  import IVSurfaceGP
from models.mlp_surface import IVSurfaceMLP
from arbitrage.checks   import run_arbitrage_checks
from viz.plotter        import (plot_iv_surface_3d, plot_iv_smiles,
                                plot_model_comparison, plot_gp_uncertainty)

os.makedirs("outputs", exist_ok=True)
np.random.seed(42)

def main():
    print("\\n" + "═"*50)
    print("   SPY IMPLIED VOLATILITY SURFACE BUILDER")
    print("═"*50 + "\\n")

    print("── STEP 1: Fetching Data ──────────────────────────")
    S, r, df_raw = fetch_spy_options("SPY")

    print("\\n── STEP 2: Extracting IVs ─────────────────────────")
    df = extract_iv_surface(df_raw, S, r)
    df.to_csv("outputs/iv_data.csv", index=False)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[Pipeline] Train: {len(df_train)} | Test: {len(df_test)}")

    print("\\n── STEP 3a: Gaussian Process ──────────────────────")
    gp = IVSurfaceGP(n_restarts=5).fit(df_train)
    gp_m = gp.evaluate(df_test)
    print(f"[GP]  MAE={gp_m[\'MAE\']:.4f}  RMSE={gp_m[\'RMSE\']:.4f}  R²={gp_m[\'R2\']:.4f}")

    print("\\n── STEP 3b: MLP ───────────────────────────────────")
    mlp = IVSurfaceMLP(epochs=300, patience=30).fit(df_train)
    mlp_m = mlp.evaluate(df_test)
    print(f"[MLP] MAE={mlp_m[\'MAE\']:.4f}  RMSE={mlp_m[\'RMSE\']:.4f}  R²={mlp_m[\'R2\']:.4f}")

    print("\\n── STEP 3c: Building Grids ────────────────────────")
    m_grid = np.linspace(0.80, 1.20, 60)
    t_grid = np.linspace(7/365, 1.0, 40)
    iv_gp, iv_gp_std = gp.predict_grid(m_grid, t_grid)
    iv_mlp            = mlp.predict_grid(m_grid, t_grid)

    print("\\n── STEP 4: Arbitrage Checks ───────────────────────")
    print("[GP Surface]");  gp_arb  = run_arbitrage_checks(iv_gp,  m_grid, t_grid)
    print("[MLP Surface]"); mlp_arb = run_arbitrage_checks(iv_mlp, m_grid, t_grid)

    print("\\n── STEP 5: Visualizations ─────────────────────────")
    gp_pred  = gp.predict(df,  return_std=False)
    mlp_pred = mlp.predict(df)
    plot_iv_surface_3d(iv_gp, iv_mlp, m_grid, t_grid)
    plot_iv_smiles(df, gp_pred, mlp_pred)
    plot_model_comparison(df, gp_pred, mlp_pred)
    plot_gp_uncertainty(iv_gp, iv_gp_std, m_grid, t_grid, fixed_tte_idx=5)

    print("\\n" + "═"*50 + "   RESULTS   " + "═"*50)
    results = pd.DataFrame({
        "Model": ["GP", "MLP"],
        "MAE":   [f"{gp_m[\'MAE\']:.4f}", f"{mlp_m[\'MAE\']:.4f}"],
        "RMSE":  [f"{gp_m[\'RMSE\']:.4f}", f"{mlp_m[\'RMSE\']:.4f}"],
        "R²":    [f"{gp_m[\'R2\']:.4f}", f"{mlp_m[\'R2\']:.4f}"],
        "Cal.Violations": [gp_arb.calendar_violations, mlp_arb.calendar_violations],
        "Butterfly.Violations": [gp_arb.butterfly_violations, mlp_arb.butterfly_violations],
    })
    print(results.to_string(index=False))
    results.to_csv("outputs/results.csv", index=False)
    print("\\n[Done] All outputs saved to ./outputs/")

if __name__ == "__main__":
    main()
'''

# ── Write all files ──────────────────────────────────────────────────────────
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Written: {path}")

print("\nProject setup complete!")
print("Next steps:")
print("  pip install -r requirements.txt")
print("  python iv/extractor.py     ← sanity check (no internet needed)")
print("  python main.py             ← full pipeline")
