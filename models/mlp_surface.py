"""
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
            nn.Linear(3, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
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
        underlying_map = {"SPY": 0.0, "QQQ": 0.5, "IWM": 1.0}
        if "underlying" in df.columns:
            u = df["underlying"].map(underlying_map).fillna(0.5).values
        else:
            u = np.full(len(df), 0.0)  # default to SPY for grid prediction
        return np.column_stack([np.log(df["moneyness"].values), df["tte"].values, u])

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
