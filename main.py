"""
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
    print("\n" + "═"*50)
    print("   SPY IMPLIED VOLATILITY SURFACE BUILDER")
    print("═"*50 + "\n")

    print("── STEP 1: Fetching Data ──────────────────────────")
    S, r, df_raw = fetch_spy_options(["SPY", "QQQ", "IWM"])

    print("\n── STEP 2: Extracting IVs ─────────────────────────")
    df = extract_iv_surface(df_raw, S, r)
    df.to_csv("outputs/iv_data.csv", index=False)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[Pipeline] Train: {len(df_train)} | Test: {len(df_test)}")

    print("\n── STEP 3a: Gaussian Process ──────────────────────")
    gp = IVSurfaceGP(n_restarts=5).fit(df_train)
    gp_m = gp.evaluate(df_test)
    print(f"[GP]  MAE={gp_m['MAE']:.4f}  RMSE={gp_m['RMSE']:.4f}  R²={gp_m['R2']:.4f}")

    print("\n── STEP 3b: MLP ───────────────────────────────────")
    mlp = IVSurfaceMLP(epochs=300, patience=30).fit(df_train)
    mlp_m = mlp.evaluate(df_test)
    print(f"[MLP] MAE={mlp_m['MAE']:.4f}  RMSE={mlp_m['RMSE']:.4f}  R²={mlp_m['R2']:.4f}")

    print("\n── STEP 3c: Building Grids ────────────────────────")
    m_grid = np.linspace(0.80, 1.20, 60)
    t_grid = np.linspace(7/365, 1.0, 40)
    iv_gp, iv_gp_std = gp.predict_grid(m_grid, t_grid)
    iv_mlp            = mlp.predict_grid(m_grid, t_grid)

    print("\n── STEP 4: Arbitrage Checks ───────────────────────")
    print("[GP Surface]");  gp_arb  = run_arbitrage_checks(iv_gp,  m_grid, t_grid)
    print("[MLP Surface]"); mlp_arb = run_arbitrage_checks(iv_mlp, m_grid, t_grid)

    print("\n── STEP 5: Visualizations ─────────────────────────")
    gp_pred  = gp.predict(df,  return_std=False)
    mlp_pred = mlp.predict(df)
    plot_iv_surface_3d(iv_gp, iv_mlp, m_grid, t_grid)
    plot_iv_smiles(df, gp_pred, mlp_pred)
    plot_model_comparison(df, gp_pred, mlp_pred)
    plot_gp_uncertainty(iv_gp, iv_gp_std, m_grid, t_grid, fixed_tte_idx=5)

    print("\n" + "═"*50 + "   RESULTS   " + "═"*50)
    results = pd.DataFrame({
        "Model": ["GP", "MLP"],
        "MAE":   [f"{gp_m['MAE']:.4f}", f"{mlp_m['MAE']:.4f}"],
        "RMSE":  [f"{gp_m['RMSE']:.4f}", f"{mlp_m['RMSE']:.4f}"],
        "R²":    [f"{gp_m['R2']:.4f}", f"{mlp_m['R2']:.4f}"],
        "Cal.Violations": [gp_arb.calendar_violations, mlp_arb.calendar_violations],
        "Butterfly.Violations": [gp_arb.butterfly_violations, mlp_arb.butterfly_violations],
    })
    print(results.to_string(index=False))
    results.to_csv("outputs/results.csv", index=False)
    print("\n[Done] All outputs saved to ./outputs/")

if __name__ == "__main__":
    main()
