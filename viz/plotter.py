"""
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
