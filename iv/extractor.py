"""
extractor.py
Black-Scholes IV extraction via Brent's root-finding method.

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
    intrinsic = max(S - K * np.exp(-r * T), 0.0) if option_type == "call"                 else max(K * np.exp(-r * T) - S, 0.0)
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
    median_iv = df["iv"].median()
    df = df[df["iv"] <= 2.0 * median_iv]
    df = df[df["iv"] >= 0.05]
    before = len(df)
    df = df.dropna(subset=["iv"])
    print(f"[IV] Success: {len(df)}/{before} | Range: {df['iv'].min():.1%}–{df['iv'].max():.1%}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    true_sigma = 0.20
    price = bs_call(S, K, T, r, true_sigma)
    recovered = extract_iv(price, S, K, T, r, "call")
    print(f"True σ: {true_sigma:.4f} | Recovered σ: {recovered:.4f}")
    assert abs(recovered - true_sigma) < 1e-5
    print("Sanity check passed ✓")
