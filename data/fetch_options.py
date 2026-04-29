"""
fetch_options.py
Pulls options chain from multiple liquid ETFs (SPY, QQQ, IWM).
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def fetch_spy_options(tickers: list = ["SPY", "QQQ", "IWM"]):
    try:
        tbill = yf.Ticker("^IRX")
        r = float(tbill.history(period="1d")["Close"].iloc[-1]) / 100.0
    except Exception:
        r = 0.05
    print(f"[Data] Risk-free rate: {r*100:.2f}%")

    today = datetime.today()
    all_records = []
    spy_spot = None

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                print(f"[Data] Skipping {ticker} — no price data")
                continue
            S = float(hist["Close"].iloc[-1])
            if ticker == "SPY":
                spy_spot = S
            print(f"[Data] {ticker} spot: ${S:.2f}")

            records = []
            for exp_str in stock.options:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                tte = (exp_date - today).days / 365.0
                if tte < 0 or tte > 2.0:
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
                    df_opt["underlying"] = ticker
                    df_opt["spot"] = S
                    records.append(df_opt)

            if records:
                df_ticker = pd.concat(records, ignore_index=True)
                all_records.append(df_ticker)
                print(f"[Data] {ticker} raw contracts: {len(df_ticker)}")

        except Exception as e:
            print(f"[Data] Failed {ticker}: {e}")
            continue

    if not all_records:
        raise RuntimeError("No options data fetched.")

    df = pd.concat(all_records, ignore_index=True)
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
    df = df[df["bid"] > 0.01]
    df = df[df["mid_price"] > 0.10]
    df["moneyness"] = df["strike"] / df["spot"]
    df = df[(df["moneyness"] >= 0.70) & (df["moneyness"] <= 1.30)]
    df = df[df["impliedVolatility"] > 0.05]
    df = df[df["impliedVolatility"] < 2.0]
    df = df[["strike","expiry","tte","mid_price","option_type",
             "moneyness","underlying","spot","impliedVolatility"]].copy()
    df.rename(columns={"impliedVolatility": "yf_iv"}, inplace=True)
    df = df.dropna(subset=["strike","tte","mid_price"]).reset_index(drop=True)

    S_ref = spy_spot if spy_spot else df["spot"].iloc[0]

    print(f"\n[Data] Total contracts: {len(df)}")
    print(df.groupby("underlying").size().to_string())
    print(f"[Data] Expirations: {df['expiry'].nunique()}")

    return S_ref, r, df


if __name__ == "__main__":
    S, r, df = fetch_spy_options()
    print(df.head())
