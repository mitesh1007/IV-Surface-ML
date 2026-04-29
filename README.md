> Built with live SPY + QQQ + IWM options data. 
> GP achieves R²=0.80. MLP surface is fully calendar-arbitrage free.

| Model | MAE | RMSE | R² | Cal. Violations | Butterfly Violations |
|---|---|---|---|---|---|
| Gaussian Process | 0.0843 | 0.1339 | 0.7981 | 458 | 901 |
| MLP | 0.1747 | 0.2284 | 0.4130 | 0 | 399 |

> *Data: SPY + QQQ + IWM options (234 contracts, 20 expirations) via yfinance. 
> GP achieves R²=0.80 with probabilistic uncertainty bands. 
> MLP surface is fully calendar-arbitrage free (0 violations).*

## Limitations
- Data sourced from yfinance (delayed, sparse). Production version would use CBOE or Bloomberg tick data.
- 234 contracts across 3 ETFs (SPY, QQQ, IWM). Richer data would improve MLP R² significantly.
- GP calendar violations occur due to kernel smoothing across underlying boundaries.