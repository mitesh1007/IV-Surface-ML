"""
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
        return "\n".join(lines)


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
