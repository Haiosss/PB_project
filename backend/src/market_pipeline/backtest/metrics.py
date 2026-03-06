from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from market_pipeline.backtest.models import BacktestResult


@dataclass(frozen=True)
class Metrics:
    trades: int
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    winrate_pct: float
    profit_factor: float


def compute_metrics(result: BacktestResult) -> Metrics:
    eq = np.array(result.equity_curve, dtype=float)
    if len(eq) < 2:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)

    #max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    max_dd = float(dd.min())

    #sharpe (per-bar, not annualized yet)
    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if len(rets) > 2 else float(np.std(rets))
    sharpe = 0.0 if std == 0 else mean / std

    pnls = np.array([t.pnl for t in result.trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    trades = int(len(pnls))
    winrate = 0.0 if trades == 0 else float((pnls > 0).sum()) / trades

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = 0.0 if gross_loss == 0 else gross_win / gross_loss

    total_return = (eq[-1] / eq[0]) - 1.0

    return Metrics(
        trades=trades,
        total_return_pct=100.0 * float(total_return),
        max_drawdown_pct=100.0 * float(max_dd),
        sharpe=float(sharpe),
        winrate_pct=100.0 * float(winrate),
        profit_factor=float(profit_factor),
    )