from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Side = Literal["long", "short"]
ExitReason = Literal["stop", "tp", "trail", "end"]


@dataclass(frozen=True)
class Trade:
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    reason: ExitReason


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[float]
    equity_timestamps: list[datetime]