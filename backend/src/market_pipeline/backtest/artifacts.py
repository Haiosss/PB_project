from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pandas as pd

from market_pipeline.backtest.models import BacktestResult


def trades_to_df(result: BacktestResult) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame(columns=["side","entry_time","exit_time","entry_price","exit_price","size","pnl","return_pct","reason"])
    rows = [asdict(t) for t in result.trades]
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def equity_to_df(result: BacktestResult) -> pd.DataFrame:
    df = pd.DataFrame(
        {"ts_utc": pd.to_datetime(result.equity_timestamps, utc=True), "equity": result.equity_curve}
    )
    return df


def save_backtest_artifacts(result: BacktestResult, out_dir: Path, prefix: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"{prefix}_trades.parquet"
    equity_path = out_dir / f"{prefix}_equity.parquet"

    trades_to_df(result).to_parquet(trades_path, index=False)
    equity_to_df(result).to_parquet(equity_path, index=False)

    return trades_path, equity_path