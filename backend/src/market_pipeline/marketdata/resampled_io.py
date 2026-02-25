from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def normalize_timeframe(tf: str) -> str:
    return tf.strip().lower()


def resampled_day_path(base_cache_dir: Path, symbol: str, timeframe: str, d: date) -> Path:
    tf = normalize_timeframe(timeframe)
    return (
        base_cache_dir
        / "candles_resampled"
        / f"symbol={symbol.upper()}"
        / f"timeframe={tf}"
        / f"day={d.isoformat()}"
        / "data.parquet"
    )


def load_resampled_day_df(base_cache_dir: Path, symbol: str, timeframe: str, d: date) -> pd.DataFrame:
    path = resampled_day_path(base_cache_dir, symbol, timeframe, d)
    if not path.exists():
        return pd.DataFrame(columns=["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"])

    df = pd.read_parquet(path)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df