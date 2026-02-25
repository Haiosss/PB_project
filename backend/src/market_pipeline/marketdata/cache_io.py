from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from market_pipeline.marketdata.resampled_io import load_resampled_day_df, normalize_timeframe


_ONE_MIN_ALIASES = {"1m", "1min", "1t", "1minute"}


def load_cached_1m_day_df(base_cache_dir: Path, symbol: str, d: date) -> pd.DataFrame:
    path = (
        base_cache_dir
        / "candles_1m"
        / f"symbol={symbol.upper()}"
        / f"day={d.isoformat()}"
        / "data.parquet"
    )
    if not path.exists():
        return pd.DataFrame(columns=["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"])

    df = pd.read_parquet(path)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def load_cached_day_df(base_cache_dir: Path, symbol: str, timeframe: str, d: date) -> pd.DataFrame:
    tf = normalize_timeframe(timeframe)
    if tf in _ONE_MIN_ALIASES:
        return load_cached_1m_day_df(base_cache_dir, symbol, d)
    return load_resampled_day_df(base_cache_dir, symbol, tf, d)