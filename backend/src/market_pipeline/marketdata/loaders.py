from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from market_pipeline.marketdata.queries import load_candles_1m_df
from market_pipeline.marketdata.resampled_io import load_resampled_day_df, normalize_timeframe


_ONE_MIN_ALIASES = {"1m", "1min", "1t", "1minute"}


def _standardize_candles_df(df: pd.DataFrame) -> pd.DataFrame:
    #ensures consistent column order/types and sorted UTC timestamps

    expected_cols = ["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"]

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    out = df.copy()

    #ensure UTC timestamp dtype
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)

    #ensure expected columns exist
    for col in expected_cols:
        if col not in out.columns:
            out[col] = pd.NA

    #reorder + sort
    out = out[expected_cols].sort_values("ts_utc").reset_index(drop=True)
    return out


def prices_int_to_float(df: pd.DataFrame, price_scale: int) -> pd.DataFrame:

    #returns a copy where bid OHLC columns are converted from int price units (108734) to float prices (1.08734)

    out = df.copy()
    for col in ["bid_o", "bid_h", "bid_l", "bid_c"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / float(price_scale)
    return out


def load_range_1m_db(
    instrument_id: int,
    d0: date,
    d1: date,
) -> pd.DataFrame:

    #load 1m candles from DB for [d0, d1) UTC

    if d1 <= d0:
        raise ValueError("d1 must be after d0")

    start = datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc)
    end = datetime(d1.year, d1.month, d1.day, tzinfo=timezone.utc)

    df = load_candles_1m_df(instrument_id=instrument_id, start=start, end=end)
    return _standardize_candles_df(df)


def load_resampled_range_parquet(
    base_cache_dir,
    symbol: str,
    timeframe: str,
    d0: date,
    d1: date,
) -> pd.DataFrame:

    #load resampled candles from parquet cache for [d0, d1) UTC

    if d1 <= d0:
        raise ValueError("d1 must be after d0")

    tf = normalize_timeframe(timeframe)

    frames: list[pd.DataFrame] = []
    d = d0
    while d < d1:
        df_day = load_resampled_day_df(base_cache_dir=base_cache_dir, symbol=symbol, timeframe=tf, d=d)
        if not df_day.empty:
            frames.append(df_day)
        d += timedelta(days=1)

    if not frames:
        return _standardize_candles_df(pd.DataFrame())

    df = pd.concat(frames, ignore_index=True)
    return _standardize_candles_df(df)


def load_range(
    *,
    instrument_id: int,
    symbol: str,
    timeframe: str,
    d0: date,
    d1: date,
    parquet_cache_dir,
    price_scale: int | None = None,
    as_float_prices: bool = False,
) -> pd.DataFrame:
   
    #loader: timeframe=1m (or alias) -> loads from DB
           # other timeframes -> loads from resampled Parquet cache

    #optionally converts integer prices to float prices using price_scale
    
    tf = normalize_timeframe(timeframe)

    if tf in _ONE_MIN_ALIASES:
        df = load_range_1m_db(instrument_id=instrument_id, d0=d0, d1=d1)
    else:
        df = load_resampled_range_parquet(
            base_cache_dir=parquet_cache_dir,
            symbol=symbol,
            timeframe=tf,
            d0=d0,
            d1=d1,
        )

    if as_float_prices:
        if price_scale is None:
            raise ValueError("price_scale is required when as_float_prices=True")
        df = prices_int_to_float(df, price_scale=price_scale)

    return df