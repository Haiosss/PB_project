from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from market_pipeline.marketdata.cache_io import load_cached_day_df
from market_pipeline.marketdata.cleaning import clean_candles_df
from market_pipeline.marketdata.queries import load_candles_1m_df
from market_pipeline.marketdata.resampled_io import normalize_timeframe


_ONE_MIN_ALIASES = {"1m", "1min", "1t", "1minute"}


def _expected_bars_per_day(timeframe: str) -> int | None:
    tf = pd.to_timedelta(timeframe)
    day_td = pd.Timedelta(days=1)
    if tf.value <= 0:
        raise ValueError("timeframe must be > 0")
    if day_td.value % tf.value != 0:
        return None
    return int(day_td.value // tf.value)


def _load_day_auto(
    *,
    instrument_id: int,
    base_cache_dir: Path,
    symbol: str,
    timeframe: str,
    d: date,
) -> pd.DataFrame:
    tf = normalize_timeframe(timeframe)
    if tf in _ONE_MIN_ALIASES:
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return load_candles_1m_df(instrument_id, start, end)

    return load_cached_day_df(base_cache_dir, symbol, tf, d)


def build_cleaning_report_range(
    *,
    instrument_id: int,
    base_cache_dir: Path,
    symbol: str,
    timeframe: str,
    d0: date,
    d1: date,
) -> tuple[pd.DataFrame, dict]:

    # Cleaning report:
    #  - 1m timeframe: reports cleaning needed on raw DB data
    #  - higher tf: reports cleaning needed on cached resampled parquet

    tf = normalize_timeframe(timeframe)
    exp_bars = None if tf in _ONE_MIN_ALIASES else _expected_bars_per_day(tf)

    rows = []
    totals = {
        "days_checked": 0,
        "days_with_data": 0,
        "days_empty": 0,
        "rows_in": 0,
        "rows_out": 0,
        "dropped_duplicates": 0,
        "dropped_null_ohlc": 0,
        "dropped_invalid_ohlc": 0,
        "dropped_inactive": 0,
        "expected_mismatch_days": 0,
    }

    d = d0
    while d < d1:
        totals["days_checked"] += 1

        raw_df = _load_day_auto(
            instrument_id=instrument_id,
            base_cache_dir=base_cache_dir,
            symbol=symbol,
            timeframe=tf,
            d=d,
        )

        cleaned_df, s = clean_candles_df(raw_df)

        if s.rows_in == 0:
            totals["days_empty"] += 1
        else:
            totals["days_with_data"] += 1

        totals["rows_in"] += s.rows_in
        totals["rows_out"] += s.rows_out
        totals["dropped_duplicates"] += s.dropped_duplicates
        totals["dropped_null_ohlc"] += s.dropped_null_ohlc
        totals["dropped_invalid_ohlc"] += s.dropped_invalid_ohlc
        totals["dropped_inactive"] += s.dropped_inactive

        expected_match = None
        if exp_bars is not None and s.rows_out > 0:
            expected_match = (s.rows_out == exp_bars)
            if not expected_match:
                totals["expected_mismatch_days"] += 1

        rows.append(
            {
                "day_utc": str(d),
                "rows_in": s.rows_in,
                "rows_out": s.rows_out,
                "dropped_duplicates": s.dropped_duplicates,
                "dropped_null_ohlc": s.dropped_null_ohlc,
                "dropped_invalid_ohlc": s.dropped_invalid_ohlc,
                "dropped_inactive": s.dropped_inactive,
                "expected_bars": exp_bars,
                "expected_match_after_cleaning": expected_match,
                "first_ts_after_cleaning": None if cleaned_df.empty else str(cleaned_df["ts_utc"].iloc[0]),
                "last_ts_after_cleaning": None if cleaned_df.empty else str(cleaned_df["ts_utc"].iloc[-1]),
            }
        )

        d += timedelta(days=1)

    return pd.DataFrame(rows), totals


def build_validate_cleaning_cache_range(
    *,
    base_cache_dir: Path,
    symbol: str,
    timeframe: str,
    d0: date,
    d1: date,
) -> tuple[pd.DataFrame, dict]:

    # Validate that cached Parquet data is already clean.
    
    tf = normalize_timeframe(timeframe)
    exp_bars = None if tf in _ONE_MIN_ALIASES else _expected_bars_per_day(tf)

    rows = []
    totals = {
        "days_checked": 0,
        "days_with_data": 0,
        "days_empty": 0,
        "rows_in": 0,
        "not_clean_days": 0,
        "total_dropped_if_recleaned": 0,
        "dropped_duplicates": 0,
        "dropped_null_ohlc": 0,
        "dropped_invalid_ohlc": 0,
        "dropped_inactive": 0,
        "expected_mismatch_days": 0,
    }

    d = d0
    while d < d1:
        totals["days_checked"] += 1

        df_cache = load_cached_day_df(base_cache_dir, symbol, tf, d)
        df_cleaned, s = clean_candles_df(df_cache)

        if s.rows_in == 0:
            totals["days_empty"] += 1
        else:
            totals["days_with_data"] += 1

        reclean_drop = s.rows_in - s.rows_out
        is_clean = (reclean_drop == 0)

        if not is_clean:
            totals["not_clean_days"] += 1

        totals["rows_in"] += s.rows_in
        totals["total_dropped_if_recleaned"] += reclean_drop
        totals["dropped_duplicates"] += s.dropped_duplicates
        totals["dropped_null_ohlc"] += s.dropped_null_ohlc
        totals["dropped_invalid_ohlc"] += s.dropped_invalid_ohlc
        totals["dropped_inactive"] += s.dropped_inactive

        expected_match = None
        if exp_bars is not None and s.rows_out > 0:
            expected_match = (s.rows_out == exp_bars)
            if not expected_match:
                totals["expected_mismatch_days"] += 1

        rows.append(
            {
                "day_utc": str(d),
                "rows_cached": s.rows_in,
                "rows_after_reclean": s.rows_out,
                "cache_is_clean": is_clean,
                "dropped_if_recleaned": reclean_drop,
                "dropped_duplicates": s.dropped_duplicates,
                "dropped_null_ohlc": s.dropped_null_ohlc,
                "dropped_invalid_ohlc": s.dropped_invalid_ohlc,
                "dropped_inactive": s.dropped_inactive,
                "expected_bars": exp_bars,
                "expected_match": expected_match,
                "first_ts": None if df_cleaned.empty else str(df_cleaned["ts_utc"].iloc[0]),
                "last_ts": None if df_cleaned.empty else str(df_cleaned["ts_utc"].iloc[-1]),
            }
        )

        d += timedelta(days=1)

    return pd.DataFrame(rows), totals