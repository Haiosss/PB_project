from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd

from market_pipeline.marketdata.resampled_io import load_resampled_day_df


@dataclass(frozen=True)
class ResampledDayValidationResult:
    rows: int
    duplicates: int
    missing_bars: int
    ohlc_violations: int
    expected_bars: int | None
    expected_match: bool | None
    first_ts: str | None
    last_ts: str | None


def _expected_bars_per_day(timeframe: str) -> int | None:
    #returns expected number of bars in a full 24h day if timeframe divides the day exactly if not divisible (7min) returns None.
    
    td = pd.to_timedelta(timeframe)
    day_td = pd.Timedelta(days=1)

    #integer nanoseconds
    tf_ns = td.value
    day_ns = day_td.value

    if tf_ns <= 0:
        raise ValueError("Timeframe must be > 0")

    if day_ns % tf_ns != 0:
        return None

    return int(day_ns // tf_ns)


def validate_resampled_day_df(df: pd.DataFrame, timeframe: str) -> ResampledDayValidationResult:
    if df.empty:
        exp = _expected_bars_per_day(timeframe)
        return ResampledDayValidationResult(
            rows=0,
            duplicates=0,
            missing_bars=0,
            ohlc_violations=0,
            expected_bars=exp,
            expected_match=None,  #no data for this day
            first_ts=None,
            last_ts=None,
        )

    df = df.sort_values("ts_utc").reset_index(drop=True)

    duplicates = int(df["ts_utc"].duplicated().sum())

    ts = pd.to_datetime(df["ts_utc"], utc=True)
    tf_td = pd.to_timedelta(timeframe)

    diffs = ts.diff().dropna()
    gaps = diffs[diffs > tf_td]

    #count how many bars are missing inside gaps
    missing_bars = 0
    for g in gaps:
        missing_bars += int((g // tf_td) - 1)

    o = df["bid_o"]
    h = df["bid_h"]
    l = df["bid_l"]
    c = df["bid_c"]
    violations = ~((l <= o) & (o <= h) & (l <= c) & (c <= h) & (l <= h))
    ohlc_violations = int(violations.sum())

    exp = _expected_bars_per_day(timeframe)
    expected_match = None if exp is None else (int(len(df)) == exp)

    return ResampledDayValidationResult(
        rows=int(len(df)),
        duplicates=duplicates,
        missing_bars=missing_bars,
        ohlc_violations=ohlc_violations,
        expected_bars=exp,
        expected_match=expected_match,
        first_ts=str(ts.iloc[0]),
        last_ts=str(ts.iloc[-1]),
    )


def validate_resampled_range(
    base_cache_dir,
    symbol: str,
    timeframe: str,
    d0: date,
    d1: date,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    days_checked = 0
    days_with_data = 0
    days_empty = 0
    total_rows = 0
    total_duplicates = 0
    total_missing_bars = 0
    total_ohlc_violations = 0
    expected_mismatch_days = 0

    d = d0
    while d < d1:
        days_checked += 1
        df = load_resampled_day_df(base_cache_dir, symbol, timeframe, d)
        res = validate_resampled_day_df(df, timeframe)

        if res.rows == 0:
            days_empty += 1
        else:
            days_with_data += 1

        if res.expected_match is False:
            expected_mismatch_days += 1

        total_rows += res.rows
        total_duplicates += res.duplicates
        total_missing_bars += res.missing_bars
        total_ohlc_violations += res.ohlc_violations

        rows.append(
            {
                "day_utc": str(d),
                "rows": res.rows,
                "duplicates": res.duplicates,
                "missing_bars": res.missing_bars,
                "ohlc_violations": res.ohlc_violations,
                "expected_bars": res.expected_bars,
                "expected_match": res.expected_match,
                "first_ts": res.first_ts,
                "last_ts": res.last_ts,
            }
        )

        d += timedelta(days=1)

    df_days = pd.DataFrame(rows)

    summary = {
        "days_checked": days_checked,
        "days_with_data": days_with_data,
        "days_empty": days_empty,
        "total_rows": total_rows,
        "total_duplicates": total_duplicates,
        "total_missing_bars": total_missing_bars,
        "total_ohlc_violations": total_ohlc_violations,
        "expected_mismatch_days": expected_mismatch_days,
    }

    return df_days, summary