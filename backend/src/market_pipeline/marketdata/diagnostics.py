from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
import pandas as pd

from market_pipeline.marketdata.queries import load_candles_1m_df
from market_pipeline.marketdata.validate import validate_day_1m


@dataclass(frozen=True)
class RangeSummary:
    days_checked: int
    days_with_data: int
    days_empty: int
    total_rows: int
    total_missing_minutes: int
    total_duplicates: int
    total_ohlc_violations: int


def validate_range_1m(instrument_id: int, d0: date, d1: date) -> tuple[pd.DataFrame, RangeSummary]:
    #Validate daily candles for [d0, d1) UTC.
    #Returns:
    #  - DataFrame with one row per day
    #  - Summary totals
 
    rows = []
    total_rows = total_missing = total_dup = total_viol = 0
    days_checked = 0
    days_with_data = 0
    days_empty = 0

    d = d0
    while d < d1:
        days_checked += 1
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        df = load_candles_1m_df(instrument_id, start, end)
        res = validate_day_1m(df)

        if res.rows == 0:
            days_empty += 1
        else:
            days_with_data += 1

        total_rows += res.rows
        total_missing += res.missing_minutes
        total_dup += res.duplicates
        total_viol += res.ohlc_violations

        rows.append(
            dict(
                day_utc=str(d),
                rows=res.rows,
                duplicates=res.duplicates,
                missing_minutes=res.missing_minutes,
                ohlc_violations=res.ohlc_violations,
                first_ts=res.first_ts,
                last_ts=res.last_ts,
            )
        )
        d += timedelta(days=1)

    df_days = pd.DataFrame(rows)

    summary = RangeSummary(
        days_checked=days_checked,
        days_with_data=days_with_data,
        days_empty=days_empty,
        total_rows=total_rows,
        total_missing_minutes=total_missing,
        total_duplicates=total_dup,
        total_ohlc_violations=total_viol,
    )
    return df_days, summary