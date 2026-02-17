from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DayValidationResult:
    rows: int
    duplicates: int
    missing_minutes: int
    ohlc_violations: int
    first_ts: str | None
    last_ts: str | None


def validate_day_1m(df: pd.DataFrame) -> DayValidationResult:
    if df.empty:
        return DayValidationResult(
            rows=0,
            duplicates=0,
            missing_minutes=0,
            ohlc_violations=0,
            first_ts=None,
            last_ts=None,
        )

    # Ensure sorted
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Duplicates
    duplicates = int(df["ts_utc"].duplicated().sum())

    # Missing minutes inside the day
    # Compute diffs between consecutive timestamps and count any gaps > 1 minute.
    ts = pd.to_datetime(df["ts_utc"], utc=True)
    diffs = ts.diff().dropna()

    missing = diffs[diffs > pd.Timedelta(minutes=1)]
    missing_minutes = int(sum((d / pd.Timedelta(minutes=1)) - 1 for d in missing))

    # OHLC sanity: low <= open/close <= high AND low <= high
    o = df["bid_o"]
    h = df["bid_h"]
    l = df["bid_l"]
    c = df["bid_c"]

    violations = ~((l <= o) & (o <= h) & (l <= c) & (c <= h) & (l <= h))
    ohlc_violations = int(violations.sum())

    return DayValidationResult(
        rows=int(len(df)),
        duplicates=duplicates,
        missing_minutes=missing_minutes,
        ohlc_violations=ohlc_violations,
        first_ts=str(ts.iloc[0]),
        last_ts=str(ts.iloc[-1]),
    )