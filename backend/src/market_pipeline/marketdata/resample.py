from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from market_pipeline.marketdata.cleaning import CleaningSummary, clean_candles_df
from market_pipeline.marketdata.queries import load_candles_1m_df


@dataclass(frozen=True)
class ResampleDayCleaningLog:
    day_utc: date
    clean_1m: CleaningSummary
    clean_tf: CleaningSummary
    tf_rows_written: int


@dataclass(frozen=True)
class ResampleExportResult:
    base_path: Path
    days_written: int
    total_rows: int
    cleaning_logs: list[ResampleDayCleaningLog]


def resample_ohlcv_day(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    #resample one day's 1m BID candles to a higher timeframe

    if df_1m.empty:
        return pd.DataFrame(columns=["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"])

    df = df_1m.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.sort_values("ts_utc").set_index("ts_utc")

    agg = {
        "bid_o": "first",
        "bid_h": "max",
        "bid_l": "min",
        "bid_c": "last",
        "bid_v": "sum",
    }

    out = (
        df.resample(
            timeframe,
            label="left",
            closed="left",
            origin="start_day",  # align bars to 00:00 UTC
        )
        .agg(agg)
        .dropna(subset=["bid_o", "bid_h", "bid_l", "bid_c"])
        .reset_index()
    )

    return out


def export_resampled_range_to_parquet(
    instrument_id: int,
    symbol: str,
    d0: date,
    d1: date,
    timeframe: str,
    out_dir: Path,
) -> ResampleExportResult:
    #resample [d0, d1) from DB and export parquet partitioned by day and returns export summary + per day cleaning summaries
    
    tf_norm = timeframe.lower()
    base = out_dir / "candles_resampled" / f"symbol={symbol.upper()}" / f"timeframe={tf_norm}"
    base.mkdir(parents=True, exist_ok=True)

    days_written = 0
    total_rows = 0
    cleaning_logs: list[ResampleDayCleaningLog] = []

    d = d0
    while d < d1:
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        # 1m raw from DB clean before resampling
        df_1m = load_candles_1m_df(instrument_id, start, end)
        df_1m, clean_1m = clean_candles_df(df_1m)

        df_tf = resample_ohlcv_day(df_1m, timeframe)

        df_tf, clean_tf = clean_candles_df(df_tf)

        rows_written_today = 0
        if not df_tf.empty:
            day_dir = base / f"day={d.isoformat()}"
            day_dir.mkdir(parents=True, exist_ok=True)
            df_tf.to_parquet(day_dir / "data.parquet", index=False)

            rows_written_today = int(len(df_tf))
            days_written += 1
            total_rows += rows_written_today

        cleaning_logs.append(
            ResampleDayCleaningLog(
                day_utc=d,
                clean_1m=clean_1m,
                clean_tf=clean_tf,
                tf_rows_written=rows_written_today,
            )
        )

        d += timedelta(days=1)

    return ResampleExportResult(
        base_path=base,
        days_written=days_written,
        total_rows=total_rows,
        cleaning_logs=cleaning_logs,
    )