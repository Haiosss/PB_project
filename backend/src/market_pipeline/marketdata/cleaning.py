from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


REQUIRED_CANDLE_COLS = ["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"]


@dataclass(frozen=True)
class CleaningSummary:
    rows_in: int
    rows_out: int
    dropped_duplicates: int
    dropped_null_ohlc: int
    dropped_invalid_ohlc: int
    dropped_inactive: int


def clean_candles_df(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    
    #Clean candle DataFrame for cached parquet usage, keeps raw DB untouched
    
    #Removes:
    #- duplicate timestamps (keep first)
    #- rows with null OHLC
    #- OHLC-invalid rows
    #- inactive placeholder bars (volume=0 and flat OHLC)
    
    if df.empty:
        empty = pd.DataFrame(columns=REQUIRED_CANDLE_COLS)
        return empty, CleaningSummary(0, 0, 0, 0, 0, 0)

    out = df.copy()

    for c in REQUIRED_CANDLE_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    for c in ["bid_o", "bid_h", "bid_l", "bid_c", "bid_v"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("ts_utc").reset_index(drop=True)

    rows_in = len(out)

    # duplicates by timestamp
    dup_mask = out["ts_utc"].duplicated(keep="first")
    dropped_duplicates = int(dup_mask.sum())
    out = out.loc[~dup_mask].copy()

    # null OHLC rows
    null_ohlc_mask = out[["bid_o", "bid_h", "bid_l", "bid_c"]].isna().any(axis=1)
    dropped_null_ohlc = int(null_ohlc_mask.sum())
    out = out.loc[~null_ohlc_mask].copy()

    # OHLC validity
    o = out["bid_o"]
    h = out["bid_h"]
    l = out["bid_l"]
    c = out["bid_c"]
    valid_ohlc_mask = (l <= o) & (o <= h) & (l <= c) & (c <= h) & (l <= h)
    dropped_invalid_ohlc = int((~valid_ohlc_mask).sum())
    out = out.loc[valid_ohlc_mask].copy()

    # inactive placeholder bars:
    # volume == 0 and OHLC all flat
    flat_mask = (
        np.isclose(out["bid_o"], out["bid_h"])
        & np.isclose(out["bid_h"], out["bid_l"])
        & np.isclose(out["bid_l"], out["bid_c"])
    )
    zero_vol_mask = out["bid_v"].fillna(0).eq(0)
    inactive_mask = flat_mask & zero_vol_mask

    dropped_inactive = int(inactive_mask.sum())
    out = out.loc[~inactive_mask].copy()

    out = out[REQUIRED_CANDLE_COLS].sort_values("ts_utc").reset_index(drop=True)

    summary = CleaningSummary(
        rows_in=rows_in,
        rows_out=len(out),
        dropped_duplicates=dropped_duplicates,
        dropped_null_ohlc=dropped_null_ohlc,
        dropped_invalid_ohlc=dropped_invalid_ohlc,
        dropped_inactive=dropped_inactive,
    )
    return out, summary