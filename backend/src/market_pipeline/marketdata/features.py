from __future__ import annotations

from datetime import date
from pathlib import Path
import math

import numpy as np
import pandas as pd


def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    out = out.sort_values("ts_utc").reset_index(drop=True)

    for c in ["bid_o", "bid_h", "bid_l", "bid_c", "bid_v"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["bid_c"]

    out["ret_pct"] = close.pct_change()
    out["ret_log"] = np.log(close / close.shift(1))

    return out


def add_ema(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    out = df.copy()
    close = out["bid_c"]

    for p in sorted(set(periods)):
        if p <= 0:
            raise ValueError(f"EMA period must be > 0, got {p}")
        out[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if period <= 0:
        raise ValueError("RSI period must be > 0")

    out = df.copy()
    close = out["bid_c"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # edge cases
    rsi = rsi.where(~((avg_gain > 0) & (avg_loss == 0)), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    out[f"rsi_{period}"] = rsi
    return out


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if period <= 0:
        raise ValueError("ATR period must be > 0")

    out = df.copy()

    high = out["bid_h"]
    low = out["bid_l"]
    close = out["bid_c"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    out["tr"] = tr
    out[f"atr_{period}"] = atr
    out[f"atr_{period}_pct"] = atr / close  # useful for volatility-normalized filtering

    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if min(fast, slow, signal) <= 0:
        raise ValueError("MACD periods must be > 0")
    if fast >= slow:
        raise ValueError("MACD fast period should be < slow period")

    out = df.copy()
    close = out["bid_c"]

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    out[f"macd_line_{fast}_{slow}"] = macd_line
    out[f"macd_signal_{fast}_{slow}_{signal}"] = macd_signal
    out[f"macd_hist_{fast}_{slow}_{signal}"] = macd_hist

    return out


def build_basic_features(
    df: pd.DataFrame,
    *,
    ema_periods: list[int] | None = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
 
    # builds a basic feature set on top of candle data.
    # expects float prices (works with ints)
 
    if ema_periods is None:
        ema_periods = [20, 50, 200]

    out = _ensure_price_columns(df)
    out = add_returns(out)
    out = add_ema(out, ema_periods)
    out = add_rsi(out, period=rsi_period)
    out = add_atr(out, period=atr_period)
    out = add_macd(out, fast=macd_fast, slow=macd_slow, signal=macd_signal)

    return out


def save_features_range_parquet(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    symbol: str,
    timeframe: str,
    date_from: date,
    date_to: date,
    feature_set_name: str = "basic_v1",
) -> Path:
 
    #save feature dataframe for a specific range into parquet cache
 
    tf = timeframe.lower().strip()
    path = (
        out_dir
        / "features"
        / f"feature_set={feature_set_name}"
        / f"symbol={symbol.upper()}"
        / f"timeframe={tf}"
        / f"range={date_from.isoformat()}_{date_to.isoformat()}"
        / "data.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path