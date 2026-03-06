from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from market_pipeline.marketdata.features import build_basic_features


@dataclass(frozen=True)
class StrategyParams:
    #trend/pullback EMAs
    ema_fast: int = 20
    ema_trend: int = 200

    #RSI
    rsi_period: int = 14
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0

    #ATR risk model
    atr_period: int = 14
    atr_sl_mult: float = 2.0
    atr_tp_mult: float = 3.0

    #MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_hist_threshold: float = 0.0

    #volatility filter (ATR% of price)
    atr_pct_min: float | None = None
    atr_pct_max: float | None = None

    #cooldown after a trade closes (bars)
    cooldown_bars: int = 0

    allow_long: bool = True
    allow_short: bool = True


def prepare_features_and_signals(candles: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    ema_periods = sorted(set([p.ema_fast, p.ema_trend]))

    df = build_basic_features(
        candles,
        ema_periods=ema_periods,
        rsi_period=p.rsi_period,
        atr_period=p.atr_period,
        macd_fast=p.macd_fast,
        macd_slow=p.macd_slow,
        macd_signal=p.macd_signal,
    ).copy()

    ema_fast_col = f"ema_{p.ema_fast}"
    ema_trend_col = f"ema_{p.ema_trend}"
    rsi_col = f"rsi_{p.rsi_period}"
    atr_col = f"atr_{p.atr_period}"
    atr_pct_col = f"atr_{p.atr_period}_pct"
    macd_hist_col = f"macd_hist_{p.macd_fast}_{p.macd_slow}_{p.macd_signal}"

    #trend direction
    trend_up = df["bid_c"] > df[ema_trend_col]
    trend_down = df["bid_c"] < df[ema_trend_col]

    #pullback around EMA fast - cross back in trend direction
    pullback_long = (df["bid_c"].shift(1) < df[ema_fast_col].shift(1)) & (df["bid_c"] > df[ema_fast_col])
    pullback_short = (df["bid_c"].shift(1) > df[ema_fast_col].shift(1)) & (df["bid_c"] < df[ema_fast_col])

    #momentum confirmation
    mom_long = df[macd_hist_col] > p.macd_hist_threshold
    mom_short = df[macd_hist_col] < -p.macd_hist_threshold

    #RSI filter
    rsi_long_ok = df[rsi_col] <= p.rsi_long_max
    rsi_short_ok = df[rsi_col] >= p.rsi_short_min

    #volatility filter
    vol_ok = pd.Series(True, index=df.index)
    if p.atr_pct_min is not None:
        vol_ok &= df[atr_pct_col] >= p.atr_pct_min
    if p.atr_pct_max is not None:
        vol_ok &= df[atr_pct_col] <= p.atr_pct_max

    signal_long = p.allow_long and (trend_up & pullback_long & mom_long & rsi_long_ok & vol_ok)
    signal_short = p.allow_short and (trend_down & pullback_short & mom_short & rsi_short_ok & vol_ok)

    df["signal_long"] = signal_long.astype(bool)
    df["signal_short"] = signal_short.astype(bool)

    df["_ema_fast_col"] = ema_fast_col
    df["_ema_trend_col"] = ema_trend_col
    df["_rsi_col"] = rsi_col
    df["_atr_col"] = atr_col
    df["_macd_hist_col"] = macd_hist_col

    return df