from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

import pandas as pd

from market_pipeline.backtest.models import BacktestResult, Trade


@dataclass
class ExecutionParams:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01
    max_leverage: float = 20.0
    pip_size: float = 0.0001
    spread_pips: float = 0.0
    commission_per_trade: float = 0.0


@dataclass
class TrailingParams:
    enabled: bool = False
    atr_trail_start_mult: float = 1.5
    atr_trail_mult: float = 2.0


def run_backtest(
    df: pd.DataFrame,
    *,
    atr_col: str,
    sl_atr_mult: float,
    tp_atr_mult: float,
    cooldown_bars: int,
    exec_params: ExecutionParams,
    trailing: TrailingParams,
) -> BacktestResult:
    
    required = ["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "signal_long", "signal_short", atr_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for backtest: {missing}")

    work = df.copy()
    work["ts_utc"] = pd.to_datetime(work["ts_utc"], utc=True)

    #drop rows without ATR or prices (warm-up)
    work = work.dropna(subset=["bid_o", "bid_h", "bid_l", "bid_c", atr_col]).reset_index(drop=True)
    if len(work) < 3:
        return BacktestResult(trades=[], equity_curve=[], equity_timestamps=[])

    equity = exec_params.initial_equity
    equity_curve: list[float] = []
    ts_list: list[datetime] = []
    trades: list[Trade] = []

    half_spread = (exec_params.spread_pips * exec_params.pip_size) / 2.0

    pos_side: str | None = None
    entry_price_eff = 0.0
    entry_time: datetime | None = None
    size = 0.0
    stop = 0.0
    tp = 0.0
    atr_entry = 0.0

    high_water = -math.inf
    low_water = math.inf

    cooldown_left = 0

    long_entry = work["signal_long"].shift(1).fillna(False).astype(bool).to_list()
    short_entry = work["signal_short"].shift(1).fillna(False).astype(bool).to_list()

    for i in range(len(work)):
        row = work.iloc[i]
        t: datetime = row["ts_utc"].to_pydatetime()
        o = float(row["bid_o"])
        h = float(row["bid_h"])
        l = float(row["bid_l"])
        c = float(row["bid_c"])
        atr = float(row[atr_col])

        if pos_side is None:
            equity_curve.append(equity)
        else:
            if pos_side == "long":
                unreal = size * ((c - half_spread) - entry_price_eff)
            else:
                unreal = size * (entry_price_eff - (c + half_spread))
            equity_curve.append(equity + unreal)

        ts_list.append(t)

        if cooldown_left > 0 and pos_side is None:
            cooldown_left -= 1

        if pos_side is not None and entry_time is not None:
            high_water = max(high_water, h)
            low_water = min(low_water, l)

            #trailing stop update
            if trailing.enabled:
                if pos_side == "long":
                    if high_water >= (entry_price_eff + trailing.atr_trail_start_mult * atr_entry):
                        trail_stop = high_water - trailing.atr_trail_mult * atr_entry
                        stop = max(stop, trail_stop)
                else:
                    if low_water <= (entry_price_eff - trailing.atr_trail_start_mult * atr_entry):
                        trail_stop = low_water + trailing.atr_trail_mult * atr_entry
                        stop = min(stop, trail_stop)

            #exit checks
            if pos_side == "long":
                stop_hit = l <= stop
                tp_hit = h >= tp
                if stop_hit or tp_hit:
                    reason = "stop" if stop_hit else "tp"
                    exit_price = stop if stop_hit else tp
                    exit_price_eff = exit_price - half_spread
                    pnl = size * (exit_price_eff - entry_price_eff) - exec_params.commission_per_trade
                    ret_pct = pnl / max(exec_params.initial_equity, 1e-9)

                    equity += pnl
                    trades.append(
                        Trade(
                            side="long",
                            entry_time=entry_time,
                            exit_time=t,
                            entry_price=entry_price_eff,
                            exit_price=exit_price_eff,
                            size=size,
                            pnl=pnl,
                            return_pct=ret_pct,
                            reason=reason,
                        )
                    )
                    pos_side = None
                    cooldown_left = max(cooldown_left, 0)
                    if cooldown_bars > 0:
                        cooldown_left = cooldown_bars
            else:
                stop_hit = h >= stop
                tp_hit = l <= tp
                if stop_hit or tp_hit:
                    reason = "stop" if stop_hit else "tp"
                    exit_price = stop if stop_hit else tp
                    exit_price_eff = exit_price + half_spread
                    pnl = size * (entry_price_eff - exit_price_eff) - exec_params.commission_per_trade
                    ret_pct = pnl / max(exec_params.initial_equity, 1e-9)

                    equity += pnl
                    trades.append(
                        Trade(
                            side="short",
                            entry_time=entry_time,
                            exit_time=t,
                            entry_price=entry_price_eff,
                            exit_price=exit_price_eff,
                            size=size,
                            pnl=pnl,
                            return_pct=ret_pct,
                            reason=reason,
                        )
                    )
                    pos_side = None
                    if cooldown_bars > 0:
                        cooldown_left = cooldown_bars

        if pos_side is None and cooldown_left == 0 and i > 0:
            go_long = long_entry[i]
            go_short = short_entry[i]

            if go_long and not go_short:
                entry_price_eff = o + half_spread
                entry_time = t
                atr_entry = atr

                stop = entry_price_eff - sl_atr_mult * atr_entry
                tp = entry_price_eff + tp_atr_mult * atr_entry

                stop_dist = entry_price_eff - stop
                if stop_dist <= 0:
                    continue

                risk_amount = equity * exec_params.risk_per_trade
                size = risk_amount / stop_dist

                #leverage cap
                notional = size * entry_price_eff
                max_notional = equity * exec_params.max_leverage
                if notional > max_notional:
                    size *= max_notional / notional

                pos_side = "long"
                high_water = o
                low_water = o

            elif go_short and not go_long:
                entry_price_eff = o - half_spread
                entry_time = t
                atr_entry = atr

                stop = entry_price_eff + sl_atr_mult * atr_entry
                tp = entry_price_eff - tp_atr_mult * atr_entry

                stop_dist = stop - entry_price_eff
                if stop_dist <= 0:
                    continue

                risk_amount = equity * exec_params.risk_per_trade
                size = risk_amount / stop_dist

                notional = size * entry_price_eff
                max_notional = equity * exec_params.max_leverage
                if notional > max_notional:
                    size *= max_notional / notional

                pos_side = "short"
                high_water = o
                low_water = o

    if pos_side is not None and entry_time is not None:
        last = work.iloc[-1]
        t = last["ts_utc"].to_pydatetime()
        c = float(last["bid_c"])
        if pos_side == "long":
            exit_eff = c - half_spread
            pnl = size * (exit_eff - entry_price_eff) - exec_params.commission_per_trade
            trades.append(Trade("long", entry_time, t, entry_price_eff, exit_eff, size, pnl, pnl / exec_params.initial_equity, "end"))
        else:
            exit_eff = c + half_spread
            pnl = size * (entry_price_eff - exit_eff) - exec_params.commission_per_trade
            trades.append(Trade("short", entry_time, t, entry_price_eff, exit_eff, size, pnl, pnl / exec_params.initial_equity, "end"))

    return BacktestResult(trades=trades, equity_curve=equity_curve, equity_timestamps=ts_list)