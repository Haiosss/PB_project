from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import threading

import numpy as np
import pandas as pd
from arch import arch_model

from market_pipeline.backtest.engine import ExecutionParams, TrailingParams, run_backtest
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.marketdata.cleaning import clean_candles_df
from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals


_SIM_LOCK = threading.Lock()


@dataclass(frozen=True)
class WickTemplate:
    upper_share: float
    wick_sigma_mult: float


@dataclass(frozen=True)
class GarchFoldContext:
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    train_returns_pct: pd.Series
    fitted_params: pd.Series

    warmup_tail: pd.DataFrame
    test_timestamps: pd.Series
    test_volumes: np.ndarray
    bar_delta: pd.Timedelta

    templates_up: list[WickTemplate]
    templates_down: list[WickTemplate]
    templates_flat: list[WickTemplate]
    templates_all: list[WickTemplate]

    scale_factor: float
    burn: int
    p: int
    q: int
    dist: str

    demean_returns: bool
    return_vol_scale: float
    wick_vol_scale: float


def _safe_clip_series(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    if s.empty:
        return s
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def _extract_vol_scaled_wick_templates(
    train_candles: pd.DataFrame,
    sigma_train_decimal: pd.Series,
) -> tuple[list[WickTemplate], list[WickTemplate], list[WickTemplate], list[WickTemplate]]:
    work = train_candles.copy().sort_values("ts_utc").reset_index(drop=True)

    work = work.iloc[1:].copy().reset_index(drop=True)
    sigma = pd.to_numeric(sigma_train_decimal.reset_index(drop=True), errors="coerce").abs()

    body_high = work[["bid_o", "bid_c"]].max(axis=1)
    body_low = work[["bid_o", "bid_c"]].min(axis=1)

    upper_ratio = ((work["bid_h"] / body_high.replace(0, np.nan)) - 1.0).clip(lower=0.0)
    lower_ratio = (1.0 - (work["bid_l"] / body_low.replace(0, np.nan))).clip(lower=0.0)

    total_wick = (upper_ratio + lower_ratio).fillna(0.0)

    sigma = sigma.replace(0, np.nan).bfill().ffill()

    if sigma.isna().all():
        sigma = pd.Series(np.full(len(sigma), 1e-8), index=sigma.index, dtype=float)

    sigma_floor = max(float(sigma.quantile(0.05)), 1e-8)
    sigma = sigma.clip(lower=sigma_floor)

    wick_sigma_mult = (total_wick / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    wick_sigma_mult = _safe_clip_series(wick_sigma_mult, 0.01, 0.99)

    upper_share = np.where(total_wick > 0, upper_ratio / total_wick.replace(0, np.nan), 0.5)
    upper_share = pd.Series(upper_share).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)

    direction = np.where(
        work["bid_c"] > work["bid_o"],
        "up",
        np.where(work["bid_c"] < work["bid_o"], "down", "flat"),
    )

    templates_all = [
        WickTemplate(float(us), float(wm))
        for us, wm in zip(upper_share, wick_sigma_mult)
    ]

    templates_up = [tpl for tpl, d in zip(templates_all, direction) if d == "up"]
    templates_down = [tpl for tpl, d in zip(templates_all, direction) if d == "down"]
    templates_flat = [tpl for tpl, d in zip(templates_all, direction) if d == "flat"]

    if not templates_up:
        templates_up = templates_all
    if not templates_down:
        templates_down = templates_all
    if not templates_flat:
        templates_flat = templates_all

    return templates_up, templates_down, templates_flat, templates_all


def build_garch_fold_context(
    *,
    train_candles: pd.DataFrame,
    test_candles: pd.DataFrame,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    warmup_bars: int = 400,
    scale_factor: float = 100.0,
    burn: int = 500,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    demean_returns: bool = True,
    return_vol_scale: float = 0.8,
    wick_vol_scale: float = 0.75,
) -> GarchFoldContext:
    train_candles, _ = clean_candles_df(train_candles)
    test_candles, _ = clean_candles_df(test_candles)

    train_candles = train_candles.sort_values("ts_utc").reset_index(drop=True)
    test_candles = test_candles.sort_values("ts_utc").reset_index(drop=True)

    if len(train_candles) < max(300, warmup_bars + 10):
        raise ValueError("Not enough train candles for GARCH fit + warmup.")
    if len(test_candles) < 5:
        raise ValueError("Not enough test candles for Monte Carlo evaluation.")

    train_close = pd.to_numeric(train_candles["bid_c"], errors="coerce")
    train_log_ret = np.log(train_close / train_close.shift(1)).dropna()

    if len(train_log_ret) < 200:
        raise ValueError("Not enough train returns for GARCH fit.")

    train_returns_pct = train_log_ret * scale_factor

    am = arch_model(
        train_returns_pct,
        mean="Zero",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)

    sigma_train_decimal = pd.Series(res.conditional_volatility, index=train_returns_pct.index) / scale_factor

    up, down, flat, all_tpl = _extract_vol_scaled_wick_templates(
        train_candles=train_candles,
        sigma_train_decimal=sigma_train_decimal,
    )

    warmup_tail = train_candles.tail(max(warmup_bars, 2)).copy().reset_index(drop=True)

    test_ts = pd.to_datetime(test_candles["ts_utc"], utc=True).reset_index(drop=True)
    test_vol = pd.to_numeric(test_candles["bid_v"], errors="coerce").fillna(0.0).to_numpy()

    if len(test_ts) >= 2:
        bar_delta = test_ts.iloc[1] - test_ts.iloc[0]
    else:
        bar_delta = pd.Timedelta(minutes=15)

    return GarchFoldContext(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        train_returns_pct=train_returns_pct,
        fitted_params=res.params,
        warmup_tail=warmup_tail,
        test_timestamps=test_ts,
        test_volumes=test_vol,
        bar_delta=bar_delta,
        templates_up=up,
        templates_down=down,
        templates_flat=flat,
        templates_all=all_tpl,
        scale_factor=scale_factor,
        burn=burn,
        p=p,
        q=q,
        dist=dist,
        demean_returns=demean_returns,
        return_vol_scale=return_vol_scale,
        wick_vol_scale=wick_vol_scale,
    )


def _simulate_returns_and_sigma(
    context: GarchFoldContext,
    nobs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    with _SIM_LOCK:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            am = arch_model(
                context.train_returns_pct,
                mean="Zero",
                vol="GARCH",
                p=context.p,
                q=context.q,
                dist=context.dist,
                rescale=False,
            )
            sim = am.simulate(context.fitted_params, nobs=nobs, burn=context.burn)
        finally:
            np.random.set_state(state)

    sim_ret = pd.to_numeric(sim["data"], errors="coerce").to_numpy(dtype=float) / context.scale_factor
    sim_sigma = pd.to_numeric(sim["volatility"], errors="coerce").to_numpy(dtype=float) / context.scale_factor
    return sim_ret, sim_sigma


def _pick_template(context: GarchFoldContext, direction: str, rng: np.random.Generator) -> WickTemplate:
    if direction == "up":
        pool = context.templates_up
    elif direction == "down":
        pool = context.templates_down
    else:
        pool = context.templates_flat

    if not pool:
        pool = context.templates_all

    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def generate_synthetic_test_candles(context: GarchFoldContext, seed: int) -> pd.DataFrame:
    nobs = len(context.test_timestamps)
    sim_log_returns, sim_sigma = _simulate_returns_and_sigma(context, nobs=nobs, seed=seed)

    if context.demean_returns:
        sim_log_returns = sim_log_returns - np.nanmean(sim_log_returns)

    sim_log_returns = sim_log_returns * context.return_vol_scale
    sim_sigma = sim_sigma * context.return_vol_scale

    rng = np.random.default_rng(seed + 12345)
    prev_close = float(context.warmup_tail["bid_c"].iloc[-1])

    rows = []

    for i in range(nobs):
        ts = context.test_timestamps.iloc[i]
        vol = float(context.test_volumes[i]) if i < len(context.test_volumes) else 0.0
        r = float(sim_log_returns[i])
        sigma_t = max(float(sim_sigma[i]), 1e-8)

        o = prev_close
        c = o * float(np.exp(r))

        if c > o:
            direction = "up"
        elif c < o:
            direction = "down"
        else:
            direction = "flat"

        tpl = _pick_template(context, direction, rng)

        total_wick_pct = max(tpl.wick_sigma_mult * sigma_t * context.wick_vol_scale, 0.0)

        total_wick_pct = min(total_wick_pct, 0.02)

        upper_ratio = total_wick_pct * tpl.upper_share
        lower_ratio = total_wick_pct * (1.0 - tpl.upper_share)

        body_high = max(o, c)
        body_low = min(o, c)

        h = body_high * (1.0 + upper_ratio)
        l = body_low * max(1.0 - lower_ratio, 1e-9)

        rows.append(
            {
                "ts_utc": ts,
                "bid_o": o,
                "bid_h": h,
                "bid_l": l,
                "bid_c": c,
                "bid_v": vol,
            }
        )

        prev_close = c

    synth_df = pd.DataFrame(rows)
    out = pd.concat([context.warmup_tail, synth_df], ignore_index=True)
    return out


def mc_summary_to_score(summary: dict) -> float:
    median_dd = abs(float(summary["median_max_drawdown_pct"]))
    if median_dd < 1e-9:
        base = float(summary["median_return_pct"])
    else:
        blended_return = 0.7 * float(summary["median_return_pct"]) + 0.3 * float(summary["q05_return_pct"])
        base = blended_return / median_dd

    penalty = 0.5 * float(summary["prob_loss"])
    return base - penalty


def evaluate_fold_monte_carlo(
    *,
    context: GarchFoldContext,
    strategy_params: StrategyParams,
    exec_params: ExecutionParams,
    trailing: TrailingParams,
    simulations: int = 30,
    seed: int = 42,
) -> dict:
    metrics_rows = []

    test_start_ts = pd.Timestamp(context.test_start, tz="UTC")
    eval_start = test_start_ts - context.bar_delta

    for sim_idx in range(simulations):
        sim_seed = seed + sim_idx
        mc_candles = generate_synthetic_test_candles(context, seed=sim_seed)

        df_feat = prepare_features_and_signals(mc_candles, strategy_params)
        df_eval = df_feat[df_feat["ts_utc"] >= eval_start].copy()

        if df_eval.empty:
            continue

        result = run_backtest(
            df_eval,
            atr_col=f"atr_{strategy_params.atr_period}",
            sl_atr_mult=strategy_params.atr_sl_mult,
            tp_atr_mult=strategy_params.atr_tp_mult,
            cooldown_bars=strategy_params.cooldown_bars,
            exec_params=exec_params,
            trailing=trailing,
        )
        m = compute_metrics(result)

        metrics_rows.append(
            {
                "trades": m.trades,
                "total_return_pct": m.total_return_pct,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe": m.sharpe,
                "winrate_pct": m.winrate_pct,
                "profit_factor": m.profit_factor,
            }
        )

    if not metrics_rows:
        return {
            "simulations": simulations,
            "successful_simulations": 0,
            "median_return_pct": -999.0,
            "q05_return_pct": -999.0,
            "median_max_drawdown_pct": -999.0,
            "prob_loss": 1.0,
            "prob_pf_gt_1": 0.0,
            "median_trades": 0.0,
            "score": -1e9,
        }

    dfm = pd.DataFrame(metrics_rows)

    summary = {
        "simulations": simulations,
        "successful_simulations": int(len(dfm)),
        "median_return_pct": float(dfm["total_return_pct"].median()),
        "q05_return_pct": float(dfm["total_return_pct"].quantile(0.05)),
        "median_max_drawdown_pct": float(dfm["max_drawdown_pct"].median()),
        "prob_loss": float((dfm["total_return_pct"] < 0.0).mean()),
        "prob_pf_gt_1": float((dfm["profit_factor"] > 1.0).mean()),
        "median_trades": float(dfm["trades"].median()),
    }
    summary["score"] = float(mc_summary_to_score(summary))
    return summary