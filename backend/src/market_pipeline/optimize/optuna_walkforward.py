from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json

import optuna
import pandas as pd

from market_pipeline.backtest.artifacts import save_backtest_artifacts
from market_pipeline.backtest.engine import ExecutionParams, TrailingParams, run_backtest
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.marketdata.cleaning import clean_candles_df
from market_pipeline.montecarlo.garch_fold_mc import (
    GarchFoldContext,
    build_garch_fold_context,
    evaluate_fold_monte_carlo,
)
from market_pipeline.optimize.splits import Fold, walkforward_splits
from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals


@dataclass(frozen=True)
class PreparedFold:
    fold: Fold
    hist_input: pd.DataFrame
    eval_start: pd.Timestamp
    mc_context: GarchFoldContext


def _historical_score(metrics) -> float:
    if metrics.trades < 20:
        return -1e9
    dd = abs(metrics.max_drawdown_pct)
    if dd < 1e-9:
        return metrics.total_return_pct
    return metrics.total_return_pct / dd


def run_optuna_walkforward(
    *,
    candles_all: pd.DataFrame,
    date_from: date,
    date_to: date,
    timeframe: str,
    train_months: int,
    test_months: int,
    trials: int,
    n_jobs: int,
    seed: int,
    study_name: str,
    spread_pips: float,
    commission_per_trade: float,
    risk_per_trade: float,
    initial_equity: float,
    max_leverage: float,
    trailing_allowed: bool,
    warmup_bars: int,
    mc_simulations: int,
    hist_weight: float,
    mc_weight: float,
    garch_dist: str,
    garch_p: int,
    garch_q: int,
    garch_burn: int,
    demean_returns: bool,
    return_vol_scale: float,
    wick_vol_scale: float,
    out_dir: Path,
    save_best_artifacts: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    candles_all, _ = clean_candles_df(candles_all)
    candles_all = candles_all.sort_values("ts_utc").reset_index(drop=True)
    candles_all["ts_utc"] = pd.to_datetime(candles_all["ts_utc"], utc=True)

    folds = walkforward_splits(date_from, date_to, train_months, test_months)
    if len(folds) < 2:
        raise ValueError("Not enough folds. Increase data range or reduce train/test months.")

    tf_delta = pd.to_timedelta(timeframe)
    warmup_delta = tf_delta * warmup_bars
    global_start = pd.Timestamp(date_from, tz="UTC")

    w_sum = hist_weight + mc_weight
    if w_sum <= 0:
        raise ValueError("hist_weight + mc_weight must be > 0")
    hist_weight = hist_weight / w_sum
    mc_weight = mc_weight / w_sum

    prepared_folds: list[PreparedFold] = []

    for f in folds:
        train_start_ts = pd.Timestamp(f.train_start, tz="UTC")
        train_end_ts = pd.Timestamp(f.train_end, tz="UTC")
        test_start_ts = pd.Timestamp(f.test_start, tz="UTC")
        test_end_ts = pd.Timestamp(f.test_end, tz="UTC")

        pre_start = test_start_ts - tf_delta - warmup_delta
        if pre_start < global_start:
            pre_start = global_start

        hist_input = candles_all[(candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < test_end_ts)].copy()
        train_candles = candles_all[(candles_all["ts_utc"] >= train_start_ts) & (candles_all["ts_utc"] < train_end_ts)].copy()
        test_candles = candles_all[(candles_all["ts_utc"] >= test_start_ts) & (candles_all["ts_utc"] < test_end_ts)].copy()

        mc_context = build_garch_fold_context(
            train_candles=train_candles,
            test_candles=test_candles,
            train_start=f.train_start,
            train_end=f.train_end,
            test_start=f.test_start,
            test_end=f.test_end,
            warmup_bars=warmup_bars,
            burn=garch_burn,
            p=garch_p,
            q=garch_q,
            dist=garch_dist,
            demean_returns=demean_returns,
            return_vol_scale=return_vol_scale,
            wick_vol_scale=wick_vol_scale,
        )

        prepared_folds.append(
            PreparedFold(
                fold=f,
                hist_input=hist_input,
                eval_start=test_start_ts - tf_delta,
                mc_context=mc_context,
            )
        )

    storage = f"sqlite:///{(out_dir / 'walkforward_mc_studies.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        ema_fast = trial.suggest_int("ema_fast", 5, 80)
        ema_trend = trial.suggest_int("ema_trend", 80, 300)
        if ema_fast >= ema_trend:
            raise optuna.TrialPruned()

        rsi_period = trial.suggest_int("rsi_period", 7, 21)
        rsi_long_max = trial.suggest_float("rsi_long_max", 55.0, 85.0)
        rsi_short_min = trial.suggest_float("rsi_short_min", 15.0, 45.0)
        if rsi_short_min >= rsi_long_max:
            raise optuna.TrialPruned()

        atr_period = trial.suggest_int("atr_period", 7, 28)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 4.0)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 1.0, 8.0)
        if atr_tp_mult < atr_sl_mult * 0.8:
            raise optuna.TrialPruned()

        macd_fast = trial.suggest_int("macd_fast", 6, 16)
        macd_slow = trial.suggest_int("macd_slow", 18, 40)
        if macd_fast >= macd_slow:
            raise optuna.TrialPruned()
        macd_signal = trial.suggest_int("macd_signal", 5, 15)
        macd_hist_threshold = trial.suggest_float("macd_hist_threshold", 0.0, 0.0002)

        cooldown_bars = trial.suggest_int("cooldown_bars", 0, 20)

        trailing_enabled = False
        atr_trail_start_mult = 1.5
        atr_trail_mult = 2.0
        if trailing_allowed:
            trailing_enabled = trial.suggest_categorical("trailing", [False, True])
            if trailing_enabled:
                atr_trail_start_mult = trial.suggest_float("atr_trail_start_mult", 0.5, 3.0)
                atr_trail_mult = trial.suggest_float("atr_trail_mult", 0.8, 4.0)

        p = StrategyParams(
            ema_fast=ema_fast,
            ema_trend=ema_trend,
            rsi_period=rsi_period,
            rsi_long_max=rsi_long_max,
            rsi_short_min=rsi_short_min,
            atr_period=atr_period,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            macd_hist_threshold=macd_hist_threshold,
            cooldown_bars=cooldown_bars,
        )

        exec_params = ExecutionParams(
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
            max_leverage=max_leverage,
            spread_pips=spread_pips,
            commission_per_trade=commission_per_trade,
        )
        trailing_params = TrailingParams(
            enabled=trailing_enabled,
            atr_trail_start_mult=atr_trail_start_mult,
            atr_trail_mult=atr_trail_mult,
        )

        fold_scores = []
        total_hist_trades = 0

        for fold_idx, pf in enumerate(prepared_folds, start=1):
            df_feat = prepare_features_and_signals(pf.hist_input, p)
            df_eval = df_feat[df_feat["ts_utc"] >= pf.eval_start].copy()

            if df_eval.empty:
                return -1e9

            hist_result = run_backtest(
                df_eval,
                atr_col=f"atr_{p.atr_period}",
                sl_atr_mult=p.atr_sl_mult,
                tp_atr_mult=p.atr_tp_mult,
                cooldown_bars=p.cooldown_bars,
                exec_params=exec_params,
                trailing=trailing_params,
            )
            hist_metrics = compute_metrics(hist_result)
            total_hist_trades += hist_metrics.trades
            hist_score = _historical_score(hist_metrics)

            mc_summary = evaluate_fold_monte_carlo(
                context=pf.mc_context,
                strategy_params=p,
                exec_params=exec_params,
                trailing=trailing_params,
                simulations=mc_simulations,
                seed=seed + trial.number * 10_000 + fold_idx * 100,
            )
            mc_score = float(mc_summary["score"])

            fold_scores.append(hist_weight * hist_score + mc_weight * mc_score)

        if total_hist_trades < 80:
            return -1e9

        return float(sum(fold_scores) / len(fold_scores))

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs, gc_after_trial=True)

    best = study.best_trial.params

    p = StrategyParams(
        ema_fast=int(best["ema_fast"]),
        ema_trend=int(best["ema_trend"]),
        rsi_period=int(best["rsi_period"]),
        rsi_long_max=float(best["rsi_long_max"]),
        rsi_short_min=float(best["rsi_short_min"]),
        atr_period=int(best["atr_period"]),
        atr_sl_mult=float(best["atr_sl_mult"]),
        atr_tp_mult=float(best["atr_tp_mult"]),
        macd_fast=int(best["macd_fast"]),
        macd_slow=int(best["macd_slow"]),
        macd_signal=int(best["macd_signal"]),
        macd_hist_threshold=float(best["macd_hist_threshold"]),
        cooldown_bars=int(best["cooldown_bars"]),
    )

    trailing_enabled = bool(best.get("trailing", False))
    atr_trail_start_mult = float(best.get("atr_trail_start_mult", 1.5))
    atr_trail_mult = float(best.get("atr_trail_mult", 2.0))

    exec_params = ExecutionParams(
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        max_leverage=max_leverage,
        spread_pips=spread_pips,
        commission_per_trade=commission_per_trade,
    )
    trailing_params = TrailingParams(
        enabled=trailing_enabled,
        atr_trail_start_mult=atr_trail_start_mult,
        atr_trail_mult=atr_trail_mult,
    )

    per_fold = []

    for idx, pf in enumerate(prepared_folds, start=1):
        df_feat = prepare_features_and_signals(pf.hist_input, p)
        df_eval = df_feat[df_feat["ts_utc"] >= pf.eval_start].copy()

        hist_result = run_backtest(
            df_eval,
            atr_col=f"atr_{p.atr_period}",
            sl_atr_mult=p.atr_sl_mult,
            tp_atr_mult=p.atr_tp_mult,
            cooldown_bars=p.cooldown_bars,
            exec_params=exec_params,
            trailing=trailing_params,
        )
        hist_metrics = compute_metrics(hist_result)
        hist_score = _historical_score(hist_metrics)

        mc_summary = evaluate_fold_monte_carlo(
            context=pf.mc_context,
            strategy_params=p,
            exec_params=exec_params,
            trailing=trailing_params,
            simulations=mc_simulations,
            seed=seed + idx * 1000,
        )

        combined_fold_score = hist_weight * hist_score + mc_weight * float(mc_summary["score"])

        trades_path = None
        equity_path = None
        if save_best_artifacts:
            fold_dir = out_dir / "best_artifacts" / f"fold_{idx:02d}_{pf.fold.test_start}_{pf.fold.test_end}"
            trades_path, equity_path = save_backtest_artifacts(
                hist_result,
                fold_dir,
                prefix=f"{study_name}_{timeframe}",
            )

        per_fold.append(
            {
                "fold": idx,
                "train": [str(pf.fold.train_start), str(pf.fold.train_end)],
                "test": [str(pf.fold.test_start), str(pf.fold.test_end)],
                "historical_test_metrics": asdict(hist_metrics),
                "historical_score": hist_score,
                "mc_summary": mc_summary,
                "combined_fold_score": combined_fold_score,
                "trades_parquet": str(trades_path) if trades_path else None,
                "equity_parquet": str(equity_path) if equity_path else None,
            }
        )

    output = {
        "study_name": study.study_name,
        "timeframe": timeframe,
        "date_from": str(date_from),
        "date_to": str(date_to),
        "train_months": train_months,
        "test_months": test_months,
        "folds": len(prepared_folds),
        "mc_simulations_per_fold": mc_simulations,
        "objective_weights": {
            "historical_weight": hist_weight,
            "mc_weight": mc_weight,
        },
        "garch_model": {
            "dist": garch_dist,
            "p": garch_p,
            "q": garch_q,
            "burn": garch_burn,
        },
        "mc_calibration": {
            "demean_returns": demean_returns,
            "return_vol_scale": return_vol_scale,
            "wick_vol_scale": wick_vol_scale,
        },
        "best_value": study.best_value,
        "best_params": best,
        "per_fold_results": per_fold,
    }
    return output


def save_walkforward_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")