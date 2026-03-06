from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import json

import optuna
import pandas as pd

from market_pipeline.optimize.splits import walkforward_splits
from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals
from market_pipeline.backtest.engine import run_backtest, ExecutionParams, TrailingParams
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.backtest.artifacts import save_backtest_artifacts
from market_pipeline.marketdata.cleaning import clean_candles_df


def _score(metrics) -> float:
    #maximize return / abs(dd), must be enough trades
    if metrics.trades < 20:
        return -1e9
    dd = abs(metrics.max_drawdown_pct)
    if dd < 1e-9:
        return metrics.total_return_pct
    return metrics.total_return_pct / dd


def _slice(df: pd.DataFrame, d0: date, d1: date) -> pd.DataFrame:
    start = pd.Timestamp(d0, tz="UTC")
    end = pd.Timestamp(d1, tz="UTC")
    return df[(df["ts_utc"] >= start) & (df["ts_utc"] < end)].copy()


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
    out_dir: Path,
    save_best_artifacts: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    candles_all, _ = clean_candles_df(candles_all)

    folds = walkforward_splits(date_from, date_to, train_months, test_months)
    if len(folds) < 2:
        raise ValueError("Not enough folds. Increase date range or reduce train/test months.")

    storage = f"sqlite:///{(out_dir / 'walkforward_studies.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    tf_delta = pd.to_timedelta(timeframe)  # e.g. 15min
    warmup_delta = tf_delta * warmup_bars

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

        scores = []
        total_trades = 0

        for f in folds:
            test_start = pd.Timestamp(f.test_start, tz="UTC")
            test_end = pd.Timestamp(f.test_end, tz="UTC")
            pre_start = test_start - tf_delta - warmup_delta

            global_start = pd.Timestamp(date_from, tz="UTC")
            if pre_start < global_start:
                pre_start = global_start

            df_slice = candles_all[(candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < test_end)].copy()
            if df_slice.empty:
                return -1e9

            df_feat = prepare_features_and_signals(df_slice, p)

            start_engine = test_start - tf_delta
            df_eval = df_feat[(df_feat["ts_utc"] >= start_engine) & (df_feat["ts_utc"] < test_end)].copy()
            if df_eval.empty:
                return -1e9

            res = run_backtest(
                df_eval,
                atr_col=f"atr_{atr_period}",
                sl_atr_mult=atr_sl_mult,
                tp_atr_mult=atr_tp_mult,
                cooldown_bars=cooldown_bars,
                exec_params=ExecutionParams(
                    initial_equity=initial_equity,
                    risk_per_trade=risk_per_trade,
                    max_leverage=max_leverage,
                    spread_pips=spread_pips,
                    commission_per_trade=commission_per_trade,
                ),
                trailing=TrailingParams(
                    enabled=trailing_enabled,
                    atr_trail_start_mult=atr_trail_start_mult,
                    atr_trail_mult=atr_trail_mult,
                ),
            )
            m = compute_metrics(res)
            total_trades += m.trades
            scores.append(_score(m))

        if total_trades < 60:
            return -1e9

        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs, gc_after_trial=True)

    best = study.best_trial.params

    per_fold = []
    best_params = best.copy()

    p = StrategyParams(
        ema_fast=int(best_params["ema_fast"]),
        ema_trend=int(best_params["ema_trend"]),
        rsi_period=int(best_params["rsi_period"]),
        rsi_long_max=float(best_params["rsi_long_max"]),
        rsi_short_min=float(best_params["rsi_short_min"]),
        atr_period=int(best_params["atr_period"]),
        atr_sl_mult=float(best_params["atr_sl_mult"]),
        atr_tp_mult=float(best_params["atr_tp_mult"]),
        macd_fast=int(best_params["macd_fast"]),
        macd_slow=int(best_params["macd_slow"]),
        macd_signal=int(best_params["macd_signal"]),
        macd_hist_threshold=float(best_params["macd_hist_threshold"]),
        cooldown_bars=int(best_params["cooldown_bars"]),
    )

    trailing_enabled = bool(best_params.get("trailing", False))
    atr_trail_start_mult = float(best_params.get("atr_trail_start_mult", 1.5))
    atr_trail_mult = float(best_params.get("atr_trail_mult", 2.0))

    for idx, f in enumerate(folds, start=1):
        test_start = pd.Timestamp(f.test_start, tz="UTC")
        test_end = pd.Timestamp(f.test_end, tz="UTC")
        pre_start = test_start - tf_delta - warmup_delta
        global_start = pd.Timestamp(date_from, tz="UTC")
        if pre_start < global_start:
            pre_start = global_start

        df_slice = candles_all[(candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < test_end)].copy()
        df_feat = prepare_features_and_signals(df_slice, p)
        start_engine = test_start - tf_delta
        df_eval = df_feat[(df_feat["ts_utc"] >= start_engine) & (df_feat["ts_utc"] < test_end)].copy()

        res = run_backtest(
            df_eval,
            atr_col=f"atr_{p.atr_period}",
            sl_atr_mult=p.atr_sl_mult,
            tp_atr_mult=p.atr_tp_mult,
            cooldown_bars=p.cooldown_bars,
            exec_params=ExecutionParams(
                initial_equity=initial_equity,
                risk_per_trade=risk_per_trade,
                max_leverage=max_leverage,
                spread_pips=spread_pips,
                commission_per_trade=commission_per_trade,
            ),
            trailing=TrailingParams(
                enabled=trailing_enabled,
                atr_trail_start_mult=atr_trail_start_mult,
                atr_trail_mult=atr_trail_mult,
            ),
        )
        m = compute_metrics(res)

        trades_path = equity_path = None
        if save_best_artifacts:
            fold_dir = out_dir / "best_artifacts" / f"fold_{idx:02d}_{f.test_start}_{f.test_end}"
            trades_path, equity_path = save_backtest_artifacts(res, fold_dir, prefix=f"{study_name}_{timeframe}")

        per_fold.append(
            {
                "fold": idx,
                "train": [str(f.train_start), str(f.train_end)],
                "test": [str(f.test_start), str(f.test_end)],
                "metrics": asdict(m),
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
        "folds": len(folds),
        "best_value": study.best_value,
        "best_params": best_params,
        "per_fold_test_metrics": per_fold,
    }
    return output


def save_walkforward_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")