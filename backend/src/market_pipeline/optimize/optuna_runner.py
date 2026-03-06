from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import json
import optuna

from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals
from market_pipeline.backtest.engine import run_backtest, ExecutionParams, TrailingParams
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.marketdata.cleaning import clean_candles_df


def _score(metrics) -> float:

    #maximize: return / abs(max_drawdown)
    #penalize too few trades

    if metrics.trades < 50:
        return -1e9

    dd = abs(metrics.max_drawdown_pct)
    if dd < 1e-9:
        return metrics.total_return_pct

    return metrics.total_return_pct / dd


def run_optuna(
    *,
    candles_train,
    candles_test,
    timeframe: str,
    trials: int,
    n_jobs: int,
    study_name: str,
    storage_path: Path,
    seed: int,
    spread_pips: float,
    commission_per_trade: float,
    risk_per_trade: float,
    initial_equity: float,
    max_leverage: float,
    trailing_allowed: bool,
) -> dict:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    candles_train, _ = clean_candles_df(candles_train)
    candles_test, _ = clean_candles_df(candles_test)

    def objective(trial: optuna.Trial) -> float:
        #strategy hyperparameters
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

        macd_hist_threshold = trial.suggest_float("macd_hist_threshold", 0.0, 0.00015)

        cooldown_bars = trial.suggest_int("cooldown_bars", 0, 20)

        #optional volatility filter (ATR%)
        use_atr_filter = trial.suggest_categorical("use_atr_filter", [False, True])
        atr_pct_min = None
        atr_pct_max = None
        if use_atr_filter:
            atr_pct_min = trial.suggest_float("atr_pct_min", 0.0, 0.002)
            atr_pct_max = trial.suggest_float("atr_pct_max", atr_pct_min, 0.01)

        #optional trailing
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
            atr_pct_min=atr_pct_min,
            atr_pct_max=atr_pct_max,
            cooldown_bars=cooldown_bars,
        )

        df = prepare_features_and_signals(candles_train, p)
        atr_col = f"atr_{atr_period}"

        result = run_backtest(
            df,
            atr_col=atr_col,
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

        m = compute_metrics(result)
        score = _score(m)

        trial.set_user_attr("trades", m.trades)
        trial.set_user_attr("return_pct", m.total_return_pct)
        trial.set_user_attr("max_dd_pct", m.max_drawdown_pct)
        trial.set_user_attr("pf", m.profit_factor)
        trial.set_user_attr("winrate", m.winrate_pct)

        return score

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs, gc_after_trial=True)

    best_params = study.best_trial.params

    def eval_on(candles, params_dict):
        p = StrategyParams(
            ema_fast=int(params_dict["ema_fast"]),
            ema_trend=int(params_dict["ema_trend"]),
            rsi_period=int(params_dict["rsi_period"]),
            rsi_long_max=float(params_dict["rsi_long_max"]),
            rsi_short_min=float(params_dict["rsi_short_min"]),
            atr_period=int(params_dict["atr_period"]),
            atr_sl_mult=float(params_dict["atr_sl_mult"]),
            atr_tp_mult=float(params_dict["atr_tp_mult"]),
            macd_fast=int(params_dict["macd_fast"]),
            macd_slow=int(params_dict["macd_slow"]),
            macd_signal=int(params_dict["macd_signal"]),
            macd_hist_threshold=float(params_dict["macd_hist_threshold"]),
            atr_pct_min=float(params_dict["atr_pct_min"]) if params_dict.get("use_atr_filter") else None,
            atr_pct_max=float(params_dict["atr_pct_max"]) if params_dict.get("use_atr_filter") else None,
            cooldown_bars=int(params_dict["cooldown_bars"]),
        )
        trailing_enabled = bool(params_dict.get("trailing", False))
        atr_trail_start_mult = float(params_dict.get("atr_trail_start_mult", 1.5))
        atr_trail_mult = float(params_dict.get("atr_trail_mult", 2.0))

        df = prepare_features_and_signals(candles, p)
        result = run_backtest(
            df,
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
        return compute_metrics(result)

    m_train = eval_on(candles_train, best_params)
    m_test = eval_on(candles_test, best_params)

    output = {
        "study_name": study.study_name,
        "timeframe": timeframe,
        "best_value": study.best_value,
        "best_params": best_params,
        "train_metrics": asdict(m_train),
        "test_metrics": asdict(m_test),
    }

    return output


def save_optuna_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")