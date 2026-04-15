from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json

import numpy as np
import optuna
import pandas as pd

from market_pipeline.backtest.artifacts import save_backtest_artifacts
from market_pipeline.backtest.engine import ExecutionParams, TrailingParams, run_backtest
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.marketdata.cleaning import clean_candles_df
from market_pipeline.montecarlo.garch_fold_mc import build_garch_fold_context, evaluate_fold_monte_carlo
from market_pipeline.optimize.region_selection import repair_params
from market_pipeline.optimize.scoring import historical_validation_score, mc_validation_score, train_objective_score
from market_pipeline.optimize.splits import Fold, walkforward_splits
from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals


@dataclass(frozen=True)
class PreparedFold:
    fold: Fold
    train_candles: pd.DataFrame
    hist_input: pd.DataFrame
    eval_start: pd.Timestamp
    mc_context: object
    recency_weight: float


def _params_to_strategy(params: dict) -> StrategyParams:
    return StrategyParams(
        ema_fast=int(params["ema_fast"]),
        ema_trend=int(params["ema_trend"]),
        rsi_period=int(params["rsi_period"]),
        rsi_long_max=float(params["rsi_long_max"]),
        rsi_short_min=float(params["rsi_short_min"]),
        atr_period=int(params["atr_period"]),
        atr_sl_mult=float(params["atr_sl_mult"]),
        atr_tp_mult=float(params["atr_tp_mult"]),
        macd_fast=int(params["macd_fast"]),
        macd_slow=int(params["macd_slow"]),
        macd_signal=int(params["macd_signal"]),
        macd_hist_threshold=float(params["macd_hist_threshold"]),
        cooldown_bars=int(params["cooldown_bars"]),
    )


def _params_to_trailing(params: dict) -> TrailingParams:
    return TrailingParams(
        enabled=bool(params.get("trailing", False)),
        atr_trail_start_mult=float(params.get("atr_trail_start_mult", 1.5)),
        atr_trail_mult=float(params.get("atr_trail_mult", 2.0)),
    )


def _suggest_params(trial: optuna.Trial, trailing_allowed: bool) -> dict:
    p = {
        "ema_fast": trial.suggest_int("ema_fast", 5, 80),
        "ema_trend": trial.suggest_int("ema_trend", 80, 300),
        "rsi_period": trial.suggest_int("rsi_period", 7, 21),
        "rsi_long_max": trial.suggest_float("rsi_long_max", 55.0, 85.0),
        "rsi_short_min": trial.suggest_float("rsi_short_min", 15.0, 45.0),
        "atr_period": trial.suggest_int("atr_period", 7, 28),
        "atr_sl_mult": trial.suggest_float("atr_sl_mult", 1.0, 4.0),
        "atr_tp_mult": trial.suggest_float("atr_tp_mult", 1.0, 8.0),
        "macd_fast": trial.suggest_int("macd_fast", 6, 16),
        "macd_slow": trial.suggest_int("macd_slow", 18, 40),
        "macd_signal": trial.suggest_int("macd_signal", 5, 15),
        "macd_hist_threshold": trial.suggest_float("macd_hist_threshold", 0.0, 0.0002),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 20),
        "trailing": False,
        "atr_trail_start_mult": 1.5,
        "atr_trail_mult": 2.0,
    }

    if trailing_allowed:
        p["trailing"] = trial.suggest_categorical("trailing", [False, True])
        if p["trailing"]:
            p["atr_trail_start_mult"] = trial.suggest_float("atr_trail_start_mult", 0.5, 3.0)
            p["atr_trail_mult"] = trial.suggest_float("atr_trail_mult", 0.8, 4.0)

    p = repair_params(p)

    if p["ema_fast"] >= p["ema_trend"]:
        raise optuna.TrialPruned()
    if p["rsi_short_min"] >= p["rsi_long_max"]:
        raise optuna.TrialPruned()
    if p["macd_fast"] >= p["macd_slow"]:
        raise optuna.TrialPruned()

    return p


def _limit_folds_evenly(folds: list[Fold], max_folds: int) -> list[Fold]:
    if len(folds) <= max_folds:
        return folds
    idx = np.linspace(0, len(folds) - 1, max_folds)
    idx = np.round(idx).astype(int)
    idx = sorted(set(idx.tolist()))
    return [folds[i] for i in idx]


def _prepare_folds(
    *,
    candles_all: pd.DataFrame,
    date_from: date,
    date_to: date,
    timeframe: str,
    train_months: int,
    test_months: int,
    warmup_bars: int,
    garch_dist: str,
    garch_p: int,
    garch_q: int,
    garch_burn: int,
    demean_returns: bool,
    return_vol_scale: float,
    wick_vol_scale: float,
    max_folds: int,
) -> list[PreparedFold]:
    candles_all = candles_all.copy().sort_values("ts_utc").reset_index(drop=True)
    candles_all["ts_utc"] = pd.to_datetime(candles_all["ts_utc"], utc=True)

    folds = walkforward_splits(date_from, date_to, train_months, test_months)
    folds = _limit_folds_evenly(folds, max_folds=max_folds)

    if len(folds) < 2:
        raise ValueError("Not enough folds after limiting.")

    tf_delta = pd.to_timedelta(timeframe)
    warmup_delta = tf_delta * warmup_bars
    global_start = pd.Timestamp(date_from, tz="UTC")

    prepared = []
    recency_weights = np.arange(1, len(folds) + 1, dtype=float)

    for i, f in enumerate(folds):
        train_start_ts = pd.Timestamp(f.train_start, tz="UTC")
        train_end_ts = pd.Timestamp(f.train_end, tz="UTC")
        test_start_ts = pd.Timestamp(f.test_start, tz="UTC")
        test_end_ts = pd.Timestamp(f.test_end, tz="UTC")

        pre_start = test_start_ts - tf_delta - warmup_delta
        if pre_start < global_start:
            pre_start = global_start

        train_candles = candles_all[
            (candles_all["ts_utc"] >= train_start_ts) & (candles_all["ts_utc"] < train_end_ts)
        ].copy()
        test_candles = candles_all[
            (candles_all["ts_utc"] >= test_start_ts) & (candles_all["ts_utc"] < test_end_ts)
        ].copy()
        hist_input = candles_all[
            (candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < test_end_ts)
        ].copy()

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

        prepared.append(
            PreparedFold(
                fold=f,
                train_candles=train_candles,
                hist_input=hist_input,
                eval_start=test_start_ts - tf_delta,
                mc_context=mc_context,
                recency_weight=float(recency_weights[i]),
            )
        )

    return prepared


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sum(v * w) / np.sum(w))


def _evaluate_params_on_fold(
    *,
    params: dict,
    pf: PreparedFold,
    exec_params: ExecutionParams,
    mc_simulations: int,
    seed: int,
    overfit_penalty_lambda: float,
) -> dict:
    sp = _params_to_strategy(params)
    tp = _params_to_trailing(params)

    # IS backtest on train part
    train_df = pf.train_candles.copy().sort_values("ts_utc").reset_index(drop=True)
    is_feat = prepare_features_and_signals(train_df, sp)
    is_result = run_backtest(
        is_feat,
        atr_col=f"atr_{sp.atr_period}",
        sl_atr_mult=sp.atr_sl_mult,
        tp_atr_mult=sp.atr_tp_mult,
        cooldown_bars=sp.cooldown_bars,
        exec_params=exec_params,
        trailing=tp,
    )
    is_metrics = compute_metrics(is_result)
    is_score = train_objective_score(is_metrics)

    # OOS backtest on real future test part
    oos_feat = prepare_features_and_signals(pf.hist_input, sp)
    oos_df = oos_feat[oos_feat["ts_utc"] >= pf.eval_start].copy()

    oos_result = run_backtest(
        oos_df,
        atr_col=f"atr_{sp.atr_period}",
        sl_atr_mult=sp.atr_sl_mult,
        tp_atr_mult=sp.atr_tp_mult,
        cooldown_bars=sp.cooldown_bars,
        exec_params=exec_params,
        trailing=tp,
    )
    oos_metrics = compute_metrics(oos_result)
    oos_score = historical_validation_score(oos_metrics)

    # MC robustness on anchored GARCH paths
    mc_summary = evaluate_fold_monte_carlo(
        context=pf.mc_context,
        strategy_params=sp,
        exec_params=exec_params,
        trailing=tp,
        simulations=mc_simulations,
        seed=seed,
    )
    mc_score = mc_validation_score(mc_summary)

    # Overfitting penalty: large IS/OOS gap is bad
    is_oos_gap = abs(is_score - oos_score)
    overfit_penalty = overfit_penalty_lambda * min(is_oos_gap, 5.0)

    # Multi-objective fold outputs
    fold_oos_objective = oos_score - overfit_penalty
    fold_mc_objective = mc_score - overfit_penalty

    return {
        "is_metrics": asdict(is_metrics),
        "is_score": is_score,
        "oos_metrics": asdict(oos_metrics),
        "oos_score": oos_score,
        "mc_summary": mc_summary,
        "mc_score": mc_score,
        "is_oos_gap": is_oos_gap,
        "overfit_penalty": overfit_penalty,
        "fold_oos_objective": fold_oos_objective,
        "fold_mc_objective": fold_mc_objective,
    }


def _evaluate_params_across_folds(
    *,
    params: dict,
    prepared_folds: list[PreparedFold],
    exec_params: ExecutionParams,
    mc_simulations: int,
    seed: int,
    overfit_penalty_lambda: float,
) -> dict:
    rows = []

    for fold_idx, pf in enumerate(prepared_folds, start=1):
        row = _evaluate_params_on_fold(
            params=params,
            pf=pf,
            exec_params=exec_params,
            mc_simulations=mc_simulations,
            seed=seed + fold_idx * 1000,
            overfit_penalty_lambda=overfit_penalty_lambda,
        )
        row["fold_index"] = fold_idx
        row["train"] = [str(pf.fold.train_start), str(pf.fold.train_end)]
        row["test"] = [str(pf.fold.test_start), str(pf.fold.test_end)]
        row["recency_weight"] = pf.recency_weight
        rows.append(row)

    weights = [r["recency_weight"] for r in rows]

    agg_is = _weighted_mean([r["is_score"] for r in rows], weights)
    agg_oos = _weighted_mean([r["oos_score"] for r in rows], weights)
    agg_mc = _weighted_mean([r["mc_score"] for r in rows], weights)
    agg_gap = _weighted_mean([r["is_oos_gap"] for r in rows], weights)
    agg_penalty = _weighted_mean([r["overfit_penalty"] for r in rows], weights)
    agg_oos_obj = _weighted_mean([r["fold_oos_objective"] for r in rows], weights)
    agg_mc_obj = _weighted_mean([r["fold_mc_objective"] for r in rows], weights)

    return {
        "per_fold_results": rows,
        "aggregate": {
            "weighted_is_score": agg_is,
            "weighted_oos_score": agg_oos,
            "weighted_mc_score": agg_mc,
            "weighted_is_oos_gap": agg_gap,
            "weighted_overfit_penalty": agg_penalty,
            "weighted_oos_objective": agg_oos_obj,
            "weighted_mc_objective": agg_mc_obj,
        },
    }


def run_optuna_walkforward_pareto(
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
    garch_dist: str,
    garch_p: int,
    garch_q: int,
    garch_burn: int,
    demean_returns: bool,
    return_vol_scale: float,
    wick_vol_scale: float,
    max_folds: int,
    overfit_penalty_lambda: float,
    pareto_select_oos_weight: float,
    pareto_select_mc_weight: float,
    out_dir: Path,
) -> dict:
    candles_all, _ = clean_candles_df(candles_all)

    prepared_folds = _prepare_folds(
        candles_all=candles_all,
        date_from=date_from,
        date_to=date_to,
        timeframe=timeframe,
        train_months=train_months,
        test_months=test_months,
        warmup_bars=warmup_bars,
        garch_dist=garch_dist,
        garch_p=garch_p,
        garch_q=garch_q,
        garch_burn=garch_burn,
        demean_returns=demean_returns,
        return_vol_scale=return_vol_scale,
        wick_vol_scale=wick_vol_scale,
        max_folds=max_folds,
    )

    exec_params = ExecutionParams(
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        max_leverage=max_leverage,
        spread_pips=spread_pips,
        commission_per_trade=commission_per_trade,
    )

    print(f"[INFO] Using {len(prepared_folds)} folds (max_folds={max_folds})", flush=True)

    storage = f"sqlite:///{(out_dir / 'wf_pareto_studies.db').as_posix()}"

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial):
        params = _suggest_params(trial, trailing_allowed=trailing_allowed)

        evaluation = _evaluate_params_across_folds(
            params=params,
            prepared_folds=prepared_folds,
            exec_params=exec_params,
            mc_simulations=mc_simulations,
            seed=seed + trial.number * 100_000,
            overfit_penalty_lambda=overfit_penalty_lambda,
        )

        agg = evaluation["aggregate"]
        trial.set_user_attr("weighted_is_score", agg["weighted_is_score"])
        trial.set_user_attr("weighted_oos_score", agg["weighted_oos_score"])
        trial.set_user_attr("weighted_mc_score", agg["weighted_mc_score"])
        trial.set_user_attr("weighted_is_oos_gap", agg["weighted_is_oos_gap"])
        trial.set_user_attr("weighted_overfit_penalty", agg["weighted_overfit_penalty"])
        trial.set_user_attr("weighted_oos_objective", agg["weighted_oos_objective"])
        trial.set_user_attr("weighted_mc_objective", agg["weighted_mc_objective"])

        return agg["weighted_oos_objective"], agg["weighted_mc_objective"]

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs, gc_after_trial=True)

    pareto_trials = study.best_trials
    pareto_rows = []

    for t in pareto_trials:
        params = repair_params(dict(t.params))
        evaluation = _evaluate_params_across_folds(
            params=params,
            prepared_folds=prepared_folds,
            exec_params=exec_params,
            mc_simulations=mc_simulations,
            seed=seed + 9_999_000 + t.number * 100_000,
            overfit_penalty_lambda=overfit_penalty_lambda,
        )
        agg = evaluation["aggregate"]

        selection_score = (
            pareto_select_oos_weight * agg["weighted_oos_objective"]
            + pareto_select_mc_weight * agg["weighted_mc_objective"]
        )

        pareto_rows.append(
            {
                "trial_number": t.number,
                "params": params,
                "optuna_values": list(t.values),
                "selection_score": selection_score,
                "aggregate": agg,
                "per_fold_results": evaluation["per_fold_results"],
            }
        )

    pareto_rows.sort(key=lambda x: x["selection_score"], reverse=True)
    selected_best = pareto_rows[0]

    return {
        "study_name": study.study_name,
        "timeframe": timeframe,
        "development_period": [str(date_from), str(date_to)],
        "train_months": train_months,
        "test_months": test_months,
        "folds_used": len(prepared_folds),
        "trials": trials,
        "parallel_optuna_jobs": n_jobs,
        "mc_simulations_per_fold": mc_simulations,
        "max_folds": max_folds,
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
        "overfitting_control": {
            "overfit_penalty_lambda": overfit_penalty_lambda,
        },
        "pareto_selection_rule": {
            "oos_weight": pareto_select_oos_weight,
            "mc_weight": pareto_select_mc_weight,
        },
        "pareto_front": pareto_rows,
        "selected_best": selected_best,
    }


def save_walkforward_pareto_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def evaluate_selected_best_on_holdout(
    *,
    candles_all: pd.DataFrame,
    result_json_path: Path,
    development_from: date,
    development_to: date,
    holdout_from: date,
    holdout_to: date,
    timeframe: str,
    spread_pips: float,
    commission_per_trade: float,
    risk_per_trade: float,
    initial_equity: float,
    max_leverage: float,
    warmup_bars: int,
    holdout_mc_simulations: int,
    save_artifacts: bool,
    out_dir: Path,
) -> dict:
    result = json.loads(result_json_path.read_text(encoding="utf-8"))
    params = result["selected_best"]["params"]

    candles_all, _ = clean_candles_df(candles_all)
    candles_all = candles_all.sort_values("ts_utc").reset_index(drop=True)
    candles_all["ts_utc"] = pd.to_datetime(candles_all["ts_utc"], utc=True)

    dev_start_ts = pd.Timestamp(development_from, tz="UTC")
    dev_end_ts = pd.Timestamp(development_to, tz="UTC")
    hold_start_ts = pd.Timestamp(holdout_from, tz="UTC")
    hold_end_ts = pd.Timestamp(holdout_to, tz="UTC")

    development_candles = candles_all[
        (candles_all["ts_utc"] >= dev_start_ts) & (candles_all["ts_utc"] < dev_end_ts)
    ].copy()
    holdout_candles = candles_all[
        (candles_all["ts_utc"] >= hold_start_ts) & (candles_all["ts_utc"] < hold_end_ts)
    ].copy()

    tf_delta = pd.to_timedelta(timeframe)
    warmup_delta = tf_delta * warmup_bars
    pre_start = hold_start_ts - tf_delta - warmup_delta
    if pre_start < dev_start_ts:
        pre_start = dev_start_ts

    hist_input = candles_all[
        (candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < hold_end_ts)
    ].copy()
    eval_start = hold_start_ts - tf_delta

    exec_params = ExecutionParams(
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        max_leverage=max_leverage,
        spread_pips=spread_pips,
        commission_per_trade=commission_per_trade,
    )

    sp = _params_to_strategy(params)
    tp = _params_to_trailing(params)

    feat = prepare_features_and_signals(hist_input, sp)
    holdout_df = feat[feat["ts_utc"] >= eval_start].copy()

    hist_result = run_backtest(
        holdout_df,
        atr_col=f"atr_{sp.atr_period}",
        sl_atr_mult=sp.atr_sl_mult,
        tp_atr_mult=sp.atr_tp_mult,
        cooldown_bars=sp.cooldown_bars,
        exec_params=exec_params,
        trailing=tp,
    )
    hist_metrics = compute_metrics(hist_result)
    hist_score = historical_validation_score(hist_metrics)

    garch_cfg = result["garch_model"]
    mc_cfg = result["mc_calibration"]

    mc_context = build_garch_fold_context(
        train_candles=development_candles,
        test_candles=holdout_candles,
        train_start=development_from,
        train_end=development_to,
        test_start=holdout_from,
        test_end=holdout_to,
        warmup_bars=warmup_bars,
        burn=int(garch_cfg["burn"]),
        p=int(garch_cfg["p"]),
        q=int(garch_cfg["q"]),
        dist=str(garch_cfg["dist"]),
        demean_returns=bool(mc_cfg["demean_returns"]),
        return_vol_scale=float(mc_cfg["return_vol_scale"]),
        wick_vol_scale=float(mc_cfg["wick_vol_scale"]),
    )

    mc_summary = evaluate_fold_monte_carlo(
        context=mc_context,
        strategy_params=sp,
        exec_params=exec_params,
        trailing=tp,
        simulations=holdout_mc_simulations,
        seed=42,
        return_details=True,
    )
    
    mc_score = mc_validation_score(mc_summary)

    selection_score = 0.7 * hist_score + 0.3 * mc_score

    trades_path = None
    equity_path = None
    if save_artifacts:
        trades_path, equity_path = save_backtest_artifacts(
            hist_result,
            out_dir,
            prefix=f"pareto_holdout_{timeframe}",
        )

    return {
        "selected_best_params": params,
        "development_period": [str(development_from), str(development_to)],
        "holdout_period": [str(holdout_from), str(holdout_to)],
        "timeframe": timeframe,
        "historical_holdout_metrics": asdict(hist_metrics),
        "historical_holdout_score": hist_score,
        "holdout_mc_summary": mc_summary,
        "holdout_mc_score": mc_score,
        "final_selection_score": selection_score,
        "trades_parquet": str(trades_path) if trades_path else None,
        "equity_parquet": str(equity_path) if equity_path else None,
    }


def save_holdout_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")