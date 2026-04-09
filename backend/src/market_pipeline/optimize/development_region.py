from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import json

import optuna
import pandas as pd

from market_pipeline.backtest.engine import ExecutionParams, TrailingParams, run_backtest
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.marketdata.cleaning import clean_candles_df
from market_pipeline.montecarlo.garch_fold_mc import build_garch_fold_context, evaluate_fold_monte_carlo
from market_pipeline.optimize.region_selection import (
    center_params_from_pool,
    generate_nearby_variants,
    params_key,
    repair_params,
)
from market_pipeline.optimize.scoring import (
    combined_fold_score,
    global_candidate_score,
    historical_validation_score,
    mc_validation_score,
    train_objective_score,
)
from market_pipeline.optimize.splits import Fold, walkforward_splits
from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals


@dataclass(frozen=True)
class PreparedFold:
    fold: Fold
    train_candles: pd.DataFrame
    hist_input: pd.DataFrame
    eval_start: pd.Timestamp
    mc_context: object


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
) -> list[PreparedFold]:
    candles_all = candles_all.copy().sort_values("ts_utc").reset_index(drop=True)
    candles_all["ts_utc"] = pd.to_datetime(candles_all["ts_utc"], utc=True)

    folds = walkforward_splits(date_from, date_to, train_months, test_months)
    if len(folds) < 2:
        raise ValueError("Not enough folds.")

    tf_delta = pd.to_timedelta(timeframe)
    warmup_delta = tf_delta * warmup_bars
    global_start = pd.Timestamp(date_from, tz="UTC")

    prepared = []

    for f in folds:
        train_start_ts = pd.Timestamp(f.train_start, tz="UTC")
        train_end_ts = pd.Timestamp(f.train_end, tz="UTC")
        test_start_ts = pd.Timestamp(f.test_start, tz="UTC")
        test_end_ts = pd.Timestamp(f.test_end, tz="UTC")

        pre_start = test_start_ts - tf_delta - warmup_delta
        if pre_start < global_start:
            pre_start = global_start

        train_candles = candles_all[(candles_all["ts_utc"] >= train_start_ts) & (candles_all["ts_utc"] < train_end_ts)].copy()
        test_candles = candles_all[(candles_all["ts_utc"] >= test_start_ts) & (candles_all["ts_utc"] < test_end_ts)].copy()
        hist_input = candles_all[(candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < test_end_ts)].copy()

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
            )
        )

    return prepared


def _evaluate_candidate_on_fold(
    *,
    params: dict,
    pf: PreparedFold,
    exec_params: ExecutionParams,
    hist_weight: float,
    mc_weight: float,
    mc_simulations: int,
    seed: int,
) -> dict:
    strategy_params = _params_to_strategy(params)
    trailing_params = _params_to_trailing(params)

    df_feat = prepare_features_and_signals(pf.hist_input, strategy_params)
    df_eval = df_feat[df_feat["ts_utc"] >= pf.eval_start].copy()

    hist_result = run_backtest(
        df_eval,
        atr_col=f"atr_{strategy_params.atr_period}",
        sl_atr_mult=strategy_params.atr_sl_mult,
        tp_atr_mult=strategy_params.atr_tp_mult,
        cooldown_bars=strategy_params.cooldown_bars,
        exec_params=exec_params,
        trailing=trailing_params,
    )
    hist_metrics = compute_metrics(hist_result)
    hist_score = historical_validation_score(hist_metrics)

    mc_summary = evaluate_fold_monte_carlo(
        context=pf.mc_context,
        strategy_params=strategy_params,
        exec_params=exec_params,
        trailing=trailing_params,
        simulations=mc_simulations,
        seed=seed,
    )
    mc_score = mc_validation_score(mc_summary)
    fold_score = combined_fold_score(hist_score, mc_score, hist_weight, mc_weight)

    return {
        "historical_test_metrics": asdict(hist_metrics),
        "historical_score": hist_score,
        "mc_summary": mc_summary,
        "mc_score": mc_score,
        "combined_fold_score": fold_score,
    }


def _run_fold_development_worker(
    *,
    fold_idx: int,
    pf: PreparedFold,
    inner_trials_per_fold: int,
    top_candidates_per_fold: int,
    seed: int,
    trailing_allowed: bool,
    exec_params: ExecutionParams,
    hist_weight: float,
    mc_weight: float,
    mc_simulations: int,
    inner_optuna_jobs: int,
) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed + fold_idx),
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, trailing_allowed=trailing_allowed)
        sp = _params_to_strategy(params)
        tp = _params_to_trailing(params)

        train_df = pf.train_candles.copy().sort_values("ts_utc").reset_index(drop=True)
        df_feat = prepare_features_and_signals(train_df, sp)

        result = run_backtest(
            df_feat,
            atr_col=f"atr_{sp.atr_period}",
            sl_atr_mult=sp.atr_sl_mult,
            tp_atr_mult=sp.atr_tp_mult,
            cooldown_bars=sp.cooldown_bars,
            exec_params=exec_params,
            trailing=tp,
        )
        metrics = compute_metrics(result)
        score = train_objective_score(metrics)

        trial.set_user_attr("params", params)
        trial.set_user_attr("train_metrics", asdict(metrics))
        return score

    study.optimize(
        objective,
        n_trials=inner_trials_per_fold,
        n_jobs=inner_optuna_jobs,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else -1e18, reverse=True)

    selected = []
    seen = set()
    for t in completed:
        params = repair_params(dict(t.user_attrs["params"]))
        key = params_key(params)
        if key in seen:
            continue
        seen.add(key)
        selected.append((params, float(t.value), dict(t.user_attrs["train_metrics"])))
        if len(selected) >= top_candidates_per_fold:
            break

    fold_rows = []
    for cand_idx, (params, train_score, train_metrics) in enumerate(selected, start=1):
        eval_row = _evaluate_candidate_on_fold(
            params=params,
            pf=pf,
            exec_params=exec_params,
            hist_weight=hist_weight,
            mc_weight=mc_weight,
            mc_simulations=mc_simulations,
            seed=seed + fold_idx * 10_000 + cand_idx * 100,
        )
        row = {
            "fold_index": fold_idx,
            "train": [str(pf.fold.train_start), str(pf.fold.train_end)],
            "test": [str(pf.fold.test_start), str(pf.fold.test_end)],
            "params": params,
            "train_objective_score": train_score,
            "train_metrics": train_metrics,
            **eval_row,
        }
        fold_rows.append(row)

    fold_rows.sort(key=lambda x: x["combined_fold_score"], reverse=True)
    return {
        "fold_index": fold_idx,
        "train": [str(pf.fold.train_start), str(pf.fold.train_end)],
        "test": [str(pf.fold.test_start), str(pf.fold.test_end)],
        "top_candidates": fold_rows,
        "fold_winner": fold_rows[0] if fold_rows else None,
        "all_local_candidates": fold_rows,
    }


def _evaluate_region_candidate_worker(
    *,
    candidate_index: int,
    params: dict,
    prepared_folds: list[PreparedFold],
    exec_params: ExecutionParams,
    hist_weight: float,
    mc_weight: float,
    mc_simulations: int,
    seed: int,
) -> dict:
    per_fold_rows = []

    for fold_idx, pf in enumerate(prepared_folds, start=1):
        eval_row = _evaluate_candidate_on_fold(
            params=params,
            pf=pf,
            exec_params=exec_params,
            hist_weight=hist_weight,
            mc_weight=mc_weight,
            mc_simulations=mc_simulations,
            seed=seed + candidate_index * 100_000 + fold_idx * 100,
        )
        per_fold_rows.append(
            {
                "fold_index": fold_idx,
                "train": [str(pf.fold.train_start), str(pf.fold.train_end)],
                "test": [str(pf.fold.test_start), str(pf.fold.test_end)],
                **eval_row,
            }
        )

    agg_score = global_candidate_score(per_fold_rows)

    return {
        "candidate_index": candidate_index,
        "candidate_type": "center" if candidate_index == 1 else "nearby_variant",
        "params": params,
        "global_score": agg_score,
        "per_fold_results": per_fold_rows,
    }


def discover_parameter_region(
    *,
    candles_all: pd.DataFrame,
    development_from: date,
    development_to: date,
    timeframe: str,
    train_months: int,
    test_months: int,
    inner_trials_per_fold: int,
    top_candidates_per_fold: int,
    nearby_variants: int,
    seed: int,
    trailing_allowed: bool,
    spread_pips: float,
    commission_per_trade: float,
    risk_per_trade: float,
    initial_equity: float,
    max_leverage: float,
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
    fold_workers: int = 3,
    inner_optuna_jobs: int = 1,
    region_workers: int = 3,
) -> dict:
    candles_all, _ = clean_candles_df(candles_all)

    prepared_folds = _prepare_folds(
        candles_all=candles_all,
        date_from=development_from,
        date_to=development_to,
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
    )

    exec_params = ExecutionParams(
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        max_leverage=max_leverage,
        spread_pips=spread_pips,
        commission_per_trade=commission_per_trade,
    )

    print(f"[INFO] Prepared {len(prepared_folds)} folds")
    print(
        f"[INFO] fold_workers={fold_workers}, inner_optuna_jobs={inner_optuna_jobs}, "
        f"region_workers={region_workers}, mc_simulations={mc_simulations}"
    )

    all_local_candidates = []
    fold_summaries = []

    # Stage 1: optimize inside each train fold
    if fold_workers <= 1:
        for fold_idx, pf in enumerate(prepared_folds, start=1):
            print(f"[INFO] Fold {fold_idx}/{len(prepared_folds)} started")
            res = _run_fold_development_worker(
                fold_idx=fold_idx,
                pf=pf,
                inner_trials_per_fold=inner_trials_per_fold,
                top_candidates_per_fold=top_candidates_per_fold,
                seed=seed,
                trailing_allowed=trailing_allowed,
                exec_params=exec_params,
                hist_weight=hist_weight,
                mc_weight=mc_weight,
                mc_simulations=mc_simulations,
                inner_optuna_jobs=inner_optuna_jobs,
            )
            print(f"[INFO] Fold {fold_idx}/{len(prepared_folds)} finished")
            fold_summaries.append(
                {
                    "fold_index": res["fold_index"],
                    "train": res["train"],
                    "test": res["test"],
                    "top_candidates": res["top_candidates"],
                    "fold_winner": res["fold_winner"],
                }
            )
            all_local_candidates.extend(res["all_local_candidates"])
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=fold_workers) as ex:
            for fold_idx, pf in enumerate(prepared_folds, start=1):
                fut = ex.submit(
                    _run_fold_development_worker,
                    fold_idx=fold_idx,
                    pf=pf,
                    inner_trials_per_fold=inner_trials_per_fold,
                    top_candidates_per_fold=top_candidates_per_fold,
                    seed=seed,
                    trailing_allowed=trailing_allowed,
                    exec_params=exec_params,
                    hist_weight=hist_weight,
                    mc_weight=mc_weight,
                    mc_simulations=mc_simulations,
                    inner_optuna_jobs=inner_optuna_jobs,
                )
                futures[fut] = fold_idx

            completed_rows = []
            for fut in as_completed(futures):
                fold_idx = futures[fut]
                print(f"[INFO] Fold {fold_idx}/{len(prepared_folds)} finished")
                res = fut.result()
                completed_rows.append(res)

        completed_rows.sort(key=lambda x: x["fold_index"])
        for res in completed_rows:
            fold_summaries.append(
                {
                    "fold_index": res["fold_index"],
                    "train": res["train"],
                    "test": res["test"],
                    "top_candidates": res["top_candidates"],
                    "fold_winner": res["fold_winner"],
                }
            )
            all_local_candidates.extend(res["all_local_candidates"])

    #Stage 2: build center + nearby region
    print(f"[INFO] Stage 1 complete: collected {len(all_local_candidates)} local candidates")
    all_local_candidates.sort(key=lambda x: x["combined_fold_score"], reverse=True)
    pool_size = min(max(40, len(prepared_folds) * top_candidates_per_fold), len(all_local_candidates))
    pool = all_local_candidates[:pool_size]

    print("[INFO] Building center set and nearby variants")
    center_params = center_params_from_pool(pool)
    nearby = generate_nearby_variants(
        pool=pool,
        center_params=center_params,
        n_variants=nearby_variants,
        seed=seed,
    )
    region_candidates = [center_params] + nearby

    # Stage 3: evaluate region candidates across ALL folds
    print(f"[INFO] Stage 2 complete: evaluating {len(region_candidates)} region candidates across all folds")
    region_ranking = []

    if region_workers <= 1:
        for idx, params in enumerate(region_candidates, start=1):
            print(f"[INFO] Region candidate {idx}/{len(region_candidates)}")
            res = _evaluate_region_candidate_worker(
                candidate_index=idx,
                params=params,
                prepared_folds=prepared_folds,
                exec_params=exec_params,
                hist_weight=hist_weight,
                mc_weight=mc_weight,
                mc_simulations=mc_simulations,
                seed=seed,
            )
            region_ranking.append(res)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=region_workers) as ex:
            for idx, params in enumerate(region_candidates, start=1):
                fut = ex.submit(
                    _evaluate_region_candidate_worker,
                    candidate_index=idx,
                    params=params,
                    prepared_folds=prepared_folds,
                    exec_params=exec_params,
                    hist_weight=hist_weight,
                    mc_weight=mc_weight,
                    mc_simulations=mc_simulations,
                    seed=seed,
                )
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                print(f"[INFO] Region candidate {idx}/{len(region_candidates)} finished")
                region_ranking.append(fut.result())

    region_ranking.sort(key=lambda x: x["global_score"], reverse=True)

    return {
        "development_period": [str(development_from), str(development_to)],
        "timeframe": timeframe,
        "train_months": train_months,
        "test_months": test_months,
        "inner_trials_per_fold": inner_trials_per_fold,
        "top_candidates_per_fold": top_candidates_per_fold,
        "region_center_plus_nearby_count": 1 + nearby_variants,
        "parallelism": {
            "fold_workers": fold_workers,
            "inner_optuna_jobs": inner_optuna_jobs,
            "region_workers": region_workers,
        },
        "objective_weights": {
            "historical_weight": hist_weight,
            "mc_weight": mc_weight,
        },
        "mc_calibration": {
            "demean_returns": demean_returns,
            "return_vol_scale": return_vol_scale,
            "wick_vol_scale": wick_vol_scale,
        },
        "garch_model": {
            "dist": garch_dist,
            "p": garch_p,
            "q": garch_q,
            "burn": garch_burn,
        },
        "fold_summaries": fold_summaries,
        "robust_region": {
            "center_params": center_params,
            "nearby_variants": nearby,
            "global_ranking": region_ranking,
        },
    }


def save_development_region(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")