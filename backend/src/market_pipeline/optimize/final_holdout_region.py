from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date
from pathlib import Path
import json

import pandas as pd

from market_pipeline.backtest.artifacts import save_backtest_artifacts
from market_pipeline.backtest.engine import ExecutionParams, TrailingParams, run_backtest
from market_pipeline.backtest.metrics import compute_metrics
from market_pipeline.marketdata.cleaning import clean_candles_df
from market_pipeline.montecarlo.garch_fold_mc import build_garch_fold_context, evaluate_fold_monte_carlo
from market_pipeline.optimize.scoring import combined_fold_score, historical_validation_score, mc_validation_score
from market_pipeline.strategy.ema_macd_atr_pullback import StrategyParams, prepare_features_and_signals


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


def _evaluate_one_holdout_candidate_worker(
    *,
    candidate_index: int,
    params: dict,
    development_candles: pd.DataFrame,
    holdout_candles: pd.DataFrame,
    hist_input: pd.DataFrame,
    eval_start: pd.Timestamp,
    development_from: date,
    development_to: date,
    holdout_from: date,
    holdout_to: date,
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
    exec_params: ExecutionParams,
) -> dict:
    sp = _params_to_strategy(params)
    tp = _params_to_trailing(params)

    mc_context = build_garch_fold_context(
        train_candles=development_candles,
        test_candles=holdout_candles,
        train_start=development_from,
        train_end=development_to,
        test_start=holdout_from,
        test_end=holdout_to,
        warmup_bars=warmup_bars,
        burn=garch_burn,
        p=garch_p,
        q=garch_q,
        dist=garch_dist,
        demean_returns=demean_returns,
        return_vol_scale=return_vol_scale,
        wick_vol_scale=wick_vol_scale,
    )

    df_feat = prepare_features_and_signals(hist_input, sp)
    df_eval = df_feat[df_feat["ts_utc"] >= eval_start].copy()

    hist_result = run_backtest(
        df_eval,
        atr_col=f"atr_{sp.atr_period}",
        sl_atr_mult=sp.atr_sl_mult,
        tp_atr_mult=sp.atr_tp_mult,
        cooldown_bars=sp.cooldown_bars,
        exec_params=exec_params,
        trailing=tp,
    )
    hist_metrics = compute_metrics(hist_result)
    hist_score = historical_validation_score(hist_metrics)

    mc_summary = evaluate_fold_monte_carlo(
        context=mc_context,
        strategy_params=sp,
        exec_params=exec_params,
        trailing=tp,
        simulations=mc_simulations,
        seed=42 + candidate_index * 1000,
    )
    mc_score = mc_validation_score(mc_summary)

    final_score = combined_fold_score(hist_score, mc_score, hist_weight, mc_weight)

    return {
        "candidate_index": candidate_index,
        "params": params,
        "holdout_historical_metrics": asdict(hist_metrics),
        "holdout_historical_score": hist_score,
        "holdout_mc_summary": mc_summary,
        "holdout_mc_score": mc_score,
        "holdout_combined_score": final_score,
        "hist_result": hist_result,
    }


def evaluate_region_on_holdout(
    *,
    candles_all: pd.DataFrame,
    development_from: date,
    development_to: date,
    holdout_from: date,
    holdout_to: date,
    timeframe: str,
    region_json_path: Path,
    spread_pips: float,
    commission_per_trade: float,
    risk_per_trade: float,
    initial_equity: float,
    max_leverage: float,
    warmup_bars: int,
    mc_simulations: int,
    hist_weight: float,
    mc_weight: float,
    save_top_artifacts: int,
    out_dir: Path,
    holdout_workers: int = 3,
) -> dict:
    region = json.loads(region_json_path.read_text(encoding="utf-8"))
    ranking = region["robust_region"]["global_ranking"]
    candidates = ranking

    candles_all, _ = clean_candles_df(candles_all)
    candles_all = candles_all.sort_values("ts_utc").reset_index(drop=True)
    candles_all["ts_utc"] = pd.to_datetime(candles_all["ts_utc"], utc=True)

    dev_start_ts = pd.Timestamp(development_from, tz="UTC")
    dev_end_ts = pd.Timestamp(development_to, tz="UTC")
    hold_start_ts = pd.Timestamp(holdout_from, tz="UTC")
    hold_end_ts = pd.Timestamp(holdout_to, tz="UTC")

    development_candles = candles_all[(candles_all["ts_utc"] >= dev_start_ts) & (candles_all["ts_utc"] < dev_end_ts)].copy()
    holdout_candles = candles_all[(candles_all["ts_utc"] >= hold_start_ts) & (candles_all["ts_utc"] < hold_end_ts)].copy()

    if development_candles.empty or holdout_candles.empty:
        raise ValueError("Development or holdout candles are empty.")

    tf_delta = pd.to_timedelta(timeframe)
    warmup_delta = tf_delta * warmup_bars
    pre_start = hold_start_ts - tf_delta - warmup_delta
    if pre_start < dev_start_ts:
        pre_start = dev_start_ts

    hist_input = candles_all[(candles_all["ts_utc"] >= pre_start) & (candles_all["ts_utc"] < hold_end_ts)].copy()
    eval_start = hold_start_ts - tf_delta

    exec_params = ExecutionParams(
        initial_equity=initial_equity,
        risk_per_trade=risk_per_trade,
        max_leverage=max_leverage,
        spread_pips=spread_pips,
        commission_per_trade=commission_per_trade,
    )

    garch_cfg = region["garch_model"]
    mc_cfg = region["mc_calibration"]

    print(f"[INFO] Holdout candidates to evaluate: {len(candidates)}", flush=True)
    print(
        f"[INFO] holdout_workers={holdout_workers}, mc_simulations={mc_simulations}, "
        f"save_top_artifacts={save_top_artifacts}",
        flush=True,
    )
    print(
        f"[INFO] development=[{development_from}, {development_to}) "
        f"holdout=[{holdout_from}, {holdout_to})",
        flush=True,
    )

    raw_rows = []

    if holdout_workers <= 1:
        for idx, row in enumerate(candidates, start=1):
            print(f"[INFO] Holdout candidate {idx}/{len(candidates)} started", flush=True)
            res = _evaluate_one_holdout_candidate_worker(
                candidate_index=idx,
                params=row["params"],
                development_candles=development_candles,
                holdout_candles=holdout_candles,
                hist_input=hist_input,
                eval_start=eval_start,
                development_from=development_from,
                development_to=development_to,
                holdout_from=holdout_from,
                holdout_to=holdout_to,
                warmup_bars=warmup_bars,
                mc_simulations=mc_simulations,
                hist_weight=hist_weight,
                mc_weight=mc_weight,
                garch_dist=str(garch_cfg["dist"]),
                garch_p=int(garch_cfg["p"]),
                garch_q=int(garch_cfg["q"]),
                garch_burn=int(garch_cfg["burn"]),
                demean_returns=bool(mc_cfg["demean_returns"]),
                return_vol_scale=float(mc_cfg["return_vol_scale"]),
                wick_vol_scale=float(mc_cfg["wick_vol_scale"]),
                exec_params=exec_params,
            )
            print(f"[INFO] Holdout candidate {idx}/{len(candidates)} finished", flush=True)
            raw_rows.append(
                {
                    "candidate_index": idx,
                    "params": row["params"],
                    "development_global_score": row["global_score"],
                    **res,
                }
            )
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=holdout_workers) as ex:
            for idx, row in enumerate(candidates, start=1):
                fut = ex.submit(
                    _evaluate_one_holdout_candidate_worker,
                    candidate_index=idx,
                    params=row["params"],
                    development_candles=development_candles,
                    holdout_candles=holdout_candles,
                    hist_input=hist_input,
                    eval_start=eval_start,
                    development_from=development_from,
                    development_to=development_to,
                    holdout_from=holdout_from,
                    holdout_to=holdout_to,
                    warmup_bars=warmup_bars,
                    mc_simulations=mc_simulations,
                    hist_weight=hist_weight,
                    mc_weight=mc_weight,
                    garch_dist=str(garch_cfg["dist"]),
                    garch_p=int(garch_cfg["p"]),
                    garch_q=int(garch_cfg["q"]),
                    garch_burn=int(garch_cfg["burn"]),
                    demean_returns=bool(mc_cfg["demean_returns"]),
                    return_vol_scale=float(mc_cfg["return_vol_scale"]),
                    wick_vol_scale=float(mc_cfg["wick_vol_scale"]),
                    exec_params=exec_params,
                )
                futures[fut] = (idx, row)

            for fut in as_completed(futures):
                idx, row = futures[fut]
                print(f"[INFO] Holdout candidate {idx}/{len(candidates)} finished", flush=True)
                res = fut.result()
                raw_rows.append(
                    {
                        "candidate_index": idx,
                        "params": row["params"],
                        "development_global_score": row["global_score"],
                        **res,
                    }
                )

    # sort by final holdout score
    raw_rows.sort(key=lambda x: x["holdout_combined_score"], reverse=True)

    candidate_rows = []
    for rank_idx, row in enumerate(raw_rows, start=1):
        trades_path = None
        equity_path = None

        if rank_idx <= save_top_artifacts:
            cand_dir = out_dir / f"candidate_{rank_idx:02d}"
            trades_path, equity_path = save_backtest_artifacts(
                row["hist_result"],
                cand_dir,
                prefix=f"holdout_{timeframe}",
            )

        candidate_rows.append(
            {
                "candidate_index": row["candidate_index"],
                "params": row["params"],
                "development_global_score": row["development_global_score"],
                "holdout_historical_metrics": row["holdout_historical_metrics"],
                "holdout_historical_score": row["holdout_historical_score"],
                "holdout_mc_summary": row["holdout_mc_summary"],
                "holdout_mc_score": row["holdout_mc_score"],
                "holdout_combined_score": row["holdout_combined_score"],
                "trades_parquet": str(trades_path) if trades_path else None,
                "equity_parquet": str(equity_path) if equity_path else None,
            }
        )

    hold_scores = [float(x["holdout_combined_score"]) for x in candidate_rows]
    hist_returns = [float(x["holdout_historical_metrics"]["total_return_pct"]) for x in candidate_rows]
    mc_medians = [float(x["holdout_mc_summary"]["median_return_pct"]) for x in candidate_rows]

    print("[INFO] Holdout region evaluation finished", flush=True)

    return {
        "development_period": [str(development_from), str(development_to)],
        "holdout_period": [str(holdout_from), str(holdout_to)],
        "timeframe": timeframe,
        "region_size": len(candidate_rows),
        "parallelism": {
            "holdout_workers": holdout_workers,
        },
        "mc_simulations": mc_simulations,
        "objective_weights": {
            "historical_weight": hist_weight,
            "mc_weight": mc_weight,
        },
        "region_group_summary": {
            "median_holdout_combined_score": float(pd.Series(hold_scores).median()),
            "median_holdout_historical_return_pct": float(pd.Series(hist_returns).median()),
            "median_holdout_mc_median_return_pct": float(pd.Series(mc_medians).median()),
            "positive_holdout_combined_frac": float((pd.Series(hold_scores) > 0).mean()),
            "positive_holdout_hist_return_frac": float((pd.Series(hist_returns) > 0).mean()),
            "positive_holdout_mc_median_return_frac": float((pd.Series(mc_medians) > 0).mean()),
        },
        "candidate_results": candidate_rows,
    }


def save_holdout_region_result(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")