from __future__ import annotations

import math
import numpy as np


def _safe_calmar(total_return_pct: float, max_drawdown_pct: float) -> float:
    dd = abs(float(max_drawdown_pct))
    if dd < 1e-9:
        return float(total_return_pct)
    return float(total_return_pct) / dd


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def train_objective_score(metrics) -> float:
    trades = int(metrics.trades)
    ret = float(metrics.total_return_pct)
    dd = float(metrics.max_drawdown_pct)
    pf = float(metrics.profit_factor)

    if trades < 15:
        return -1e9

    calmar = _safe_calmar(ret, dd)
    pf_term = _clip(pf - 1.0, -1.0, 2.0)
    trade_term = min(math.log1p(trades) / math.log1p(100), 1.0)

    penalty = 0.0
    if abs(dd) > 25.0:
        penalty += 0.15 * (abs(dd) - 25.0)
    if pf < 0.95:
        penalty += 2.0 * (0.95 - pf)
    if ret < 0:
        penalty += abs(ret) / 20.0

    return calmar + 0.40 * pf_term + 0.15 * trade_term - penalty


def historical_validation_score(metrics) -> float:
    trades = int(metrics.trades)
    ret = float(metrics.total_return_pct)
    dd = float(metrics.max_drawdown_pct)
    pf = float(metrics.profit_factor)
    win = float(metrics.winrate_pct)

    if trades < 8:
        return -1e9

    calmar = _safe_calmar(ret, dd)
    pf_term = _clip(pf - 1.0, -1.0, 2.5)
    trade_term = min(math.log1p(trades) / math.log1p(50), 1.0)
    win_term = _clip((win - 25.0) / 25.0, -1.0, 1.0)

    penalty = 0.0
    if abs(dd) > 20.0:
        penalty += 0.20 * (abs(dd) - 20.0)
    if ret < 0:
        penalty += abs(ret) / 12.0
    if pf < 0.95:
        penalty += 1.5 * (0.95 - pf)

    return calmar + 0.50 * pf_term + 0.10 * trade_term + 0.05 * win_term - penalty


def mc_validation_score(summary: dict) -> float:
    median_ret = float(summary["median_return_pct"])
    q05_ret = float(summary["q05_return_pct"])
    median_dd = abs(float(summary["median_max_drawdown_pct"]))
    prob_loss = float(summary["prob_loss"])
    prob_pf_gt_1 = float(summary["prob_pf_gt_1"])
    median_trades = float(summary["median_trades"])

    if median_dd < 1e-9:
        base = median_ret
    else:
        robust_ret = 0.55 * median_ret + 0.45 * q05_ret
        base = robust_ret / median_dd

    pf_bonus = 0.75 * (prob_pf_gt_1 - 0.5)
    loss_penalty = 1.00 * prob_loss
    trade_penalty = 0.30 if median_trades < 10 else 0.0

    return base + pf_bonus - loss_penalty - trade_penalty


def combined_fold_score(
    historical_score: float,
    mc_score: float,
    hist_weight: float = 0.7,
    mc_weight: float = 0.3,
) -> float:
    w_sum = hist_weight + mc_weight
    if w_sum <= 0:
        raise ValueError("hist_weight + mc_weight must be > 0")
    hw = hist_weight / w_sum
    mw = mc_weight / w_sum
    return hw * historical_score + mw * mc_score


def global_candidate_score(per_fold_rows: list[dict]) -> float:
    if not per_fold_rows:
        return -1e9

    arr = np.array([float(x["combined_fold_score"]) for x in per_fold_rows], dtype=float)
    hist_arr = np.array([float(x["historical_score"]) for x in per_fold_rows], dtype=float)
    mc_arr = np.array([float(x["mc_score"]) for x in per_fold_rows], dtype=float)

    mean_score = float(np.mean(arr))
    median_score = float(np.median(arr))
    q25_score = float(np.quantile(arr, 0.25))
    std_score = float(np.std(arr))
    positive_frac = float((arr > 0).mean())
    hist_positive_frac = float((hist_arr > 0).mean())
    mc_positive_frac = float((mc_arr > 0).mean())

    score = (
        0.35 * mean_score
        + 0.25 * median_score
        + 0.15 * q25_score
        + 0.15 * positive_frac
        + 0.05 * hist_positive_frac
        + 0.05 * mc_positive_frac
        - 0.10 * std_score
    )
    return score