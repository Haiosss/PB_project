from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

import numpy as np

PARAM_SPECS = {
    "ema_fast": ("int", 5, 80),
    "ema_trend": ("int", 80, 300),
    "rsi_period": ("int", 7, 21),
    "rsi_long_max": ("float", 55.0, 85.0),
    "rsi_short_min": ("float", 15.0, 45.0),
    "atr_period": ("int", 7, 28),
    "atr_sl_mult": ("float", 1.0, 4.0),
    "atr_tp_mult": ("float", 1.0, 8.0),
    "macd_fast": ("int", 6, 16),
    "macd_slow": ("int", 18, 40),
    "macd_signal": ("int", 5, 15),
    "macd_hist_threshold": ("float", 0.0, 0.0002),
    "cooldown_bars": ("int", 0, 20),
    "trailing": ("bool", None, None),
    "atr_trail_start_mult": ("float", 0.5, 3.0),
    "atr_trail_mult": ("float", 0.8, 4.0),
}


def params_key(params: dict) -> str:
    norm = deepcopy(params)
    for k, v in list(norm.items()):
        if isinstance(v, float):
            norm[k] = round(v, 10)
    return json.dumps(norm, sort_keys=True)


def repair_params(params: dict) -> dict:
    p = deepcopy(params)

    # hard bounds
    for name, spec in PARAM_SPECS.items():
        typ, lo, hi = spec
        if name not in p:
            continue

        if typ == "int":
            p[name] = int(round(float(p[name])))
            p[name] = max(int(lo), min(int(hi), p[name]))
        elif typ == "float":
            p[name] = float(p[name])
            p[name] = max(float(lo), min(float(hi), p[name]))
        elif typ == "bool":
            p[name] = bool(p[name])

    # relational constraints
    if p["ema_fast"] >= p["ema_trend"]:
        p["ema_fast"] = max(5, min(p["ema_trend"] - 1, p["ema_fast"]))

    if p["rsi_short_min"] >= p["rsi_long_max"]:
        mid = 0.5 * (p["rsi_short_min"] + p["rsi_long_max"])
        p["rsi_short_min"] = max(15.0, min(mid - 1.0, 44.0))
        p["rsi_long_max"] = min(85.0, max(mid + 1.0, 56.0))

    if p["macd_fast"] >= p["macd_slow"]:
        p["macd_fast"] = max(6, min(p["macd_slow"] - 1, p["macd_fast"]))

    if p["atr_tp_mult"] < p["atr_sl_mult"] * 0.8:
        p["atr_tp_mult"] = min(8.0, max(p["atr_sl_mult"] * 0.8, p["atr_tp_mult"]))

    if not p.get("trailing", False):
        p["atr_trail_start_mult"] = 1.5
        p["atr_trail_mult"] = 2.0

    return p


def center_params_from_pool(pool: list[dict]) -> dict:
    if not pool:
        raise ValueError("Pool is empty")

    center = {}
    for name, spec in PARAM_SPECS.items():
        typ, _, _ = spec
        vals = [x["params"][name] for x in pool if name in x["params"]]

        if typ == "bool":
            center[name] = Counter(bool(v) for v in vals).most_common(1)[0][0]
        elif typ == "int":
            center[name] = int(round(float(np.median(vals))))
        elif typ == "float":
            center[name] = float(np.median(vals))

    return repair_params(center)


def _numeric_spread(vals: list[float], lo: float, hi: float) -> tuple[float, float, float]:
    arr = np.array(vals, dtype=float)
    q10 = float(np.quantile(arr, 0.10))
    q90 = float(np.quantile(arr, 0.90))
    center = float(np.median(arr))

    if q90 <= q10:
        q10, q90 = float(arr.min()), float(arr.max())

    if q90 <= q10:
        q10 = max(lo, center - 0.05 * (hi - lo))
        q90 = min(hi, center + 0.05 * (hi - lo))

    return q10, center, q90


def generate_nearby_variants(
    *,
    pool: list[dict],
    center_params: dict,
    n_variants: int = 20,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)

    # robust local ranges
    param_ranges = {}
    for name, spec in PARAM_SPECS.items():
        typ, lo, hi = spec
        vals = [x["params"][name] for x in pool if name in x["params"]]

        if typ == "bool":
            param_ranges[name] = {"type": "bool", "mode": center_params[name], "vals": vals}
        else:
            q10, center, q90 = _numeric_spread([float(v) for v in vals], float(lo), float(hi))
            param_ranges[name] = {
                "type": typ,
                "lo": float(lo),
                "hi": float(hi),
                "q10": q10,
                "q90": q90,
                "center": center,
                "spread": max(q90 - q10, 1e-12),
            }

    variants = []
    seen = {params_key(center_params)}

    attempts = 0
    while len(variants) < n_variants and attempts < n_variants * 100:
        attempts += 1
        p = deepcopy(center_params)

        for name, meta in param_ranges.items():
            if meta["type"] == "bool":
                p[name] = bool(center_params[name])
                continue

            c = float(meta["center"])
            s = float(meta["spread"])
            lo = float(meta["lo"])
            hi = float(meta["hi"])
            q10 = float(meta["q10"])
            q90 = float(meta["q90"])

            val = c + rng.normal(0.0, 0.30 * s)
            val = max(lo, min(hi, val))
            val = max(q10, min(q90, val))

            if meta["type"] == "int":
                p[name] = int(round(val))
            else:
                p[name] = float(val)

        p = repair_params(p)
        key = params_key(p)
        if key in seen:
            continue
        seen.add(key)
        variants.append(p)

    while len(variants) < n_variants:
        p = deepcopy(center_params)
        idx = len(variants) + 1

        p["ema_fast"] = max(5, min(80, p["ema_fast"] + (-1 if idx % 2 else 1)))
        p["ema_trend"] = max(80, min(300, p["ema_trend"] + (2 if idx % 3 == 0 else -2)))
        p["atr_sl_mult"] = max(1.0, min(4.0, p["atr_sl_mult"] + (0.05 if idx % 2 else -0.05)))
        p["atr_tp_mult"] = max(1.0, min(8.0, p["atr_tp_mult"] + (0.10 if idx % 2 else -0.10)))
        p["macd_hist_threshold"] = max(0.0, min(0.0002, p["macd_hist_threshold"] + (1e-6 if idx % 2 else -1e-6)))
        p = repair_params(p)

        key = params_key(p)
        if key not in seen:
            seen.add(key)
            variants.append(p)

    return variants