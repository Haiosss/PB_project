from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from market_pipeline.montecarlo.garch_fold_mc import (
    GarchFoldContext,
    generate_synthetic_test_candles,
)


@dataclass(frozen=True)
class McValidationArtifacts:
    summary_json: Path
    chart_png: Path
    real_parquet: Path
    synthetic_parquet: Path

def _rebase_candles(df: pd.DataFrame, base_value: float = 100.0) -> pd.DataFrame:
    out = df.copy().sort_values("ts_utc").reset_index(drop=True)
    first_close = float(out["bid_c"].iloc[0])

    if first_close == 0:
        return out

    scale = base_value / first_close

    for col in ["bid_o", "bid_h", "bid_l", "bid_c"]:
        out[col] = pd.to_numeric(out[col], errors="coerce") * scale

    return out

def _bar_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values("ts_utc").reset_index(drop=True)

    o = pd.to_numeric(work["bid_o"], errors="coerce")
    h = pd.to_numeric(work["bid_h"], errors="coerce")
    l = pd.to_numeric(work["bid_l"], errors="coerce")
    c = pd.to_numeric(work["bid_c"], errors="coerce")

    ret_log = np.log(c / c.shift(1))
    range_pct = (h - l) / o.replace(0, np.nan)
    body_pct = (c - o).abs() / o.replace(0, np.nan)

    upper_wick = (h - np.maximum(o, c)) / o.replace(0, np.nan)
    lower_wick = (np.minimum(o, c) - l) / o.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(work["ts_utc"], utc=True),
            "ret_log": ret_log,
            "range_pct": range_pct,
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick,
            "lower_wick_pct": lower_wick,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "q05": None,
            "median": None,
            "q95": None,
        }

    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "q05": float(s.quantile(0.05)),
        "median": float(s.quantile(0.50)),
        "q95": float(s.quantile(0.95)),
    }


def compare_real_vs_synthetic(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    real_f = _bar_features(real_df)
    synth_f = _bar_features(synth_df)

    metrics = ["ret_log", "range_pct", "body_pct", "upper_wick_pct", "lower_wick_pct"]

    summary = {}
    for m in metrics:
        real_stats = _stats(real_f[m])
        synth_stats = _stats(synth_f[m])

        mean_diff = None
        std_diff = None
        if real_stats["mean"] is not None and synth_stats["mean"] is not None:
            mean_diff = float(synth_stats["mean"] - real_stats["mean"])
        if real_stats["std"] is not None and synth_stats["std"] is not None:
            std_diff = float(synth_stats["std"] - real_stats["std"])

        summary[m] = {
            "real": real_stats,
            "synthetic": synth_stats,
            "differences": {
                "mean_diff": mean_diff,
                "std_diff": std_diff,
            },
        }

    return summary


def _draw_candles(ax, df: pd.DataFrame, title: str, bars: int = 200) -> None:
    plot_df = df.copy().sort_values("ts_utc").tail(bars).reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.6

    for i, row in plot_df.iterrows():
        o = float(row["bid_o"])
        h = float(row["bid_h"])
        l = float(row["bid_l"])
        c = float(row["bid_c"])

        if c > o:
            color = "#16a34a"
        elif c < o:
            color = "#dc2626"
        else:
            color = "#6b7280"

        ax.vlines(i, l, h, color=color, linewidth=1)

        body_low = min(o, c)
        body_high = max(o, c)
        body_height = body_high - body_low

        if body_height == 0:
            body_height = max(abs(o) * 1e-6, 1e-6)

        rect = Rectangle(
            (i - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlim(-1, len(plot_df))
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelbottom=False)


def save_mc_validation_artifacts(
    *,
    context: GarchFoldContext,
    out_dir: Path,
    seed: int = 42,
    bars_to_plot: int = 200,
    prefix: str = "mc_validation",
) -> McValidationArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_full = generate_synthetic_test_candles(context, seed=seed)
    real_test = pd.concat(
        [
            context.warmup_tail,
            pd.DataFrame(
                {
                    "ts_utc": context.test_timestamps,
                    "bid_o": pd.NA,
                    "bid_h": pd.NA,
                    "bid_l": pd.NA,
                    "bid_c": pd.NA,
                    "bid_v": context.test_volumes,
                }
            ),
        ],
        ignore_index=True,
    )
    synth_test = synth_full.iloc[len(context.warmup_tail):].copy().reset_index(drop=True)

    summary = {
        "seed": seed,
        "bars_to_plot": bars_to_plot,
        "notes": "Use CLI wrapper to compare against actual real test candles."
    }

    summary_path = out_dir / f"{prefix}_summary.json"
    chart_path = out_dir / f"{prefix}_chart.png"
    real_path = out_dir / f"{prefix}_real_placeholder.parquet"
    synth_path = out_dir / f"{prefix}_synthetic.parquet"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    real_test.to_parquet(real_path, index=False)
    synth_test.to_parquet(synth_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_candles(ax, synth_test, title="Synthetic test candles", bars=bars_to_plot)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return McValidationArtifacts(
        summary_json=summary_path,
        chart_png=chart_path,
        real_parquet=real_path,
        synthetic_parquet=synth_path,
    )


def save_real_vs_synthetic_validation(
    *,
    real_test_candles: pd.DataFrame,
    synthetic_test_candles: pd.DataFrame,
    out_dir: Path,
    seed: int = 42,
    bars_to_plot: int = 200,
    prefix: str = "mc_validation",
) -> McValidationArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)

    real_test = real_test_candles.copy().sort_values("ts_utc").reset_index(drop=True)
    synth_test = synthetic_test_candles.copy().sort_values("ts_utc").reset_index(drop=True)

    comparison = compare_real_vs_synthetic(real_test, synth_test)

    summary = {
        "seed": seed,
        "bars_to_plot": bars_to_plot,
        "real_rows": int(len(real_test)),
        "synthetic_rows": int(len(synth_test)),
        "comparison": comparison,
    }

    summary_path = out_dir / f"{prefix}_summary.json"
    chart_path = out_dir / f"{prefix}_chart.png"
    real_path = out_dir / f"{prefix}_real.parquet"
    synth_path = out_dir / f"{prefix}_synthetic.parquet"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    real_test.to_parquet(real_path, index=False)
    synth_test.to_parquet(synth_path, index=False)


    real_rebased = _rebase_candles(real_test, base_value=100.0)
    synth_rebased = _rebase_candles(synth_test, base_value=100.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)


    real_plot_abs = real_test.tail(bars_to_plot)
    synth_plot_abs = synth_test.tail(bars_to_plot)

    y_min_abs = float(min(real_plot_abs["bid_l"].min(), synth_plot_abs["bid_l"].min()))
    y_max_abs = float(max(real_plot_abs["bid_h"].max(), synth_plot_abs["bid_h"].max()))
    pad_abs = (y_max_abs - y_min_abs) * 0.05 if y_max_abs > y_min_abs else max(abs(y_max_abs) * 0.01, 1e-6)

    _draw_candles(axes[0, 0], real_test, title="Real test candles (absolute)", bars=bars_to_plot)
    _draw_candles(axes[0, 1], synth_test, title="Synthetic test candles (absolute)", bars=bars_to_plot)

    axes[0, 0].set_ylim(y_min_abs - pad_abs, y_max_abs + pad_abs)
    axes[0, 1].set_ylim(y_min_abs - pad_abs, y_max_abs + pad_abs)


    real_plot_reb = real_rebased.tail(bars_to_plot)
    synth_plot_reb = synth_rebased.tail(bars_to_plot)

    y_min_reb = float(min(real_plot_reb["bid_l"].min(), synth_plot_reb["bid_l"].min()))
    y_max_reb = float(max(real_plot_reb["bid_h"].max(), synth_plot_reb["bid_h"].max()))
    pad_reb = (y_max_reb - y_min_reb) * 0.05 if y_max_reb > y_min_reb else max(abs(y_max_reb) * 0.01, 1e-6)

    _draw_candles(axes[1, 0], real_rebased, title="Real test candles (rebased to 100)", bars=bars_to_plot)
    _draw_candles(axes[1, 1], synth_rebased, title="Synthetic test candles (rebased to 100)", bars=bars_to_plot)

    axes[1, 0].set_ylim(y_min_reb - pad_reb, y_max_reb + pad_reb)
    axes[1, 1].set_ylim(y_min_reb - pad_reb, y_max_reb + pad_reb)

    fig.suptitle("Real vs synthetic candlestick comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return McValidationArtifacts(
        summary_json=summary_path,
        chart_png=chart_path,
        real_parquet=real_path,
        synthetic_parquet=synth_path,
    )