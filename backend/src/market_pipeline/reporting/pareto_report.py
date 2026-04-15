from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def build_pareto_report(
    *,
    pareto_result_json: Path,
    holdout_result_json: Path,
    output_pdf: Path,
) -> Path:
    pareto = json.loads(pareto_result_json.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_result_json.read_text(encoding="utf-8"))

    selected = pareto["selected_best"]
    params = selected["params"]
    agg = selected["aggregate"]
    per_fold = selected["per_fold_results"]
    pareto_front = pareto["pareto_front"]

    hold_metrics = holdout["historical_holdout_metrics"]
    hold_mc = holdout["holdout_mc_summary"]
    hold_mc_details = hold_mc.get("details", [])

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        # Page 1: summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")

        lines = [
            "Final Pareto Walk-Forward Report",
            "",
            f"Development period: {pareto['development_period'][0]} -> {pareto['development_period'][1]}",
            f"Holdout period: {holdout['holdout_period'][0]} -> {holdout['holdout_period'][1]}",
            f"Timeframe: {pareto['timeframe']}",
            f"Folds used: {pareto['folds_used']}",
            f"Pareto front size: {len(pareto_front)}",
            "",
            "Selected final parameter set:",
        ]
        for k, v in params.items():
            lines.append(f"  {k}: {v}")

        lines += [
            "",
            "Development weighted scores:",
            f"  weighted IS score: {_fmt(agg['weighted_is_score'])}",
            f"  weighted OOS score: {_fmt(agg['weighted_oos_score'])}",
            f"  weighted MC score: {_fmt(agg['weighted_mc_score'])}",
            f"  weighted IS/OOS gap: {_fmt(agg['weighted_is_oos_gap'])}",
            f"  weighted OOS objective: {_fmt(agg['weighted_oos_objective'])}",
            f"  weighted MC objective: {_fmt(agg['weighted_mc_objective'])}",
            f"  Pareto selection score: {_fmt(selected['selection_score'])}",
            "",
            "Final holdout results:",
            f"  holdout return %: {_fmt(hold_metrics['total_return_pct'])}",
            f"  holdout max drawdown %: {_fmt(hold_metrics['max_drawdown_pct'])}",
            f"  holdout Sharpe: {_fmt(hold_metrics['sharpe'])}",
            f"  holdout winrate %: {_fmt(hold_metrics['winrate_pct'])}",
            f"  holdout profit factor: {_fmt(hold_metrics['profit_factor'])}",
            "",
            "Holdout Monte Carlo (300 paths):",
            f"  median return %: {_fmt(hold_mc['median_return_pct'])}",
            f"  q05 return %: {_fmt(hold_mc['q05_return_pct'])}",
            f"  median max drawdown %: {_fmt(hold_mc['median_max_drawdown_pct'])}",
            f"  prob. loss: {_fmt(hold_mc['prob_loss'])}",
            f"  prob. PF > 1: {_fmt(hold_mc['prob_pf_gt_1'])}",
            f"  median trades: {_fmt(hold_mc['median_trades'])}",
            f"  MC score: {_fmt(holdout['holdout_mc_score'])}",
            "",
            f"Final selection score on holdout: {_fmt(holdout['final_selection_score'])}",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Pareto front
        fig, ax = plt.subplots(figsize=(10, 6))
        xs = [row["aggregate"]["weighted_oos_objective"] for row in pareto_front]
        ys = [row["aggregate"]["weighted_mc_objective"] for row in pareto_front]
        ax.scatter(xs, ys)
        ax.set_xlabel("Weighted OOS objective")
        ax.set_ylabel("Weighted MC objective")
        ax.set_title("Pareto Front")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: fold-by-fold development scores
        fig, ax = plt.subplots(figsize=(11, 6))
        fold_idx = [row["fold_index"] for row in per_fold]
        is_scores = [row["is_score"] for row in per_fold]
        oos_scores = [row["oos_score"] for row in per_fold]
        mc_scores = [row["mc_score"] for row in per_fold]

        ax.plot(fold_idx, is_scores, marker="o", label="IS score")
        ax.plot(fold_idx, oos_scores, marker="o", label="OOS score")
        ax.plot(fold_idx, mc_scores, marker="o", label="MC score")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_title("Selected Final Parameter Set Across Development Folds")
        ax.legend()
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: holdout equity curve
        equity_path = holdout.get("equity_parquet")
        if equity_path:
            equity_df = pd.read_parquet(equity_path)
            if "equity" in equity_df.columns:
                fig, ax = plt.subplots(figsize=(11, 6))
                x = equity_df["ts_utc"] if "ts_utc" in equity_df.columns else range(len(equity_df))
                ax.plot(x, equity_df["equity"])
                ax.set_title("Holdout Equity Curve")
                ax.set_xlabel("Time")
                ax.set_ylabel("Equity")
                ax.grid(True, alpha=0.3)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        # Page 5: MC returns histogram
        if hold_mc_details:
            df_mc = pd.DataFrame(hold_mc_details)

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.hist(df_mc["total_return_pct"], bins=30)
            ax.set_title("Monte Carlo Distribution: Holdout Total Return %")
            ax.set_xlabel("Total Return %")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 6: MC drawdown histogram
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.hist(df_mc["max_drawdown_pct"], bins=30)
            ax.set_title("Monte Carlo Distribution: Holdout Max Drawdown %")
            ax.set_xlabel("Max Drawdown %")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Page 7: MC scatter return vs drawdown
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.scatter(df_mc["max_drawdown_pct"], df_mc["total_return_pct"])
            ax.set_title("Monte Carlo: Return vs Max Drawdown")
            ax.set_xlabel("Max Drawdown %")
            ax.set_ylabel("Total Return %")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output_pdf