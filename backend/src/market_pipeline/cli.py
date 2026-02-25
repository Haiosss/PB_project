from __future__ import annotations
from pathlib import Path

import typer
from sqlalchemy import inspect, text

from market_pipeline.config import get_settings
from market_pipeline.db.session import engine

from datetime import date, datetime, timezone
import asyncio

from market_pipeline.marketdata.dukascopy.client import download_bi5_to_path
from market_pipeline.marketdata.dukascopy.parser import parse_candles_1m_bi5
from market_pipeline.marketdata.repository import ensure_instrument, upsert_candles_1m, write_ingest_log, count_candles_in_range, count_candles_1m

from datetime import timedelta

from dataclasses import dataclass

from market_pipeline.marketdata.queries import load_candles_1m_df
from market_pipeline.marketdata.validate import validate_day_1m

from market_pipeline.marketdata.diagnostics import validate_range_1m

from market_pipeline.marketdata.export import export_1m_to_parquet

from market_pipeline.marketdata.resample import export_resampled_range_to_parquet

from market_pipeline.marketdata.validate_resampled import validate_resampled_range

from market_pipeline.marketdata.loaders import load_range

from market_pipeline.marketdata.features import build_basic_features, save_features_range_parquet

from market_pipeline.marketdata.cleaning import clean_candles_df

from market_pipeline.marketdata.cleaning_reports import build_cleaning_report_range, build_validate_cleaning_cache_range

app = typer.Typer(help="Market pipeline CLI")


@app.callback()
def _root() -> None:
    # Market pipeline CLI
    pass


@app.command()
def db_check() -> None:
    # Check DB connectivity and list tables
    settings = get_settings()
    typer.echo(f"DATABASE_URL: {settings.database_url}")

    with engine.connect() as conn:
        v = conn.execute(text("select version()")).scalar_one()
        typer.echo(f"Postgres: {v}")

    tables = inspect(engine).get_table_names()
    typer.echo(f"Tables: {tables}")
    
@app.command("ingest-day")
def ingest_day(day: str) -> None:

    #Download + parse + insert one day of EURUSD 1m BID candles

    settings = get_settings()
    d = date.fromisoformat(day)

    # raw cache path
    base = (
        settings.raw_cache_dir
        / "dukascopy"
        / settings.symbol
        / f"{d.year:04d}"
        / f"{(d.month - 1):02d}"
        / f"{d.day:02d}"
    )
    bid_path = base / "BID_candles_min_1.bi5"

    async def _run():
        return await download_bi5_to_path(settings.symbol, d, "BID", bid_path)

    bid_res = asyncio.run(_run())

    day_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    if bid_res.status != "ok":
        write_ingest_log(
            settings.symbol,
            day_dt,
            "failed",
            f"BID download: {bid_res.status} {bid_res.message or ''}".strip(),
        )
        raise typer.Exit(code=1)

    # parse BID candles
    bid_rows = parse_candles_1m_bi5(bid_path.read_bytes(), d)

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    # build DB rows (ASK price = NULL)
    rows_for_db: list[dict] = []
    for b in bid_rows:
        rows_for_db.append(
            dict(
                instrument_id=instrument_id,
                ts_utc=b.ts_utc,
                bid_o=b.o,
                bid_h=b.h,
                bid_l=b.l,
                bid_c=b.c,
                bid_v=b.v,
                source="dukascopy",
            )
        )

    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    
    before = count_candles_1m(instrument_id, start, end)
    upsert_candles_1m(instrument_id, rows_for_db)
    after = count_candles_1m(instrument_id, start, end)
    
    inserted = after - before
    
    write_ingest_log(settings.symbol, start, "ok", f"Inserted {inserted} rows. Total now {after}. BID={len(bid_rows)}")
    typer.echo(f"Done. Inserted {inserted} rows. Total now {after}. BID={len(bid_rows)}")

    inserted = upsert_candles_1m(instrument_id, rows_for_db)

    write_ingest_log(
        settings.symbol,
        day_dt,
        "ok",
        f"Inserted approx {inserted} rows. BID={len(bid_rows)}",
    )

    typer.echo(f"Done. Inserted approx {inserted} rows. BID={len(bid_rows)}")
    
@dataclass(frozen=True)
class _DayDownload:
    day: date
    path: str
    status: str
    message: str | None


@app.command("ingest-range")
def ingest_range(date_from: str, date_to: str) -> None:
    # Download+ingest a date range (date_from, date_to) in UTC
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    # Build list of days
    days: list[date] = []
    d = d0
    while d < d1:
        days.append(d)
        d += timedelta(days=1)

    typer.echo(f"Range days: {len(days)} (from {d0} to {d1}, exclusive)")

    sem = asyncio.Semaphore(settings.max_download_workers)

    async def _download_one(d: date) -> _DayDownload:
        base = (
            settings.raw_cache_dir
            / "dukascopy"
            / settings.symbol
            / f"{d.year:04d}"
            / f"{(d.month - 1):02d}"
            / f"{d.day:02d}"
        )
        bid_path = base / "BID_candles_min_1.bi5"

        # Skip download if already cached
        if bid_path.exists() and bid_path.stat().st_size > 0:
            return _DayDownload(d, str(bid_path), "cached", None)

        async with sem:
            res = await download_bi5_to_path(settings.symbol, d, "BID", bid_path)

        return _DayDownload(d, str(res.path), res.status, res.message)

    async def _download_all() -> list[_DayDownload]:
        tasks = [asyncio.create_task(_download_one(d)) for d in days]
        return await asyncio.gather(*tasks)

    downloads = asyncio.run(_download_all())

    ok = sum(1 for x in downloads if x.status in ("ok", "cached"))
    miss = sum(1 for x in downloads if x.status == "missing")
    err = sum(1 for x in downloads if x.status == "error")
    typer.echo(f"Download summary: ok/cached={ok}, missing={miss}, error={err}")

    # Ingest day-by-day
    for item in downloads:
        d = item.day
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        if item.status in ("missing", "error"):
            write_ingest_log(settings.symbol, start, "skipped", f"download {item.status}: {item.message or ''}".strip())
            typer.echo(f"{d} -> skipped (download {item.status})")
            continue

        path = item.path
        try:
            bid_rows = parse_candles_1m_bi5(Path(path).read_bytes(), d)

            rows_for_db = [
                dict(
                    instrument_id=instrument_id,
                    ts_utc=b.ts_utc,
                    bid_o=b.o,
                    bid_h=b.h,
                    bid_l=b.l,
                    bid_c=b.c,
                    bid_v=b.v,
                    source="dukascopy",
                )
                for b in bid_rows
            ]

            before = count_candles_1m(instrument_id, start, end)
            upsert_candles_1m(instrument_id, rows_for_db)
            after = count_candles_1m(instrument_id, start, end)
            inserted = after - before

            write_ingest_log(settings.symbol, start, "ok", f"Inserted {inserted}. Total now {after}.")
            typer.echo(f"{d} -> inserted {inserted} (total {after})")

        except Exception as e:
            write_ingest_log(settings.symbol, start, "failed", str(e))
            typer.echo(f"{d} -> FAILED: {e}")

@app.command("validate-day")
def validate_day(day: str) -> None:

    # Validate one UTC day of 1m BID candles

    settings = get_settings()
    d = date.fromisoformat(day)

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    df = load_candles_1m_df(instrument_id, start, end)
    res = validate_day_1m(df)

    typer.echo(f"Symbol: {settings.symbol} Day(UTC): {day}")
    typer.echo(f"Rows: {res.rows}")
    typer.echo(f"Duplicates: {res.duplicates}")
    typer.echo(f"Missing minutes (internal gaps): {res.missing_minutes}")
    typer.echo(f"OHLC violations: {res.ohlc_violations}")
    typer.echo(f"First ts: {res.first_ts}")
    typer.echo(f"Last ts: {res.last_ts}")

@app.command("validate-range")
def validate_range(date_from: str, date_to: str) -> None:

    # Validate daily 1m BID candle range
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    df_days, summary = validate_range_1m(instrument_id, d0, d1)

    typer.echo(f"Symbol: {settings.symbol} Range: [{d0}, {d1}) UTC")
    typer.echo(f"Days checked: {summary.days_checked}")
    typer.echo(f"Days with data: {summary.days_with_data}")
    typer.echo(f"Days empty: {summary.days_empty}")
    typer.echo(f"Total rows: {summary.total_rows}")
    typer.echo(f"Total missing minutes: {summary.total_missing_minutes}")
    typer.echo(f"Total duplicates: {summary.total_duplicates}")
    typer.echo(f"Total OHLC violations: {summary.total_ohlc_violations}")

    # Save diagnostics to Parquet
    out_path = settings.parquet_cache_dir / "diagnostics" / f"validate_1m_{d0}_{d1}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_days.to_parquet(out_path, index=False)

    typer.echo(f"Saved daily diagnostics to: {out_path}")


@app.command("export-parquet")
def export_parquet(date_from: str, date_to: str) -> None:

    # Export 1m BID candles range from DB to Parquet cache

    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    base = export_1m_to_parquet(
        instrument_id=instrument_id,
        symbol=settings.symbol,
        d0=d0,
        d1=d1,
        out_dir=settings.parquet_cache_dir,
    )

    typer.echo(f"Exported to: {base}")
    
@app.command("resample-range")
def resample_range(date_from: str, date_to: str, timeframe: str) -> None:
    # Resample 1m BID candles from DB to a higher timeframe and save parquet cache
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    try:
        result = export_resampled_range_to_parquet(
            instrument_id=instrument_id,
            symbol=settings.symbol,
            d0=d0,
            d1=d1,
            timeframe=timeframe,
            out_dir=settings.parquet_cache_dir,
        )
    except ValueError as e:
        raise typer.BadParameter(f"Invalid timeframe '{timeframe}': {e}") from e

    # export summary
    typer.echo(f"Resampled timeframe: {timeframe}")
    typer.echo(f"Days written: {result.days_written}")
    typer.echo(f"Total resampled rows: {result.total_rows}")
    typer.echo(f"Saved to: {result.base_path}")

    # cleaning summary
    def _has_drops(s) -> bool:
        return (
            s.dropped_duplicates
            or s.dropped_null_ohlc
            or s.dropped_invalid_ohlc
            or s.dropped_inactive
        )

    total_1m_inactive = 0
    total_1m_dup = 0
    total_1m_null = 0
    total_1m_invalid = 0

    total_tf_inactive = 0
    total_tf_dup = 0
    total_tf_null = 0
    total_tf_invalid = 0

    for log in result.cleaning_logs:
        c1 = log.clean_1m
        ct = log.clean_tf

        total_1m_inactive += c1.dropped_inactive
        total_1m_dup += c1.dropped_duplicates
        total_1m_null += c1.dropped_null_ohlc
        total_1m_invalid += c1.dropped_invalid_ohlc

        total_tf_inactive += ct.dropped_inactive
        total_tf_dup += ct.dropped_duplicates
        total_tf_null += ct.dropped_null_ohlc
        total_tf_invalid += ct.dropped_invalid_ohlc

    typer.echo("")
    typer.echo("Cleaning summary (aggregate):")
    typer.echo(
        f"  1m source: dropped_inactive={total_1m_inactive}, "
        f"dropped_dup={total_1m_dup}, dropped_null={total_1m_null}, dropped_invalid={total_1m_invalid}"
    )
    typer.echo(
        f"  {timeframe} resampled: dropped_inactive={total_tf_inactive}, "
        f"dropped_dup={total_tf_dup}, dropped_null={total_tf_null}, dropped_invalid={total_tf_invalid}"
    )

    # per-day logs only if something was dropped
    printed_any = False
    for log in result.cleaning_logs:
        c1 = log.clean_1m
        ct = log.clean_tf

        if not (_has_drops(c1) or _has_drops(ct)):
            continue

        printed_any = True
        typer.echo(
            f"- {log.day_utc.isoformat()} | "
            f"1m(in={c1.rows_in}, out={c1.rows_out}, drop_inactive={c1.dropped_inactive}, "
            f"drop_dup={c1.dropped_duplicates}, drop_null={c1.dropped_null_ohlc}, drop_invalid={c1.dropped_invalid_ohlc}) | "
            f"{timeframe}(in={ct.rows_in}, out={ct.rows_out}, rows_written={log.tf_rows_written}, "
            f"drop_inactive={ct.dropped_inactive}, drop_dup={ct.dropped_duplicates}, "
            f"drop_null={ct.dropped_null_ohlc}, drop_invalid={ct.dropped_invalid_ohlc})"
        )

    if not printed_any:
        typer.echo("No per-day cleaning events (nothing was removed).")
    
@app.command("validate-resampled-range")
def validate_resampled_range_cmd(date_from: str, date_to: str, timeframe: str) -> None:
  
    #validate resampled parquet candles for [date_from, date_to)
   
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    df_days, summary = validate_resampled_range(
        base_cache_dir=settings.parquet_cache_dir,
        symbol=settings.symbol,
        timeframe=timeframe,
        d0=d0,
        d1=d1,
    )

    typer.echo(f"Symbol: {settings.symbol}")
    typer.echo(f"Timeframe: {timeframe}")
    typer.echo(f"Range: [{d0}, {d1}) UTC")
    typer.echo(f"Days checked: {summary['days_checked']}")
    typer.echo(f"Days with data: {summary['days_with_data']}")
    typer.echo(f"Days empty: {summary['days_empty']}")
    typer.echo(f"Total rows: {summary['total_rows']}")
    typer.echo(f"Total duplicates: {summary['total_duplicates']}")
    typer.echo(f"Total missing bars (internal gaps): {summary['total_missing_bars']}")
    typer.echo(f"Total OHLC violations: {summary['total_ohlc_violations']}")
    typer.echo(f"Days with expected-bars mismatch: {summary['expected_mismatch_days']}")

    out_path = (
        settings.parquet_cache_dir
        / "diagnostics"
        / f"validate_resampled_{timeframe}_{d0}_{d1}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_days.to_parquet(out_path, index=False)

    typer.echo(f"Saved diagnostics to: {out_path}")
    
@app.command("inspect-range")
def inspect_range(
    date_from: str,
    date_to: str,
    timeframe: str = typer.Argument("1m"),
    as_float: bool = typer.Option(False, "--as-float"),
) -> None:

    # Load candles and print a quick summary
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    df = load_range(
        instrument_id=instrument_id,
        symbol=settings.symbol,
        timeframe=timeframe,
        d0=d0,
        d1=d1,
        parquet_cache_dir=settings.parquet_cache_dir,
        price_scale=settings.price_scale,
        as_float_prices=as_float,
    )

    typer.echo(f"Symbol: {settings.symbol}")
    typer.echo(f"Timeframe: {timeframe}")
    typer.echo(f"Range: [{d0}, {d1}) UTC")
    typer.echo(f"Rows: {len(df)}")
    typer.echo(f"Columns: {list(df.columns)}")

    if df.empty:
        typer.echo("DataFrame is empty.")
        return

    typer.echo(f"First ts: {df['ts_utc'].iloc[0]}")
    typer.echo(f"Last ts:  {df['ts_utc'].iloc[-1]}")
    typer.echo("")
    typer.echo("Head (first 5 rows):")
    typer.echo(df.head(5).to_string(index=False))
    
@app.command("build-features")
def build_features_cmd(
    date_from: str,
    date_to: str,
    timeframe: str = typer.Argument("15min"),
    ema_periods: str = typer.Option("20,50,200", help="Comma-separated EMA periods, e.g. 20,50,200"),
    rsi_period: int = typer.Option(14, help="RSI period"),
    atr_period: int = typer.Option(14, help="ATR period"),
    macd_fast: int = typer.Option(12, help="MACD fast EMA period"),
    macd_slow: int = typer.Option(26, help="MACD slow EMA period"),
    macd_signal: int = typer.Option(9, help="MACD signal EMA period"),
) -> None:

    # Build and cache feature set for a date range + timeframe.
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    try:
        ema_list = [int(x.strip()) for x in ema_periods.split(",") if x.strip()]
        if not ema_list:
            raise ValueError
    except ValueError:
        raise typer.BadParameter("ema_periods must be comma-separated integers, e.g. 20,50,200")

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    # Load candles as FLOAT prices
    df = load_range(
        instrument_id=instrument_id,
        symbol=settings.symbol,
        timeframe=timeframe,
        d0=d0,
        d1=d1,
        parquet_cache_dir=settings.parquet_cache_dir,
        price_scale=settings.price_scale,
        as_float_prices=True,
    )
    
    df, clean_summary = clean_candles_df(df)

    typer.echo(
        f"Cleaning: in={clean_summary.rows_in}, out={clean_summary.rows_out}, "
        f"dropped_inactive={clean_summary.dropped_inactive}, "
        f"dropped_dup={clean_summary.dropped_duplicates}, "
        f"dropped_null_ohlc={clean_summary.dropped_null_ohlc}, "
        f"dropped_invalid_ohlc={clean_summary.dropped_invalid_ohlc}"
    )

    if df.empty:
        typer.echo("No candle data found for this range/timeframe.")
        raise typer.Exit(code=1)

    feat_df = build_basic_features(
        df,
        ema_periods=ema_list,
        rsi_period=rsi_period,
        atr_period=atr_period,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    )

    out_path = save_features_range_parquet(
        feat_df,
        out_dir=settings.parquet_cache_dir,
        symbol=settings.symbol,
        timeframe=timeframe,
        date_from=d0,
        date_to=d1,
        feature_set_name="basic_v1",
    )

    typer.echo(f"Built features for {settings.symbol} {timeframe} [{d0}, {d1})")
    typer.echo(f"Rows: {len(feat_df)}")
    typer.echo(f"Columns: {len(feat_df.columns)}")
    typer.echo(f"Saved to: {out_path}")

    # Show some key columns if present
    show_cols = ["ts_utc", "bid_c", "ret_pct"]
    for c in feat_df.columns:
        if c.startswith("ema_"):
            show_cols.append(c)
    for c in feat_df.columns:
        if c.startswith("rsi_"):
            show_cols.append(c)
    for c in feat_df.columns:
        if c.startswith("atr_") and not c.endswith("_pct"):
            show_cols.append(c)
    for c in feat_df.columns:
        if c.startswith("macd_hist_"):
            show_cols.append(c)

    # Keep unique order
    show_cols = [c for i, c in enumerate(show_cols) if c in feat_df.columns and c not in show_cols[:i]]

    typer.echo("")
    typer.echo("Preview:")
    typer.echo(feat_df[show_cols].head(10).to_string(index=False))
    
@app.command("cleaning-report-range")
def cleaning_report_range_cmd(date_from: str, date_to: str, timeframe: str = typer.Argument("1m")) -> None:
  
    # Report what cleaning removes over a range
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    instrument_id = ensure_instrument(settings.symbol, settings.price_scale)

    df_days, s = build_cleaning_report_range(
        instrument_id=instrument_id,
        base_cache_dir=settings.parquet_cache_dir,
        symbol=settings.symbol,
        timeframe=timeframe,
        d0=d0,
        d1=d1,
    )

    typer.echo(f"Cleaning report | Symbol={settings.symbol} Timeframe={timeframe} Range=[{d0}, {d1})")
    typer.echo(f"Days checked: {s['days_checked']}")
    typer.echo(f"Days with data: {s['days_with_data']}")
    typer.echo(f"Days empty: {s['days_empty']}")
    typer.echo(f"Rows in: {s['rows_in']}")
    typer.echo(f"Rows out (after cleaning): {s['rows_out']}")
    typer.echo(f"Dropped duplicates: {s['dropped_duplicates']}")
    typer.echo(f"Dropped null OHLC: {s['dropped_null_ohlc']}")
    typer.echo(f"Dropped invalid OHLC: {s['dropped_invalid_ohlc']}")
    typer.echo(f"Dropped inactive bars: {s['dropped_inactive']}")
    typer.echo(f"Expected-bars mismatch days (after cleaning): {s['expected_mismatch_days']}")

    out_path = (
        settings.parquet_cache_dir
        / "diagnostics"
        / f"cleaning_report_{timeframe}_{d0}_{d1}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_days.to_parquet(out_path, index=False)
    typer.echo(f"Saved report to: {out_path}")
    
@app.command("validate-cleaning")
def validate_cleaning_cmd(date_from: str, date_to: str, timeframe: str = typer.Argument("1m")) -> None:
 
    # Validate that cached parquet candles are already clean
    
    settings = get_settings()
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)

    if d1 <= d0:
        raise typer.BadParameter("date_to must be after date_from")

    df_days, s = build_validate_cleaning_cache_range(
        base_cache_dir=settings.parquet_cache_dir,
        symbol=settings.symbol,
        timeframe=timeframe,
        d0=d0,
        d1=d1,
    )

    typer.echo(f"Validate cleaning | Symbol={settings.symbol} Timeframe={timeframe} Range=[{d0}, {d1})")
    typer.echo(f"Days checked: {s['days_checked']}")
    typer.echo(f"Days with data: {s['days_with_data']}")
    typer.echo(f"Days empty: {s['days_empty']}")
    typer.echo(f"Rows cached: {s['rows_in']}")
    typer.echo(f"Days not clean: {s['not_clean_days']}")
    typer.echo(f"Total rows dropped if re-cleaned: {s['total_dropped_if_recleaned']}")
    typer.echo(f"Would-drop duplicates: {s['dropped_duplicates']}")
    typer.echo(f"Would-drop null OHLC: {s['dropped_null_ohlc']}")
    typer.echo(f"Would-drop invalid OHLC: {s['dropped_invalid_ohlc']}")
    typer.echo(f"Would-drop inactive bars: {s['dropped_inactive']}")
    typer.echo(f"Expected-bars mismatch days: {s['expected_mismatch_days']}")

    out_path = (
        settings.parquet_cache_dir
        / "diagnostics"
        / f"validate_cleaning_{timeframe}_{d0}_{d1}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_days.to_parquet(out_path, index=False)
    typer.echo(f"Saved validation to: {out_path}")

def main() -> None:
    app()


if __name__ == "__main__":
    main()
