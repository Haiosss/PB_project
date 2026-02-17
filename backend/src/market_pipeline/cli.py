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

def main() -> None:
    app()


if __name__ == "__main__":
    main()
