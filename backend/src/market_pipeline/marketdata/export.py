from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from sqlalchemy import select

from market_pipeline.db.session import get_session
from market_pipeline.db.models import Candle1m

from market_pipeline.marketdata.cleaning import clean_candles_df


def export_1m_to_parquet(
    instrument_id: int,
    symbol: str,
    d0: date,
    d1: date,
    out_dir: Path,
) -> Path:
    #Export candles to Parquet files partitioned by day
    #Structure:
    #  out_dir/candles_1m/symbol=EURUSD/day=YYYY-MM-DD/data.parquet
    #Returns the base export folder

    base = out_dir / "candles_1m" / f"symbol={symbol.upper()}"
    base.mkdir(parents=True, exist_ok=True)

    with get_session() as s:
        d = d0
        while d < d1:
            start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            end = start + timedelta(days=1)

            stmt = (
                select(
                    Candle1m.ts_utc,
                    Candle1m.bid_o,
                    Candle1m.bid_h,
                    Candle1m.bid_l,
                    Candle1m.bid_c,
                    Candle1m.bid_v,
                )
                .where(
                    Candle1m.instrument_id == instrument_id,
                    Candle1m.ts_utc >= start,
                    Candle1m.ts_utc < end,
                )
                .order_by(Candle1m.ts_utc)
            )

            rows = s.execute(stmt).all()
            if rows:
                df = pd.DataFrame(rows, columns=["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"])
                df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

                # CLEAN before caching
                df, _clean = clean_candles_df(df)

                if not df.empty:
                    day_dir = base / f"day={d.isoformat()}"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(day_dir / "data.parquet", index=False)

            d += timedelta(days=1)

    return base