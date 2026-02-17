from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from market_pipeline.db.session import get_session
from market_pipeline.db.models import Instrument, Candle1m, IngestLog

from sqlalchemy import select, func


def ensure_instrument(symbol: str, price_scale: int) -> int:
    with get_session() as s:
        inst = s.execute(select(Instrument).where(Instrument.symbol == symbol)).scalar_one_or_none()
        if inst:
            return inst.id

        now = datetime.now(timezone.utc)
        inst = Instrument(symbol=symbol, price_scale=price_scale, created_at=now)
        s.add(inst)
        s.commit()
        s.refresh(inst)
        return inst.id


def write_ingest_log(symbol: str, day_utc: datetime, status: str, message: str | None = None) -> None:
    with get_session() as s:
        now = datetime.now(timezone.utc)
        log = IngestLog(symbol=symbol, day_utc=day_utc, status=status, message=message, created_at=now)
        s.add(log)
        s.commit()


def upsert_candles_1m(
    instrument_id: int,
    rows: list[dict],
) -> int:
    #rows: list of dicts matching Candle1m columns (except id)
    #Returns inserted row count

    if not rows:
        return 0

    stmt = insert(Candle1m).values(rows)
    # if same instrument_id+ts_utc exists, do nothing
    stmt = stmt.on_conflict_do_nothing(index_elements=["instrument_id", "ts_utc"])

    with get_session() as s:
        res = s.execute(stmt)
        s.commit()
        # res.rowcount can be -1 depending on driver
        return int(res.rowcount or 0)

def count_candles_in_range(instrument_id: int, start: datetime, end: datetime) -> int:
    with get_session() as s:
        q = (
            select(func.count())
            .select_from(Candle1m)
            .where(
                Candle1m.instrument_id == instrument_id,
                Candle1m.ts_utc >= start,
                Candle1m.ts_utc < end,
            )
        )
        return int(s.execute(q).scalar_one())
    
def count_candles_1m(instrument_id: int, start: datetime, end: datetime) -> int:
    with get_session() as s:
        q = (
            select(func.count())
            .select_from(Candle1m)
            .where(
                Candle1m.instrument_id == instrument_id,
                Candle1m.ts_utc >= start,
                Candle1m.ts_utc < end,
            )
        )
        return int(s.execute(q).scalar_one())