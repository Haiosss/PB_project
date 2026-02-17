from __future__ import annotations

from datetime import datetime
import pandas as pd
from sqlalchemy import select

from market_pipeline.db.session import get_session
from market_pipeline.db.models import Candle1m


def load_candles_1m_df(instrument_id: int, start: datetime, end: datetime) -> pd.DataFrame:

    #Loads candles for [start, end) into a pandas DataFrame sorted by ts_utc

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

    with get_session() as s:
        rows = s.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"])

    df = pd.DataFrame(rows, columns=["ts_utc", "bid_o", "bid_h", "bid_l", "bid_c", "bid_v"])
    return df