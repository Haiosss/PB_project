from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
import lzma
import struct
from typing import Iterable


# Commonly used candle record format for Dukascopy candle files

CANDLE_FMT = ">5if"  # time + 4 ints + float volume
RECORD_SIZE = struct.calcsize(CANDLE_FMT)


@dataclass(frozen=True)
class CandleRowSide:
    ts_utc: datetime
    o: int
    h: int
    l: int
    c: int
    v: float | None


def parse_candles_1m_bi5(content: bytes, day: date) -> list[CandleRowSide]:
    raw = lzma.decompress(content)

    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    out: list[CandleRowSide] = []

    n = len(raw) // RECORD_SIZE

    for i in range(n):
        off = i * RECORD_SIZE
        sec, o, c, l, h, v = struct.unpack_from(CANDLE_FMT, raw, off)  # NOTE order!
        ts = base + timedelta(seconds=int(sec))
        out.append(CandleRowSide(ts_utc=ts, o=int(o), h=int(h), l=int(l), c=int(c), v=float(v)))

    return out