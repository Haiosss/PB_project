from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx


def _month_0_based(d: date) -> int:
    # Dukascopy uses 0-based month folders: Jan=00, Feb=01, ...
    return d.month - 1


def build_candle_1m_url(symbol: str, d: date, side: str) -> str:
    mm = _month_0_based(d)
    # Example:
    # https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/15/BID_candles_min_1.bi5
    return (
        f"https://datafeed.dukascopy.com/datafeed/"
        f"{symbol.upper()}/{d.year:04d}/{mm:02d}/{d.day:02d}/"
        f"{side.upper()}_candles_min_1.bi5"
    )


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    status: str  # "ok", "missing", "error"
    http_status: int | None = None
    message: str | None = None


async def download_bi5_to_path(
    symbol: str,
    d: date,
    side: str,
    out_path: Path,
    timeout_s: float = 30.0,
) -> DownloadResult:
    url = build_candle_1m_url(symbol, d, side)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(url)

        if r.status_code == 404:
            return DownloadResult(path=out_path, status="missing", http_status=404)

        r.raise_for_status()
        out_path.write_bytes(r.content)
        return DownloadResult(path=out_path, status="ok", http_status=r.status_code)

    except Exception as e:
        return DownloadResult(path=out_path, status="error", message=str(e))