from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Loads backend/.env
load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str
    raw_cache_dir: Path
    parquet_cache_dir: Path
    max_download_workers: int
    symbol: str
    price_scale: int


def get_settings() -> Settings:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Create backend/.env and set DATABASE_URL=...")

    raw_cache = Path(os.environ.get("RAW_CACHE_DIR", "../data/raw")).resolve()
    parquet_cache = Path(os.environ.get("PARQUET_CACHE_DIR", "../data/cache_parquet")).resolve()

    # Create cache directories if missing
    raw_cache.mkdir(parents=True, exist_ok=True)
    parquet_cache.mkdir(parents=True, exist_ok=True)

    return Settings(
        database_url=db_url,
        raw_cache_dir=raw_cache,
        parquet_cache_dir=parquet_cache,
        max_download_workers=int(os.environ.get("MAX_DOWNLOAD_WORKERS", "10")),
        symbol=os.environ.get("SYMBOL", "EURUSD"),
        price_scale=int(os.environ.get("PRICE_SCALE", "100000")),
    )