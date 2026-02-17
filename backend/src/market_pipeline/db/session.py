from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from market_pipeline.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # helps detect broken connections
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# helper for scripts/CLI: with get_session() as s
def get_session():
    return SessionLocal()
