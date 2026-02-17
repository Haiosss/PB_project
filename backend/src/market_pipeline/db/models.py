from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from market_pipeline.db.base import Base


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    price_scale: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    candles_1m = relationship("Candle1m", back_populates="instrument")


class Candle1m(Base):
    __tablename__ = "candles_1m"
    __table_args__ = (
        UniqueConstraint("instrument_id", "ts_utc", name="uq_candles_1m_instrument_ts"),
        Index("ix_candles_1m_ts_utc", "ts_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), nullable=False)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    bid_o: Mapped[int] = mapped_column(Integer, nullable=False)
    bid_h: Mapped[int] = mapped_column(Integer, nullable=False)
    bid_l: Mapped[int] = mapped_column(Integer, nullable=False)
    bid_c: Mapped[int] = mapped_column(Integer, nullable=False)
    bid_v: Mapped[float | None] = mapped_column(Float, nullable=True)

    ask_o: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ask_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ask_l: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ask_c: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ask_v: Mapped[float | None] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="dukascopy")

    instrument = relationship("Instrument", back_populates="candles_1m")


class IngestLog(Base):
    __tablename__ = "ingest_log"
    __table_args__ = (
        Index("ix_ingest_log_symbol_day", "symbol", "day_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    day_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # ok/failed/skipped
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)