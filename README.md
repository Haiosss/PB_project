# PB_project
Testing trading strategies using Monte Carlo simulation and hyperparameter optimization

## Infrastructure / setup commands:

#### # docker compose up -d
Starts PostgreSQL in Docker (from infra/)

#### # docker compose down
Stops PostgreSQL container

#### # docker compose down -v
Stops PostgreSQL and deletes DB volume (full reset)

#### # alembic upgrade head
Applies DB migrations (creates/updates tables)


## Main CLI commands (python -m market_pipeline.cli)

### Database and connectivity

#### # python -m market_pipeline.cli db-check
Checks DB connection, prints Postgres version, and lists tables

### Ingestion (Dukascopy → Postgres)

#### # python -m market_pipeline.cli ingest-day 2024-01-16
Downloads and ingests one day of 1-minute EURUSD BID candles into Postgres

#### # python -m market_pipeline.cli ingest-range 2024-01-15 2024-01-20
Downloads and ingests a date range [date_from, date_to) into Postgres

### Validation of raw 1m data (Postgres)

#### # python -m market_pipeline.cli validate-day 2024-01-16
Validates one day of 1m data (duplicates, missing minutes, OHLC correctness)

#### # python -m market_pipeline.cli validate-range 2024-01-15 2024-01-20
Validates a date range of 1m data and prints aggregated diagnostics

### Export 1m cache to Parquet (cleaned cache)

#### # python -m market_pipeline.cli export-parquet 2024-01-15 2024-01-20
Exports 1m candles from Postgres to cleaned Parquet cache (partitioned by day)

### Resampling (1m → higher timeframe)

#### # python -m market_pipeline.cli resample-range 2024-01-15 2024-01-20 5min
#### # python -m market_pipeline.cli resample-range 2024-01-15 2024-01-20 12min
#### # python -m market_pipeline.cli resample-range 2024-01-15 2024-01-20 15min ...
Resamples cleaned 1m candles into a chosen timeframe and saves cleaned Parquet cache

### Validation of resampled data (Parquet)

#### # python -m market_pipeline.cli validate-resampled-range 2024-01-15 2024-01-20 15min
Validates resampled candles (duplicates, gaps, OHLC correctness, expected bars/day)

### Unified loader inspection (debug / sanity check)

#### # python -m market_pipeline.cli inspect-range 2024-01-15 2024-01-20 1m
#### # python -m market_pipeline.cli inspect-range 2024-01-15 2024-01-20 15min --as-float
Loads candles through the unified loader and prints row count, timestamps, columns, and sample rows

### Feature engineering (indicators)

#### # python -m market_pipeline.cli build-features 2024-01-15 2024-01-20 15min
Builds and caches feature dataset (returns, EMA, RSI, ATR, MACD histogram) for the chosen timeframe

#### # python -m market_pipeline.cli build-features 2024-01-15 2024-01-20 15min --ema-periods 9,21,50
Same as above, but with custom EMA periods

### Cleaning diagnostics

#### # python -m market_pipeline.cli cleaning-report-range 2024-01-15 2024-01-20 1m
Reports what the cleaning process would remove (duplicates, invalid OHLC, inactive bars)  
For 1m, it reports cleaning needed on raw DB data

#### # python -m market_pipeline.cli validate-cleaning 2024-01-15 2024-01-20 1m
#### # python -m market_pipeline.cli validate-cleaning 2024-01-15 2024-01-20 5min
Checks whether cached Parquet data is already clean (should ideally report zero rows dropped if re-cleaned)